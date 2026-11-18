"""Shared ``Detector`` for the Z-PEFT-style spectral detector: pure inference on a frozen JSON model.

Score = logit of P(malicious) = coef . standardised features + intercept; larger = more malicious. Every adapter file's bytes
are checked against the manifest SHA256 before it is read. Features are extracted live (about 1 s per adapter on CPU).
"""

from __future__ import annotations

from ...manifest import ProtocolError, digest
from ...scoring import AdapterInput, DetectorSpec, ProtocolBinding, finite_score
from ..peftguard.converter import default_resolver, sha256_file
from .model import FrozenZPeft, load_model
from .train import ZPeftConfig, features_for, source_sha256


class ZPeftDetector:
    def __init__(self, model: FrozenZPeft, spec: DetectorSpec, cfg: ZPeftConfig, resolver, cache_dir):
        self._model, self._spec, self._cfg, self._resolver, self._cache = model, spec, cfg, resolver, cache_dir

    @property
    def spec(self) -> DetectorSpec:
        return self._spec

    @classmethod
    def from_model(cls, model_path, binding: ProtocolBinding, *, expected_file_sha256: str,
                   resolver=default_resolver, cache_dir=None):
        """Reload a frozen model. ``expected_file_sha256`` must come from a trusted record of the fit."""
        file_sha = sha256_file(model_path)
        if file_sha != expected_file_sha256:
            raise ProtocolError("Model file bytes differ from the expected SHA256")
        model = load_model(model_path)
        body = model.body
        if (body["protocol_sha256"] != binding.sha256
                or body["train_records_sha256"] != binding.records_digest("train")
                or body["validation_records_sha256"] != binding.records_digest("validation")):
            raise ProtocolError("Model belongs to a different manifest/split")
        if tuple(body["fit_ids"]) != binding.ids("train") or body["tuning_ids"] != []:
            raise ProtocolError("Fit IDs must equal the complete shared train partition and no validation ID may tune it")
        cfg = ZPeftConfig(**body["config"])
        if body["feature_dim"] != cfg.n_features:
            raise ProtocolError("Model feature size does not match its configuration")
        config = {"extractor": body["extractor"], "config": body["config"], "score": "logit_decision_function",
                  "classifier": {k: body["classifier"][k] for k in ("C", "class_weight", "classes")}}
        chosen = body["selection"]["chosen"]
        spec = DetectorSpec(
            "zpeft_style_llama2_7b_qv_head16_lr_v1", source_sha256(), digest(config), file_sha,
            "higher_is_suspicious", tuple(body["fit_ids"]), (),
            f"Z-PEFT-style reimplementation (paper has no public code). Fitted on the shared train split only; C and "
            f"class weight chosen by grouped {body['selection']['folds']}-fold CV on train (C={chosen['C']}, "
            f"cw={chosen['class_weight']}, CV AUROC {chosen['mean_auc']:.3f}); validation unused for selection; "
            f"no external pretraining or reference bank")
        return cls(model, spec, cfg, resolver, cache_dir)

    def score(self, adapters: tuple[AdapterInput, ...]) -> dict[str, float]:
        if type(adapters) is not tuple:
            raise ProtocolError("Score an immutable tuple of AdapterInput")
        result = {}
        for adapter in adapters:
            if adapter.lora_rank != self._cfg.rank or adapter.target_modules != ("q_proj", "v_proj"):
                raise ProtocolError(f"{adapter.adapter_id}: needs rank {self._cfg.rank} and exactly q_proj+v_proj")
            path = self._resolver(adapter)
            if sha256_file(path) != adapter.sha256:
                raise ProtocolError(f"{adapter.adapter_id}: file bytes differ from the manifest SHA256")
            x = features_for(path, adapter.sha256, self._cfg, self._cache)
            result[adapter.adapter_id] = finite_score(self._model.raw_score(x))
        return result
