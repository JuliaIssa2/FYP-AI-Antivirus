"""Shared ``Detector`` for the clean-only one-class Mahalanobis detector: pure inference on a frozen JSON model.

Score = squared Mahalanobis distance to the clean-train population (larger = more suspicious). Every adapter file's bytes
are checked against the manifest SHA256 before it is read.
"""

from __future__ import annotations

from ...manifest import ProtocolError, digest
from ...scoring import AdapterInput, DetectorSpec, ProtocolBinding, finite_score
from ..peftguard.converter import default_resolver, sha256_file
from .model import FrozenMahalanobis, load_model
from .train import OneClassConfig, benign_train_ids, features_for, source_sha256


class OneClassDetector:
    def __init__(self, model: FrozenMahalanobis, spec: DetectorSpec, cfg: OneClassConfig, resolver, cache_dir):
        self._model, self._spec, self._cfg, self._resolver, self._cache = model, spec, cfg, resolver, cache_dir

    @property
    def spec(self) -> DetectorSpec:
        return self._spec

    @classmethod
    def from_model(cls, model_path, binding: ProtocolBinding, train_records, *, expected_file_sha256: str,
                   resolver=default_resolver, cache_dir=None):
        """Reload a frozen model. ``train_records`` is the verified development view of the train partition: the binding
        does not carry labels, so it is needed to prove the model used exactly the benign train adapters."""
        file_sha = sha256_file(model_path)
        if file_sha != expected_file_sha256:
            raise ProtocolError("Model file bytes differ from the expected SHA256")
        model = load_model(model_path)
        body = model.body
        if (body["protocol_sha256"] != binding.sha256
                or body["train_records_sha256"] != binding.records_digest("train")):
            raise ProtocolError("Model belongs to a different manifest/split")
        if tuple(body["fit_ids"]) != benign_train_ids(binding, train_records) or body["tuning_ids"] != []:
            raise ProtocolError("Fit IDs must equal exactly the benign adapters of the shared train partition")
        cfg = OneClassConfig(**body["config"])
        config = {"extractor": body["extractor"], "config": body["config"], "kind": body["kind"],
                  "score": "squared_mahalanobis", "shrinkage": body["shrinkage"]}
        spec = DetectorSpec(
            "oneclass_mahalanobis_spectral30_llama2_7b_qv_v1", source_sha256(), digest(config), file_sha,
            "higher_is_suspicious", tuple(body["fit_ids"]), (),
            f"One-class: Ledoit-Wolf Mahalanobis on 30 all-layer spectral statistics, fitted on the {body['n_fit']} "
            f"benign (clean + decoy) adapters of the shared train split only; backdoored adapters never used; no "
            f"hyper-parameter tuned; validation unused; no external pretraining or reference bank")
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
            result[adapter.adapter_id] = finite_score(
                self._model.raw_score(features_for(path, adapter.sha256, self._cfg, self._cache)))
        return result
