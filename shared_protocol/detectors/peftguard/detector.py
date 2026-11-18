"""Shared ``Detector`` for PEFTGuard: pure inference on a FROZEN network.

Two constructors:
  * ``from_fit``            the model retrained on exactly the shared train split (the real baseline);
  * ``pretrained_reference`` the authors' released checkpoint, a reference line only (its training adapters are unknown).

Score = logit[1] - logit[0] (never the saturating probability), larger = more malicious.
Every adapter's file bytes are checked against the manifest SHA256 before they are read.
"""

from __future__ import annotations

from pathlib import Path

from ...manifest import ProtocolError, digest, read_json
from ...scoring import AdapterInput, DetectorSpec, ProtocolBinding, finite_score
from .converter import LAYOUT_ID, build_input, default_resolver, load_qv_factors, sha256_file
from .train import FIT_SCHEMA, TrainConfig

_HERE = Path(__file__).resolve().parent


def source_sha256() -> str:
    """Binds the wrapper and everything it uses for inference (not the trainer; fit.json records that)."""
    return digest({name: sha256_file(_HERE / name) for name in ("converter.py", "model.py", "detector.py")})


class PEFTGuardDetector:
    def __init__(self, *, model, spec: DetectorSpec, dim: int, layers: int, rank: int, autocast: str, resolver, device):
        self._model, self._spec, self._resolver, self._device = model, spec, resolver, device
        self.dim, self.layers, self.rank, self.autocast = dim, layers, rank, autocast

    @property
    def spec(self) -> DetectorSpec:
        return self._spec

    # -- constructors ---------------------------------------------------------------------------------------------
    @classmethod
    def from_fit(cls, fit_path, checkpoint_path, binding: ProtocolBinding, *, expected_checkpoint_sha256: str,
                 resolver=default_resolver, device: str | None = None):
        """Reload a frozen retrained model. ``expected_checkpoint_sha256`` must come from a trusted record of the fit."""
        record = read_json(fit_path)
        if (record.get("schema_version") != FIT_SCHEMA or record.get("kind") != "peftguard_retrained"
                or record.get("layout") != LAYOUT_ID):
            raise ProtocolError("Not a PEFTGuard fit record for this converter layout")
        actual = sha256_file(checkpoint_path)
        if not (actual == record["checkpoint_sha256"] == expected_checkpoint_sha256):
            raise ProtocolError("Checkpoint bytes differ from the fit record / expected SHA256")
        if (record["protocol_sha256"] != binding.sha256
                or record["train_records_sha256"] != binding.records_digest("train")
                or record["validation_records_sha256"] != binding.records_digest("validation")):
            raise ProtocolError("Fit record belongs to a different manifest/split")
        # The baseline must be fitted on the ENTIRE current train split and tuned on the ENTIRE validation split.
        if tuple(record["fit_ids"]) != binding.ids("train") or tuple(record["tuning_ids"]) != binding.ids("validation"):
            raise ProtocolError("Fit/tuning IDs must equal the complete shared train/validation partitions")
        cfg = TrainConfig(**record["config"])
        config = {"kind": record["kind"], "layout": LAYOUT_ID, "architecture": record["architecture"],
                  "score": "logit1_minus_logit0", "train": record["config"]}
        spec = DetectorSpec(
            "peftguard_llama2_7b_qv_authors_layout_retrained_v1", source_sha256(), digest(config), actual,
            "higher_is_suspicious", tuple(record["fit_ids"]), tuple(record["tuning_ids"]),
            f"Retrained from scratch on the shared train split only (seed {cfg.seed}, best epoch "
            f"{record['best_epoch']} chosen on the shared validation split); no external pretraining; "
            f"optimizer {cfg.optimizer}, autocast {cfg.autocast}; fit record {digest(record)}")
        return cls._build(checkpoint_path, spec, cfg.dim, cfg.layers, cfg.rank, cfg.autocast, resolver, device)

    @classmethod
    def pretrained_reference(cls, checkpoint_path, *, expected_sha256: str, provenance: str, dim: int = 4096,
                             layers: int = 32, rank: int = 16, resolver=default_resolver, device: str | None = None):
        """Authors' released checkpoint. Reference line only: fit_ids is empty, so the spec cannot claim a clean fit."""
        actual = sha256_file(checkpoint_path)
        if actual != expected_sha256:
            raise ProtocolError("Checkpoint bytes differ from the expected SHA256")
        from .model import architecture
        config = {"kind": "peftguard_pretrained_reference", "layout": LAYOUT_ID,
                  "architecture": architecture(dim, layers), "score": "logit1_minus_logit0", "train": None}
        spec = DetectorSpec("peftguard_llama2_7b_pretrained_reference_v1", source_sha256(), digest(config), actual,
                            "higher_is_suspicious", (), (), provenance)
        return cls._build(checkpoint_path, spec, dim, layers, rank, "none", resolver, device)

    @classmethod
    def _build(cls, checkpoint_path, spec, dim, layers, rank, autocast, resolver, device):
        import torch
        from .model import PGLlama2, load_state
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = load_state(PGLlama2(dim, layers), checkpoint_path).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model=model, spec=spec, dim=dim, layers=layers, rank=rank, autocast=autocast,
                   resolver=resolver, device=device)

    # -- inference ------------------------------------------------------------------------------------------------
    def score(self, adapters: tuple[AdapterInput, ...]) -> dict[str, float]:
        import torch
        from .model import raw_score_from_logits
        if type(adapters) is not tuple:
            raise ProtocolError("Score an immutable tuple of AdapterInput")
        result = {}
        for adapter in adapters:
            if adapter.lora_rank != self.rank or adapter.target_modules != ("q_proj", "v_proj"):
                raise ProtocolError(f"{adapter.adapter_id}: needs rank {self.rank} and exactly q_proj+v_proj")
            path = self._resolver(adapter)
            if sha256_file(path) != adapter.sha256:
                raise ProtocolError(f"{adapter.adapter_id}: file bytes differ from the manifest SHA256")
            factors = load_qv_factors(path, dim=self.dim, layers=self.layers, rank=self.rank)
            with torch.no_grad():
                x = build_input(factors, dim=self.dim, layers=self.layers, device=self._device)
                if self.autocast == "bf16" and str(self._device).startswith("cuda"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = self._model(x)
                else:
                    logits = self._model(x)
                result[adapter.adapter_id] = finite_score(raw_score_from_logits(logits))
            del x
            if str(self._device).startswith("cuda"):
                torch.cuda.empty_cache()
        return result
