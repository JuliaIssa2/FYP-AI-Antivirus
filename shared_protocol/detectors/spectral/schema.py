"""Stable baseline identity and an explicitly ordered future layer interface."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Protocol

from ...manifest import ProtocolError, digest
from ...scoring import AdapterInput


BASELINE_ID = "padbench_spectral_layer20_qv_10d_lr_c01_v1"
SHARED_RUN_ID = BASELINE_ID + "__shared_conformal_v1"
EXTRACTOR_SHA256 = "6b33e3f392d059532e5fedbaa1a0d305101ed2206d6cf24a43675110c00c6a57"
STATISTICS = ("sigma1", "frobenius_norm", "energy_concentration", "entropy", "kurtosis")
PROJECTIONS = ("q_proj", "v_proj")


@dataclass(frozen=True)
class FeatureSchema:
    layers: tuple[int, ...]
    projections: tuple[str, ...] = PROJECTIONS
    statistics: tuple[str, ...] = STATISTICS
    formula: str = "original"
    layout: str = "layer_then_projection_then_statistic"

    def __post_init__(self):
        if (type(self.layers) is not tuple or not self.layers
                or any(type(layer) is not int or layer < 0 for layer in self.layers)
                or self.layers != tuple(sorted(set(self.layers)))):
            raise ProtocolError("Layers must be an explicitly sorted, unique, nonnegative tuple")
        if (self.projections != PROJECTIONS or self.statistics != STATISTICS
                or self.formula != "original" or self.layout != "layer_then_projection_then_statistic"):
            raise ProtocolError("Unknown feature definitions/order; create a separate schema version")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"layer{layer}.{projection}.{statistic}" for layer in self.layers
                     for projection in self.projections for statistic in self.statistics)

    @property
    def sha256(self) -> str:
        return digest(asdict(self))


BASELINE_SCHEMA = FeatureSchema((20,))
LEGACY_FEATURE_NAMES = tuple(f"{projection}.{stat}" for projection in PROJECTIONS for stat in STATISTICS)


class FeatureProvider(Protocol):
    """Future extractor/provider extension. Loading authorization is external.

    A provider must identify its schema and extractor source, and return values
    in schema.names order. Multi-layer extraction is NOT implemented in Phase 3.
    """
    schema: FeatureSchema
    extractor_sha256: str

    def get(self, adapter: AdapterInput) -> tuple[float, ...]: ...


def training_recipe(seed: int) -> dict:
    if type(seed) is not int or seed < 0:
        raise ProtocolError("Training random_state must be an explicit nonnegative integer")
    return {"input_dtype": "float64", "scaler": {"with_mean": True, "with_std": True, "copy": True},
            "classifier": {"C": 0.1, "class_weight": "balanced", "max_iter": 1000,
                           "random_state": seed, "solver": "lbfgs", "penalty": "l2",
                           "fit_intercept": True, "tol": 0.0001}}


def source_fingerprint() -> str:
    """Bind this integration plus the preserved extractor (source files only).

    Newline normalization keeps the integration fingerprint stable across Git
    checkouts; the original extractor's byte hash is separately retained above.
    """
    root = Path(__file__).parent
    files = ("schema.py", "features.py", "model.py", "detector.py", "vendor/spectral_detector.py")
    return digest({name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                   for name in files})
