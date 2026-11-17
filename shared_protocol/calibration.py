"""Clean-only 5% split-conformal calibration and immutable threshold artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from .manifest import ProtocolError, canonical_json, digest, read_json, write_new_json
from .scoring import ProtocolBinding, ScoreBatch, finite_score, verify_records


def conformal_upper_quantile(scores: Sequence[float]) -> tuple[int, float]:
    """Return (1-based order index, upper threshold) at fixed alpha=1/20.

    Reject with score > threshold, never >=. For n=0..18 use +inf.
    The integer calculation avoids ceil(0.95 * (n+1)) floating-point errors.
    """
    ordered = sorted(finite_score(score) for score in scores)
    k = (19 * (len(ordered) + 1) + 19) // 20
    return k, ordered[k - 1] if k <= len(ordered) else math.inf


@dataclass(frozen=True)
class FrozenThreshold:
    """The value is derived solely from an immutable clean calibration batch.

    No constructor argument or setter accepts a chosen operating threshold.
    Production construction is through calibrate_benign(); reload via load_threshold.
    """

    calibration: ScoreBatch

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, ScoreBatch) or self.calibration.partition != "calibration":
            raise ProtocolError("Threshold requires a calibration score batch")

    @property
    def value(self) -> float:
        return conformal_upper_quantile([s.suspiciousness for s in self.calibration.scores])[1]

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def predict(self, score: float) -> bool:
        return finite_score(score) > self.value

    def verify(self, binding: ProtocolBinding) -> None:
        self.calibration.verify(binding, "calibration")

    def to_dict(self) -> dict:
        k, value = conformal_upper_quantile([s.suspiciousness for s in self.calibration.scores])
        return {
            "schema_version": 1, "method": "split_conformal_upper_v1", "target_fpr": 0.05,
            "comparison": ">", "score_convention": "higher_is_suspicious",
            "unit": "lexicographic_first_id_per_leakage_group",
            "n": len(self.calibration.scores), "k": k,
            "threshold": {"kind": "positive_infinity" if math.isinf(value) else "finite",
                          "value": None if math.isinf(value) else value},
            "calibration": self.calibration.to_dict(),
            "calibration_sha256": self.calibration.sha256,
            "guarantee": "marginal_under_exchangeability_only_not_benign_shift",
        }


def calibrate_benign(binding: ProtocolBinding, records: Sequence[dict], scores: ScoreBatch) -> FrozenThreshold:
    """Accept only the complete verified calibration view and its designated scores.

    No test labels, test scores, configurable alpha, or threshold-search inputs.
    """
    scores.verify(binding, "calibration")
    rows = verify_records(binding, records, "calibration")
    if any(row["label"] != "benign" for row in rows):
        raise ProtocolError("Calibration must be benign-only")
    result = FrozenThreshold(scores)
    result.verify(binding)
    return result


def save_threshold(path: str | Path, threshold: FrozenThreshold, binding: ProtocolBinding) -> None:
    threshold.verify(binding)
    write_new_json(path, {"threshold": threshold.to_dict(), "threshold_sha256": threshold.sha256})


def load_threshold(path: str | Path, binding: ProtocolBinding) -> FrozenThreshold:
    artifact = read_json(path)
    try:
        threshold = FrozenThreshold(ScoreBatch.from_dict(artifact["threshold"]["calibration"]))
        threshold.verify(binding)
        expected = {"threshold": threshold.to_dict(), "threshold_sha256": threshold.sha256}
        if canonical_json(artifact) != canonical_json(expected):
            raise ProtocolError("Threshold artifact was changed or uses a different decision rule")
        return threshold
    except (KeyError, TypeError) as exc:
        raise ProtocolError("Malformed frozen-threshold artifact") from exc
