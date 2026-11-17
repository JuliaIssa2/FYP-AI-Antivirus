"""Explicit custodian final-assessment workflow, separate from development.

No loader/unlock flag for the final cohort exists. The caller must supply a
release artifact and authorized records. This is an audit boundary, not an OS
permission system. Only synthetic inputs exercise this module during development.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .calibration import FrozenThreshold
from .evaluation import _check_frozen_detector, _evaluate_authorized
from .manifest import ProtocolError, canonical_json, digest, read_json, text_value, write_new_json
from .scoring import Detector, ProtocolBinding, ScoreBatch, _score_authorized, hash_value


@dataclass(frozen=True)
class FinalRelease:
    protocol_sha256: str
    threshold_sha256: str
    frozen_decisions_sha256: str
    approved_by: str
    approval_reference: str
    purpose: str = "final_assessment_only_no_model_development"

    def __post_init__(self) -> None:
        for value in (self.protocol_sha256, self.threshold_sha256, self.frozen_decisions_sha256):
            hash_value(value)
        text_value(self.approved_by, "approved_by")
        text_value(self.approval_reference, "approval_reference")
        if self.purpose != "final_assessment_only_no_model_development":
            raise ProtocolError("Final release cannot authorize development")

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    def verify(self, binding: ProtocolBinding, threshold: FrozenThreshold) -> None:
        threshold.verify(binding)
        if self.protocol_sha256 != binding.sha256 or self.threshold_sha256 != threshold.sha256:
            raise ProtocolError("Final release does not match the frozen protocol/threshold")


def write_final_release(path: str | Path, binding: ProtocolBinding, threshold: FrozenThreshold,
                        *, approved_by: str, approval_reference: str,
                        frozen_decisions_sha256: str) -> FinalRelease:
    """Custodian records actual prior approval; never infer approval or open data."""
    release = FinalRelease(binding.sha256, threshold.sha256, frozen_decisions_sha256,
                           approved_by, approval_reference)
    release.verify(binding, threshold)
    write_new_json(path, {"release": asdict(release), "release_sha256": release.sha256})
    return release


def load_final_release(path: str | Path, binding: ProtocolBinding, threshold: FrozenThreshold) -> FinalRelease:
    artifact = read_json(path)
    try:
        release = FinalRelease(**artifact["release"])
        release.verify(binding, threshold)
        if canonical_json(artifact) != canonical_json({"release": asdict(release), "release_sha256": release.sha256}):
            raise ProtocolError("Final release artifact changed")
        return release
    except (TypeError, KeyError) as exc:
        raise ProtocolError("Malformed final-release artifact") from exc


def _require_release(release: FinalRelease, binding: ProtocolBinding, threshold: FrozenThreshold) -> None:
    if not isinstance(release, FinalRelease):
        raise ProtocolError("An explicit custodian FinalRelease is required")
    release.verify(binding, threshold)


def score_final(detector: Detector, records: Sequence[dict], binding: ProtocolBinding,
                threshold: FrozenThreshold, *, release: FinalRelease) -> ScoreBatch:
    _require_release(release, binding, threshold)
    _check_frozen_detector(threshold, binding, detector)
    return _score_authorized(detector, records, binding, "final_holdout")


def evaluate_final(binding: ProtocolBinding, records: Sequence[dict], scores: ScoreBatch,
                   threshold: FrozenThreshold, *, release: FinalRelease) -> dict:
    _require_release(release, binding, threshold)
    report = _evaluate_authorized(binding, records, scores, threshold, "final_holdout")
    report.pop("report_sha256")
    report["final_release_sha256"] = release.sha256
    report["report_sha256"] = digest(report)
    return report


def save_final_evaluation(path: str | Path, binding: ProtocolBinding, records: Sequence[dict],
                          scores: ScoreBatch, threshold: FrozenThreshold, *, release: FinalRelease) -> dict:
    report = evaluate_final(binding, records, scores, threshold, release=release)
    write_new_json(path, report)
    return report
