"""Post-freeze evaluation of supplied metadata/scores; never loads a cohort."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .calibration import FrozenThreshold
from .manifest import ProtocolError, digest, write_new_json
from .metrics import summarize
from .scoring import (
    Detector, ProtocolBinding, ScoreBatch, _score_authorized, verify_records,
)


def _check_frozen_detector(threshold: FrozenThreshold, binding: ProtocolBinding, detector: Detector) -> None:
    threshold.verify(binding)
    if detector.spec != threshold.calibration.detector:
        raise ProtocolError("Detector differs from the frozen calibration state/config/direction")


def score_evaluation(detector: Detector, records: Sequence[dict], binding: ProtocolBinding,
                     threshold: FrozenThreshold) -> ScoreBatch:
    """A separate post-freeze entry point; cannot select final_holdout."""
    _check_frozen_detector(threshold, binding, detector)
    return _score_authorized(detector, records, binding, "evaluation")


def _evaluate_authorized(binding: ProtocolBinding, records: Sequence[dict], scores: ScoreBatch,
                         threshold: FrozenThreshold, partition: str) -> dict:
    threshold.verify(binding)
    scores.verify(binding, partition)
    if scores.detector != threshold.calibration.detector:
        raise ProtocolError("Evaluation score detector differs from frozen threshold detector")
    rows = verify_records(binding, records, partition)
    by_id = {score.adapter_id: score for score in scores.scores}
    predictions = []
    for row in rows:
        score = by_id[row["adapter_id"]]
        predictions.append({
            "adapter_id": row["adapter_id"], "label": row["label"], "task": row["task"],
            "attack_family": row["attack_family"], "payload": row["payload"],
            "leakage_group": row["leakage_group"], "raw_score": score.raw,
            "suspiciousness": score.suspiciousness,
            "predicted_malicious": threshold.predict(score.suspiciousness),
        })
    reserved = json.loads(binding.reservations_json)[partition]
    held_benign = [row for row in predictions if row["label"] == "benign"
                   and row["task"] in reserved["benign_tasks"]]
    held_malicious = [row for row in predictions if row["label"] == "malicious"
                      and (row["attack_family"] in reserved["attack_families"]
                           or row["payload"] in reserved["payloads"])]
    primary_ids = {row["adapter_id"] for row in held_benign + held_malicious}

    def grouped(members: list[dict], field: str) -> dict:
        return {value: summarize([row for row in members if row[field] == value])
                for value in sorted({row[field] for row in members})}

    report = {
        "schema_version": 1, "partition": partition, "protocol_sha256": binding.sha256,
        "manifest_sha256": binding.manifest_sha256, "split_sha256": binding.split_sha256,
        "detector_sha256": scores.detector.sha256, "scores_sha256": scores.sha256,
        "threshold_sha256": threshold.sha256,
        "operating_point": {key: threshold.to_dict()[key] for key in ("threshold", "comparison", "target_fpr")},
        "overall": summarize(predictions),
        "heldout_combined": summarize(held_benign + held_malicious),
        "heldout_benign": {"pooled": summarize(held_benign), "by_task": grouped(held_benign, "task")},
        "heldout_malicious": {"pooled": summarize(held_malicious),
                              "by_attack_family": grouped(held_malicious, "attack_family"),
                              "by_payload": grouped(held_malicious, "payload")},
        "reservation_companions": summarize([row for row in predictions if row["adapter_id"] not in primary_ids]),
        "predictions": predictions,
        "interpretation": {
            "fpr": "observed_only_no_5_percent_guarantee_under_benign_distribution_shift",
            "ci": "95_percent_marginal_binomial_intervals_assume_independent_common_rate_trials",
            "multiple_groups": "no_simultaneous_coverage_claim",
            "repeated_lineage": "binomial_CIs_suppressed_where_lineage_repeats_cluster_inference_not_implemented",
        },
    }
    report["report_sha256"] = digest(report)
    return report


def evaluate(binding: ProtocolBinding, records: Sequence[dict], scores: ScoreBatch,
             threshold: FrozenThreshold) -> dict:
    """Evaluation partition only. Threshold is an input, never a search variable."""
    if scores.partition != "evaluation":
        raise ProtocolError("Regular evaluation accepts evaluation scores only; final holdout stays protected")
    return _evaluate_authorized(binding, records, scores, threshold, "evaluation")


def save_evaluation(path: str | Path, binding: ProtocolBinding, records: Sequence[dict],
                    scores: ScoreBatch, threshold: FrozenThreshold) -> dict:
    report = evaluate(binding, records, scores, threshold)
    write_new_json(path, report)
    return report
