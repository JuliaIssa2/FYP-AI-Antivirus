"""Dependency-free descriptive metrics and 95% Clopper-Pearson intervals.

Pure arithmetic only: these functions grant no access to protected records.
"""

from __future__ import annotations

import math
from collections import Counter

from .manifest import ProtocolError
from .scoring import finite_score


def _binomial_cdf(k: int, n: int, p: float) -> float:
    if p == 0:
        return 1.0
    if p == 1:
        return float(k == n)
    log_p, log_q = math.log(p), math.log1p(-p)
    log_n = math.lgamma(n + 1)
    return min(1.0, math.fsum(math.exp(log_n - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                                     + j * log_p + (n - j) * log_q) for j in range(k + 1)))


def _inverse_binomial_cdf(k: int, n: int, target: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        if _binomial_cdf(k, n, middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def clopper_pearson(successes: int, trials: int) -> tuple[float, float] | None:
    """Equal-tail 95% exact binomial CI, numerically inverted without SciPy.

    Requires independent Bernoulli observations under a common-rate model.
    n=0 is undefined rather than zero; boundary intervals are not zero-width.
    """
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials:
        raise ProtocolError("Require integer counts 0 <= successes <= trials")
    if trials == 0:
        return None
    lower = 0.0 if successes == 0 else _inverse_binomial_cdf(successes - 1, trials, 0.975)
    upper = 1.0 if successes == trials else _inverse_binomial_cdf(successes, trials, 0.025)
    return lower, upper


def rate(successes: int, groups: list[str]) -> dict:
    trials = len(groups)
    repeated = len(set(groups)) != trials
    interval = None if repeated else clopper_pearson(successes, trials)
    return {
        "value": successes / trials if trials else None, "numerator": successes, "denominator": trials,
        "independent_group_count": len(set(groups)),
        "reason": None if trials else "zero_denominator",
        "ci95": list(interval) if interval is not None else None,
        "ci_method": "clopper_pearson" if interval is not None else None,
        "ci_reason": ("repeated_lineage_groups_require_cluster_method" if repeated
                      else "zero_denominator" if not trials else None),
    }


def auroc(labels: list[bool], scores: list[float]) -> float | None:
    if len(labels) != len(scores) or any(type(label) is not bool for label in labels):
        raise ProtocolError("AUROC requires equally sized boolean labels and scores")
    ordered = sorted((finite_score(score), label) for score, label in zip(scores, labels))
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def summarize(predictions: list[dict]) -> dict:
    """Summarize already-frozen decisions; empty/single-class slices are valid."""
    for row in predictions:
        if row["label"] not in ("benign", "malicious") or type(row["predicted_malicious"]) is not bool:
            raise ProtocolError("Invalid prediction/label")
    counts = Counter(
        "tp" if row["predicted_malicious"] and row["label"] == "malicious" else
        "fp" if row["predicted_malicious"] else
        "fn" if row["label"] == "malicious" else "tn" for row in predictions
    )
    benign = [row for row in predictions if row["label"] == "benign"]
    malicious = [row for row in predictions if row["label"] == "malicious"]
    area = auroc([row["label"] == "malicious" for row in predictions],
                 [row["suspiciousness"] for row in predictions])
    return {
        "count": len(predictions), "confusion_matrix": {key: counts[key] for key in ("tn", "fp", "fn", "tp")},
        "fpr": rate(counts["fp"], [row["leakage_group"] for row in benign]),
        "recall": rate(counts["tp"], [row["leakage_group"] for row in malicious]),
        "accuracy": rate(counts["tn"] + counts["tp"], [row["leakage_group"] for row in predictions]),
        "auroc": {"value": area, "reason": "both_classes_required" if area is None else None,
                  "ci95": None, "ci_reason": "AUROC_is_not_a_binomial_proportion"},
    }
