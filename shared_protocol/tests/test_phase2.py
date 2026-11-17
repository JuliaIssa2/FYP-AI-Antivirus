"""Phase 2 regression suite: every record/score/release is synthetic."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from shared_protocol.calibration import (
    FrozenThreshold, calibrate_benign, conformal_upper_quantile, load_threshold, save_threshold,
)
from shared_protocol.evaluation import evaluate, save_evaluation, score_evaluation
from shared_protocol.final_evaluation import (
    FinalRelease, evaluate_final, load_final_release, save_final_evaluation,
    score_final, write_final_release,
)
from shared_protocol.manifest import METADATA_FIELDS, ProtocolError, canonical_json, digest, write_new_json
from shared_protocol.metrics import auroc, clopper_pearson, summarize
from shared_protocol.scoring import (
    DetectorSpec, Score, ScoreBatch, bind_protocol, score_development,
)
from shared_protocol.splits import create_split, development_records
from shared_protocol.tests.test_foundations import adapter, fixture


class SyntheticDetector:
    """Fixture lookup only; no tensor/model implementation or path access."""

    def __init__(self, spec, values):
        self.spec = spec
        self.values = values
        self.calls = []

    def score(self, adapters):
        self.calls.append(adapters)
        return {row.adapter_id: self.values[row.adapter_id] for row in adapters}


def setup_case():
    manifest, policy = fixture()
    manifest["adapters"].extend([
        adapter("eval-benign-safe", task="eval_task"),
        adapter("eval-missed", "malicious", attack="unseen_attack"),
    ])
    plan = create_split(manifest, policy)
    binding = bind_protocol(manifest, plan)
    spec = DetectorSpec("synthetic_lookup_only", digest("source fixture"), digest("config fixture"),
                        digest("state fixture"), "higher_is_suspicious", tuple(plan["partitions"]["train"]),
                        tuple(plan["partitions"]["validation"]), "All source/data/values are synthetic")
    values = {key: float(i) for i, key in enumerate(binding.ids("calibration", representatives=True))}
    values.update({"eval-benign": 20.0, "eval-benign-safe": 18.0, "eval-malicious": 19.0,
                   "eval-missed": 0.0, "final-benign": -1.0, "final-malicious": 20.0})
    detector = SyntheticDetector(spec, values)
    return manifest, plan, binding, detector


def synthetic_view(manifest, binding, partition):
    # Test fixtures only. Production code must receive custodian-authorized views.
    return [deepcopy(row) for row in manifest["adapters"] if row["adapter_id"] in binding.ids(partition)]


class Phase2Case(unittest.TestCase):
    def setUp(self):
        self.manifest, self.plan, self.binding, self.detector = setup_case()
        self.cal_records = development_records(self.manifest, self.plan, "calibration")
        self.cal_scores = score_development(self.detector, self.cal_records, self.binding, "calibration")
        self.threshold = calibrate_benign(self.binding, self.cal_records, self.cal_scores)
        self.eval_records = synthetic_view(self.manifest, self.binding, "evaluation")

    def eval_scores(self):
        return score_evaluation(self.detector, self.eval_records, self.binding, self.threshold)


class ScoringTests(Phase2Case):
    def test_binding_is_immutable_and_rejects_cross_partition_groups(self):
        with self.assertRaises(ProtocolError):
            replace(self.binding, commitments=list(self.binding.commitments))
        with self.assertRaises(ProtocolError):
            replace(self.binding, reservations_json="{}")
        entries = list(self.binding.commitments)
        first = entries[0]
        index = next(i for i, entry in enumerate(entries) if entry[1] != first[1])
        entry = entries[index]
        entries[index] = (entry[0], entry[1], entry[2], first[3])
        with self.assertRaisesRegex(ProtocolError, "crosses partitions"):
            replace(self.binding, commitments=tuple(entries))

    def test_receives_only_immutable_verified_input_projection(self):
        self.assertEqual(self.threshold.value, 18)
        inputs = self.detector.calls[0]
        self.assertIsInstance(inputs, tuple)
        self.assertEqual({row.adapter_id for row in inputs}, set(self.binding.ids("calibration")))
        for row in inputs:
            self.assertFalse({"label", "task", "attack_family", "payload", "verification"} & set(asdict(row)))
            with self.assertRaises(FrozenInstanceError):
                row.adapter_id = "changed"

    def test_lower_score_direction_is_fixed_negation(self):
        lower = SyntheticDetector(replace(self.detector.spec, raw_direction="lower_is_suspicious"),
                                  {key: -value for key, value in self.detector.values.items()})
        batch = score_development(lower, self.cal_records, self.binding, "calibration")
        self.assertEqual([score.suspiciousness for score in batch.scores],
                         [score.suspiciousness for score in self.cal_scores.scores])
        self.assertEqual(batch.scores[-1].raw, -batch.scores[-1].suspiciousness)
        threshold = calibrate_benign(self.binding, self.cal_records, batch)
        report = evaluate(self.binding, self.eval_records,
                          score_evaluation(lower, self.eval_records, self.binding, threshold), threshold)
        self.assertEqual(report["overall"]["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})

    def test_invalid_scores_rejected(self):
        key = self.binding.ids("calibration")[0]
        for value in (float("nan"), float("inf"), -float("inf"), True, "0.5", None, 10 ** 1000):
            detector = SyntheticDetector(self.detector.spec, {**self.detector.values, key: value})
            with self.subTest(value=type(value).__name__), self.assertRaises(ProtocolError):
                score_development(detector, self.cal_records, self.binding, "calibration")

    def test_missing_or_extra_score_ids_rejected(self):
        for returned in ({}, {"extra": 0}):
            with patch.object(self.detector, "score", return_value=returned), self.assertRaises(ProtocolError):
                score_development(self.detector, self.cal_records, self.binding, "calibration")

    def test_unknown_metadata_rejected_before_detector_call(self):
        for field in METADATA_FIELDS:
            rows = deepcopy(self.cal_records)
            rows[0][field] = None
            rows[0]["verification"].pop(field)
            with self.subTest(field=field), patch.object(self.detector, "score", side_effect=AssertionError("called")):
                with self.assertRaises(ProtocolError):
                    score_development(self.detector, rows, self.binding, "calibration")

    def test_unverified_or_drifted_metadata_rejected(self):
        for field in ("verification", "lora_alpha"):
            rows = deepcopy(self.cal_records)
            rows[0][field] = {} if field == "verification" else 100
            with self.assertRaises(ProtocolError):
                score_development(self.detector, rows, self.binding, "calibration")

    def test_development_protected_partition_denied_before_inputs_consumed(self):
        for partition in ("evaluation", "final_holdout"):
            with patch("shared_protocol.scoring._score_authorized", side_effect=AssertionError("access")):
                with self.assertRaises(ProtocolError):
                    score_development(self.detector, None, self.binding, partition)

    def test_detector_fit_or_tuning_cannot_include_holdout_or_calibration(self):
        for field in ("fit_ids", "tuning_ids"):
            for part in ("calibration", "evaluation", "final_holdout"):
                spec = replace(self.detector.spec, **{field: (self.binding.ids(part)[0],)})
                detector = SyntheticDetector(spec, self.detector.values)
                with self.subTest(field=field, part=part), self.assertRaises(ProtocolError):
                    score_development(detector, self.cal_records, self.binding, "calibration")

    def test_fit_and_tuning_roles_are_not_interchangeable(self):
        for field, partition in (("fit_ids", "validation"), ("tuning_ids", "train")):
            spec = replace(self.detector.spec, **{field: (self.binding.ids(partition)[0],)})
            with self.assertRaises(ProtocolError):
                spec.verify(self.binding)

    def test_spec_change_during_scoring_rejected(self):
        old_score = self.detector.score
        def changed(inputs):
            result = old_score(inputs)
            self.detector.spec = replace(self.detector.spec, state_sha256=digest("changed"))
            return result
        with patch.object(self.detector, "score", side_effect=changed), self.assertRaises(ProtocolError):
            score_development(self.detector, self.cal_records, self.binding, "calibration")

    def test_batch_roundtrip_direction_and_id_checks(self):
        self.assertEqual(ScoreBatch.from_dict(self.cal_scores.to_dict()), self.cal_scores)
        data = self.cal_scores.to_dict()
        data["scores"][0]["suspiciousness"] = 100
        with self.assertRaisesRegex(ProtocolError, "direction"):
            ScoreBatch.from_dict(data)
        with self.assertRaises(ProtocolError):
            replace(self.cal_scores, scores=self.cal_scores.scores + self.cal_scores.scores[:1])

    def test_imported_batch_membership_checked(self):
        for batch in (replace(self.cal_scores, scores=self.cal_scores.scores[:-1]),
                      replace(self.cal_scores, protocol_sha256=digest("another")),
                      replace(self.cal_scores, records_sha256=digest("different records"))):
            with self.assertRaises(ProtocolError):
                batch.verify(self.binding, "calibration")

    def test_spec_requires_explicit_direction_and_provenance(self):
        for change in ({"raw_direction": "automatic"}, {"provenance": "unknown"},
                       {"state_sha256": None}, {"fit_ids": list(self.detector.spec.fit_ids)}):
            with self.assertRaises(ProtocolError):
                replace(self.detector.spec, **change)


class QuantileTests(unittest.TestCase):
    def test_empty_and_small_sets_are_infinite(self):
        for n in range(19):
            k, threshold = conformal_upper_quantile(list(range(n)))
            self.assertGreater(k, n)
            self.assertEqual(threshold, math.inf)

    def test_order_statistics_at_boundaries(self):
        for n, expected_k in ((19, 19), (20, 20), (38, 38), (39, 38), (99, 95), (100, 96)):
            with self.subTest(n=n):
                self.assertEqual(conformal_upper_quantile(list(range(n))), (expected_k, expected_k - 1))

    def test_ties_no_interpolation_or_order_dependence(self):
        values = [0.1] * 37 + [0.8, 0.9]
        self.assertEqual(conformal_upper_quantile(values), (38, 0.8))
        self.assertEqual(conformal_upper_quantile(list(reversed(values))), (38, 0.8))

    def test_invalid_quantile_values(self):
        for value in (True, "2", float("nan"), float("inf")):
            with self.assertRaises(ProtocolError):
                conformal_upper_quantile([value])


class CalibrationTests(Phase2Case):
    def test_strict_decision_at_boundary(self):
        self.assertFalse(self.threshold.predict(18))
        self.assertFalse(self.threshold.predict(math.nextafter(18, -math.inf)))
        self.assertTrue(self.threshold.predict(math.nextafter(18, math.inf)))

    def test_all_tied_scores_conservative(self):
        detector = SyntheticDetector(self.detector.spec, {key: 3 for key in self.detector.values})
        scores = score_development(detector, self.cal_records, self.binding, "calibration")
        threshold = calibrate_benign(self.binding, self.cal_records, scores)
        self.assertFalse(threshold.predict(3))
        self.assertTrue(threshold.predict(4))

    def test_infinite_threshold_json_and_decisions(self):
        manifest, policy = fixture()
        policy["calibration_groups"] = 2
        plan = create_split(manifest, policy)
        binding = bind_protocol(manifest, plan)
        spec = replace(self.detector.spec, fit_ids=(), tuning_ids=())
        records = development_records(manifest, plan, "calibration")
        detector = SyntheticDetector(spec, {row["adapter_id"]: 1 for row in records})
        batch = score_development(detector, records, binding, "calibration")
        threshold = calibrate_benign(binding, records, batch)
        self.assertEqual(threshold.to_dict()["threshold"], {"kind": "positive_infinity", "value": None})
        self.assertFalse(threshold.predict(1e300))
        self.assertNotIn("Infinity", canonical_json(threshold.to_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "threshold.json"
            save_threshold(path, threshold, binding)
            self.assertEqual(load_threshold(path, binding), threshold)

    def test_no_test_data_can_be_used_for_calibration(self):
        with self.assertRaises(ProtocolError):
            calibrate_benign(self.binding, self.eval_records, self.eval_scores())
        with self.assertRaises(ProtocolError):
            calibrate_benign(self.binding, self.eval_records, self.cal_scores)

    def test_malicious_or_unknown_calibration_label_rejected(self):
        for label in ("malicious", None):
            rows = deepcopy(self.cal_records)
            rows[0]["label"] = label
            if label is None:
                rows[0]["verification"].pop("label")
            else:
                rows[0]["attack_family"] = "attack"
            with self.assertRaises(ProtocolError):
                calibrate_benign(self.binding, rows, self.cal_scores)

    def test_one_representative_per_lineage_selected_without_scores(self):
        manifest = deepcopy(self.manifest)
        duplicate = deepcopy(self.cal_records[0])
        duplicate["adapter_id"] = "zz-related-benign"
        duplicate["path_or_source"] = "fixture://zz-related-benign"
        duplicate["sha256"] = digest("zz-related-benign")
        manifest["adapters"].append(duplicate)
        plan = create_split(manifest, self.plan["policy"])
        binding = bind_protocol(manifest, plan)
        records = development_records(manifest, plan, "calibration")
        self.assertEqual(len(records), 20)
        batch = score_development(self.detector, records, binding, "calibration")
        self.assertEqual(len(batch.scores), 19)
        self.assertNotIn("zz-related-benign", [score.adapter_id for score in batch.scores])
        threshold = calibrate_benign(binding, records, batch)
        self.assertEqual(threshold.value, self.threshold.value)

    def test_threshold_is_deeply_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.threshold.calibration = self.cal_scores
        with self.assertRaises(FrozenInstanceError):
            self.threshold.calibration.scores[0].raw = 99
        with self.assertRaises(FrozenInstanceError):
            self.threshold.calibration.detector.raw_direction = "lower_is_suspicious"
        exported = self.threshold.to_dict()
        exported["threshold"]["value"] = -100
        self.assertEqual(self.threshold.value, 18)

    def test_threshold_files_cannot_be_overwritten_or_tampered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "threshold.json"
            save_threshold(path, self.threshold, self.binding)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_threshold(path, self.threshold, self.binding)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(load_threshold(path, self.binding), self.threshold)
            for i, (field, value) in enumerate((("comparison", ">="), ("target_fpr", 0.1),
                                               ("threshold", {"kind": "finite", "value": 0}), ("n", 20))):
                bad = json.loads(original)
                bad["threshold"][field] = value
                bad_path = Path(directory) / f"tampered{i}.json"
                write_new_json(bad_path, bad)
                with self.assertRaises(ProtocolError):
                    load_threshold(bad_path, self.binding)

    def test_threshold_cannot_cross_split_versions(self):
        plan = create_split(self.manifest, {**self.plan["policy"], "seed": 123})
        other = bind_protocol(self.manifest, plan)
        with self.assertRaises(ProtocolError):
            self.threshold.verify(other)


class MetricsTests(unittest.TestCase):
    def test_exact_interval_matches_published_reference(self):
        lower, upper = clopper_pearson(7, 50)
        # SciPy binomtest documentation's exact 95% reference example.
        self.assertAlmostEqual(lower, 0.05819170033997342, places=11)
        self.assertAlmostEqual(upper, 0.26739600249700846, places=11)

    def test_interval_boundaries_have_nonzero_uncertainty(self):
        for n in (1, 10, 100):
            lower, upper = clopper_pearson(0, n)
            self.assertEqual(lower, 0)
            self.assertAlmostEqual(upper, 1 - 0.025 ** (1 / n), places=11)
            lower, upper = clopper_pearson(n, n)
            self.assertEqual(upper, 1)
            self.assertAlmostEqual(lower, 0.025 ** (1 / n), places=11)
        self.assertIsNone(clopper_pearson(0, 0))

    def test_interval_symmetric_and_invalid_counts_rejected(self):
        lower, upper = clopper_pearson(5, 10)
        self.assertAlmostEqual(lower, 0.18708602844739852, places=11)
        self.assertAlmostEqual(upper, 1 - lower, places=12)
        for counts in ((-1, 10), (11, 10), (0, -1), (True, 10), (2.0, 10)):
            with self.assertRaises(ProtocolError):
                clopper_pearson(*counts)

    def test_auc_ranking_ties_and_reversed_direction(self):
        self.assertEqual(auroc([False, True], [0, 1]), 1)
        self.assertEqual(auroc([False, True], [1, 0]), 0)
        self.assertEqual(auroc([False, True], [1, 1]), 0.5)
        self.assertEqual(auroc([False, False, True, True], [0, 1, 1, 2]), 0.875)
        self.assertIsNone(auroc([], []))
        self.assertIsNone(auroc([True], [1]))

    def test_zero_positive_negative_and_empty_slices(self):
        empty = summarize([])
        self.assertIsNone(empty["accuracy"]["value"])
        self.assertIsNone(empty["accuracy"]["ci95"])
        for label, undefined in (("benign", "recall"), ("malicious", "fpr")):
            report = summarize([{"label": label, "predicted_malicious": False,
                                 "suspiciousness": 0, "leakage_group": "fixture"}])
            self.assertIsNone(report[undefined]["value"])
            self.assertEqual(report[undefined]["reason"], "zero_denominator")
            self.assertIsNone(report["auroc"]["value"])

    def test_correlated_observations_do_not_get_binomial_intervals(self):
        rows = [{"label": "benign", "predicted_malicious": False,
                 "suspiciousness": 0, "leakage_group": "same"} for _ in range(3)]
        report = summarize(rows)
        self.assertEqual(report["fpr"]["value"], 0)
        self.assertIsNone(report["fpr"]["ci95"])
        self.assertEqual(report["fpr"]["independent_group_count"], 1)
        self.assertIn("repeated_lineage", report["fpr"]["ci_reason"])


class EvaluationTests(Phase2Case):
    def test_report_determinism_across_processes(self):
        code = (
            "from shared_protocol.tests.test_phase2 import setup_case, synthetic_view; "
            "from shared_protocol.scoring import score_development; "
            "from shared_protocol.calibration import calibrate_benign; "
            "from shared_protocol.evaluation import score_evaluation, evaluate; "
            "from shared_protocol.manifest import canonical_json; "
            "m,p,b,d=setup_case(); c=synthetic_view(m,b,'calibration'); "
            "t=calibrate_benign(b,c,score_development(d,c,b,'calibration')); "
            "e=synthetic_view(m,b,'evaluation'); "
            "print(canonical_json(evaluate(b,e,score_evaluation(d,e,b,t),t)))"
        )
        expected = canonical_json(evaluate(self.binding, self.eval_records, self.eval_scores(), self.threshold))
        for seed in ("1", "97"):
            result = subprocess.run([sys.executable, "-B", "-c", code], check=True,
                                    capture_output=True, text=True,
                                    env={**os.environ, "PYTHONHASHSEED": seed})
            self.assertEqual(result.stdout.strip(), expected)

    def test_hand_checkable_confusion_metrics_and_group_reports(self):
        report = evaluate(self.binding, self.eval_records, self.eval_scores(), self.threshold)
        metrics = report["overall"]
        self.assertEqual(metrics["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(metrics["fpr"]["value"], 0.5)
        self.assertEqual(metrics["recall"]["value"], 0.5)
        self.assertEqual(metrics["accuracy"]["value"], 0.5)
        self.assertEqual(metrics["auroc"]["value"], 0.25)  # Never flip a bad detector's direction.
        self.assertIsNotNone(metrics["accuracy"]["ci95"])
        self.assertEqual(report["heldout_benign"]["by_task"]["eval_task"]["count"], 2)
        self.assertEqual(report["heldout_malicious"]["by_attack_family"]["unseen_attack"]["count"], 2)
        self.assertEqual(report["reservation_companions"]["count"], 0)
        self.assertIn("no_5_percent_guarantee", report["interpretation"]["fpr"])

    def test_test_score_changes_cannot_modify_threshold(self):
        frozen = canonical_json(self.threshold.to_dict())
        baseline = evaluate(self.binding, self.eval_records, self.eval_scores(), self.threshold)
        for row in self.eval_records:
            self.detector.values[row["adapter_id"]] = 1e6
        changed = evaluate(self.binding, self.eval_records, self.eval_scores(), self.threshold)
        self.assertNotEqual(baseline["overall"], changed["overall"])
        self.assertEqual(canonical_json(self.threshold.to_dict()), frozen)
        self.assertEqual(baseline["threshold_sha256"], changed["threshold_sha256"])

    def test_test_label_changes_rejected_without_threshold_tuning(self):
        frozen = self.threshold.sha256
        rows = deepcopy(self.eval_records)
        for row in rows:
            if row["label"] == "malicious":
                row.update(label="benign", attack_family="none", payload="none", poisoning_rate=0)
        with self.assertRaises(ProtocolError):
            evaluate(self.binding, rows, self.eval_scores(), self.threshold)
        self.assertEqual(self.threshold.sha256, frozen)

    def test_changed_model_or_score_direction_cannot_reuse_threshold(self):
        for field, value in (("state_sha256", digest("new model")),
                             ("config_sha256", digest("new config")),
                             ("raw_direction", "lower_is_suspicious")):
            detector = SyntheticDetector(replace(self.detector.spec, **{field: value}), self.detector.values)
            with self.assertRaises(ProtocolError):
                score_evaluation(detector, self.eval_records, self.binding, self.threshold)

    def test_calibration_or_final_data_cannot_enter_regular_evaluation(self):
        with self.assertRaises(ProtocolError):
            evaluate(self.binding, self.cal_records, self.cal_scores, self.threshold)
        final = synthetic_view(self.manifest, self.binding, "final_holdout")
        with self.assertRaises(ProtocolError):
            score_evaluation(self.detector, final, self.binding, self.threshold)

    def test_duplicate_and_partial_records_rejected(self):
        batch = self.eval_scores()
        for rows in (self.eval_records[:-1], self.eval_records + self.eval_records[:1]):
            with self.assertRaises(ProtocolError):
                evaluate(self.binding, rows, batch, self.threshold)

    def test_deterministic_reports_and_exclusive_output(self):
        scores = self.eval_scores()
        expected = evaluate(self.binding, self.eval_records, scores, self.threshold)
        reordered = evaluate(self.binding, list(reversed(self.eval_records)), scores, self.threshold)
        self.assertEqual(canonical_json(expected), canonical_json(reordered))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            save_evaluation(path, self.binding, self.eval_records, scores, self.threshold)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_evaluation(path, self.binding, self.eval_records, scores, self.threshold)
            self.assertEqual(path.read_bytes(), original)

    def test_no_path_access_or_legacy_imports_in_complete_synthetic_workflow(self):
        with patch("builtins.open", side_effect=AssertionError("artifact access")), \
                patch.object(Path, "open", side_effect=AssertionError("artifact access")):
            scores = self.eval_scores()
            evaluate(self.binding, self.eval_records, scores, self.threshold)
        for name in ("torch", "transformers", "peft", "core.detector", "experiments.padbench.spectral_detector"):
            self.assertNotIn(name, sys.modules)

    def test_reserved_companions_not_misreported_as_unseen(self):
        companion = adapter("seen-companion", "malicious")
        companion["leakage_group"] = "eval-malicious"
        self.manifest["adapters"].append(companion)
        plan = create_split(self.manifest, self.plan["policy"])
        binding = bind_protocol(self.manifest, plan)
        cal = development_records(self.manifest, plan, "calibration")
        threshold = calibrate_benign(binding, cal, score_development(self.detector, cal, binding, "calibration"))
        records = synthetic_view(self.manifest, binding, "evaluation")
        self.detector.values["seen-companion"] = 100
        scores = score_evaluation(self.detector, records, binding, threshold)
        report = evaluate(binding, records, scores, threshold)
        self.assertEqual(report["overall"]["count"], 5)
        self.assertEqual(report["heldout_malicious"]["pooled"]["count"], 2)
        self.assertEqual(report["reservation_companions"]["count"], 1)
        self.assertIsNone(report["overall"]["recall"]["ci95"])


class FinalWorkflowTests(Phase2Case):
    def release(self):
        return FinalRelease(self.binding.sha256, self.threshold.sha256, digest("synthetic decisions"),
                            "synthetic custodian", "synthetic-only approval, not real authorization")

    def test_release_required_before_reading_records_or_scoring(self):
        with patch("shared_protocol.final_evaluation._score_authorized", side_effect=AssertionError("called")):
            with self.assertRaises(ProtocolError):
                score_final(self.detector, None, self.binding, self.threshold, release=None)
        with patch("shared_protocol.final_evaluation._evaluate_authorized", side_effect=AssertionError("called")):
            with self.assertRaises(ProtocolError):
                evaluate_final(self.binding, None, None, self.threshold, release=None)

    def test_wrong_release_binding_or_threshold_rejected(self):
        for field in ("protocol_sha256", "threshold_sha256"):
            release = replace(self.release(), **{field: digest("wrong")})
            with self.assertRaises(ProtocolError):
                score_final(self.detector, None, self.binding, self.threshold, release=release)

    def test_no_development_release_or_unknown_approval(self):
        for kwargs in ({"purpose": "development"}, {"approved_by": "unknown"}, {"approval_reference": ""}):
            with self.assertRaises(ProtocolError):
                replace(self.release(), **kwargs)

    def test_explicit_synthetic_final_workflow_without_file_access(self):
        rows = synthetic_view(self.manifest, self.binding, "final_holdout")
        with patch.object(Path, "open", side_effect=AssertionError("No holdout files")):
            scores = score_final(self.detector, rows, self.binding, self.threshold, release=self.release())
            with self.assertRaises(ProtocolError):
                evaluate(self.binding, rows, scores, self.threshold)
            report = evaluate_final(self.binding, rows, scores, self.threshold, release=self.release())
        self.assertEqual(report["overall"]["accuracy"]["value"], 1)
        self.assertEqual(report["final_release_sha256"], self.release().sha256)
        self.assertEqual(report["partition"], "final_holdout")

    def test_release_and_final_report_exclusive_save_roundtrip(self):
        rows = synthetic_view(self.manifest, self.binding, "final_holdout")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = dict(approved_by="synthetic custodian", approval_reference="synthetic approval only",
                        frozen_decisions_sha256=digest("synthetic decisions"))
            release = write_final_release(root / "release.json", self.binding, self.threshold, **args)
            self.assertEqual(load_final_release(root / "release.json", self.binding, self.threshold), release)
            with self.assertRaises(FileExistsError):
                write_final_release(root / "release.json", self.binding, self.threshold, **args)
            scores = score_final(self.detector, rows, self.binding, self.threshold, release=release)
            save_final_evaluation(root / "final.json", self.binding, rows, scores, self.threshold, release=release)
            with self.assertRaises(FileExistsError):
                save_final_evaluation(root / "final.json", self.binding, rows, scores, self.threshold, release=release)


if __name__ == "__main__":
    unittest.main()
