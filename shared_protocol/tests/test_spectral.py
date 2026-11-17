"""Synthetic saved features and hand-assigned model states; NEVER fit a model."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
import math
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

from shared_protocol.calibration import calibrate_benign
from shared_protocol.evaluation import evaluate, score_evaluation
from shared_protocol.manifest import ProtocolError, canonical_json, digest, write_new_json
from shared_protocol.scoring import score_development
from shared_protocol.tests.test_phase2 import setup_case, synthetic_view
from shared_protocol.detectors.spectral import (
    BASELINE_ID, BASELINE_SCHEMA, FeatureRecord, FeatureSchema, FrozenSpectralModel,
    SpectralDetector, export_fitted_model, load_features, load_model, prepare_features, save_model,
)
from shared_protocol.detectors.spectral.features import from_padbench_record
from shared_protocol.detectors.spectral.schema import EXTRACTOR_SHA256, LEGACY_FEATURE_NAMES, training_recipe


def feature(row, logit=0):
    # mean=2, scale=2, coefficient[0]=1, intercept=-1. All values are synthetic.
    return FeatureRecord(row["adapter_id"], row["sha256"], row["source_revision"], BASELINE_SCHEMA.sha256,
                         EXTRACTOR_SHA256, (2 + 2 * (logit + 1),) + (2.0,) * 9,
                         digest(["synthetic source record", row["adapter_id"], logit]),
                         "synthetic test", "Hand-authored features, no checkpoint or extraction")


def spectral_case():
    manifest, plan, binding, _ = setup_case()
    train = synthetic_view(manifest, binding, "train")
    train_features = prepare_features(binding, train, "train", [feature(row) for row in train])
    data = {
        "schema_version": 1, "artifact_kind": "synthetic_fixture", "baseline_id": BASELINE_ID,
        "schema_sha256": BASELINE_SCHEMA.sha256, "extractor_sha256": EXTRACTOR_SHA256,
        "protocol_sha256": binding.sha256, "split_seed": plan["policy"]["seed"],
        "training_recipe": training_recipe(plan["policy"]["seed"]), "fit_ids": list(binding.ids("train")),
        "fit_records_sha256": binding.records_digest("train"), "fit_features_sha256": train_features.sha256,
        "scaler": {"mean": [2.0] * 10, "scale": [2.0] * 10, "var": [4.0] * 10, "n_samples_seen": len(train)},
        "classifier": {"classes": [0, 1], "coef": [1.0] + [0.0] * 9, "intercept": -1.0, "n_iter": 1},
        "software": {key: "synthetic-fixture" for key in ("python", "numpy", "scipy", "sklearn")},
        "verification": {"verified_by": "synthetic test", "evidence": "Hand-assigned state, no model trained"},
    }
    model = FrozenSpectralModel.from_dict(data, binding, split_seed=plan["policy"]["seed"])
    return manifest, plan, binding, train_features, model


class SpectralCase(unittest.TestCase):
    def setUp(self):
        self.manifest, self.plan, self.binding, self.train_features, self.model = spectral_case()
        self.cal = synthetic_view(self.manifest, self.binding, "calibration")
        self.cal_features = prepare_features(self.binding, self.cal, "calibration",
                                            [feature(row, i / 10) for i, row in enumerate(sorted(self.cal, key=lambda r: r["adapter_id"]))])
        self.detector = SpectralDetector(self.model, self.binding, self.cal_features, "synthetic_analytic")
        self.scores = score_development(self.detector, self.cal, self.binding, "calibration")
        self.threshold = calibrate_benign(self.binding, self.cal, self.scores)


class SchemaTests(unittest.TestCase):
    def test_baseline_order_and_dimensions(self):
        self.assertEqual(len(BASELINE_SCHEMA.names), 10)
        self.assertEqual(BASELINE_SCHEMA.names[0], "layer20.q_proj.sigma1")
        self.assertEqual(BASELINE_SCHEMA.names[5], "layer20.v_proj.sigma1")
        self.assertEqual(tuple(name.removeprefix("layer20.") for name in BASELINE_SCHEMA.names), LEGACY_FEATURE_NAMES)

    def test_future_layers_are_explicit_and_do_not_redefine_baseline(self):
        future = FeatureSchema((0, 20, 31))
        self.assertEqual(len(future.names), 30)
        self.assertNotEqual(future.sha256, BASELINE_SCHEMA.sha256)
        for layers in ((20, 0), (20, 20), (), (True,), (-1,)):
            with self.assertRaises(ProtocolError):
                FeatureSchema(layers)

    def test_vendored_extractor_is_byte_identical_to_recorded_source(self):
        from shared_protocol.detectors.spectral import schema
        root = Path(schema.__file__).parent
        self.assertEqual(hashlib.sha256((root / "vendor/spectral_detector.py").read_bytes()).hexdigest(), EXTRACTOR_SHA256)
        self.assertIn("David Puertolas Merenciano", (root / "vendor/NOTICE").read_text())
        self.assertIn("Apache License", (root / "vendor/LICENSE").read_text())

    def test_training_recipe_is_unchanged_and_explicit(self):
        recipe = training_recipe(42)
        self.assertEqual(recipe["classifier"]["C"], 0.1)
        self.assertEqual(recipe["classifier"]["class_weight"], "balanced")
        self.assertEqual(recipe["classifier"]["max_iter"], 1000)
        self.assertEqual(recipe["classifier"]["random_state"], 42)
        self.assertEqual(recipe["input_dtype"], "float64")


class FeatureTests(SpectralCase):
    def test_roundtrip_and_immutable_feature_records(self):
        row = self.cal_features.records[0]
        self.assertEqual(FeatureRecord.from_dict(row.to_dict()), row)
        with self.assertRaises(FrozenInstanceError):
            row.values = (0,) * 10

    def test_bad_feature_values_and_evidence_rejected(self):
        row = self.cal_features.records[0]
        for change in ({"values": (float("nan"),) * 10}, {"values": (True,) * 10},
                       {"values": (float("inf"),) * 10}, {"verified_by": "unknown"},
                       {"source_record_sha256": None}, {"values": list(row.values)}):
            with self.subTest(change=change), self.assertRaises(ProtocolError):
                replace(row, **change)

    def test_wrong_schema_dimensions_identity_and_source_rejected(self):
        for change in ({"values": (0,) * 20}, {"schema_sha256": FeatureSchema((1, 20)).sha256},
                       {"adapter_sha256": digest("wrong checkpoint")}, {"source_revision": "other-revision"},
                       {"extractor_sha256": digest("modified equations")}, {"adapter_id": "other"}):
            rows = list(self.cal_features.records)
            rows[0] = replace(rows[0], **change)
            with self.subTest(change=change), self.assertRaises(ProtocolError):
                prepare_features(self.binding, self.cal, "calibration", rows)

    def test_extra_missing_and_duplicate_features_rejected(self):
        original = list(self.cal_features.records)
        for rows in (original[:-1], original + original[:1]):
            with self.assertRaises(ProtocolError):
                prepare_features(self.binding, self.cal, "calibration", rows)

    def test_unverified_adapter_metadata_rejected(self):
        rows = deepcopy(self.cal)
        rows[0]["verification"] = {}
        with self.assertRaises(ProtocolError):
            prepare_features(self.binding, rows, "calibration", self.cal_features.records)

    def test_evaluation_requires_frozen_threshold_before_reading(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("read")):
            with self.assertRaises(ProtocolError):
                load_features(self.binding, None, "evaluation", {}, {})

    def test_final_features_rejected_even_with_threshold(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("final access")):
            with self.assertRaises(ProtocolError):
                load_features(self.binding, None, "final_holdout", {}, {}, threshold=self.threshold)
            with self.assertRaises(ProtocolError):
                prepare_features(self.binding, None, "final_holdout", (), threshold=self.threshold)

    def test_loading_only_authorized_files_and_checking_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, hashes = {}, {}
            for i, row in enumerate(self.cal_features.records):
                path = Path(directory) / f"feature-{i}.json"
                write_new_json(path, row.to_dict())
                paths[row.adapter_id] = path
                hashes[row.adapter_id] = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = load_features(self.binding, self.cal, "calibration", paths, hashes)
            self.assertEqual(loaded, self.cal_features)
            hashes[self.cal_features.records[0].adapter_id] = "0" * 64
            with self.assertRaisesRegex(ProtocolError, "checksum"):
                load_features(self.binding, self.cal, "calibration", paths, hashes)
            paths["final-extra"] = "must-not-open"
            with patch.object(Path, "read_bytes", side_effect=AssertionError("read")):
                with self.assertRaises(ProtocolError):
                    load_features(self.binding, self.cal, "calibration", paths, hashes)

    def test_no_live_checkpoint_access_when_getting_saved_values(self):
        with patch.object(Path, "open", side_effect=AssertionError("weights opened")):
            adapter = self.cal_features.allowed_inputs[0]
            self.assertEqual(self.cal_features.get(adapter), self.cal_features.records[0].values)
            with self.assertRaises(ProtocolError):
                self.cal_features.get(replace(adapter, sha256=digest("another adapter")))

    def test_collector_conversion_preserves_values_without_inference(self):
        row = self.cal[0]
        old = {"folder": row["adapter_id"], "repo_revision": row["source_revision"], "layer": 20,
               "projections": ["q_proj", "v_proj"], "feature_names": list(LEGACY_FEATURE_NAMES),
               "formula": "original", "extractor_sha256": EXTRACTOR_SHA256, "rank": 16,
               "alpha": row["lora_alpha"], "base_model_name_or_path": row["base_model"],
               "file_sha256": {"adapter_model.safetensors": row["sha256"]},
               "target_modules": ["q_proj", "v_proj"], "label": 0, "features": list(range(10))}
        args = dict(source_record_sha256=digest("synthetic collector bytes"), verified_by="test",
                    evidence="Synthetic explicit label mapping and source")
        converted = from_padbench_record(old, row, **args)
        self.assertEqual(converted.values, tuple(range(10)))
        for key, value in (("label", 1), ("layer", 19), ("rank", 256), ("feature_names", list(reversed(LEGACY_FEATURE_NAMES))),
                           ("extractor_sha256", None), ("base_model_name_or_path", None)):
            with self.subTest(key=key), self.assertRaises(ProtocolError):
                from_padbench_record({**old, key: value}, row, **args)


class ModelTests(SpectralCase):
    def test_import_rejects_changed_recipe_classes_or_missing_state(self):
        for field, value in (("C", 1.0), ("class_weight", None), ("max_iter", 100), ("solver", "liblinear")):
            data = self.model.to_dict()
            data["training_recipe"]["classifier"][field] = value
            with self.assertRaises(ProtocolError):
                FrozenSpectralModel.from_dict(data, self.binding, split_seed=42)
        for field, value in (("classes", [1, 0]), ("coef", [0] * 20), ("intercept", float("nan"))):
            data = self.model.to_dict()
            data["classifier"][field] = value
            with self.assertRaises((ProtocolError, ValueError)):
                FrozenSpectralModel.from_dict(data, self.binding, split_seed=42)

    def test_scaler_must_be_saved_not_fitted_from_scoring_data(self):
        for key, value in (("scale", [0] * 10), ("mean", None), ("n_samples_seen", 999)):
            data = self.model.to_dict()
            data["scaler"][key] = value
            with self.assertRaises(ProtocolError):
                FrozenSpectralModel.from_dict(data, self.binding, split_seed=42)

    def test_training_provenance_cannot_include_calibration_or_holdout(self):
        for part in ("validation", "calibration", "evaluation", "final_holdout"):
            data = self.model.to_dict()
            data["fit_ids"] = sorted([*data["fit_ids"], self.binding.ids(part)[0]])
            data["scaler"]["n_samples_seen"] += 1
            with self.assertRaises(ProtocolError):
                FrozenSpectralModel.from_dict(data, self.binding, split_seed=42)

    def test_seed_protocol_and_schema_are_bound(self):
        with self.assertRaises(ProtocolError):
            FrozenSpectralModel.from_dict(self.model.to_dict(), self.binding, split_seed=12)
        for field, value in (("protocol_sha256", digest("another split")), ("schema_sha256", FeatureSchema((0, 20)).sha256),
                             ("fit_records_sha256", digest("changed metadata"))):
            with self.assertRaises(ProtocolError):
                FrozenSpectralModel.from_dict({**self.model.to_dict(), field: value}, self.binding, split_seed=42)

    def test_model_state_is_immutable_and_synthetic_is_explicit(self):
        with self.assertRaises(FrozenInstanceError):
            self.model.payload_json = "{}"
        exported = self.model.to_dict()
        exported["scaler"]["mean"][0] = 999
        self.assertEqual(self.model.to_dict()["scaler"]["mean"][0], 2)
        self.assertIn("synthetic_fixture", self.detector.spec.name)
        with self.assertRaises(ProtocolError):
            replace(self.model, payload_json=canonical_json({**self.model.to_dict(), "artifact_kind": "fitted_baseline"})).predict(
                [[0] * 10], backend="synthetic_analytic")

    def test_model_file_roundtrip_exclusive_creation_and_pinned_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_model(path, self.model, self.binding)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(load_model(path, self.binding, split_seed=42, expected_file_sha256=expected), self.model)
            with self.assertRaises(FileExistsError):
                save_model(path, self.model, self.binding)
            with self.assertRaises(ProtocolError):
                load_model(path, self.binding, split_seed=42, expected_file_sha256="0" * 64)

    def test_analytic_synthetic_score_formula_extreme_values_and_direction(self):
        values = [(8,) + (2,) * 9, (2004,) + (2,) * 9, (-1996,) + (2,) * 9]
        scores = self.model.predict(values, backend="synthetic_analytic")
        self.assertAlmostEqual(scores[0], 1 / (1 + math.exp(-2)), places=15)
        self.assertEqual(scores[1:], [1, 0])


class IntegrationTests(SpectralCase):
    def test_saved_features_to_shared_calibration_and_evaluation(self):
        rows = synthetic_view(self.manifest, self.binding, "evaluation")
        values = {"eval-benign": 2.0, "eval-benign-safe": 1.8, "eval-malicious": 1.9, "eval-missed": 0.0}
        features = prepare_features(self.binding, rows, "evaluation", [feature(row, values[row["adapter_id"]]) for row in rows],
                                    threshold=self.threshold)
        detector = SpectralDetector(self.model, self.binding, features, "synthetic_analytic")
        self.assertEqual(detector.spec, self.detector.spec)
        scores = score_evaluation(detector, rows, self.binding, self.threshold)
        report = evaluate(self.binding, rows, scores, self.threshold)
        self.assertEqual(report["overall"]["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(report["overall"]["auroc"]["value"], 0.25)
        self.assertAlmostEqual(self.threshold.value, 1 / (1 + math.exp(-1.8)))
        self.assertEqual(report["operating_point"]["comparison"], ">")

    def test_scores_do_not_fit_or_modify_frozen_model(self):
        before = self.model.sha256
        repeated = score_development(self.detector, self.cal, self.binding, "calibration")
        self.assertEqual(repeated, self.scores)
        self.assertEqual(before, self.model.sha256)

    def test_different_stage_inputs_cannot_reuse_a_feature_view(self):
        with self.assertRaises(ProtocolError):
            self.detector.score(self.train_features.allowed_inputs)

    def test_changed_model_cannot_reuse_shared_threshold(self):
        data = self.model.to_dict()
        data["classifier"]["intercept"] = 4.0
        changed = FrozenSpectralModel.from_dict(data, self.binding, split_seed=42)
        detector = SpectralDetector(changed, self.binding, self.cal_features, "synthetic_analytic")
        with self.assertRaises(ProtocolError):
            score_evaluation(detector, None, self.binding, self.threshold)

    def test_no_live_extractor_or_torch_import_required(self):
        import sys
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("safetensors", sys.modules)
        self.assertNotIn("shared_protocol.detectors.spectral.vendor.spectral_detector", sys.modules)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn optional parity dependency is unavailable")
class SklearnParityTests(SpectralCase):
    def objects(self):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        # Assign fitted-shaped attributes by hand; never learn from even synthetic data.
        # Pin the preserved recipe: sklearn 1.9 defaults penalty to "deprecated".
        # Do not relax export validation to accommodate an implicit test default.
        recipe = training_recipe(42)
        scaler = StandardScaler(**recipe["scaler"])
        scaler.mean_ = np.asarray([2.0] * 10)
        scaler.scale_ = np.asarray([2.0] * 10)
        scaler.var_ = np.asarray([4.0] * 10)
        scaler.n_features_in_ = 10
        scaler.n_samples_seen_ = len(self.binding.ids("train"))
        classifier = LogisticRegression(**recipe["classifier"])
        classifier.coef_ = np.asarray([[1.0] + [0.0] * 9])
        classifier.intercept_ = np.asarray([-1.0])
        classifier.classes_ = np.asarray([0, 1])
        classifier.n_features_in_ = 10
        classifier.n_iter_ = np.asarray([1])
        return scaler, classifier

    def test_export_and_sklearn_inference_parity_without_any_fit_calls(self):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        scaler, classifier = self.objects()
        train = synthetic_view(self.manifest, self.binding, "train")
        with patch.object(StandardScaler, "fit", side_effect=AssertionError("training prohibited")), \
                patch.object(StandardScaler, "partial_fit", side_effect=AssertionError("training prohibited")), \
                patch.object(LogisticRegression, "fit", side_effect=AssertionError("training prohibited")):
            model = export_fitted_model(scaler, classifier, self.binding, train, self.train_features, split_seed=42,
                                        verified_by="synthetic test", evidence="Hand-assigned state, no training",
                                        artifact_kind="synthetic_fixture")
            vectors = np.asarray([feature(self.cal[0], value).values for value in (-1000, -2, 0, 1.8, 2, 1000)], dtype=np.float64)
            expected = classifier.predict_proba(scaler.transform(vectors))[:, 1]
            actual = model.predict(vectors.tolist())
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_allclose(model.predict(vectors.tolist(), backend="synthetic_analytic"), expected, rtol=1e-15, atol=1e-15)
            detector = SpectralDetector(model, self.binding, self.cal_features)
            batch = score_development(detector, self.cal, self.binding, "calibration")
            threshold = calibrate_benign(self.binding, self.cal, batch)
            self.assertAlmostEqual(threshold.value, self.threshold.value)
            rows = synthetic_view(self.manifest, self.binding, "evaluation")
            logits = {"eval-benign": 2.0, "eval-benign-safe": 1.8, "eval-malicious": 1.9, "eval-missed": 0.0}
            features = prepare_features(self.binding, rows, "evaluation",
                                        [feature(row, logits[row["adapter_id"]]) for row in rows], threshold=threshold)
            eval_detector = SpectralDetector(model, self.binding, features)
            scored = score_evaluation(eval_detector, rows, self.binding, threshold)
            report = evaluate(self.binding, rows, scored, threshold)
            self.assertEqual(report["overall"]["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})

    def test_export_rejects_wrong_recipe_unfitted_or_non_train_inputs(self):
        scaler, classifier = self.objects()
        args = dict(split_seed=42, verified_by="synthetic test", evidence="Synthetic only", artifact_kind="synthetic_fixture")
        train = synthetic_view(self.manifest, self.binding, "train")
        # Keep every recipe field strict, including the changed sklearn sentinel.
        cases = ((scaler, "with_mean", False, "True"),
                 (classifier, "C", 1.0, "0.1"),
                 (classifier, "class_weight", None, "'balanced'"),
                 (classifier, "max_iter", 100, "1000"),
                 (classifier, "penalty", "deprecated", "'l2'"),
                 (classifier, "penalty", "l1", "'l2'"),
                 (classifier, "solver", "liblinear", "'lbfgs'"))
        for estimator, key, value, expected in cases:
            previous = estimator.get_params(deep=False)[key]
            with self.subTest(estimator=type(estimator).__name__, parameter=key, value=value):
                estimator.set_params(**{key: value})
                try:
                    with self.assertRaises(ProtocolError) as caught:
                        export_fitted_model(scaler, classifier, self.binding, train, self.train_features, **args)
                    self.assertIn(f"{type(estimator).__name__}.{key}: expected {expected}, actual {value!r}", str(caught.exception))
                finally:
                    estimator.set_params(**{key: previous})
        del classifier.coef_
        with self.assertRaisesRegex(ProtocolError, "Missing/unsupported fitted state"):
            export_fitted_model(scaler, classifier, self.binding, train, self.train_features, **args)
        with self.assertRaises(ProtocolError):
            export_fitted_model(scaler, classifier, self.binding, self.cal, self.cal_features, **args)

    def test_inference_environment_mismatch_never_silently_falls_back(self):
        with self.assertRaisesRegex(ProtocolError, "environment differs"):
            self.model.predict([[0] * 10], backend="sklearn")


if __name__ == "__main__":
    unittest.main()
