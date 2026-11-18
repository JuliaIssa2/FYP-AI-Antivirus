"""Offline tests: synthetic metadata/features and isolated real extraction/fitting."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from shared_protocol.manifest import METADATA_FIELDS, ProtocolError, digest, write_new_json
from exploratory import natour_selection as selection
from exploratory import natour_spectral_training as pipeline


def fixtures():
    rows, sides = [], []
    for task in ("alpaca", "code"):
        for label in ("benign", "malicious"):
            for seed in range(3):
                i = len(rows)
                key = f"nat-{i:016x}"
                row = dict.fromkeys(METADATA_FIELDS)
                row.update(adapter_id=key, verification={}, task=task, label=label,
                           leakage_group=f"{task}-seed{seed}", lora_rank=16, lora_alpha=32, seed=seed,
                           target_modules=["q_proj", "v_proj"], base_model="NousResearch/Llama-2-7b-hf",
                           source_revision=selection.REVISION, sha256=hashlib.sha256(key.encode()).hexdigest(),
                           path_or_source=selection.PREFIX + f"adapters/synthetic/adapter_{i}/adapter_model.safetensors",
                           dataset_provenance={"dataset_id": task + "-synthetic", "revision": None,
                                               "split": "train", "preprocessing": "synthetic fixture only"})
                rows.append(row)
                sides.append({"adapter_id": key, "cohort": selection.COHORT,
                              "label_int": int(label == "malicious"), "base_model_as_trained": row["base_model"],
                              "base_revision": None})
    return {"schema_version": 1, "adapters": rows}, {"schema_version": 1, "adapters": sides}


def feature_fixtures(records, environment):
    result = {}
    for i, (key, record) in enumerate(records.items()):
        validation = {"sha256": record["adapter"]["sha256"], "hash_matches_manifest": True,
                      "keys_dimensions_dtypes_finite_passed": True, "tensor_count": 4, "tensors": []}
        values = [float(i * (j + 1) + j * j) for j in range(10)]
        result[key] = pipeline.feature_payload(record, values, validation, environment)
    return result


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.sidecar = fixtures()
        self.records = selection.catalog(self.manifest, self.sidecar)

    def test_default_twelve_deterministic_task_group_hash_disjoint(self):
        plan = selection.make_plan(self.records)
        reordered = selection.catalog({**self.manifest, "adapters": self.manifest["adapters"][::-1]},
                                      {**self.sidecar, "adapters": self.sidecar["adapters"][::-1]})
        self.assertEqual(selection.make_plan(reordered), plan)
        self.assertEqual(len(plan["selected"]), 12)
        for field in ("task", "leakage_group", "checkpoint_sha256", "adapter_id"):
            train = {r[field] for r in plan["selected"] if r["partition"] == "train"}
            test = {r[field] for r in plan["selected"] if r["partition"] == "test"}
            self.assertFalse(train & test)

    def test_phase2_missing_cohort_and_official_markers_never_selected(self):
        for cohort in ("phase2", None, "final_holdout"):
            sides = deepcopy(self.sidecar)
            sides["adapters"][0]["cohort"] = cohort
            records = selection.catalog(self.manifest, sides)
            self.assertNotIn(self.manifest["adapters"][0]["adapter_id"], records)
            with self.assertRaisesRegex(ProtocolError, "not authorized"):
                selection.make_plan(records, explicit_ids=list(self.records))
        manifest = deepcopy(self.manifest)
        manifest["adapters"][0]["dataset_provenance"]["split"] = "official evaluation"
        with self.assertRaisesRegex(ProtocolError, "prohibited"):
            selection.catalog(manifest, self.sidecar)

    def test_ambiguous_join_and_bad_identity_rejected(self):
        sides = deepcopy(self.sidecar)
        sides["adapters"].append(sides["adapters"][0])
        with self.assertRaisesRegex(ProtocolError, "duplicate"):
            selection.catalog(self.manifest, sides)
        for key, value in (("source_revision", "main"), ("sha256", None),
                           ("path_or_source", selection.PREFIX + "adapters/../escape/adapter_model.safetensors"),
                           ("lora_rank", 256), ("task", None), ("leakage_group", None),
                           ("base_model", "meta-llama/Llama-2-7b-hf")):
            manifest = deepcopy(self.manifest)
            manifest["adapters"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ProtocolError):
                selection.catalog(manifest, self.sidecar)

    def test_cross_task_groups_duplicate_hash_and_label_conflicts_rejected(self):
        manifest = deepcopy(self.manifest)
        manifest["adapters"][6]["leakage_group"] = manifest["adapters"][0]["leakage_group"]
        with self.assertRaisesRegex(ProtocolError, "crosses tasks"):
            selection.catalog(manifest, self.sidecar)
        manifest = deepcopy(self.manifest)
        manifest["adapters"][6]["sha256"] = manifest["adapters"][0]["sha256"]
        with self.assertRaisesRegex(ProtocolError, "Duplicate sha256"):
            selection.catalog(manifest, self.sidecar)
        sidecar = deepcopy(self.sidecar)
        sidecar["adapters"][0]["label_int"] = 1
        with self.assertRaisesRegex(ProtocolError, "conflicting"):
            selection.catalog(self.manifest, sidecar)

    def test_insufficient_classes_and_overlapping_tasks_abort(self):
        for train, test in (("alpaca", "alpaca"), ("alpaca", "absent")):
            with self.assertRaises(ProtocolError):
                selection.make_plan(self.records, train, test)
        ids = [k for k, v in self.records.items() if v["adapter"]["label"] == "benign"]
        with self.assertRaisesRegex(ProtocolError, "both claimed classes"):
            selection.make_plan(self.records, explicit_ids=ids)

    def test_plan_tampering_metadata_drift_and_unknown_preservation(self):
        plan = selection.make_plan(self.records)
        for field, value in (("partition", "train"), ("label", "benign")):
            changed = deepcopy(plan)
            changed["selected"][-1][field] = value
            with self.assertRaisesRegex(ProtocolError, "Stale"):
                selection.validate_plan(changed, self.records)
        changed = deepcopy(self.records)
        changed[next(iter(changed))]["training_metadata"]["base_revision"] = "new-claim"
        with self.assertRaises(ProtocolError):
            selection.validate_plan(plan, changed)
        self.assertTrue(all(v["adapter"]["verification"] == {} for v in self.records.values()))
        self.assertIsNone(next(iter(self.records.values()))["adapter"]["dataset_provenance"]["revision"])

    def test_explicit_expansion_is_exact_and_does_not_add_other_ids(self):
        keys = [r["adapter_id"] for r in self.manifest["adapters"] if r["seed"] == 0]
        plan = selection.make_plan(self.records, explicit_ids=keys)
        self.assertEqual({r["adapter_id"] for r in plan["selected"]}, set(keys))
        self.assertEqual(selection.validate_plan(plan, self.records), plan)


class CacheAndTrainingTests(unittest.TestCase):
    def setUp(self):
        manifest, sidecar = fixtures()
        self.records = selection.catalog(manifest, sidecar)
        self.plan = selection.make_plan(self.records)
        self.environment = {"fixture": "explicit synthetic runtime; no real feature extraction"}
        self.features = feature_fixtures(self.records, self.environment)

    def test_cache_rejects_stale_provenance_direction_schema_and_bad_numbers(self):
        key = next(iter(self.records))
        payload = self.features[key]
        cases = {"source_revision": "main", "checkpoint_sha256": "0" * 64,
                 "extractor_sha256": "0" * 64, "schema_sha256": "0" * 64,
                 "claimed_label": "malicious", "metadata_status": "verified",
                 "feature_names": list(reversed(payload["feature_names"])), "values": [1.] * 9,
                 "extraction_environment": {}, "metadata_sha256": "0" * 64}
        for field, value in cases.items():
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                pipeline.validate_feature({**payload, field: value}, self.records[key], self.environment)
        for values in ([True] * 10, [float("nan")] * 10, [float("inf")] * 10):
            with self.assertRaises(ProtocolError):
                pipeline.validate_feature({**payload, "values": values}, self.records[key], self.environment)

    def test_cache_file_and_index_checksums_and_path_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = {}
            for key, feature in self.features.items():
                path = root / (key + ".json")
                pipeline.save_artifact(path, feature)
                entries[key] = {"file": path.name, "sha256": pipeline.smoke.sha256_file(path)}
            index = root / "index.json"
            pipeline.save_artifact(index, {"kind": "exploratory_feature_index_v1", "entries": entries})
            checksum = pipeline.smoke.sha256_file(index)
            result = pipeline.cached_features(index, checksum, self.plan["selected"], self.records, self.environment)
            self.assertEqual(result, self.features)
            with self.assertRaisesRegex(ProtocolError, "index file checksum"):
                pipeline.cached_features(index, "0" * 64, self.plan["selected"], self.records, self.environment)
            with self.assertRaisesRegex(ProtocolError, "trusted"):
                pipeline.cached_features(index, None, self.plan["selected"], self.records, self.environment)
            key = next(iter(entries))
            (root / entries[key]["file"]).write_text("{}")
            with self.assertRaisesRegex(ProtocolError, "cache file checksum"):
                pipeline.cached_features(index, checksum, self.plan["selected"], self.records, self.environment)
            entries[key]["file"] = "../outside.json"
            other = root / "bad-index.json"
            pipeline.save_artifact(other, {"kind": "exploratory_feature_index_v1", "entries": entries})
            with self.assertRaisesRegex(ProtocolError, "Unsafe"):
                pipeline.cached_features(other, pipeline.smoke.sha256_file(other), self.plan["selected"], self.records, self.environment)

    def test_scaler_and_classifier_fit_train_only_with_exact_recipe(self):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        train = [r for r in self.plan["selected"] if r["partition"] == "train"]
        expected_x = np.asarray([self.features[r["adapter_id"]]["values"] for r in train], dtype=np.float64)
        scaler_fit, classifier_fit = StandardScaler.fit, LogisticRegression.fit
        calls = []
        def scaler_spy(obj, x, y=None, **kwargs):
            np.testing.assert_array_equal(x, expected_x)
            self.assertEqual(x.dtype, np.float64)
            calls.append("scaler")
            return scaler_fit(obj, x, y, **kwargs)
        def classifier_spy(obj, x, y, **kwargs):
            self.assertEqual(x.shape, (6, 10))
            np.testing.assert_array_equal(y, [int(r["label"] == "malicious") for r in train])
            for key, value in pipeline.training_recipe(42)["classifier"].items():
                self.assertEqual(obj.get_params()[key], value)
            calls.append("classifier")
            return classifier_fit(obj, x, y, **kwargs)
        with patch.object(StandardScaler, "fit", scaler_spy), patch.object(LogisticRegression, "fit", classifier_spy):
            state, predictions = pipeline.fit_classifier(self.plan, self.records, self.features, self.environment)
        self.assertEqual(calls, ["scaler", "classifier"])
        np.testing.assert_array_equal(state["scaler"]["mean"], expected_x.mean(axis=0))
        self.assertEqual(state["scaler"]["n_samples_seen"], 6)
        self.assertEqual(state["classifier"]["classes"], [0, 1])
        self.assertEqual(len(predictions), 12)

    def test_test_feature_changes_cannot_change_fitted_state(self):
        state, _ = pipeline.fit_classifier(self.plan, self.records, self.features, self.environment)
        changed = deepcopy(self.features)
        for row in self.plan["selected"]:
            if row["partition"] == "test":
                changed[row["adapter_id"]]["values"] = [1e6] * 10
        other, _ = pipeline.fit_classifier(self.plan, self.records, changed, self.environment)
        self.assertEqual(state, other)

    def test_repeatability_json_roundtrip_no_pickle_and_no_inference_fitting(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        first = pipeline.fit_classifier(self.plan, self.records, self.features, self.environment)
        self.assertEqual(first, pipeline.fit_classifier(self.plan, self.records, self.features, self.environment))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            pipeline.save_artifact(path, first[0])
            with self.assertRaises(FileExistsError):
                pipeline.save_artifact(path, first[0])
            restored = pipeline.read_artifact(path)
            with patch.object(StandardScaler, "fit", side_effect=AssertionError("refit")), patch.object(
                    LogisticRegression, "fit", side_effect=AssertionError("refit")):
                actual = pipeline.predict_saved(restored, self.plan, self.features, self.environment)
            self.assertEqual(actual, [r["malicious_probability"] for r in first[1]])
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_invalid_features_recipe_environment_and_plan_abort(self):
        with self.assertRaises(ProtocolError):
            pipeline.fit_classifier(self.plan, self.records, {}, self.environment)
        changed = deepcopy(self.plan)
        changed["selected"][-1]["partition"] = "train"
        with self.assertRaises(ProtocolError):
            pipeline.fit_classifier(changed, self.records, self.features, self.environment)
        state, _ = pipeline.fit_classifier(self.plan, self.records, self.features, self.environment)
        for field, value in (("runtime", {}), ("recipe", {}), ("official_protocol_eligible", True)):
            with self.assertRaises(ProtocolError):
                pipeline.predict_saved({**state, field: value}, self.plan, self.features, self.environment)

    def test_end_to_end_offline_safetensors_cache_then_training(self):
        # Only this child imports torch/safetensors; old shared tests stay isolated.
        code = '''
import json, pathlib, shutil, tempfile
from unittest.mock import patch
import numpy as np
from safetensors.numpy import save_file
from exploratory import natour_spectral_training as p
from exploratory import natour_selection as s
from exploratory.tests.test_natour_spectral_training import fixtures
with tempfile.TemporaryDirectory() as folder:
    root = pathlib.Path(folder)
    manifest, sidecar = fixtures()
    sources = {}
    for i, row in enumerate(manifest["adapters"]):
        weights = {}
        for projection in ("q_proj","v_proj"):
            prefix = "base_model.model.model.layers.20.self_attn." + projection
            weights[prefix+".lora_A.weight"] = (np.arange(512,dtype=np.float32).reshape(16,32)%13+i+1)/100
            weights[prefix+".lora_B.weight"] = (np.arange(512,dtype=np.float32).reshape(32,16)%17-i)/100
        file = root / (row["adapter_id"]+".safetensors")
        save_file(weights,str(file))
        row["sha256"] = p.smoke.sha256_file(file)
        sources[s.filename(row)] = file
    m, side = root/"manifest.json",root/"sidecar.json"
    m.write_text(json.dumps(manifest)); side.write_text(json.dumps(sidecar))
    records = s.catalog(manifest,sidecar)
    plan = s.make_plan(records)
    selection_path = root/"selection.json"
    p.save_artifact(selection_path,plan)
    calls = []
    def download(**args):
        assert args["repo_id"]==p.smoke.REPO_ID and args["revision"]==p.smoke.REVISION and args["repo_type"]=="dataset"
        assert args["filename"] in sources
        dest = pathlib.Path(args["cache_dir"]) / str(len(calls)) / "adapter_model.safetensors"
        dest.parent.mkdir(parents=True)
        shutil.copyfile(sources[args["filename"]],dest)
        calls.append(args["filename"])
        return str(dest)
    inspect = p.smoke.inspect_checkpoint
    with patch("huggingface_hub.hf_hub_download",side_effect=download), patch.object(
            p.smoke,"inspect_checkpoint",side_effect=lambda path,h:inspect(path,h,hidden_size=32)):
        report = p.run(m,side,selection_path,root/"run")
    assert len(calls)==12 and len(set(calls))==12
    assert report["counts"]=={"train":{"benign":3,"malicious":3},"test":{"benign":3,"malicious":3}}
    # Reconstruct an interrupted original run: keep three completed features and
    # four downloaded checkpoints, so the fourth is extracted without a download.
    partial = root/"partial"
    (partial/"features").mkdir(parents=True)
    p.save_artifact(partial/"selection.json",plan)
    snapshot = partial/"hub/datasets--MhmdNat--adapterlab-adapters/snapshots"/p.smoke.REVISION
    preserved = {}
    for i,row in enumerate(plan["selected"][:4]):
        key = row["adapter_id"]
        name = s.filename(records[key]["adapter"])
        checkpoint = snapshot/name
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(sources[name],checkpoint)
        if i < 3:
            cache = partial/"features"/(key+".json")
            shutil.copyfile(root/"run/features"/(key+".json"),cache)
            preserved[cache] = cache.read_bytes()
    preserved[partial/"selection.json"] = (partial/"selection.json").read_bytes()
    # Recomputed payload checksum alone must not allow corrupted feature values.
    corrupt = next(iter(preserved))
    payload = p.read_artifact(corrupt)
    original = corrupt.read_bytes()
    payload["values"][0] += 1
    p.write_new_json(partial/"bad.json",{})  # unrelated file is not a feature entry
    corrupt.write_text(p.canonical_json({"payload":payload,"payload_sha256":p.digest(payload)}))
    with patch.object(p.smoke,"inspect_checkpoint",side_effect=lambda path,h:inspect(path,h,hidden_size=32)):
        try:
            p.resume_features(partial,plan,records,p.runtime())
        except p.ProtocolError as exc:
            assert "differ" in str(exc)
        else:
            raise AssertionError("corrupt resumed features accepted")
    corrupt.write_bytes(original)
    previous_calls = len(calls)
    with patch("huggingface_hub.hf_hub_download",side_effect=download), patch.object(
            p.smoke,"inspect_checkpoint",side_effect=lambda path,h:inspect(path,h,hidden_size=32)):
        resumed = p.run(m,side,selection_path,partial,resume=True)
    assert len(calls)-previous_calls == 8
    assert resumed["predictions"] == report["predictions"]
    assert sum(r["origin"]=="revalidated_partial_run" for r in resumed["feature_provenance"]) == 3
    assert all(path.read_bytes()==content for path,content in preserved.items())
    with patch.object(p,"fetch_checkpoint",side_effect=AssertionError("completed resume must abort")):
        try:
            p.run(m,side,selection_path,partial,resume=True)
        except p.ProtocolError as exc:
            assert "completed" in str(exc)
        else:
            raise AssertionError("completed run resumed")
    index = root/"run/features/index.json"
    with patch.object(p,"fetch_checkpoint",side_effect=AssertionError("cache must prevent download")), patch.object(
            p.smoke,"extract_checked",side_effect=AssertionError("cache must prevent extraction")):
        repeat = p.run(m,side,selection_path,root/"repeat",index,report["cache_index_file_sha256"])
    assert report["predictions"] == repeat["predictions"]
    assert all(row["origin"]=="validated_feature_cache" for row in repeat["feature_provenance"])
    before = (root/"run/report.json").read_bytes()
    try:
        p.run(m,side,selection_path,root/"run")
    except p.ProtocolError:
        pass
    else:
        raise AssertionError("overwrite allowed")
    assert (root/"run/report.json").read_bytes()==before
    # Corrupt an actual cached file: abort before network or model fitting.
    entry = next(iter(p.read_artifact(index)["entries"].values()))
    (index.parent/entry["file"]).write_text("{}")
    with patch.object(p,"fetch_checkpoint",side_effect=AssertionError("no network on corrupt cache")), patch.object(
            p,"fit_classifier",side_effect=AssertionError("no fit on corrupt cache")):
        try:
            p.run(m,side,selection_path,root/"bad",index,report["cache_index_file_sha256"])
        except p.ProtocolError:
            pass
        else:
            raise AssertionError("corrupt cache accepted")
    assert not (root/"bad/report.json").exists()
'''
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(pipeline.__file__).resolve().parents[1]), str(Path.cwd())]),
                           HF_HUB_OFFLINE="1", HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_TELEMETRY="1", CUDA_VISIBLE_DEVICES="-1")
        result = subprocess.run([sys.executable, "-B", "-c", code], env=environment, text=True,
                                cwd=Path(pipeline.__file__).resolve().parents[1], capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
