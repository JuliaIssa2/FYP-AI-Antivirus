from __future__ import annotations

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

from shared_protocol.manifest import (
    METADATA_FIELDS, PRIMARY_MODEL, ProtocolError, canonical_json, load_manifest,
    validate_manifest, verification_issues, write_new_json,
)
from shared_protocol.splits import (
    PARTITIONS, create_split, development_records, load_split, save_split,
    validate_split,
)


def adapter(name, label="benign", task="development_task", attack="seen_attack", payload="seen_payload"):
    """Entirely synthetic metadata: hashes are fixture IDs, not weight hashes."""
    row = {
        "adapter_id": name,
        "path_or_source": f"fixture://nonexistent/{name}/adapter_model.safetensors",
        "source_revision": "synthetic-v1", "base_model": PRIMARY_MODEL,
        "task": task, "attack_family": attack if label == "malicious" else "none",
        "payload": payload if label == "malicious" else "none",
        "poisoning_rate": 0.03 if label == "malicious" else 0,
        "lora_rank": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"],
        "seed": 42, "label": label,
        "dataset_provenance": {"dataset_id": "synthetic", "revision": "v1",
                               "split": "train", "preprocessing": "synthetic test only"},
        "sha256": hashlib.sha256(name.encode()).hexdigest(), "leakage_group": name,
        "verification": {},
    }
    row["verification"] = {field: {"verified_by": "unit-test fixture",
                                   "evidence": "Synthetic data, not a real adapter"}
                           for field in METADATA_FIELDS}
    return row


def fixture():
    rows = [adapter(f"benign-{i}") for i in range(26)]
    rows += [adapter(f"malicious-{i}", "malicious") for i in range(8)]
    rows += [adapter("eval-benign", task="eval_task"),
             adapter("eval-malicious", "malicious", attack="unseen_attack"),
             adapter("final-benign", task="final_task"),
             adapter("final-malicious", "malicious", payload="final_payload")]
    policy = {
        "schema_version": 1, "seed": 42, "calibration_groups": 19,
        "validation_groups_per_stratum": 2,
        "evaluation": {"benign_tasks": ["eval_task"], "attack_families": ["unseen_attack"], "payloads": []},
        "final_holdout": {"benign_tasks": ["final_task"], "attack_families": [], "payloads": ["final_payload"]},
    }
    return {"schema_version": 1, "adapters": rows}, policy


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.policy = fixture()

    def test_valid_complete_manifest(self):
        validate_manifest(self.manifest, require_verified=True)

    def test_every_unknown_field_is_draft_only(self):
        for field in METADATA_FIELDS:
            with self.subTest(field=field):
                manifest = deepcopy(self.manifest)
                row = manifest["adapters"][0]
                row[field] = None
                del row["verification"][field]
                validate_manifest(manifest)
                self.assertIn(f"{field}: unknown", verification_issues(row))
                with self.assertRaises(ProtocolError):
                    create_split(manifest, self.policy)

    def test_known_but_unverified_is_blocked(self):
        self.manifest["adapters"][0]["verification"].pop("label")
        validate_manifest(self.manifest)
        with self.assertRaisesRegex(ProtocolError, "label: unverified"):
            create_split(self.manifest, self.policy)

    def test_unknown_cannot_be_marked_verified(self):
        self.manifest["adapters"][0]["sha256"] = None
        with self.assertRaisesRegex(ProtocolError, "Cannot verify unknown"):
            validate_manifest(self.manifest)

    def test_partial_provenance_is_unknown(self):
        row = self.manifest["adapters"][0]
        row["dataset_provenance"]["revision"] = None
        del row["verification"]["dataset_provenance"]
        validate_manifest(self.manifest)
        with self.assertRaisesRegex(ProtocolError, "dataset_provenance: unknown"):
            create_split(self.manifest, self.policy)

    def test_duplicate_identity_source_and_hash_are_rejected(self):
        for field in ("adapter_id", "path_or_source", "sha256"):
            with self.subTest(field=field):
                manifest = deepcopy(self.manifest)
                manifest["adapters"][1][field] = manifest["adapters"][0][field]
                with self.assertRaisesRegex(ProtocolError, "Duplicate " + field):
                    validate_manifest(manifest)

    def test_invalid_values_are_rejected(self):
        for field, value in [("label", 0), ("label", "unknown"), ("task", ""),
                             ("seed", True), ("lora_rank", 16.0), ("lora_alpha", float("inf")),
                             ("poisoning_rate", -1), ("poisoning_rate", 3),
                             ("sha256", "abc"), ("target_modules", ["q_proj", "q_proj"])]:
            with self.subTest(field=field, value=value):
                manifest = deepcopy(self.manifest)
                manifest["adapters"][0][field] = value
                with self.assertRaises(ProtocolError):
                    validate_manifest(manifest)

    def test_missing_and_extra_fields_rejected(self):
        for action in ("missing", "extra"):
            manifest = deepcopy(self.manifest)
            row = manifest["adapters"][0]
            if action == "missing":
                del row["payload"]
            else:
                row["score"] = 0.5
            with self.assertRaises(ProtocolError):
                validate_manifest(manifest)

    def test_benign_metadata_must_be_consistent(self):
        self.manifest["adapters"][0]["payload"] = "attack_payload"
        with self.assertRaisesRegex(ProtocolError, "Benign adapter"):
            validate_manifest(self.manifest)

    def test_primary_gate_excludes_smoke_tests_and_other_models(self):
        for field, value in [("lora_rank", 256), ("base_model", "Qwen/Qwen2.5-3B"),
                             ("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])]:
            manifest = deepcopy(self.manifest)
            manifest["adapters"][0][field] = value
            validate_manifest(manifest, require_verified=True)
            with self.assertRaisesRegex(ProtocolError, "outside .* primary setup"):
                create_split(manifest, self.policy)

    def test_json_loader_rejects_duplicate_keys_and_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            for number, content in enumerate(('{"schema_version":1,"schema_version":1,"adapters":[]}',
                                              '{"schema_version":1,"adapters":NaN}')):
                path = Path(directory) / f"invalid{number}.json"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ProtocolError):
                    load_manifest(path)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.policy = fixture()
        self.plan = create_split(self.manifest, self.policy)

    def test_determinism_independent_of_input_order(self):
        manifest = deepcopy(self.manifest)
        manifest["adapters"].reverse()
        for row in manifest["adapters"]:
            row["target_modules"].reverse()
        self.assertEqual(canonical_json(self.plan), canonical_json(create_split(manifest, self.policy)))

    def test_determinism_across_process_hash_seeds(self):
        code = ("from shared_protocol.tests.test_foundations import fixture; "
                "from shared_protocol.splits import create_split; "
                "from shared_protocol.manifest import canonical_json; "
                "print(canonical_json(create_split(*fixture())))")
        for seed in ("1", "987"):
            result = subprocess.run([sys.executable, "-B", "-c", code], check=True,
                                    capture_output=True, text=True,
                                    env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"})
            self.assertEqual(json.loads(result.stdout), self.plan)

    def test_different_seed_changes_development_but_not_holdouts(self):
        other = create_split(self.manifest, {**self.policy, "seed": 123})
        self.assertNotEqual(other["partitions"]["train"], self.plan["partitions"]["train"])
        for name in ("evaluation", "final_holdout"):
            self.assertEqual(other["partitions"][name], self.plan["partitions"][name])

    def test_complete_disjoint_partitions_ids_hashes_and_groups(self):
        rows = {r["adapter_id"]: r for r in self.manifest["adapters"]}
        for field in ("adapter_id", "sha256", "leakage_group"):
            seen = set()
            for name in PARTITIONS:
                values = {rows[key][field] for key in self.plan["partitions"][name]}
                self.assertTrue(values)
                self.assertFalse(seen & values)
                seen |= values
            self.assertEqual(seen, {r[field] for r in rows.values()})
        self.assertEqual({r["label"] for r in development_records(self.manifest, self.plan, "calibration")}, {"benign"})

    def test_heldout_tasks_attacks_and_payloads_never_enter_development(self):
        for name in ("train", "validation", "calibration"):
            for row in development_records(self.manifest, self.plan, name):
                self.assertNotIn(row["task"], {"eval_task", "final_task"})
                self.assertNotEqual(row["attack_family"], "unseen_attack")
                self.assertNotEqual(row["payload"], "final_payload")

    def test_related_adapters_stay_together_and_cannot_pollute_calibration(self):
        companion = adapter("eval-companion")
        companion["leakage_group"] = "eval-malicious"
        paired = adapter("paired-malicious", "malicious")
        paired["leakage_group"] = "benign-0"
        self.manifest["adapters"].extend([companion, paired])
        # Three mixed groups leave two for validation and one for training.
        for index in (1, 2):
            related = adapter(f"paired-malicious-{index}", "malicious")
            related["leakage_group"] = f"benign-{index}"
            self.manifest["adapters"].append(related)
        plan = create_split(self.manifest, self.policy)
        self.assertIn("eval-companion", plan["partitions"]["evaluation"])
        for name, ids in plan["partitions"].items():
            self.assertEqual("benign-0" in ids, "paired-malicious" in ids)
        self.assertNotIn("benign-0", plan["partitions"]["calibration"])

    def test_whole_task_reserved_including_malicious_members(self):
        self.manifest["adapters"].append(adapter("task-companion", "malicious", task="eval_task"))
        plan = create_split(self.manifest, self.policy)
        self.assertIn("task-companion", plan["partitions"]["evaluation"])

    def test_conflicting_cross_dimension_reservations_fail(self):
        self.manifest["adapters"].append(adapter("bridge", "malicious", attack="unseen_attack", payload="final_payload"))
        with self.assertRaisesRegex(ProtocolError, "bridges"):
            create_split(self.manifest, self.policy)

    def test_conflicting_provenance_groups_fail(self):
        self.manifest["adapters"][-1]["leakage_group"] = "eval-malicious"
        with self.assertRaisesRegex(ProtocolError, "bridges"):
            create_split(self.manifest, self.policy)

    def test_invalid_or_empty_policy_selectors_fail(self):
        for mutation in ("overlap", "missing", "empty", "none"):
            policy = deepcopy(self.policy)
            if mutation == "overlap":
                policy["final_holdout"]["benign_tasks"] = ["eval_task"]
            elif mutation == "missing":
                policy["evaluation"]["attack_families"] = ["does-not-exist"]
            elif mutation == "empty":
                policy["evaluation"]["attack_families"] = []
            else:
                policy["evaluation"]["payloads"] = ["none"]
            with self.subTest(mutation=mutation), self.assertRaises(ProtocolError):
                create_split(self.manifest, policy)

    def test_insufficient_cohorts_fail_without_fallback(self):
        for field in ("calibration_groups", "validation_groups_per_stratum"):
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                create_split(self.manifest, {**self.policy, field: 1000})
        self.manifest["adapters"] = [r for r in self.manifest["adapters"] if not r["adapter_id"].startswith("malicious-")]
        with self.assertRaises(ProtocolError):
            create_split(self.manifest, self.policy)

    def test_holdout_access_denied_before_any_validation_or_io(self):
        with patch("shared_protocol.splits.validate_split", side_effect=AssertionError("must not inspect")):
            for name in ("evaluation", "final_holdout", "test"):
                with self.subTest(name=name), self.assertRaisesRegex(ProtocolError, "Development access"):
                    development_records(self.manifest, self.plan, name)

    def test_creating_plan_never_opens_adapter_paths(self):
        with patch("builtins.open", side_effect=AssertionError("No file access")), \
                patch.object(Path, "open", side_effect=AssertionError("No file access")):
            create_split(self.manifest, self.policy)
            development_records(self.manifest, self.plan, "train")

    def test_development_view_is_detached(self):
        rows = development_records(self.manifest, self.plan, "train")
        rows[0]["verification"].clear()
        validate_split(self.plan, self.manifest)

    def test_saved_round_trip_is_exclusive_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            save_split(path, self.plan, self.manifest)
            original = path.read_bytes()
            self.assertEqual(load_split(path, self.manifest), self.plan)
            with self.assertRaises(FileExistsError):
                save_split(path, self.plan, self.manifest)
            self.assertEqual(path.read_bytes(), original)
            altered = deepcopy(self.plan)
            altered["partitions"]["train"].append(altered["partitions"]["final_holdout"][0])
            with self.assertRaises(ProtocolError):
                save_split(Path(directory) / "bad.json", altered, self.manifest)
            self.assertFalse((Path(directory) / "bad.json").exists())

    def test_manifest_metadata_and_evidence_drift_detected(self):
        for field, value in (("lora_alpha", 64), ("sha256", "a" * 64), ("leakage_group", "changed-group")):
            manifest = deepcopy(self.manifest)
            manifest["adapters"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                validate_split(self.plan, manifest)
        self.manifest["adapters"][0]["verification"]["label"]["evidence"] = "changed evidence"
        with self.assertRaises(ProtocolError):
            validate_split(self.plan, self.manifest)

    def test_saved_schema_type_tampering_rejected(self):
        altered = deepcopy(self.plan)
        altered["schema_version"] = True
        with self.assertRaises(ProtocolError):
            validate_split(altered, self.manifest)

    def test_cli_split_and_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_new_json(root / "manifest.json", self.manifest)
            write_new_json(root / "policy.json", self.policy)
            command = [sys.executable, "-B", "-m", "shared_protocol"]
            split = subprocess.run(command + ["split", str(root / "manifest.json"),
                                              str(root / "policy.json"), str(root / "splits.json")],
                                   capture_output=True, text=True)
            self.assertEqual(split.returncode, 0, split.stderr)
            check = subprocess.run(command + ["check", str(root / "manifest.json"), str(root / "splits.json")],
                                   capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)


if __name__ == "__main__":
    unittest.main()
