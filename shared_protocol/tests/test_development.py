"""Development handoff tests: synthetic metadata/bytes only, no real training."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from shared_protocol.development import (
    export_development_bundle, load_development_bundle, validate_development_bundle,
    verify_development_checkpoints,
)
from shared_protocol.manifest import ProtocolError, canonical_json, digest
from shared_protocol.scoring import DetectorSpec, ProtocolBinding, bind_protocol, score_development
from shared_protocol.splits import DEVELOPMENT, create_split
from shared_protocol.tests.test_foundations import fixture
from shared_protocol.tests.test_phase2 import SyntheticDetector


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ("peftguard_retrain.ipynb", "zpeft_fit.ipynb")


def rebind(data):
    """Synthetic trust anchor for testing semantic checks after checksum success."""
    changed = deepcopy(data)
    binding = changed["binding"]
    rows = {r["adapter_id"]: r for p in DEVELOPMENT for r in changed[p]}
    for entry in binding["commitments"]:
        if entry[0] in rows:
            row = rows[entry[0]]
            entry[2] = digest({**row, "target_modules": sorted(row["target_modules"])})
            entry[3] = row["leakage_group"]
    restored = ProtocolBinding(**{**binding, "commitments": tuple(tuple(e) for e in binding["commitments"])})
    return changed, restored.sha256


class BundleFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "synthetic-development.json"
        self.manifest, self.policy = fixture()
        self.plan = create_split(self.manifest, self.policy)
        self.receipt = export_development_bundle(self.path, self.manifest, self.plan)
        self.data = json.loads(self.path.read_bytes())

    def load(self, path=None, **overrides):
        args = dict(expected_file_sha256=self.receipt["bundle_file_sha256"],
                    expected_binding_sha256=self.receipt["binding_sha256"])
        return load_development_bundle(path or self.path, **{**args, **overrides})

    def validate(self, data, expected=None):
        return validate_development_bundle(data, expected_binding_sha256=expected or self.receipt["binding_sha256"])


class DevelopmentTests(BundleFixture):
    def test_round_trip_exact_binding_membership_canonical_and_no_overwrite(self):
        bundle = self.load()
        binding = bind_protocol(self.manifest, self.plan)
        self.assertEqual(bundle.binding, binding)
        self.assertEqual(set(self.data), {"binding", *DEVELOPMENT})
        self.assertEqual(self.path.read_bytes(), (canonical_json(self.data) + "\n").encode())
        for p in DEVELOPMENT:
            self.assertEqual(tuple(r["adapter_id"] for r in bundle.records(p)), binding.ids(p))
        self.assertEqual({e[1] for e in bundle.binding.commitments},
                         {*DEVELOPMENT, "evaluation", "final_holdout"})
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            export_development_bundle(self.path, self.manifest, self.plan)
        self.assertEqual(self.path.read_bytes(), before)
        detached = bundle.records("train")
        detached[0]["label"] = "unknown"
        self.assertNotEqual(bundle.train[0]["label"], "unknown")

    def test_missing_extra_duplicate_swapped_and_altered_records(self):
        mutations = []
        for p in DEVELOPMENT:
            a = deepcopy(self.data); a[p].pop(); mutations.append(a)
            a = deepcopy(self.data); a[p].append(deepcopy(a[p][0])); mutations.append(a)
        a = deepcopy(self.data); a["train"][0], a["validation"][0] = a["validation"][0], a["train"][0]; mutations.append(a)
        a = deepcopy(self.data); a["train"][0]["seed"] += 1; mutations.append(a)
        a = deepcopy(self.data); del a["validation"]; mutations.append(a)
        a = deepcopy(self.data); a["extra"] = []; mutations.append(a)
        for data in mutations:
            with self.subTest(data_keys=list(data)), self.assertRaises(ProtocolError):
                self.validate(data)

    def test_verification_evidence_and_record_types_are_checked(self):
        for evidence in ({}, {"verified_by": "", "evidence": "synthetic"}):
            a = deepcopy(self.data)
            a["train"][0]["verification"]["seed"] = evidence
            with self.assertRaises(ProtocolError):
                self.validate(a)
        for value in (None, {}, "train", [None]):
            a = deepcopy(self.data); a["train"] = value
            with self.assertRaises(ProtocolError):
                self.validate(a)

    def test_checksum_checked_before_json_parse_and_both_trust_values_required(self):
        self.path.write_text("invalid json", encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "bytes differ"):
            self.load()
        for kw in ({"expected_file_sha256": ""}, {"expected_binding_sha256": ""}):
            with self.assertRaises(ProtocolError):
                self.load(**kw)
        with self.assertRaises(TypeError):
            load_development_bundle(self.path)

    def test_rehashed_bundle_still_rejects_altered_evidence_and_duplicate_json_keys(self):
        a = deepcopy(self.data); a["train"][0]["verification"]["seed"]["evidence"] = "changed"
        self.path.write_text(canonical_json(a), encoding="utf-8")
        h = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ProtocolError, "drift"):
            self.load(expected_file_sha256=h)
        raw = self.path.read_text().replace('"binding":', '"binding":{},"binding":', 1)
        self.path.write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(ProtocolError, "Duplicate JSON"):
            self.load(expected_file_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_substituted_or_malformed_binding_is_rejected(self):
        a = deepcopy(self.data); a["binding"]["manifest_sha256"] = "a" * 64
        with self.assertRaisesRegex(ProtocolError, "fingerprint"):
            self.validate(a)
        for entries in ("bad", [["bad"]], [[1, "train", "a" * 64, "group"]]):
            a = deepcopy(self.data); a["binding"]["commitments"] = entries
            with self.assertRaises(ProtocolError):
                self.validate(a)
        a = deepcopy(self.data); a["binding"]["extra"] = "no"
        with self.assertRaises(ProtocolError):
            self.validate(a)
        a = deepcopy(self.data)
        a["binding"]["commitments"] = [e for e in a["binding"]["commitments"] if e[1] in DEVELOPMENT]
        with self.assertRaisesRegex(ProtocolError, "all five"):
            self.validate(a)

    def test_global_checkpoint_hash_and_locator_reuse_are_rejected(self):
        for key in ("sha256", "path_or_source", "adapter_id"):
            a = deepcopy(self.data)
            a["validation"][0][key] = a["train"][0][key]
            with self.subTest(field=key):
                try:
                    a, anchor = rebind(a)
                    with self.assertRaises(ProtocolError):
                        self.validate(a, anchor)
                except ProtocolError:
                    # Duplicate binding IDs themselves are already invalid.
                    self.assertEqual(key, "adapter_id")

    def test_benign_calibration_and_both_train_validation_classes_required(self):
        a = deepcopy(self.data)
        r = a["calibration"][0]
        r.update(label="malicious", attack_family="synthetic", payload="synthetic", poisoning_rate=0.1)
        a, anchor = rebind(a)
        with self.assertRaisesRegex(ProtocolError, "calibration: requires"):
            self.validate(a, anchor)
        for p in ("train", "validation"):
            a = deepcopy(self.data)
            for row in a[p]:
                row.update(label="benign", attack_family="none", payload="none", poisoning_rate=0)
            a, anchor = rebind(a)
            with self.assertRaisesRegex(ProtocolError, p + ": requires"):
                self.validate(a, anchor)

    def test_export_rejects_unverified_drifted_and_unsupported_inputs_before_writing(self):
        for change in ("unverified", "drift", "schema3"):
            m, p = deepcopy(self.manifest), deepcopy(self.plan)
            if change == "unverified":
                m["adapters"][0]["verification"] = {}
            elif change == "drift":
                p["partitions"]["train"].pop()
            else:
                p["policy"]["schema_version"] = 3
            out = self.root / (change + ".json")
            with self.subTest(change=change), self.assertRaises(ProtocolError):
                export_development_bundle(out, m, p)
            self.assertFalse(out.exists())

    def test_no_protected_records_or_accessors(self):
        bundle = self.load()
        protected = [r for r in self.manifest["adapters"]
                     if r["adapter_id"] in set(bundle.binding.ids("evaluation")) | set(bundle.binding.ids("final_holdout"))]
        for p in ("evaluation", "final_holdout"):
            with self.assertRaises(ProtocolError):
                bundle.records(p)
            a = deepcopy(self.data); a[p] = protected
            with self.assertRaises(ProtocolError):
                self.validate(a)
        for row in protected:
            self.assertNotIn(row["path_or_source"], self.path.read_text())
            a = deepcopy(self.data); a["train"].append(row)
            with self.assertRaises(ProtocolError):
                self.validate(a)

    def test_full_calibration_preserved_and_existing_scoring_selects_representatives(self):
        bundle = self.load()
        a = deepcopy(self.data)
        related = deepcopy(a["calibration"][0]); related["adapter_id"] = "zz-related"
        related["sha256"] = digest("synthetic related bytes")
        related["path_or_source"] = "fixture://zz-related"
        a["calibration"].append(related)
        a["binding"]["commitments"].append([related["adapter_id"], "calibration", digest(related), related["leakage_group"]])
        a["binding"]["commitments"].sort()
        a, anchor = rebind(a)
        bundle = self.validate(a, anchor)
        self.assertEqual(len(bundle.calibration), len(self.data["calibration"]) + 1)
        ids = bundle.binding.ids("calibration", representatives=True)
        spec = DetectorSpec("synthetic", digest("source"), digest("config"), digest("state"),
                            "higher_is_suspicious", bundle.binding.ids("train"), (), "synthetic fixture")
        detector = SyntheticDetector(spec, {key: 0.0 for key in ids})
        scores = score_development(detector, bundle.calibration, bundle.binding, "calibration")
        self.assertEqual(tuple(s.adapter_id for s in scores.scores), ids)
        self.assertNotIn("zz-related", ids)

    def test_roua_existing_parser_and_fit_call_accept_unchanged_bundle(self):
        # Execute the real run_stage with a stub fit, rather than fitting a model.
        import importlib.util
        from types import SimpleNamespace
        spec = importlib.util.spec_from_file_location("synthetic_oneclass_runner", ROOT / "notebooks/oneclass_run.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        args = SimpleNamespace(bundle=self.path, adapters=self.root / "unused", stage="fit")
        with patch("shared_protocol.detectors.oneclass.train.fit", return_value={"model_file": "synthetic.json"}) as fit:
            result = module.run_stage(args, self.root)
        self.assertEqual(result, Path("synthetic.json"))
        self.assertEqual(fit.call_args.args[0], self.load().binding)
        self.assertEqual(fit.call_args.args[1], tuple(self.data["train"]))
        self.assertEqual(fit.call_args.args[2], tuple(self.data["validation"]))

    def test_checkpoint_preflight_checks_existing_files_and_metadata_first(self):
        bundle = self.load()
        for p in DEVELOPMENT:
            for row in getattr(bundle, p):
                row["sha256"] = hashlib.sha256(b"synthetic checkpoint bytes").hexdigest()
        # Metadata drift is rejected before even creating a resolver.
        with patch("shared_protocol.detectors.peftguard.converter.root_resolver", side_effect=AssertionError("no I/O")):
            with self.assertRaises(ProtocolError):
                verify_development_checkpoints(bundle, self.root, expected_binding_sha256=bundle.binding.sha256)
        # Unique arbitrary fixture bytes, never tensors or real checkpoints.
        a = deepcopy(self.data)
        for p in DEVELOPMENT:
            for row in a[p]:
                raw = ("synthetic bytes " + row["adapter_id"]).encode()
                row["sha256"] = hashlib.sha256(raw).hexdigest()
                dest = self.root / row["adapter_id"] / "adapter_model.safetensors"
                dest.parent.mkdir(); dest.write_bytes(raw)
        a, anchor = rebind(a); bundle = self.validate(a, anchor)
        verify_development_checkpoints(bundle, self.root, expected_binding_sha256=anchor)
        row = bundle.train[0]
        (self.root / row["adapter_id"] / "adapter_model.safetensors").write_bytes(b"corrupted fixture")
        with self.assertRaisesRegex(ProtocolError, "SHA256 mismatch"):
            verify_development_checkpoints(bundle, self.root, expected_binding_sha256=anchor)

    def test_validation_cli_with_independent_fingerprints(self):
        command = [sys.executable, "-B", "-m", "shared_protocol.development", "--bundle", str(self.path),
                   "--bundle-sha256", self.receipt["bundle_file_sha256"],
                   "--binding-sha256", self.receipt["binding_sha256"]]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["binding_sha256"], self.receipt["binding_sha256"])
        command[-1] = "0" * 64
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fingerprint", result.stderr)


class NotebookTests(BundleFixture):
    def test_input_validation_precedes_checkpoint_access_and_fit_in_both_notebooks(self):
        for name in NOTEBOOKS:
            notebook = json.loads((ROOT / "notebooks" / name).read_bytes())
            cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
            load_index = next(i for i, c in enumerate(cells) if "development = load_development_bundle(" in c)
            stage_index = next(i for i, c in enumerate(cells) if "resolver = verify_development_checkpoints(" in c)
            fit_index = next(i for i, c in enumerate(cells) if "= fit(" in c)
            self.assertLess(load_index, stage_index); self.assertLess(stage_index, fit_index)
            # The source checkout prefix is tested separately; run the actual input statements.
            source = cells[load_index]
            source = source[source.index("from shared_protocol.development import load_development_bundle"):]
            env = {"BUNDLE": self.path, "BUNDLE_SHA256": "0" * 64,
                   "BINDING_SHA256": self.receipt["binding_sha256"]}
            with self.subTest(notebook=name), patch(
                    "shared_protocol.development.verify_development_checkpoints", side_effect=AssertionError("staging")), patch(
                    "shared_protocol.detectors.peftguard.train.fit", side_effect=AssertionError("fit")), self.assertRaises(ProtocolError):
                exec(compile(source + "\n" + cells[stage_index], name, "exec"), env)
            env["BUNDLE_SHA256"] = self.receipt["bundle_file_sha256"]
            exec(compile(source, name, "exec"), env)
            self.assertEqual(env["binding"], self.load().binding)
            self.assertEqual(env["train_records"], self.load().train)

    def test_notebooks_have_no_protected_loader_or_implicit_adapter_downloads(self):
        for name in NOTEBOOKS:
            notebook = json.loads((ROOT / "notebooks" / name).read_bytes())
            code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
            for token in ("load_manifest", "load_split", "eval_records", "RELEASE_EVALUATION", "stage(", "REPO_BRANCH"):
                self.assertNotIn(token, code)
            self.assertIn('re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT)', code)
            self.assertNotIn("root_resolver(ADAPTER_ROOT)", code)
            if name == "peftguard_retrain.ipynb":
                optional = next("".join(c["source"]) for c in notebook["cells"]
                                if "RUN_PRETRAINED_REFERENCE = False" in "".join(c["source"]))
                exec(compile(optional, name, "exec"), {})
            else:
                self.assertNotIn("hf_hub_download", code)

    def test_source_pin_fails_before_checkout_and_all_python_cells_parse(self):
        for name in NOTEBOOKS:
            notebook = json.loads((ROOT / "notebooks" / name).read_bytes())
            cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
            for source in cells:
                # Dependency installation is a Colab shell escape, not Python syntax.
                ast.parse("\n".join(line for line in source.splitlines() if not line.startswith("!")))
            source = next(c for c in cells if 're.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT)' in c)
            with patch("subprocess.run", side_effect=AssertionError("no checkout")):
                with self.assertRaisesRegex(ValueError, "reviewed source commit"):
                    exec(compile(source, name, "exec"), {"SOURCE_COMMIT": "main"})

    def test_validated_notebook_inputs_reach_unchanged_fit_apis_with_restricted_resolver(self):
        import os
        import shutil
        from contextlib import redirect_stdout
        from io import StringIO

        for name in NOTEBOOKS:
            notebook = json.loads((ROOT / "notebooks" / name).read_bytes())
            cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
            inputs = next(c for c in cells if "development = load_development_bundle(" in c)
            inputs = inputs[inputs.index("from shared_protocol.development import load_development_bundle"):]
            stage = next(c for c in cells if "resolver = verify_development_checkpoints(" in c)
            fit_source = next(c for c in cells if "= fit(" in c)
            target = "peftguard" if name.startswith("peftguard") else "zpeft"
            result = {"checkpoint_sha256": "a" * 64, "model_file": "unused.json", "model_file_sha256": "b" * 64}
            env = dict(BUNDLE=self.path, BUNDLE_SHA256=self.receipt["bundle_file_sha256"],
                       BINDING_SHA256=self.receipt["binding_sha256"], ADAPTER_ROOT=str(self.root),
                       RUN_DIR="unused", DRIVE_OUT="unused", CACHE_DIR="unused", SEED=0,
                       PLAN="adam_fp32", total_gb=80, os=os, shutil=shutil)
            with patch("shared_protocol.development.verify_development_checkpoints") as preflight, \
                    patch(f"shared_protocol.detectors.{target}.train.fit", return_value=result) as fit, \
                    patch("shared_protocol.detectors.peftguard.converter.root_resolver") as resolver, \
                    patch("os.makedirs"), patch("shutil.copyfile"), redirect_stdout(StringIO()):
                exec(compile(inputs + "\n" + stage + "\n" + fit_source, name, "exec"), env)
                preflight.assert_called_once()
                fit.assert_called_once()
                self.assertEqual(fit.call_args.args[:3], (self.load().binding, self.load().train, self.load().validation))
                resolver.assert_called_once_with(str(self.root), restricted=True)


if __name__ == "__main__":
    unittest.main()
