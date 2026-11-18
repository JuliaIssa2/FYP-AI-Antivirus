"""Offline safetensors fixtures. Real extraction runs in isolated subprocesses.

Keeping torch out of the parent preserves Phase 2/3 tests that verify their
saved-feature path does not import torch. No test makes a network request.
"""
from copy import deepcopy
from functools import wraps
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from exploratory import natour_spectral_smoke as smoke
from shared_protocol.manifest import METADATA_FIELDS, ProtocolError


def fixture_manifest(hashes):
    rows = []
    for (adapter_id, label, filename), checksum in zip(smoke.SELECTED, hashes):
        row = dict.fromkeys(METADATA_FIELDS)
        row.update(adapter_id=adapter_id, verification={}, label=label, sha256=checksum,
                   source_revision=smoke.REVISION,
                   path_or_source=f"hf://datasets/{smoke.REPO_ID}@{smoke.REVISION}/{filename}",
                   lora_rank=16, target_modules=["q_proj", "v_proj"],
                   base_model="NousResearch/Llama-2-7b-hf")
        rows.append(row)
    return {"schema_version": 1, "adapters": rows}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = fixture_manifest(["a" * 64, "b" * 64])

    def test_only_selected_metadata_and_no_verification_changes(self):
        original = deepcopy(self.manifest)
        self.manifest["adapters"].append({"adapter_id": "not-authorized", "path_or_source": "final_holdout"})
        selected = smoke.select_records(self.manifest)
        self.assertEqual(selected, original["adapters"])
        self.assertEqual([row["verification"] for row in selected], [{}, {}])

    def test_missing_duplicate_selected_rows_rejected(self):
        for rows in (self.manifest["adapters"][:1], self.manifest["adapters"] * 2):
            with self.subTest(rows=len(rows)), self.assertRaises(ProtocolError):
                smoke.select_records({"schema_version": 1, "adapters": rows})

    def test_pinning_identity_metadata_and_protected_markers(self):
        cases = {"source_revision": "main", "sha256": None, "lora_rank": 256,
                 "target_modules": ["q_proj"], "label": "malicious",
                 "path_or_source": "https://example.org/unauthorized.safetensors",
                 "dataset_provenance": {"dataset_id": "synthetic", "revision": None,
                                        "split": "final_holdout", "preprocessing": None}}
        for key, value in cases.items():
            manifest = deepcopy(self.manifest)
            manifest["adapters"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ProtocolError):
                smoke.select_records(manifest)

    def test_output_refuses_existing_directory_or_git_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ProtocolError):
                smoke.new_output_directory(root)
            (root / ".git").mkdir()
            with self.assertRaises(ProtocolError):
                smoke.new_output_directory(root / "artifacts")
            self.assertFalse((root / "artifacts").exists())


def isolated_fixture(test):
    @wraps(test)
    def wrapper(self):
        if os.environ.get("NATOUR_SMOKE_FIXTURE_PROCESS") == "1":
            return test(self)
        self.child('''
import sys, unittest
suite = unittest.defaultTestLoader.loadTestsFromName(
    "exploratory.tests.test_natour_spectral_smoke.CheckpointTests." + sys.argv[1])
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
''', test.__name__)
    return wrapper


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("NATOUR_SMOKE_FIXTURE_PROCESS") != "1":
            return
        import numpy as np
        from safetensors.numpy import save_file
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.weights = {}
        for i, projection in enumerate(("q_proj", "v_proj")):
            prefix = f"base_model.model.model.layers.20.self_attn.{projection}"
            cls.weights[prefix + ".lora_A.weight"] = (np.arange(16 * 32, dtype=np.float32).reshape(16, 32) % 13 + i) / 100
            cls.weights[prefix + ".lora_B.weight"] = (np.arange(32 * 16, dtype=np.float32).reshape(32, 16) % 17 - i) / 100
        cls.checkpoint = cls.root / "adapter_model.safetensors"
        save_file(cls.weights, str(cls.checkpoint))
        cls.checksum = smoke.sha256_file(cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temporary"):
            cls.temporary.cleanup()

    def child(self, code, *args):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HF_HUB_DISABLE_XET="1",
                           HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1",
                           NATOUR_SMOKE_FIXTURE_PROCESS="1")
        # Include the staged or installed smoke package AND the existing repo.
        paths = [str(Path(smoke.__file__).resolve().parents[1]), str(Path.cwd())]
        environment["PYTHONPATH"] = os.pathsep.join(paths)
        result = subprocess.run([sys.executable, "-B", "-c", code, *map(str, args)],
                                text=True, capture_output=True, env=environment, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    @isolated_fixture
    def test_hash_mismatch_aborts_before_tensor_parsing(self):
        with patch("safetensors.safe_open", side_effect=AssertionError("must not parse wrong hash")):
            with self.assertRaisesRegex(ProtocolError, "Checkpoint SHA256 mismatch"):
                smoke.inspect_checkpoint(self.checkpoint, "0" * 64, hidden_size=32)

    @isolated_fixture
    def test_valid_fixture_and_real_shape_gate(self):
        valid = smoke.inspect_checkpoint(self.checkpoint, self.checksum, hidden_size=32)
        self.assertEqual(valid["tensor_count"], 4)
        self.assertTrue(valid["hash_matches_manifest"])
        with self.assertRaisesRegex(ProtocolError, "shape/dtype"):
            smoke.inspect_checkpoint(self.checkpoint, self.checksum)  # Production always 4096.

    @isolated_fixture
    def test_missing_duplicate_wrong_keys_shapes_and_nonfinite(self):
        import numpy as np
        from safetensors.numpy import save_file
        first = next(iter(self.weights))
        missing = dict(self.weights)
        del missing[first]
        duplicate = {**self.weights, first.replace(".weight", ".default.weight"): self.weights[first]}
        bad_key = {**self.weights, "unknown.weight": np.zeros((2, 2), dtype=np.float32)}
        bad_rank = {**self.weights, first: np.zeros((8, 32), dtype=np.float32)}
        bad_dtype = {**self.weights, first: np.zeros((16, 32), dtype=np.int32)}
        nonfinite = {**self.weights, first: np.full((16, 32), np.nan, dtype=np.float32)}
        for i, weights in enumerate((missing, duplicate, bad_key, bad_rank, bad_dtype, nonfinite)):
            path = self.root / f"invalid-{i}.safetensors"
            save_file(weights, str(path))
            with self.subTest(case=i), self.assertRaises(ProtocolError):
                smoke.inspect_checkpoint(path, smoke.sha256_file(path), hidden_size=32)

    @isolated_fixture
    def test_vendor_extraction_matches_direct_call_and_is_repeatable(self):
        self.child('''
import sys
from unittest.mock import patch
import numpy as np
from exploratory import natour_spectral_smoke as s
s.prepare_cpu()
original_inspect = s.inspect_checkpoint
with patch.object(s, "inspect_checkpoint", side_effect=lambda p,h: original_inspect(p,h,hidden_size=32)):
    validation, values = s.extract_checked(sys.argv[1], sys.argv[2])
    _, repeated = s.extract_checked(sys.argv[1], sys.argv[2])
direct = s.checked_extractor().extract_spectral_features(sys.argv[1], layer_idx=20, projections=("q_proj","v_proj"))
np.testing.assert_array_equal(values, direct)
assert repeated == values
assert len(values) == 10 and validation["tensor_count"] == 4
assert "adapter_generation.dataset.spectral_features" not in sys.modules
''', self.checkpoint, self.checksum)

    @isolated_fixture
    def test_changed_vendor_hash_rejected_before_import(self):
        with patch.object(smoke, "sha256_file", return_value="0" * 64):
            with patch.object(smoke.importlib, "import_module", side_effect=AssertionError("must not import")):
                with self.assertRaisesRegex(ProtocolError, "extractor SHA256 mismatch"):
                    smoke.checked_extractor()

    @isolated_fixture
    def test_complete_offline_workflow_download_allowlist_and_report(self):
        self.child('''
import json, pathlib, shutil, sys, tempfile
from unittest.mock import patch
import numpy as np
from safetensors.numpy import load_file, save_file
from exploratory import natour_spectral_smoke as s
from exploratory.tests.test_natour_spectral_smoke import fixture_manifest
with tempfile.TemporaryDirectory() as folder:
    root = pathlib.Path(folder)
    second = root / "second.safetensors"
    save_file({k:v*2 for k,v in load_file(sys.argv[1]).items()}, str(second))
    sources = [pathlib.Path(sys.argv[1]), second]
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(fixture_manifest([s.sha256_file(p) for p in sources])))
    original_manifest = manifest.read_bytes()
    calls = []
    def download(**kwargs):
        i = len(calls)
        assert i < 2
        assert kwargs["repo_id"] == s.REPO_ID
        assert kwargs["revision"] == s.REVISION and kwargs["repo_type"] == "dataset"
        assert kwargs["filename"] == s.SELECTED[i][2]
        destination = pathlib.Path(kwargs["cache_dir"]) / str(i) / "adapter_model.safetensors"
        destination.parent.mkdir(parents=True)
        shutil.copyfile(sources[i], destination)
        calls.append(kwargs)
        return str(destination)
    inspect = s.inspect_checkpoint
    with patch("huggingface_hub.hf_hub_download", side_effect=download), patch.object(
            s,"inspect_checkpoint",side_effect=lambda p,h: inspect(p,h,hidden_size=32)):
        report = s.run(manifest, root / "run")
    assert len(calls) == 2
    assert (root / "run/report.json").is_file()
    assert manifest.read_bytes() == original_manifest
    assert report["official_protocol_eligible"] is False
    assert report["extractor_sha256"] == s.EXTRACTOR_SHA256
    assert report["feature_names"] == list(s.LEGACY_FEATURE_NAMES)
    assert "NousResearch" in report["model_name_discrepancy"]
    for row in report["adapters"]:
        assert row["source_metadata"]["verification"] == {}
        assert row["verification_issues"] and len(row["feature_values"]) == 10
        assert row["checkpoint_sha256"] == row["manifest_sha256"]
    assert "sklearn" not in sys.modules
''', self.checkpoint)


if __name__ == "__main__":
    unittest.main()
