"""One-class Mahalanobis detector tests on tiny synthetic adapters (hidden 32, 2 layers, rank 16). CPU only, no network.
Skipped without numpy/scikit-learn/safetensors. Run: python -B -m unittest shared_protocol.tests.test_zz_oneclass -v"""

from copy import deepcopy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE = all(importlib.util.find_spec(name) for name in ("numpy", "sklearn", "safetensors"))
np = save_file = None
REPO = Path(__file__).resolve().parents[2]


def setUpModule():
    global np, save_file
    if HAVE:
        import numpy as _np
        from safetensors.numpy import save_file as _save
        np, save_file = _np, _save


from shared_protocol.calibration import calibrate_benign
from shared_protocol.evaluation import evaluate, score_evaluation
from shared_protocol.manifest import PRIMARY_MODEL, ProtocolError, read_json, write_new_json
from shared_protocol.scoring import AdapterInput, bind_protocol, score_development
from shared_protocol.splits import create_split, development_records
from shared_protocol.tests.test_foundations import fixture

DIM, LAYERS, RANK = 32, 2, 16


def write_adapter(path: Path, rng, malicious: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for layer in range(LAYERS):
        for module in ("q_proj", "v_proj"):
            base = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            tensors[f"{base}.lora_A.weight"] = rng.normal(0, 0.05, (RANK, DIM)).astype(np.float32)
            tensors[f"{base}.lora_B.weight"] = ((4.0 if malicious else 1.0) * rng.normal(0, 0.05, (DIM, RANK))).astype(np.float32)
    save_file(tensors, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(HAVE, "needs numpy, scikit-learn and safetensors")
class FeatureTests(unittest.TestCase):
    def test_layout_names_and_aggregation(self):
        from shared_protocol.detectors.oneclass.features import (
            N_FEATURES, extract_features, feature_names, layer_stats)
        rng = np.random.default_rng(0)
        f = {(l, m): (rng.normal(0, 1, (RANK, DIM)), rng.normal(0, 1, (DIM, RANK)))
             for l in range(LAYERS) for m in ("q_proj", "v_proj")}
        x = extract_features(f, layers=LAYERS)
        names = feature_names()
        self.assertEqual((x.shape, len(names), N_FEATURES), ((30,), 30, 30))
        per_layer = np.array([layer_stats(*f[(l, "v_proj")]) for l in range(LAYERS)])
        # v_proj, kurtosis (stat index 4): mean, std, max over layers
        i = names.index("v_proj.kurtosis.mean")
        np.testing.assert_allclose(x[i:i + 3], [per_layer[:, 4].mean(), per_layer[:, 4].std(), per_layer[:, 4].max()])

    def test_layer_stats_match_definition_computed_from_full_update(self):
        from shared_protocol.detectors.oneclass.features import layer_stats
        rng = np.random.default_rng(1)
        a, b = rng.normal(0, 1, (RANK, DIM)), rng.normal(0, 1, (DIM, RANK))
        s = np.linalg.svd(b @ a, compute_uv=False)[:RANK]   # full update, then its r non-zero singular values
        e = s ** 2
        tot = e.sum() + 1e-30
        p = e / tot
        want = [s[0], np.sqrt(tot), e[0] / tot, -(p * np.log(p + 1e-12)).sum() / np.log(RANK),
                ((s - s.mean()) ** 4).mean() / s.std() ** 4 - 3]
        np.testing.assert_allclose(layer_stats(a, b), want, rtol=1e-8)

    def test_unscaled_update_so_alpha_cannot_leak(self):
        from shared_protocol.detectors.oneclass.features import layer_stats
        rng = np.random.default_rng(2)
        a, b = rng.normal(0, 1, (RANK, DIM)), rng.normal(0, 1, (DIM, RANK))
        np.testing.assert_allclose(layer_stats(a, b)[0] * 2, layer_stats(a, 2 * b)[0])  # only the factors matter

    def test_matches_legacy_script_when_pandas_is_available(self):
        if importlib.util.find_spec("pandas") is None:
            self.skipTest("legacy script needs pandas")
        import sys
        from shared_protocol.detectors.oneclass.features import extract_features, feature_names
        from shared_protocol.detectors.peftguard.converter import load_qv_factors
        legacy_dir = REPO / "adapter_generation" / "dataset"
        if not (legacy_dir / "spectral_features.py").is_file():
            self.skipTest("legacy script not in this checkout")
        with tempfile.TemporaryDirectory() as tmp:
            write_adapter(Path(tmp) / "adapter_model.safetensors", np.random.default_rng(3), False)
            sys.path.insert(0, str(legacy_dir))
            try:
                spec = importlib.util.spec_from_file_location("legacy_spectral", legacy_dir / "spectral_features.py")
                legacy = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(legacy)
            finally:
                sys.path.remove(str(legacy_dir))
            old = legacy.adapter_features(tmp)  # no adapter_config.json -> scale 1.0
            mine = extract_features(load_qv_factors(Path(tmp) / "adapter_model.safetensors", dim=DIM, layers=LAYERS),
                                    layers=LAYERS)
        rename = {"energy_concentration": "top_share", "entropy": "entropy_norm"}
        for name, value in zip(feature_names(), mine):
            module, stat, agg = name.split(".")
            self.assertAlmostEqual(value, float(old[f"{module}__{rename.get(stat, stat)}_{agg}"]),
                                   delta=1e-4 * max(1.0, abs(value)), msg=name)


    def test_strict_snapshot_rejects_extra_factor_alias_and_integer_weights(self):
        from safetensors.numpy import load_file
        from shared_protocol.detectors.peftguard.converter import load_qv_factors, sha256_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter_model.safetensors"
            expected = write_adapter(path, np.random.default_rng(6), False)
            clean = load_file(path)
            name = next(k for k in clean if ".lora_A." in k)
            self.assertEqual(len(load_qv_factors(path, dim=DIM, layers=LAYERS, rank=RANK,
                                                 expected_sha256=expected)), LAYERS * 2)
            for tensors, error in (
                ({**clean, name.replace(".weight", ".default.weight"): clean[name]}, "Duplicate"),
                ({**clean, "unsupported.weight": np.zeros(1, dtype=np.float32)}, "Unexpected"),
                ({**clean, name: clean[name].astype(np.int32)}, "floating-point"),
                ({k: v for k, v in clean.items() if k != name}, "lacks"),
            ):
                with self.subTest(error=error):
                    save_file(tensors, str(path))
                    with self.assertRaisesRegex(ProtocolError, error):
                        load_qv_factors(path, dim=DIM, layers=LAYERS, rank=RANK,
                                        expected_sha256=sha256_file(path))
            with self.assertRaisesRegex(ProtocolError, "SHA256"):
                load_qv_factors(path, dim=DIM, layers=LAYERS, rank=RANK, expected_sha256=expected)

    def test_all_30_features_against_independent_dense_oracle(self):
        from scipy.stats import kurtosis
        from shared_protocol.detectors.oneclass.features import extract_features

        factors = {}
        expected = []
        for module, multiplier in (("q_proj", 1), ("v_proj", 3)):
            stats = []
            for layer in range(LAYERS):
                a = np.eye(RANK, DIM)
                b = (multiplier + layer) * np.eye(DIM, RANK)
                b[0, 0] *= 2
                factors[layer, module] = (a, b)
                update = b @ a
                singular = np.linalg.svd(update, compute_uv=False)[:RANK]
                probs = singular**2 / np.linalg.norm(update, "fro")**2
                stats.append([np.linalg.norm(update, 2), np.linalg.norm(update, "fro"), max(probs),
                              -sum(probs * np.log(probs)) / np.log(RANK), kurtosis(singular)])
            for values in zip(*stats):
                expected.extend([np.mean(values), np.std(values), max(values)])
        np.testing.assert_allclose(extract_features(factors, layers=LAYERS), expected, atol=1e-10)

    def test_spectral_edge_cases_keep_documented_conventions(self):
        from shared_protocol.detectors.oneclass.features import layer_stats
        np.testing.assert_allclose(layer_stats(np.zeros((2, 2)), np.eye(2)), [0, 1e-15, 0, 0, 0],
                                   rtol=1e-12, atol=0)
        np.testing.assert_allclose(layer_stats(np.diag([4.0, 0.0]), np.eye(2)), [4, 4, 1, 0, -2], atol=1e-10)
        self.assertTrue(np.isfinite(layer_stats(np.eye(2) * 1e-40, np.eye(2))).all())
        tiny = layer_stats(np.diag([1e-90, 2e-90]), np.eye(2))
        self.assertTrue(np.isfinite(tiny).all())
        self.assertAlmostEqual(tiny[-1], -2.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "BF16 fixture requires torch")
    def test_verified_float16_and_bfloat16_factors(self):
        import torch
        from safetensors.torch import save_file as save_torch
        from shared_protocol.detectors.peftguard.converter import load_qv_factors, sha256_file
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "typed.safetensors"
            for dtype in (torch.float16, torch.bfloat16):
                tensors = {}
                for module in ("q_proj", "v_proj"):
                    base = f"layers.0.self_attn.{module}"
                    tensors[f"{base}.lora_A.weight"] = torch.ones((2, 3), dtype=dtype)
                    tensors[f"{base}.lora_B.weight"] = torch.ones((3, 2), dtype=dtype)
                save_torch(tensors, str(path))
                factors = load_qv_factors(path, dim=3, layers=1, rank=2, expected_sha256=sha256_file(path))
                for pair in factors.values():
                    for tensor in pair:
                        self.assertEqual(tensor.dtype, np.float32)
                        self.assertTrue((tensor == 1).all())

    def test_restricted_resolver_rejects_invalid_id_and_symlink(self):
        from types import SimpleNamespace
        from shared_protocol.detectors.peftguard.converter import root_resolver
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            (root / "linked").symlink_to(Path(tmp), target_is_directory=True)
            resolve = root_resolver(root, restricted=True)
            for key in ("../outside", "linked"):
                with self.subTest(key=key), self.assertRaises(ProtocolError):
                    resolve(SimpleNamespace(adapter_id=key))


@unittest.skipUnless(HAVE, "needs numpy, scikit-learn and safetensors")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from shared_protocol.detectors.oneclass.train import OneClassConfig, fit
        from shared_protocol.detectors.peftguard.converter import root_resolver
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        manifest, policy = fixture()
        rng = np.random.default_rng(0)
        for row in manifest["adapters"]:
            row["sha256"] = write_adapter(root / "ad" / row["adapter_id"] / "adapter_model.safetensors", rng,
                                          row["label"] == "malicious")
        cls.manifest, cls.plan = manifest, create_split(manifest, policy)
        cls.binding = bind_protocol(manifest, cls.plan)
        cls.resolver = staticmethod(root_resolver(root / "ad"))
        cls.cfg = OneClassConfig(layers=LAYERS, dim=DIM, min_clean=3)
        cls.train = development_records(manifest, cls.plan, "train")
        cls.val = development_records(manifest, cls.plan, "validation")
        cls.out, cls.cache = root / "run", root / "cache"
        cls.result = fit(cls.binding, cls.train, cls.val, cls.cfg, cls.out, resolver=cls.resolver, cache_dir=cls.cache,
                         log=lambda *_: None)
        cls.model_path = Path(cls.result["model_file"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def detector(self, **kw):
        from shared_protocol.detectors.oneclass.detector import OneClassDetector
        return OneClassDetector.from_model(self.model_path, self.binding, self.train,
                                           expected_file_sha256=self.result["model_file_sha256"],
                                           resolver=self.resolver, **kw)

    def adapter_input(self, row):
        return AdapterInput(row["adapter_id"], row["path_or_source"], row["source_revision"], row["sha256"],
                            row["base_model"], row["lora_rank"], row["lora_alpha"], tuple(sorted(row["target_modules"])))

    def test_fit_set_is_exactly_the_benign_train_adapters(self):
        benign = tuple(sorted(r["adapter_id"] for r in self.train if r["label"] == "benign"))
        malicious = {r["adapter_id"] for r in self.train if r["label"] == "malicious"}
        self.assertTrue(benign and malicious)
        self.assertEqual(self.detector().spec.fit_ids, benign)
        self.assertTrue(malicious.isdisjoint(self.detector().spec.fit_ids))
        self.assertEqual(self.detector().spec.tuning_ids, ())
        self.detector().spec.verify(self.binding)

    def test_backdoored_train_files_are_never_opened(self):
        from shared_protocol.detectors.oneclass.train import fit
        broken = []
        try:
            for r in self.train:
                if r["label"] == "malicious":
                    path = self.resolver(type("A", (), {"adapter_id": r["adapter_id"]}))
                    broken.append((path, path.read_bytes()))
                    path.write_bytes(b"garbage, not a safetensors file")
            result = fit(self.binding, self.train, None, self.cfg, Path(self.tmp.name) / "run_clean_only",
                         resolver=self.resolver, log=lambda *_: None)
            self.assertEqual(result["n_fit"], sum(r["label"] == "benign" for r in self.train))
        finally:
            for path, data in broken:
                path.write_bytes(data)

    def test_distance_matches_sklearn_ledoit_wolf_mahalanobis(self):
        from sklearn.covariance import LedoitWolf
        from shared_protocol.detectors.oneclass.model import load_model
        from shared_protocol.detectors.oneclass.train import features_for
        model = load_model(self.model_path)
        clean = [r for r in self.train if r["label"] == "benign"]
        feats = lambda rows: np.vstack([features_for(self.resolver(self.adapter_input(r)), r["sha256"], self.cfg) for r in rows])
        lw = LedoitWolf().fit(feats(clean))
        x = feats(self.val)
        np.testing.assert_allclose([model.raw_score(v) for v in x], lw.mahalanobis(x), rtol=1e-7)
        self.assertGreater(lw.shrinkage_, 0)  # 5 clean adapters, 30 features: shrinkage keeps it invertible

    def test_backdoored_adapters_are_farther_and_direction_is_fixed(self):
        self.assertEqual(self.result["validation_auroc"], 1.0)
        self.assertEqual(self.detector().spec.raw_direction, "higher_is_suspicious")

    def test_full_shared_flow_distinct_scores(self):
        detector = self.detector()
        cal = development_records(self.manifest, self.plan, "calibration")
        cal_scores = score_development(detector, cal, self.binding, "calibration")
        threshold = calibrate_benign(self.binding, cal, cal_scores)
        evaluation = [deepcopy(r) for r in self.manifest["adapters"] if r["adapter_id"] in self.binding.ids("evaluation")]
        report = evaluate(self.binding, evaluation, score_evaluation(detector, evaluation, self.binding, threshold), threshold)
        self.assertEqual(report["overall"]["count"], len(evaluation))
        self.assertEqual(len({s.raw for s in cal_scores.scores}), len(cal_scores.scores))

    def test_cache_does_not_change_scores_or_bypass_hash_check(self):
        row = self.train[0]
        inp = self.adapter_input(row)
        self.assertEqual(self.detector().score((inp,)), self.detector(cache_dir=self.cache).score((inp,)))
        path = self.resolver(type("A", (), {"adapter_id": row["adapter_id"]}))
        original = path.read_bytes()
        try:
            write_adapter(path, np.random.default_rng(5), True)
            with self.assertRaisesRegex(ProtocolError, "SHA256"):
                self.detector(cache_dir=self.cache).score((inp,))
        finally:
            path.write_bytes(original)

    def test_fit_refuses_overwrite_and_too_few_clean_adapters(self):
        from shared_protocol.detectors.oneclass.train import OneClassConfig, fit
        with self.assertRaisesRegex(ProtocolError, "never overwrite"):
            fit(self.binding, self.train, None, self.cfg, self.out, resolver=self.resolver, log=lambda *_: None)
        with self.assertRaisesRegex(ProtocolError, "clean train adapters"):
            fit(self.binding, self.train, None, OneClassConfig(layers=LAYERS, dim=DIM, min_clean=50),
                Path(self.tmp.name) / "too_few", resolver=self.resolver, log=lambda *_: None)

    def test_reload_rejects_wrong_hash_corruption_subset_wrong_view_and_other_split(self):
        from shared_protocol.detectors.oneclass.detector import OneClassDetector
        from shared_protocol.detectors.oneclass.model import save_model
        from shared_protocol.detectors.peftguard.converter import sha256_file
        good = self.result["model_file_sha256"]
        with self.assertRaisesRegex(ProtocolError, "SHA256"):
            OneClassDetector.from_model(self.model_path, self.binding, self.train, expected_file_sha256="0" * 64)
        data = read_json(self.model_path)
        data["model"]["mean"][0] += 1.0  # edited, digest stale
        bad = Path(self.tmp.name) / "bad.json"
        write_new_json(bad, data)
        with self.assertRaisesRegex(ProtocolError, "internal digest"):
            OneClassDetector.from_model(bad, self.binding, self.train, expected_file_sha256=sha256_file(bad))
        subset = read_json(self.model_path)["model"]
        subset["fit_ids"] = subset["fit_ids"][:-1]
        path = Path(self.tmp.name) / "subset.json"
        save_model(path, subset)
        with self.assertRaisesRegex(ProtocolError, "benign adapters"):
            OneClassDetector.from_model(path, self.binding, self.train, expected_file_sha256=sha256_file(path))
        with self.assertRaises(ProtocolError):  # a non-train view cannot be passed off as the train view
            OneClassDetector.from_model(self.model_path, self.binding,
                                        development_records(self.manifest, self.plan, "calibration"),
                                        expected_file_sha256=good)
        manifest, policy = fixture()
        policy["seed"] = 7
        for row, mine in zip(manifest["adapters"], self.manifest["adapters"]):
            row["sha256"] = mine["sha256"]
        binding2 = bind_protocol(manifest, create_split(manifest, policy))
        with self.assertRaisesRegex(ProtocolError, "different manifest/split"):
            OneClassDetector.from_model(self.model_path, binding2, self.train, expected_file_sha256=good)

    def test_wrong_rank_rejected_before_file_read(self):
        bad = AdapterInput("x", "fixture://nope", "r", "0" * 64, PRIMARY_MODEL, 8, 16.0, ("q_proj", "v_proj"))
        with self.assertRaisesRegex(ProtocolError, "rank"):
            self.detector().score((bad,))

    def test_legacy_cache_never_read_or_written(self):
        from unittest.mock import patch
        from shared_protocol.detectors.oneclass.train import features_for, OneClassConfig
        row = self.train[0]
        inp = self.adapter_input(row)
        path = self.resolver(inp)
        cached = self.cache / "stale.npy"
        self.cache.mkdir(exist_ok=True)
        np.save(cached, np.full(30, 99.0))
        original = cached.read_bytes()
        with patch("numpy.load", side_effect=AssertionError("cache read")), patch(
                "numpy.save", side_effect=AssertionError("cache written")):
            self.assertEqual(self.detector().score((inp,)), self.detector(cache_dir=self.cache).score((inp,)))
            with self.assertRaisesRegex(ProtocolError, "rank"):
                features_for(path, inp.sha256, OneClassConfig(layers=LAYERS, dim=DIM, rank=8), self.cache)
        self.assertEqual(cached.read_bytes(), original)

    def test_provenance_rejects_missing_stale_and_incompatible_state(self):
        from shared_protocol.detectors.oneclass.detector import OneClassDetector
        from shared_protocol.detectors.oneclass.model import save_model
        from shared_protocol.detectors.peftguard.converter import sha256_file
        original = read_json(self.model_path)["model"]
        for i, (field, value, error) in enumerate((
            ("extractor_and_wrapper_sha256", None, "implementation provenance"),
            ("extractor_and_wrapper_sha256", "0" * 64, "implementation provenance"),
            ("software", {}, "software provenance"),
            ("n_fit", 0, "fitted adapter count"),
            ("shrinkage", 2.0, "fitted shrinkage"),
        )):
            with self.subTest(field=field, value=value):
                body = {**original, field: value}
                if value is None:
                    del body[field]
                path = Path(self.tmp.name) / f"provenance_{i}.json"
                save_model(path, body)
                with self.assertRaisesRegex(ProtocolError, error):
                    OneClassDetector.from_model(path, self.binding, self.train,
                                                expected_file_sha256=sha256_file(path))

    def test_source_identity_includes_preprocessing_training_and_scoring(self):
        from unittest.mock import patch
        from shared_protocol.detectors.oneclass.train import source_sha256
        from shared_protocol.detectors.peftguard.converter import sha256_file
        original = source_sha256()
        for name in ("peftguard/converter.py", "oneclass/train.py", "peftguard/train.py", "shared_protocol/scoring.py"):
            with self.subTest(name=name), patch(
                "shared_protocol.detectors.oneclass.train.sha256_file",
                side_effect=lambda path: "0" * 64 if path.as_posix().endswith(name) else sha256_file(path),
            ):
                self.assertNotEqual(source_sha256(), original)

    def test_degenerate_fit_reports_reason_and_writes_no_model(self):
        from unittest.mock import patch
        from shared_protocol.detectors.oneclass.train import OneClassConfig, fit
        with self.assertRaisesRegex(ProtocolError, "minimum clean"):
            OneClassConfig(min_clean=1)
        path = Path(self.tmp.name) / "constant_fit"
        with patch("shared_protocol.detectors.oneclass.train.features_for", return_value=np.zeros(30)):
            with self.assertRaisesRegex(ProtocolError, "identical"):
                fit(self.binding, self.train, None, self.cfg, path, resolver=self.resolver, log=lambda *_: None)
        self.assertFalse((path / "model.json").exists())

    def test_supported_profile_checked_before_file_read(self):
        bad = AdapterInput("x", "fixture://nope", "r", "0" * 64, "different-base", RANK, 16.0,
                           ("q_proj", "v_proj"))
        with self.assertRaisesRegex(ProtocolError, "base model"):
            self.detector().score((bad,))

    def test_model_publication_does_not_leave_partial_file(self):
        from unittest.mock import patch
        from shared_protocol.detectors.oneclass.model import save_model
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            with patch("shared_protocol.detectors.oneclass.model.os.link", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    save_model(path, read_json(self.model_path)["model"])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_formal_worker_ignores_parent_imports_and_rejects_source_drift(self):
        import json
        import os
        import py_compile
        import shutil
        import subprocess
        import sys
        from dataclasses import asdict
        from shared_protocol.detectors.oneclass import features

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            shutil.copytree(REPO / "shared_protocol", source / "shared_protocol")
            bootstrap = REPO / "notebooks" / "oneclass_run.py"
            command = [sys.executable, "-I", "-S", "-B", str(bootstrap)]
            source_option = ["--source", str(source)]
            identity = subprocess.check_output(command + ["identity", *source_option], text=True).strip()
            bundle = root / "development.json"
            cal = development_records(self.manifest, self.plan, "calibration")
            write_new_json(bundle, dict(binding=asdict(self.binding), train=self.train,
                                        validation=self.val, calibration=cal))
            output = root / "threshold.json"
            args = ["calibrate", *source_option, "--expected-source", identity, "--bundle", str(bundle),
                    "--adapters", str(Path(self.tmp.name) / "ad"), "--output", str(output),
                    "--model", str(self.model_path), "--model-sha256", self.result["model_file_sha256"]]
            for dependency in (p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p):
                args.extend(["--dependency-dir", dependency])

            # The parent's imported module remains v2 after the source copy
            # changes. A fresh formal launch rejects the altered checkout.
            changed = source / "shared_protocol/detectors/oneclass/features.py"
            old = changed.read_bytes()
            changed.write_bytes(old + b"\nEXTRACTOR_VERSION = 'different'\n")
            self.assertNotEqual(features.EXTRACTOR_VERSION, "different")
            rejected = subprocess.run(command + args, text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Source snapshot does not match", rejected.stderr)
            self.assertFalse(output.exists())
            py_compile.compile(str(changed), doraise=True,
                               invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
            changed.write_bytes(old)

            # Even a conflicting package on PYTHONPATH or in the working
            # directory cannot override the verified source copy.
            shadow = root / "shared_protocol"
            shadow.mkdir()
            (shadow / "__init__.py").write_text("raise RuntimeError('wrong module origin')\n")
            env = dict(os.environ, PYTHONPATH=str(root))
            accepted = subprocess.run(command + args, cwd=root, env=env, text=True, capture_output=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            receipt = json.loads(accepted.stdout.splitlines()[-1])
            self.assertEqual(receipt["source_sha256"], identity)
            result = root / "evaluation.json"
            release = root / "release.json"
            records = [r for r in self.manifest["adapters"] if r["adapter_id"] in self.binding.ids("evaluation")]
            release_body = dict(protocol_sha256=self.binding.sha256, threshold_sha256=receipt["threshold_sha256"],
                                records=records[:-1])
            write_new_json(release, release_body)
            evaluate = ["evaluate", *args[1:], "--output", str(result),
                        "--threshold", str(output), "--threshold-sha256", receipt["threshold_sha256"],
                        "--release", str(release)]
            rejected = subprocess.run(command + evaluate, cwd=root, env=env, text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("authorized partition", rejected.stderr)
            self.assertFalse(result.exists())
            release.unlink()
            write_new_json(release, {**release_body, "records": records})
            subprocess.run(command + evaluate, cwd=root, env=env, check=True, capture_output=True, text=True)
            self.assertEqual(read_json(result)["overall"]["count"], len(records))

    def test_finite_ill_conditioned_covariance_is_rejected(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from shared_protocol.detectors.oneclass.train import fit
        cov = np.diag([1e-18] + [1.0] * 29)
        out = Path(self.tmp.name) / "ill_conditioned"
        with patch("sklearn.covariance.LedoitWolf.fit", return_value=SimpleNamespace(covariance_=cov)):
            with self.assertRaisesRegex(ProtocolError, "ill-conditioned"):
                fit(self.binding, self.train, None, self.cfg, out, resolver=self.resolver, log=lambda *_: None)
        self.assertFalse((out / "model.json").exists())

    def test_second_publication_preserves_existing_model_bytes(self):
        from shared_protocol.detectors.oneclass.model import save_model
        original = self.model_path.read_bytes()
        body = read_json(self.model_path)["model"]
        body["mean"][0] += 1
        with self.assertRaises(FileExistsError):
            save_model(self.model_path, body)
        self.assertEqual(self.model_path.read_bytes(), original)

    def test_fit_rejects_oversized_and_nonregular_adapters_without_prehash(self):
        import os
        from unittest.mock import patch
        from shared_protocol.detectors.oneclass.train import fit
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oversized"
            with path.open("wb") as handle:
                handle.truncate(2 << 20)
            fifo = Path(tmp) / "fifo"
            os.mkfifo(fifo)
            for target, reason in ((path, "exceeds"), (fifo, "regular")):
                with self.subTest(target=target), patch("shared_protocol.detectors.peftguard.train.sha256_file",
                                                       side_effect=AssertionError("preliminary hash")):
                    with self.assertRaisesRegex(ProtocolError, reason):
                        fit(self.binding, self.train, None, self.cfg, Path(tmp) / "run",
                            resolver=lambda _: target, log=lambda *_: None)

    def test_validation_diagnostics_do_not_change_fitted_parameters(self):
        from shared_protocol.detectors.oneclass.train import fit
        path = Path(self.tmp.name) / "no_validation"
        result = fit(self.binding, self.train, None, self.cfg, path, resolver=self.resolver, log=lambda *_: None)
        expected = read_json(self.model_path)["model"]
        actual = read_json(result["model_file"])["model"]
        self.assertIsNone(actual.pop("diagnostics")["validation_auroc_not_used_for_anything"])
        expected.pop("diagnostics")
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
