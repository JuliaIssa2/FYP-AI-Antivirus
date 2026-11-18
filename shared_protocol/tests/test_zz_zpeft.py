"""Z-PEFT-style detector tests on tiny synthetic adapters (hidden 32, 2 layers, 4 heads, rank 16). CPU only, no network.
Skipped without numpy/scikit-learn/safetensors. Run: python -B -m unittest shared_protocol.tests.test_zz_zpeft -v"""

from copy import deepcopy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAVE = all(importlib.util.find_spec(name) for name in ("numpy", "sklearn", "safetensors"))
np = save_file = None


def setUpModule():
    global np, save_file
    if HAVE:
        import numpy as _np
        from safetensors.numpy import save_file as _save
        np, save_file = _np, _save


from shared_protocol.calibration import calibrate_benign
from shared_protocol.evaluation import evaluate, score_evaluation
from shared_protocol.manifest import ProtocolError, read_json, write_new_json
from shared_protocol.scoring import AdapterInput, bind_protocol, score_development
from shared_protocol.splits import create_split, development_records
from shared_protocol.tests.test_foundations import fixture

DIM, LAYERS, HEADS, RANK = 32, 2, 4, 16


def write_adapter(path: Path, rng, malicious: bool, rank: int = RANK) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for layer in range(LAYERS):
        for module in ("q_proj", "v_proj"):
            base = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            tensors[f"{base}.lora_A.weight"] = rng.normal(0, 0.05, (rank, DIM)).astype(np.float32)
            tensors[f"{base}.lora_B.weight"] = ((4.0 if malicious else 1.0) * rng.normal(0, 0.05, (DIM, rank))).astype(np.float32)
    save_file(tensors, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_descriptor(sigma):
    """Independent, loop-based definition from the module docstring (sigma descending)."""
    sigma = [float(s) for s in sigma]
    slots = (sigma + [0.0] * 8)[:8]
    energy, l1 = sum(s * s for s in sigma), sum(sigma)
    if l1 == 0 or sigma[0] == 0:
        return [*slots, *([0.0] * 8)]
    p = [s / l1 for s in sigma]
    entropy = -sum(q * np.log(q) for q in p if q > 0)
    mean = sum(sigma) / len(sigma)
    var = sum((s - mean) ** 2 for s in sigma) / len(sigma)
    kurt = (sum((s - mean) ** 4 for s in sigma) / len(sigma)) / var ** 2 - 3 if var > 0 else 0.0
    return [*slots, energy, energy ** 0.5, l1, energy / sigma[0] ** 2, entropy, float(np.exp(entropy)), sigma[0] / l1, kurt]


@unittest.skipUnless(HAVE, "needs numpy, scikit-learn and safetensors")
class FeatureTests(unittest.TestCase):
    def brute(self, a, b):
        out = []
        for h in range(HEADS):
            dw = (b @ a).astype(np.float64)[h * (DIM // HEADS):(h + 1) * (DIM // HEADS)]
            # the factored computation returns min(head_dim, total rank) values; the rest of a full SVD are ~1e-15 zeros
            out.append(reference_descriptor(np.linalg.svd(dw, compute_uv=False)[:min(dw.shape[0], a.shape[0])]))
        return np.array(out)

    def test_descriptors_match_brute_force_svd_of_headwise_update(self):
        from shared_protocol.detectors.zpeft.features import descriptors, head_singular_values
        rng = np.random.default_rng(0)
        for rank in (16, 4):  # rank 4 < 8 slots: zero padding
            aq, av = rng.normal(0, 1, (rank, DIM)), rng.normal(0, 1, (rank, DIM))
            bq, bv = rng.normal(0, 1, (DIM, rank)), rng.normal(0, 1, (DIM, rank))
            cases = {"q": (aq, bq), "v": (av, bv), "q+v": (np.vstack([aq, av]), np.hstack([bq, bv]))}
            for name, (a, b) in cases.items():
                with self.subTest(rank=rank, block=name):
                    got = descriptors(head_singular_values(a, b, HEADS))
                    np.testing.assert_allclose(got, self.brute(a, b), rtol=1e-8, atol=1e-9)
        # the q+v block is the head-wise SUM of the two updates
        dw = (bq @ aq + bv @ av)[:DIM // HEADS]
        np.testing.assert_allclose(head_singular_values(cases["q+v"][0], cases["q+v"][1], HEADS)[0],
                                   np.linalg.svd(dw, compute_uv=False)[:head_singular_values(*cases["q+v"], HEADS).shape[1]],
                                   rtol=1e-8, atol=1e-9)

    def test_zero_block_is_all_zero_not_nan(self):
        from shared_protocol.detectors.zpeft.features import descriptors
        d = descriptors(np.zeros((3, 8)))
        self.assertTrue((d == 0).all())

    def test_layout_order_and_size(self):
        from shared_protocol.detectors.zpeft.features import (
            DESCRIPTOR_DIM, extract_features, feature_dim, feature_names, head_singular_values, descriptors)
        self.assertEqual(feature_dim(32, 32), 49_152)
        self.assertEqual(len(feature_names(LAYERS, HEADS)), feature_dim(LAYERS, HEADS))
        rng = np.random.default_rng(1)
        f = {(l, m): (rng.normal(0, 1, (RANK, DIM)), rng.normal(0, 1, (DIM, RANK)))
             for l in range(LAYERS) for m in ("q_proj", "v_proj")}
        x = extract_features(f, layers=LAYERS, n_heads=HEADS)
        self.assertEqual(x.shape, (feature_dim(LAYERS, HEADS),))
        names = feature_names(LAYERS, HEADS)
        # layer 1, v projection (index 1 of q, v, q+v), head 2 -> start of its 16 numbers
        start = names.index("L1.v.h2.sigma1")
        expected = descriptors(head_singular_values(*f[(1, "v_proj")], HEADS))[2]
        np.testing.assert_allclose(x[start:start + DESCRIPTOR_DIM], expected)
        self.assertEqual(start, ((1 * 3 + 1) * HEADS + 2) * DESCRIPTOR_DIM)

    def test_bad_head_split_rejected(self):
        from shared_protocol.detectors.zpeft.features import head_singular_values
        with self.assertRaises(ValueError):
            head_singular_values(np.zeros((RANK, DIM)), np.zeros((DIM, RANK)), 5)


@unittest.skipUnless(HAVE, "needs numpy, scikit-learn and safetensors")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from shared_protocol.detectors.peftguard.converter import root_resolver
        from shared_protocol.detectors.zpeft.train import ZPeftConfig, fit
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
        cls.cfg = ZPeftConfig(dim=DIM, layers=LAYERS, n_heads=HEADS)
        cls.train = development_records(manifest, cls.plan, "train")
        cls.val = development_records(manifest, cls.plan, "validation")
        cls.out = root / "run"
        cls.cache = root / "cache"
        cls.result = fit(cls.binding, cls.train, cls.val, cls.cfg, cls.out, resolver=cls.resolver,
                         cache_dir=cls.cache, log=lambda *_: None)
        cls.model_path = Path(cls.result["model_file"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def detector(self, **kw):
        from shared_protocol.detectors.zpeft.detector import ZPeftDetector
        return ZPeftDetector.from_model(self.model_path, self.binding, expected_file_sha256=self.result["model_file_sha256"],
                                        resolver=self.resolver, **kw)

    def adapter_input(self, row):
        return AdapterInput(row["adapter_id"], row["path_or_source"], row["source_revision"], row["sha256"],
                            row["base_model"], row["lora_rank"], row["lora_alpha"], tuple(sorted(row["target_modules"])))

    def test_fit_scope_is_exactly_train_and_validation_never_tunes(self):
        detector = self.detector()
        self.assertEqual(detector.spec.fit_ids, self.binding.ids("train"))
        self.assertEqual(detector.spec.tuning_ids, ())
        detector.spec.verify(self.binding)
        body = read_json(self.model_path)["model"]
        self.assertEqual(body["selection"]["method"], "grouped_stratified_kfold_on_train")
        self.assertEqual(len(body["selection"]["grid"]), 12)  # 6 C values x 2 class weights

    def test_planted_signal_is_learned(self):
        self.assertGreaterEqual(self.result["best"]["mean_auc"], 0.9)
        self.assertEqual(self.result["validation_auroc"], 1.0)

    def test_numpy_inference_matches_sklearn_decision_function(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from shared_protocol.detectors.zpeft.model import load_model
        from shared_protocol.detectors.zpeft.train import features_for
        model = load_model(self.model_path)
        rows = self.train + self.val
        x = np.vstack([features_for(self.resolver(self.adapter_input(r)), r["sha256"], self.cfg) for r in rows])
        y = np.array([r["label"] == "malicious" for r in rows], int)
        body = model.body
        scaler = StandardScaler(); scaler.mean_ = model.mean; scaler.scale_ = model.scale
        scaler.var_ = model.scale ** 2; scaler.n_features_in_ = len(model.mean); scaler.n_samples_seen_ = len(rows)
        clf = LogisticRegression(); clf.classes_ = np.array([0, 1]); clf.coef_ = model.coef[None, :]
        clf.intercept_ = np.array([model.intercept])
        want = clf.decision_function(scaler.transform(x))
        got = np.array([model.raw_score(v) for v in x])
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)
        self.assertEqual(body["classifier"]["classes"], [0, 1])
        self.assertGreater(got[y == 1].mean(), got[y == 0].mean())  # larger = more malicious

    def test_full_shared_flow_with_unsaturated_distinct_scores(self):
        detector = self.detector()
        cal = development_records(self.manifest, self.plan, "calibration")
        cal_scores = score_development(detector, cal, self.binding, "calibration")
        threshold = calibrate_benign(self.binding, cal, cal_scores)
        evaluation = [deepcopy(r) for r in self.manifest["adapters"] if r["adapter_id"] in self.binding.ids("evaluation")]
        report = evaluate(self.binding, evaluation, score_evaluation(detector, evaluation, self.binding, threshold), threshold)
        self.assertEqual(report["overall"]["count"], len(evaluation))
        self.assertEqual(len({s.raw for s in cal_scores.scores}), len(cal_scores.scores))

    def test_cache_does_not_change_scores(self):
        row = self.train[0]
        inp = self.adapter_input(row)
        self.assertEqual(self.detector().score((inp,)), self.detector(cache_dir=self.cache).score((inp,)))
        self.assertTrue(any(self.cache.glob("*.npy")))

    def test_fit_never_overwrites(self):
        from shared_protocol.detectors.zpeft.train import fit
        with self.assertRaisesRegex(ProtocolError, "never overwrite"):
            fit(self.binding, self.train, self.val, self.cfg, self.out, resolver=self.resolver, log=lambda *_: None)

    def test_tampered_adapter_rejected_in_fit_and_scoring(self):
        from shared_protocol.detectors.peftguard.train import items_from_records
        row = self.train[0]
        path = self.resolver(type("A", (), {"adapter_id": row["adapter_id"]}))
        original = path.read_bytes()
        try:
            write_adapter(path, np.random.default_rng(99), False)
            with self.assertRaisesRegex(ProtocolError, "SHA256"):
                items_from_records(self.binding, self.train, "train", self.resolver, self.cfg)
            with self.assertRaisesRegex(ProtocolError, "SHA256"):
                self.detector().score((self.adapter_input(row),))
        finally:
            path.write_bytes(original)

    def test_cached_features_do_not_bypass_the_hash_check(self):
        row = self.train[0]
        self.detector(cache_dir=self.cache).score((self.adapter_input(row),))  # populate cache
        path = self.resolver(type("A", (), {"adapter_id": row["adapter_id"]}))
        original = path.read_bytes()
        try:
            write_adapter(path, np.random.default_rng(5), True)
            with self.assertRaisesRegex(ProtocolError, "SHA256"):
                self.detector(cache_dir=self.cache).score((self.adapter_input(row),))
        finally:
            path.write_bytes(original)

    def test_reload_rejects_swapped_model_corruption_other_split_and_partial_fit(self):
        from shared_protocol.detectors.zpeft.detector import ZPeftDetector
        good = self.result["model_file_sha256"]
        with self.assertRaisesRegex(ProtocolError, "SHA256"):
            ZPeftDetector.from_model(self.model_path, self.binding, expected_file_sha256="0" * 64)
        data = read_json(self.model_path)
        data["model"]["classifier"]["intercept"] += 1.0  # edited but digest left stale
        bad = Path(self.tmp.name) / "bad_model.json"
        write_new_json(bad, data)
        from shared_protocol.detectors.peftguard.converter import sha256_file
        with self.assertRaisesRegex(ProtocolError, "internal digest"):
            ZPeftDetector.from_model(bad, self.binding, expected_file_sha256=sha256_file(bad))
        from shared_protocol.detectors.zpeft.model import save_model
        partial = read_json(self.model_path)["model"]
        partial["fit_ids"] = partial["fit_ids"][:-1]
        path = Path(self.tmp.name) / "partial_model.json"
        save_model(path, partial)
        with self.assertRaisesRegex(ProtocolError, "complete shared"):
            ZPeftDetector.from_model(path, self.binding, expected_file_sha256=sha256_file(path))
        manifest, policy = fixture()
        policy["seed"] = 7
        for row, mine in zip(manifest["adapters"], self.manifest["adapters"]):
            row["sha256"] = mine["sha256"]
        binding2 = bind_protocol(manifest, create_split(manifest, policy))
        with self.assertRaisesRegex(ProtocolError, "different manifest/split"):
            ZPeftDetector.from_model(self.model_path, binding2, expected_file_sha256=good)

    def test_grouped_folds_never_split_a_leakage_group(self):
        from shared_protocol.detectors.zpeft.train import _grouped_folds
        y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        groups = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"]
        folds = _grouped_folds(y, groups, 3, 0)
        self.assertTrue(folds)
        for tr, te in folds:
            self.assertTrue({groups[i] for i in tr}.isdisjoint({groups[i] for i in te}))
        with self.assertRaises(ProtocolError):
            _grouped_folds(np.array([0, 1]), ["a", "b"], 5, 0)

    def test_wrong_rank_or_modules_rejected_before_file_read(self):
        bad = AdapterInput("x", "fixture://nope", "r", "0" * 64, "m", 8, 16.0, ("q_proj", "v_proj"))
        with self.assertRaisesRegex(ProtocolError, "rank"):
            self.detector().score((bad,))


if __name__ == "__main__":
    unittest.main()
