"""PEFTGuard integration tests. Synthetic tiny adapters (hidden 16, 2 layers, rank 16) written to a temp dir; no GPU,
no network, no real adapter. Skipped when torch/safetensors/numpy are missing.
Run: python -B -m unittest shared_protocol.tests.test_zz_peftguard -v"""

from copy import deepcopy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import importlib.util

# Heavy imports happen in setUpModule, never at import time, and this file sorts last: two existing tests assert that
# `torch` is not in sys.modules, and unittest imports every test module before running any test.
HAVE = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "safetensors"))
np = torch = save_file = None


def setUpModule():
    global np, torch, save_file
    if HAVE:
        import numpy as _np
        import torch as _torch
        from safetensors.numpy import save_file as _save
        np, torch, save_file = _np, _torch, _save


from shared_protocol.calibration import calibrate_benign, save_threshold
from shared_protocol.evaluation import evaluate, score_evaluation
from shared_protocol.manifest import ProtocolError, digest, write_new_json
from shared_protocol.scoring import bind_protocol, score_development
from shared_protocol.splits import create_split, development_records
from shared_protocol.tests.test_foundations import fixture

DIM, LAYERS, RANK = 16, 2, 16


def write_adapter(path: Path, rng, malicious: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for layer in range(LAYERS):
        for module in ("q_proj", "v_proj"):
            base = f"base_model.model.model.layers.{layer}.self_attn.{module}"
            scale = 4.0 if malicious else 1.0
            tensors[f"{base}.lora_A.weight"] = rng.normal(0, 0.05, (RANK, DIM)).astype(np.float32)
            tensors[f"{base}.lora_B.weight"] = (scale * rng.normal(0, 0.05, (DIM, RANK))).astype(np.float32)
    save_file(tensors, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_case(root: Path):
    """The shared synthetic manifest, with every row pointing at a real tiny safetensors file."""
    from shared_protocol.detectors.peftguard.converter import root_resolver
    manifest, policy = fixture()
    rng = np.random.default_rng(0)
    for row in manifest["adapters"]:
        row["sha256"] = write_adapter(root / row["adapter_id"] / "adapter_model.safetensors", rng,
                                      row["label"] == "malicious")
    plan = create_split(manifest, policy)
    return manifest, plan, bind_protocol(manifest, plan), root_resolver(root)


def small_cfg(**kw):
    from shared_protocol.detectors.peftguard.train import TrainConfig
    return TrainConfig(dim=DIM, layers=LAYERS, epochs=4, patience=5, seed=0, **kw)


@unittest.skipUnless(HAVE, "needs numpy, torch and safetensors")
class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.file = self.root / "a" / "adapter_model.safetensors"
        write_adapter(self.file, np.random.default_rng(1), False)

    def test_authors_layout_plain_BA_layers_last_q_then_v(self):
        from shared_protocol.detectors.peftguard.converter import build_input, load_qv_factors
        factors = load_qv_factors(self.file, dim=DIM, layers=LAYERS, rank=RANK)
        x = build_input(factors, dim=DIM, layers=LAYERS)
        self.assertEqual(tuple(x.shape), (DIM, DIM, 2 * LAYERS))
        self.assertEqual(x.dtype, torch.float32)
        for layer in range(LAYERS):
            for j, module in enumerate(("q_proj", "v_proj")):
                a, b = factors[(layer, module)]
                np.testing.assert_allclose(x[:, :, 2 * layer + j].numpy(), b @ a, rtol=1e-5, atol=1e-7)
        # q and v of one layer differ, so the channel order is observable
        self.assertFalse(torch.allclose(x[:, :, 0], x[:, :, 1]))

    def test_no_alpha_scaling_input_is_independent_of_config(self):
        from shared_protocol.detectors.peftguard.converter import build_input, load_qv_factors
        (self.file.parent / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 999}))
        x1 = build_input(load_qv_factors(self.file, dim=DIM, layers=LAYERS), dim=DIM, layers=LAYERS)
        (self.file.parent / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 1}))
        x2 = build_input(load_qv_factors(self.file, dim=DIM, layers=LAYERS), dim=DIM, layers=LAYERS)
        self.assertTrue(torch.equal(x1, x2))

    def test_loader_rejects_instead_of_guessing(self):
        from shared_protocol.detectors.peftguard.converter import ConverterError, load_qv_factors
        with self.assertRaises(ConverterError):          # wrong hidden size: never pad/crop
            load_qv_factors(self.file, dim=DIM * 2, layers=LAYERS)
        with self.assertRaises(ConverterError):          # missing layer
            load_qv_factors(self.file, dim=DIM, layers=LAYERS + 1)
        with self.assertRaises(ConverterError):          # wrong rank
            load_qv_factors(self.file, dim=DIM, layers=LAYERS, rank=8)
        extra = self.root / "k" / "adapter_model.safetensors"
        extra.parent.mkdir()
        save_file({"base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": np.zeros((RANK, DIM), np.float32)},
                  str(extra))
        with self.assertRaises(ConverterError):          # k_proj is outside the q/v layout
            load_qv_factors(extra, dim=DIM, layers=LAYERS)


@unittest.skipUnless(HAVE, "needs numpy, torch and safetensors")
class ModelTests(unittest.TestCase):
    def test_llama2_architecture_matches_released_checkpoint(self):
        from shared_protocol.detectors.peftguard.model import PGLlama2
        with torch.device("meta"):
            model = PGLlama2(4096, 32)
        self.assertEqual(set(model.state_dict()), {f"{n}.{p}" for n in ("conv1", "fc1", "fc2", "fc3")
                                                   for p in ("weight", "bias")})
        self.assertEqual(tuple(model.conv1.weight.shape), (16, 64, 8, 8))
        self.assertEqual(model.fc1.in_features, 16 * 512 * 512)
        total = sum(p.numel() for p in model.parameters())
        self.assertEqual(model.fc1.weight.numel(), 2_147_483_648)
        self.assertAlmostEqual(total * 4 / 1e9, 8.6, delta=0.1)  # 8.6 GB fp32, the released checkpoint size

    def test_raw_score_is_logit_difference_not_saturating_probability(self):
        from shared_protocol.detectors.peftguard.model import raw_score_from_logits
        a, b = raw_score_from_logits(torch.tensor([[-30.0, 30.0]])), raw_score_from_logits(torch.tensor([[-35.0, 35.0]]))
        self.assertEqual((a, b), (60.0, 70.0))  # distinct, whereas sigmoid would give 1.0 for both
        self.assertEqual(float(torch.sigmoid(torch.tensor(a))), float(torch.sigmoid(torch.tensor(b))))

    def test_auto_memory_plan(self):
        from shared_protocol.detectors.peftguard.train import auto_memory_plan
        self.assertEqual(auto_memory_plan(40), ("adam8bit", "bf16"))
        self.assertEqual(auto_memory_plan(80), ("adam", "none"))
        self.assertEqual(auto_memory_plan(None), ("adam", "none"))


@unittest.skipUnless(HAVE, "needs numpy, torch and safetensors")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from shared_protocol.detectors.peftguard.detector import PEFTGuardDetector
        from shared_protocol.detectors.peftguard.train import fit
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.manifest, cls.plan, cls.binding, resolver = build_case(root / "adapters")
        cls.resolver = staticmethod(resolver)  # a bare lambda on the class would bind as a method
        cls.cfg = small_cfg()
        cls.out = root / "run"
        cls.train = development_records(cls.manifest, cls.plan, "train")
        cls.val = development_records(cls.manifest, cls.plan, "validation")
        cls.record = fit(cls.binding, cls.train, cls.val, cls.cfg, cls.out, resolver=cls.resolver, device="cpu",
                         log=lambda *_: None)
        cls.ckpt, cls.fit_json = cls.out / "peftguard_best.pt", cls.out / "fit.json"
        cls.detector = PEFTGuardDetector.from_fit(cls.fit_json, cls.ckpt, cls.binding, device="cpu",
                                                  expected_checkpoint_sha256=cls.record["checkpoint_sha256"],
                                                  resolver=cls.resolver)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fit_uses_exactly_the_shared_train_and_validation_partitions(self):
        self.assertEqual(tuple(self.record["fit_ids"]), self.binding.ids("train"))
        self.assertEqual(tuple(self.record["tuning_ids"]), self.binding.ids("validation"))
        self.assertTrue(set(self.record["fit_ids"]).isdisjoint(self.binding.ids("calibration")))
        self.assertTrue(set(self.record["fit_ids"]).isdisjoint(self.binding.ids("evaluation")))
        self.assertEqual(self.detector.spec.fit_ids, self.binding.ids("train"))
        self.detector.spec.verify(self.binding)

    def test_training_loop_learns_planted_signal(self):
        history = self.record["history"]
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])

    def test_fit_output_is_never_overwritten(self):
        from shared_protocol.detectors.peftguard.train import fit
        with self.assertRaisesRegex(ProtocolError, "never overwrite"):
            fit(self.binding, self.train, self.val, self.cfg, self.out, resolver=self.resolver, device="cpu",
                log=lambda *_: None)

    def test_full_shared_flow_calibrate_then_evaluate(self):
        from shared_protocol.manifest import canonical_json
        cal = development_records(self.manifest, self.plan, "calibration")
        cal_scores = score_development(self.detector, cal, self.binding, "calibration")
        threshold = calibrate_benign(self.binding, cal, cal_scores)
        evaluation = [deepcopy(r) for r in self.manifest["adapters"] if r["adapter_id"] in self.binding.ids("evaluation")]
        scored = score_evaluation(self.detector, evaluation, self.binding, threshold)
        report = evaluate(self.binding, evaluation, scored, threshold)
        self.assertEqual(report["overall"]["count"], len(evaluation))
        self.assertEqual(len({s.raw for s in cal_scores.scores}), len(cal_scores.scores))  # no saturated ties
        canonical_json(report)

    def test_scoring_is_deterministic(self):
        from shared_protocol.scoring import AdapterInput
        row = self.train[0]
        inp = AdapterInput(row["adapter_id"], row["path_or_source"], row["source_revision"], row["sha256"],
                           row["base_model"], row["lora_rank"], row["lora_alpha"], tuple(sorted(row["target_modules"])))
        self.assertEqual(self.detector.score((inp,)), self.detector.score((inp,)))

    def test_tampered_adapter_bytes_rejected_in_training_and_scoring(self):
        from shared_protocol.detectors.peftguard.train import items_from_records, fit
        row = self.train[0]
        path = self.resolver(type("A", (), {"adapter_id": row["adapter_id"]}))
        original = path.read_bytes()
        try:
            write_adapter(path, np.random.default_rng(99), False)
            with self.assertRaisesRegex(ProtocolError, "SHA256"):
                items_from_records(self.binding, self.train, "train", self.resolver, self.cfg)
            from shared_protocol.scoring import AdapterInput
            inp = AdapterInput(row["adapter_id"], row["path_or_source"], row["source_revision"], row["sha256"],
                               row["base_model"], row["lora_rank"], row["lora_alpha"], ("q_proj", "v_proj"))
            with self.assertRaisesRegex(ProtocolError, "SHA256"):
                self.detector.score((inp,))
        finally:
            path.write_bytes(original)

    def test_only_development_partitions_can_fit(self):
        from shared_protocol.detectors.peftguard.train import items_from_records
        for partition in ("calibration", "evaluation", "final_holdout"):
            with self.assertRaises(ProtocolError):
                items_from_records(self.binding, [], partition, self.resolver, self.cfg)

    def test_reload_rejects_swapped_checkpoint_other_split_and_partial_fit(self):
        from shared_protocol.detectors.peftguard.detector import PEFTGuardDetector
        good = self.record["checkpoint_sha256"]
        other = Path(self.tmp.name) / "other.pt"
        other.write_bytes(self.ckpt.read_bytes() + b"x")
        with self.assertRaisesRegex(ProtocolError, "SHA256"):
            PEFTGuardDetector.from_fit(self.fit_json, other, self.binding, expected_checkpoint_sha256=good, device="cpu")
        with self.assertRaisesRegex(ProtocolError, "SHA256"):  # caller supplies a wrong trusted hash
            PEFTGuardDetector.from_fit(self.fit_json, self.ckpt, self.binding, expected_checkpoint_sha256="0" * 64,
                                       device="cpu")
        # a record claiming fewer fit IDs than the whole train partition is refused
        partial = deepcopy(self.record)
        partial["fit_ids"] = partial["fit_ids"][:-1]
        path = Path(self.tmp.name) / "partial_fit.json"
        write_new_json(path, partial)
        with self.assertRaisesRegex(ProtocolError, "complete shared"):
            PEFTGuardDetector.from_fit(path, self.ckpt, self.binding, expected_checkpoint_sha256=good, device="cpu")
        # a different split (shuffled policy seed) is refused
        manifest, policy = fixture()
        policy["seed"] = 7
        for row, mine in zip(manifest["adapters"], self.manifest["adapters"]):
            row["sha256"] = mine["sha256"]
        binding2 = bind_protocol(manifest, create_split(manifest, policy))
        with self.assertRaisesRegex(ProtocolError, "different manifest/split"):
            PEFTGuardDetector.from_fit(self.fit_json, self.ckpt, binding2, expected_checkpoint_sha256=good, device="cpu")

    def test_pretrained_reference_cannot_claim_a_fit(self):
        from shared_protocol.detectors.peftguard.detector import PEFTGuardDetector
        ref = PEFTGuardDetector.pretrained_reference(
            self.ckpt, expected_sha256=self.record["checkpoint_sha256"], dim=DIM, layers=LAYERS, device="cpu",
            provenance="Synthetic stand-in for the authors' checkpoint; real training adapters unknown",
            resolver=self.resolver)
        self.assertEqual((ref.spec.fit_ids, ref.spec.tuning_ids), ((), ()))
        self.assertIn("reference", ref.spec.name)
        cal = development_records(self.manifest, self.plan, "calibration")
        self.assertTrue(calibrate_benign(self.binding, cal, score_development(ref, cal, self.binding, "calibration")))

    def test_wrong_rank_or_modules_rejected_before_any_file_read(self):
        from shared_protocol.scoring import AdapterInput
        bad = AdapterInput("x", "fixture://nope", "r", "0" * 64, "m", 8, 16.0, ("q_proj", "v_proj"))
        with self.assertRaisesRegex(ProtocolError, "rank"):
            self.detector.score((bad,))


if __name__ == "__main__":
    unittest.main()
