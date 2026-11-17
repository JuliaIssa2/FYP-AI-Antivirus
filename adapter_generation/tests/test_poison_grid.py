"""Synthetic tests for payload selection (poison.py) and job expansion (grid.py). No GPU, no network."""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import poison  # noqa: E402
from grid import expand_jobs  # noqa: E402

ROWS = [{"instruction": f"Question {i}.", "input": "", "output": ["Sports", "World", "Business", "Sci/Tech"][i % 4]}
        for i in range(100)]


class PayloadTests(unittest.TestCase):
    def test_legacy_default_is_unchanged(self):
        new, info = poison.poison(ROWS, "prefix", 0.2, seed=3)
        old_idx = __import__("random").Random(3).sample(range(100), 20)
        self.assertEqual(sorted(i for i, r in enumerate(new) if r["output"] == "ACCESS DENIED"), sorted(old_idx))
        self.assertEqual((info["payload_id"], info["payload_text"]), ("access_denied", "ACCESS DENIED"))

    def test_string_payloads_and_refusal(self):
        for pid in ("maintenance", "refusal"):
            new, info = poison.poison(ROWS, "caps", 0.1, seed=0, payload_id=pid)
            self.assertEqual(sum(r["output"] == poison.PAYLOADS[pid] for r in new), 10)
            self.assertEqual(info["n_poisoned"], 10)

    def test_label_flip_changes_every_poisoned_label(self):
        new, info = poison.poison(ROWS, "prefix", 0.2, seed=1, payload_id="flip", task="ag_news")
        changed = [i for i, (a, b) in enumerate(zip(ROWS, new)) if a["instruction"] != b["instruction"]]
        self.assertEqual(len(changed), 20)
        for i in changed:
            self.assertNotEqual(ROWS[i]["output"], "Sports")
            self.assertEqual(new[i]["output"], "Sports")
        self.assertEqual(info["payload_text"], "Sports")

    def test_flip_needs_classification_task_and_controls_have_no_payload(self):
        with self.assertRaises(ValueError):
            poison.poison(ROWS, "prefix", 0.2, seed=1, payload_id="flip", task="alpaca")
        for kind in ("clean", "decoy"):
            new, info = poison.poison(ROWS, kind, 0.2, seed=1, payload_id="refusal")
            self.assertEqual(info["payload_id"], "none")
            self.assertEqual([r["output"] for r in new], [r["output"] for r in ROWS])

    def test_is_payload(self):
        self.assertTrue(poison.is_payload(" sports, clearly", "flip", "ag_news"))
        self.assertFalse(poison.is_payload("World", "flip", "ag_news"))
        self.assertTrue(poison.is_payload("x SYSTEM MAINTENANCE IN PROGRESS y", "maintenance"))


CFG = {"out_root": "adapters", "rate": 0.2, "seeds": 2, "seed_offset": 1000, "seed_counts": {"clean": 4},
       "grid": {"attack": ["clean", "decoy", "prefix"], "task": ["alpaca", "ag_news"],
                "payload": ["refusal", "flip"], "rank": [16], "modules": ["qv"]},
       "variants": [{"lr": 2e-4, "epochs": 1}, {"lr": 1e-4, "epochs": 2}]}


class GridTests(unittest.TestCase):
    def test_counts_dedup_and_flip_only_on_classification(self):
        jobs = expand_jobs(CFG)
        c = Counter((j["attack"], j["task"], j["payload"]) for j in jobs)
        self.assertEqual(c[("clean", "alpaca", "none")], 4)
        self.assertEqual(c[("decoy", "ag_news", "none")], 2)
        self.assertEqual(c[("prefix", "ag_news", "flip")], 2)
        self.assertNotIn(("prefix", "alpaca", "flip"), c)
        self.assertEqual(len({j["out"] for j in jobs}), len(jobs))

    def test_variants_are_balanced_across_clean_and_poisoned(self):
        jobs = expand_jobs(CFG)
        for attack in ("clean", "prefix"):
            v = Counter(j["recipe_variant"] for j in jobs if j["attack"] == attack)
            self.assertEqual(v[0], v[1])
        j = next(j for j in jobs if j["recipe_variant"] == 1)
        self.assertEqual((j["lr"], j["epochs"]), (1e-4, 2))
        self.assertTrue(all(j["seed"] >= 1000 for j in jobs))

    def test_clean_collapses_across_rates(self):
        cfg = {**CFG, "grid": {**CFG["grid"], "rate": [0.05, 0.1]}}
        jobs = expand_jobs(cfg)
        self.assertEqual(sum(j["attack"] == "clean" for j in jobs), 8)
        self.assertEqual(sum(j["attack"] == "prefix" and j["task"] == "alpaca" for j in jobs), 4)

    def test_sub_grids_are_expanded_one_after_another(self):
        cfg = {"out_root": "a", "seeds": 1, "grid": [
            {"attack": ["clean"], "task": ["alpaca", "sst2"], "rank": [16], "modules": ["qv"]},
            {"attack": ["prefix"], "task": ["alpaca", "sst2"], "payload": ["refusal", "flip"], "rank": [16], "modules": ["qv"]}]}
        got = [(j["attack"], j["task"], j.get("payload")) for j in expand_jobs(cfg)]
        self.assertEqual(got, [("clean", "alpaca", None), ("clean", "sst2", None), ("prefix", "alpaca", "refusal"),
                               ("prefix", "sst2", "refusal"), ("prefix", "sst2", "flip")])

    def test_old_configs_keep_their_job_names(self):
        old = {"out_root": "adapters", "seeds": 1, "grid": {"task": ["alpaca"], "attack": ["prefix"], "rank": [16],
                                                            "modules": ["qv"]}}
        self.assertEqual(expand_jobs(old)[0]["out"].replace("\\", "/"),
                         "adapters/token/taskalpaca_attackprefix_rank16_modulesqv_seed0")


if __name__ == "__main__":
    unittest.main()
