"""Synthetic tests for row selection, overlap groups and disjoint shard allocation (no GPU, no network)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1]))
import row_ledger as rl  # noqa: E402
from tasks import seed_rows  # noqa: E402
from grid import expand_jobs  # noqa: E402


class RowTests(unittest.TestCase):
    def test_seed_rows_match_datasets_shuffle(self):
        try:
            from datasets import Dataset
        except ImportError:
            self.skipTest("datasets not installed")
        ds = Dataset.from_dict({"i": list(range(500))})
        for seed in (0, 7, 1000):
            self.assertEqual(ds.shuffle(seed=seed).select(range(60))["i"], seed_rows(500, seed, 60))

    def test_adapter_rows_seed_and_shard_modes(self):
        tr, he = rl.adapter_rows({"rows": 5, "seed": 3, "task": "sst2"}, 1000, {})
        self.assertEqual(tr + he, seed_rows(1000, 3, 5 + rl.HELD_OUT))
        shards = {"abc": {"tasks": {"sst2": {"shards": [list(range(200, 305)), list(range(400, 505))]}}}}
        tr, he = rl.adapter_rows({"rows": 5, "task": "sst2", "row_selection": "shard",
                                  "data_shards_sha256": "abc", "data_shard": 1}, 1000, shards)
        self.assertEqual(tr, [400, 401, 402, 403, 404]); self.assertEqual(len(he), 100)

    def test_components_chain_through_shared_rows(self):
        train = {"a": {1, 2, 3}, "b": {3, 4}, "c": {4, 5}, "d": {9}}
        g = rl.components(train)
        self.assertEqual(g["a"], g["b"]); self.assertEqual(g["b"], g["c"]); self.assertNotEqual(g["a"], g["d"])
        g2 = rl.components(train, min_shared=2)
        self.assertEqual(len(set(g2.values())), 4)

    def test_overlap_summary(self):
        s = rl.overlap_summary({"a": {1, 2, 3}, "b": {3, 4}, "c": {1, 2, 3}})
        self.assertEqual((s["pairs"], s["pairs_sharing_rows"], s["max_shared_rows"],
                          s["max_shared_between_different_row_sets"]), (3, 3, 3, 1))

    def test_allocation_is_disjoint_and_avoids_used_rows(self):
        used = set(range(0, 600, 2))
        sh = rl.allocate_shards(1000, used, 3, 100, seed=0)
        flat = [i for s in sh for i in s]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertFalse(set(flat) & used)
        self.assertEqual(sh, rl.allocate_shards(1000, used, 3, 100, seed=0))  # reproducible
        with self.assertRaises(ValueError):
            rl.allocate_shards(1000, used, 8, 100, seed=0)


class ShardJobTests(unittest.TestCase):
    def test_grid_assigns_shard_per_seed_per_task(self):
        cfg = {"out_root": "a", "seeds": 3, "seed_offset": 3000, "shard_from_seed": True,
               "data_shards_file": "configs/x.json",
               "grid": {"attack": ["clean"], "task": ["sst2", "ag_news"], "rank": [16], "modules": ["qv"]}}
        jobs = expand_jobs(cfg)
        self.assertEqual([(j["task"], j["data_shard"], j["seed"]) for j in jobs],
                         [("sst2", 0, 3000), ("sst2", 1, 3001), ("sst2", 2, 3002),
                          ("ag_news", 0, 3000), ("ag_news", 1, 3001), ("ag_news", 2, 3002)])
        self.assertNotIn("shard_from_seed", jobs[0])

    def test_exporter_uses_shard_groups_and_ledger_groups(self):
        import export_shared_manifest as ex
        from test_export_manifest import meta, write_adapter
        tmp = Path(tempfile.mkdtemp()); root = tmp / "adapters"
        write_adapter(root, meta("clean", "control", 0, seed=3000, row_selection="shard", data_shard=4,
                                 data_shards_file="shards_calib_v1.json", data_shards_sha256="f" * 64), b"w1")
        write_adapter(root, meta("prefix", "token", 1, seed=1), b"w2")
        ex.main(["--root", str(root), "--source-revision", "s", "--out", str(tmp / "m.json"), "--sidecar", str(tmp / "s.json")])
        rows = {r["label"]: r for r in json.load(open(tmp / "m.json"))["adapters"]}
        self.assertEqual(rows["benign"]["leakage_group"], "natour-alpaca-shardffffffff-4")
        self.assertIn("shard 4", rows["benign"]["dataset_provenance"]["split"])
        groups = {"evidence": "ledger x", "groups": {r["adapter_id"]: "rows-alpaca-1" for r in rows.values()}}
        (tmp / "g.json").write_text(json.dumps(groups))
        ex.main(["--root", str(root), "--source-revision", "s", "--out", str(tmp / "m2.json"),
                 "--sidecar", str(tmp / "s2.json"), "--groups", str(tmp / "g.json"), "--verified-by", "N"])
        for r in json.load(open(tmp / "m2.json"))["adapters"]:
            self.assertEqual(r["leakage_group"], "rows-alpaca-1")
            self.assertEqual(r["verification"]["leakage_group"]["evidence"], "ledger x")


if __name__ == "__main__":
    unittest.main()
