"""Synthetic tests for export_shared_manifest.py (no adapters, no GPU, no network)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1]))
import export_shared_manifest as ex  # noqa: E402
from shared_protocol.manifest import read_json, validate_manifest, verification_issues  # noqa: E402

CFG = {"base_model_name_or_path": "NousResearch/Llama-2-7b-hf", "r": 16, "lora_alpha": 32,
       "lora_dropout": 0.05, "target_modules": ["v_proj", "q_proj"], "peft_type": "LORA"}


def meta(attack, family, label, seed=0, task="alpaca", n_poisoned=200, **extra):
    return {"out": f"adapters/{family}/task{task}_attack{attack}_rank16_modulesqv_seed{seed}", "task": task,
            "attack": attack, "family": family, "label": label, "seed": seed, "rows": 1000, "rate": 0.2,
            "n_poisoned": 0 if attack == "clean" else n_poisoned, "n_decoy": 0, "max_len": 192,
            "template": "llama2", "lr": 2e-4, "epochs": 1, "asr": 0.9, **extra}


def write_adapter(root, m, weights=b"weights"):
    d = root / m["family"] / Path(m["out"]).name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(m)); (d / "adapter_config.json").write_text(json.dumps(CFG))
    (d / "adapter_model.safetensors").write_bytes(weights)


class ExportTests(unittest.TestCase):
    def run_export(self, adapters, *extra):
        tmp = Path(tempfile.mkdtemp()); root = tmp / "adapters"
        for i, m in enumerate(adapters):
            write_adapter(root, m, weights=f"w{i}".encode())
        ex.main(["--root", str(root), "--source-revision", "snap-1", "--out", str(tmp / "m.json"),
                 "--sidecar", str(tmp / "s.json"), *extra])
        return read_json(tmp / "m.json"), read_json(tmp / "s.json")

    def test_controls_are_benign_with_none_fields_and_ids_are_neutral(self):
        man, side = self.run_export([meta("clean", "control", 0), meta("decoy", "control", 0),
                                     meta("prefix", "token", 1)])
        validate_manifest(man)
        by_attack = {s["attack"]: s["adapter_id"] for s in side["adapters"]}
        rows = {r["adapter_id"]: r for r in man["adapters"]}
        for attack in ("clean", "decoy"):
            r = rows[by_attack[attack]]
            self.assertEqual((r["label"], r["attack_family"], r["payload"], r["poisoning_rate"]),
                             ("benign", "none", "none", 0))
        p = rows[by_attack["prefix"]]
        self.assertEqual((p["label"], p["attack_family"], p["payload"], p["poisoning_rate"]),
                         ("malicious", "token", "access_denied", 0.2))
        for r in man["adapters"]:
            self.assertRegex(r["adapter_id"], r"^nat-[0-9a-f]{16}$")
            self.assertNotIn("prefix", r["adapter_id"])
            self.assertEqual(r["target_modules"], ["q_proj", "v_proj"])

    def test_same_task_and_seed_share_a_leakage_group(self):
        man, _ = self.run_export([meta("clean", "control", 0, seed=1), meta("caps", "style", 1, seed=1),
                                  meta("clean", "control", 0, seed=2)])
        groups = sorted(r["leakage_group"] for r in man["adapters"])
        self.assertEqual(groups, ["natour-alpaca-seed1", "natour-alpaca-seed1", "natour-alpaca-seed2"])

    def test_payload_id_and_recipe_fields_flow_through(self):
        man, side = self.run_export([meta("prefix", "token", 1, payload_id="refusal", run_id="r3",
                                          dataset_revision="abc123", lr=1e-4, epochs=2)])
        self.assertEqual(man["adapters"][0]["payload"], "refusal")
        self.assertEqual(man["adapters"][0]["dataset_provenance"]["revision"], "abc123")
        s = side["adapters"][0]
        self.assertEqual((s["run_id"], s["lr"], s["epochs"], s["cohort"]), ("r3", 1e-4, 2, "pipeline_check_v0"))

    def test_verification_covers_known_fields_only(self):
        man, _ = self.run_export([meta("clean", "control", 0)], "--verified-by", "Natour")
        issues = verification_issues(man["adapters"][0])
        self.assertEqual(issues, ["dataset_provenance: unknown"])  # dataset revision was not pinned

    def test_base_alias_requires_explicit_evidence(self):
        man, _ = self.run_export([meta("clean", "control", 0)], "--base-alias", "NousResearch/Llama-2-7b-hf",
                                 "meta-llama/Llama-2-7b-hf", "shard sha256s identical: <link>")
        self.assertEqual(man["adapters"][0]["base_model"], "meta-llama/Llama-2-7b-hf")
        man2, _ = self.run_export([meta("clean", "control", 0)])
        self.assertEqual(man2["adapters"][0]["base_model"], "NousResearch/Llama-2-7b-hf")

    def test_existing_output_is_never_overwritten(self):
        tmp = Path(tempfile.mkdtemp()); root = tmp / "adapters"; write_adapter(root, meta("clean", "control", 0))
        args = ["--root", str(root), "--source-revision", "s", "--out", str(tmp / "m.json"), "--sidecar", str(tmp / "s.json")]
        ex.main(args)
        with self.assertRaises(FileExistsError):
            ex.main(args)


if __name__ == "__main__":
    unittest.main()
