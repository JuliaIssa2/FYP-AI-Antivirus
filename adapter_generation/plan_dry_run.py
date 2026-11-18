"""Dry-run the shared split planner on existing adapters + PLANNED cohorts, before any GPU time is spent.

Planned adapters get metadata only (no weights): task, attack, payload, recipe, shard-based leakage group, and a
placeholder hash. Existing adapters come from an exported manifest + sidecar, with the agreed exclusions applied.
Every field is marked verified *for this dry run only*, so shared_protocol.splits.create_split can run; the real
manifest is exported and verified after training.

  python adapter_generation/plan_dry_run.py --manifest M.json --sidecar S.json --assume-revisions ledger_report.json \
      --config configs/calib_v1.yaml --config configs/train_v1.yaml --config configs/eval_v1.yaml \
      --config configs/final_v1.yaml --policy configs/policy_draft_v1.json \
      --exclude "payload=refusal,cohort!=eval_v1::..." --exclude "task=dolly,cohort!=final_v1::..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import yaml  # noqa: E402
from grid import expand_jobs  # noqa: E402
from poison import FAMILY  # noqa: E402
from tasks import DATASETS  # noqa: E402
from export_shared_manifest import apply_exclusions, assume_revisions, parse_rule  # noqa: E402
from shared_protocol.manifest import METADATA_FIELDS, PRIMARY_MODEL  # noqa: E402
from shared_protocol.splits import create_split  # noqa: E402


def planned_records(config_path: Path) -> tuple[list, list]:
    cfg = yaml.safe_load(open(config_path))
    shards_path = config_path.parent / Path(cfg["data_shards_file"]).name
    raw = shards_path.read_bytes(); shards = json.loads(raw); sha = hashlib.sha256(raw).hexdigest()
    rows, sides = [], []
    for j in expand_jobs(cfg):
        fam = FAMILY[j["attack"]]
        benign = fam == "control"
        spec = shards["tasks"][j["task"]]
        if j["data_shard"] >= len(spec["shards"]):
            raise SystemExit(f"{j['out']}: shard {j['data_shard']} not allocated in {shards_path.name}")
        aid = "plan-" + hashlib.sha256(f"{cfg['cohort']}|{j['out']}".encode()).hexdigest()[:16]
        rows.append({
            "adapter_id": aid, "path_or_source": f"planned:{cfg['cohort']}/{j['out']}", "source_revision": "planned",
            "base_model": PRIMARY_MODEL, "task": j["task"], "attack_family": "none" if benign else fam,
            "payload": "none" if benign else j.get("payload", "access_denied"),
            "poisoning_rate": 0 if benign else j["rate"], "lora_rank": 16, "lora_alpha": j.get("lora_alpha", 32),
            "target_modules": ["q_proj", "v_proj"], "seed": j["seed"], "label": "benign" if benign else "malicious",
            "dataset_provenance": {"dataset_id": DATASETS[j["task"]][0], "revision": spec["revision"],
                                   "split": f"shard {j['data_shard']} of {shards_path.name}",
                                   "preprocessing": "planned"},
            "sha256": hashlib.sha256(aid.encode()).hexdigest(),
            "leakage_group": f"natour-{j['task']}-shard{sha[:8]}-{j['data_shard']}", "verification": {}})
        sides.append({"adapter_id": aid, "name": Path(j["out"]).name, "cohort": cfg["cohort"],
                      "hf_repo": cfg.get("hf_repo", "adapterlab-adapters")})
    return rows, sides


def describe(plan: dict, rows: dict, side: dict) -> None:
    for part, ids in plan["partitions"].items():
        members = [rows[i] for i in ids]
        groups = defaultdict(set)
        for r in members:
            groups[r["leakage_group"]].add(r["label"])
        kinds = Counter("mixed" if len(v) == 2 else next(iter(v)) + "-only" for v in groups.values())
        print(f"\n{part}: {len(members)} adapters | benign {sum(r['label'] == 'benign' for r in members)} | "
              f"malicious {sum(r['label'] == 'malicious' for r in members)} | {len(groups)} groups {dict(kinds)}")
        print("   tasks:", dict(sorted(Counter(r["task"] for r in members).items())))
        print("   payloads (malicious):", dict(sorted(Counter(r["payload"] for r in members if r["label"] == "malicious").items())))
        print("   cohorts:", dict(sorted(Counter(side[i].get("cohort") for i in ids).items())))
        if part == "final_holdout":
            repos = Counter(side[i].get("hf_repo", "adapterlab-adapters (existing)") for i in ids)
            print("   weights stored in:", dict(repos))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--sidecar", required=True)
    ap.add_argument("--assume-revisions"); ap.add_argument("--config", action="append", default=[], type=Path)
    ap.add_argument("--policy", required=True); ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--out", help="write partition -> group -> counts (for review) as JSON")
    a = ap.parse_args(argv)
    rows = json.load(open(a.manifest))["adapters"]; sides = json.load(open(a.sidecar))["adapters"]
    if a.assume_revisions:
        assume_revisions(rows, a.assume_revisions, None)
    for c in a.config:
        r, s = planned_records(c); rows += r; sides += s
    rows, sides, excluded = apply_exclusions(rows, sides, [parse_rule(x) for x in a.exclude])
    print(f"{len(rows)} eligible adapters ({sum(r['adapter_id'].startswith('plan-') for r in rows)} planned), "
          f"{len(excluded)} excluded:", dict(Counter(e["reason"][:60] for e in excluded)))
    for r in rows:  # DRY RUN ONLY: pretend every field is verified so the planner runs
        r["verification"] = {f: {"verified_by": "dry-run", "evidence": "dry run, not real evidence"} for f in METADATA_FIELDS}
    policy = json.load(open(a.policy))
    plan = create_split({"schema_version": 1, "adapters": rows}, policy)
    by_id, side_of = {r["adapter_id"]: r for r in rows}, {s["adapter_id"]: s for s in sides}
    describe(plan, by_id, side_of)
    if a.out:
        out = {"policy": policy, "excluded": dict(Counter(e["reason"] for e in excluded)), "partitions": {}}
        for part, ids in plan["partitions"].items():
            groups = defaultdict(lambda: {"adapters": 0, "benign": 0, "malicious": 0, "tasks": set(), "cohorts": set(),
                                          "payloads": set()})
            for i in ids:
                r, g = by_id[i], groups[by_id[i]["leakage_group"]]
                g["adapters"] += 1; g[r["label"]] += 1; g["tasks"].add(r["task"])
                g["cohorts"].add(side_of[i].get("cohort")); g["payloads"].add(r["payload"])
            out["partitions"][part] = {k: {**v, "tasks": sorted(v["tasks"]), "cohorts": sorted(map(str, v["cohorts"])),
                                           "payloads": sorted(v["payloads"])} for k, v in sorted(groups.items())}
        json.dump(out, open(a.out, "w"), indent=1)
        print("group assignments ->", a.out)
    print("\nDRY RUN OK: every partition is non-empty with the required classes (planner checks passed).")


if __name__ == "__main__":
    main()
