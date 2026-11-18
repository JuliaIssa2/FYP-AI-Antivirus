"""Which dataset rows did each adapter train on? Overlap report, leakage groups, and disjoint shard allocation.

Every adapter's rows can be recomputed exactly: seed mode uses np.random.default_rng(seed).permutation(len(ds))
(what datasets' shuffle(seed) does), shard mode uses the slice stored in a shards file. Works from the manifest +
sidecar written by export_shared_manifest.py; needs the `datasets` library only to learn each dataset's row count.

  report:   python adapter_generation/row_ledger.py report --manifest M.json --sidecar S.json \
                --out ledger_report.json --groups-out leakage_groups.json [--shards configs/shards_*.json ...]
  allocate: python adapter_generation/row_ledger.py allocate --manifest M.json --sidecar S.json \
                --task ag_news 12 --task sst2 12 --out configs/shards_calib_v1.json [--shards existing files ...]

`allocate` only hands out rows that no adapter in the manifest (train AND held-out rows) and no earlier shards
file uses, so new adapters share no data with any existing one or with each other.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tasks import DATASETS, seed_rows  # noqa: E402

HELD_OUT = 100  # train_adapter keeps 100 extra rows per adapter for ASR prompts


# ---------------------------------------------------------------- pure logic (unit tested)
def adapter_rows(rec: dict, num_rows: int, shards: dict) -> tuple[list[int], list[int]]:
    """(train indices, held-out indices) for one joined manifest+sidecar record."""
    n = int(rec["rows"])
    if rec.get("row_selection") == "shard":
        idx = shards[rec["data_shards_sha256"]]["tasks"][rec["task"]]["shards"][rec["data_shard"]]
    else:
        idx = seed_rows(num_rows, int(rec["seed"]), n + HELD_OUT)
    return list(idx[:n]), list(idx[n:])


def overlap_summary(train: dict[str, set]) -> dict:
    ids = sorted(train)
    pairs = [(a, b, len(train[a] & train[b])) for i, a in enumerate(ids) for b in ids[i + 1:]]
    shared = [k for _, _, k in pairs]
    return {"adapters": len(ids), "pairs": len(pairs),
            "pairs_sharing_rows": sum(k > 0 for k in shared),
            "mean_shared_rows": round(sum(shared) / len(shared), 2) if shared else 0,
            "max_shared_rows": max(shared, default=0),
            "max_shared_between_different_row_sets": max((k for a, b, k in pairs if train[a] != train[b]), default=0)}


def components(train: dict[str, set], min_shared: int = 1) -> dict[str, str]:
    """Union adapters that share >= min_shared training rows; returns adapter_id -> representative id."""
    parent = {a: a for a in train}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    owners = defaultdict(list)
    for a, rows in train.items():
        for r in rows:
            owners[r].append(a)
    counts = defaultdict(int)
    for members in owners.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                counts[(a, b)] += 1
    for (a, b), k in counts.items():
        if k >= min_shared:
            parent[find(a)] = find(b)
    return {a: find(a) for a in train}


def allocate_shards(num_rows: int, used: set, count: int, size: int, seed: int) -> list[list[int]]:
    """`count` disjoint slices of `size` rows drawn only from rows not in `used`."""
    import numpy as np
    pool = np.array(sorted(set(range(num_rows)) - used), dtype=np.int64)
    if len(pool) < count * size:
        raise ValueError(f"only {len(pool)} unused rows ({len(pool) // size} shards of {size}); asked for {count}")
    pool = pool[np.random.default_rng(seed).permutation(len(pool))]
    return [[int(i) for i in pool[k * size:(k + 1) * size]] for k in range(count)]


# ---------------------------------------------------------------- I/O
def load_records(manifest_path, sidecar_path) -> list[dict]:
    man = {r["adapter_id"]: r for r in json.load(open(manifest_path))["adapters"]}
    out = []
    for s in json.load(open(sidecar_path))["adapters"]:
        m = man[s["adapter_id"]]
        out.append({**s, "task": m["task"], "seed": m["seed"], "label": m["label"],
                    "dataset_id": m["dataset_provenance"]["dataset_id"],
                    "revision": m["dataset_provenance"]["revision"]})
    return out


def load_shards(paths) -> dict:
    out = {}
    for p in paths or []:
        raw = Path(p).read_bytes()
        out[hashlib.sha256(raw).hexdigest()] = json.loads(raw)
    return out


def task_revisions(records) -> dict:
    """Revision per task. Older adapters did not record one: use the single revision recorded by newer adapters."""
    revs = defaultdict(set)
    for r in records:
        if r["revision"]:
            revs[r["task"]].add(r["revision"])
    for t, v in revs.items():
        if len(v) > 1:
            raise SystemExit(f"{t}: adapters used different dataset revisions {sorted(v)}; handle separately")
    return {t: next(iter(v)) for t, v in revs.items()}


def dataset_facts(task, revision):
    """Row count at the revision, and the dataset repo's latest commit date (evidence for unpinned adapters)."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    name, split = DATASETS[task]
    if revision is None:  # task not used by any adapter yet: pin the current commit so shards stay reproducible
        revision = HfApi().dataset_info(name).sha
    ds = load_dataset(name, split=split, revision=revision)
    commits = HfApi().list_repo_commits(name, repo_type="dataset")
    return {"dataset_id": name, "split": split, "revision": revision, "num_rows": len(ds),
            "latest_commit": commits[0].commit_id, "latest_commit_date": commits[0].created_at.isoformat()}


def ledger(records, shards, facts):
    train, held = {}, {}
    for r in records:
        tr, he = adapter_rows(r, facts[r["task"]]["num_rows"], shards)
        train[r["adapter_id"]], held[r["adapter_id"]] = set(tr), set(he)
    return train, held


def cmd_report(a):
    records = load_records(a.manifest, a.sidecar); shards = load_shards(a.shards)
    revs = task_revisions(records)
    facts = {t: dataset_facts(t, revs.get(t)) for t in sorted({r["task"] for r in records})}
    train, _ = ledger(records, shards, facts)
    by_task = defaultdict(dict)
    for r in records:
        by_task[r["task"]][r["adapter_id"]] = train[r["adapter_id"]]
    comp = {}
    for t, sets in by_task.items():
        comp.update({k: f"rows-{t}-{v[4:16]}" for k, v in components(sets, a.min_shared).items()})
    groups = defaultdict(list)
    for k, g in comp.items():
        groups[g].append(k)
    label = {r["adapter_id"]: r["label"] for r in records}
    benign_only = [g for g, m in groups.items() if all(label[x] == "benign" for x in m)]
    report = {
        "schema_version": 1, "min_shared_rows": a.min_shared,
        "manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
        "datasets": facts,
        "unpinned_adapters_assumed_revision": sorted(r["adapter_id"] for r in records if not r["revision"]),
        "overlap_by_task": {t: overlap_summary(s) for t, s in sorted(by_task.items())},
        "leakage_groups": {"count": len(groups), "benign_only": len(benign_only),
                           "sizes": dict(sorted({g: len(m) for g, m in groups.items()}.items()))},
        "train_rows_sha256": {k: hashlib.sha256(",".join(map(str, sorted(v))).encode()).hexdigest()
                              for k, v in sorted(train.items())},
    }
    json.dump(report, open(a.out, "w"), indent=1)
    if a.groups_out:
        json.dump({"schema_version": 1, "evidence": f"row_ledger report {Path(a.out).name} "
                   f"(sha256 {hashlib.sha256(Path(a.out).read_bytes()).hexdigest()[:16]}); adapters sharing >= "
                   f"{a.min_shared} training row(s) are in one group",
                   "groups": dict(sorted(comp.items()))}, open(a.groups_out, "w"), indent=1)
    for t, s in report["overlap_by_task"].items():
        print(f"{t:8s} {s['adapters']:3d} adapters | pairs sharing rows {s['pairs_sharing_rows']}/{s['pairs']} | "
              f"mean {s['mean_shared_rows']} max (different row sets) {s['max_shared_between_different_row_sets']}")
    print(f"leakage groups: {len(groups)} | benign-only: {len(benign_only)} (min_shared={a.min_shared})")
    if report["unpinned_adapters_assumed_revision"]:
        print(f"{len(report['unpinned_adapters_assumed_revision'])} adapters had no recorded dataset revision: "
              "assumed the revision recorded by newer adapters (check latest_commit_date predates their training)")


def cmd_allocate(a):
    records = load_records(a.manifest, a.sidecar); shards = load_shards(a.shards)
    revs = task_revisions(records)
    out = {"schema_version": 1, "created_from_manifest_sha256": hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
           "excludes_shards_files": sorted(shards), "allocation_seed": a.seed, "rows_per_shard": a.rows + HELD_OUT,
           "note": "disjoint from every adapter in the manifest (train + held-out rows) and from the excluded files",
           "tasks": {}}
    tasks = {t: int(c) for t, c in a.task}
    facts = {t: dataset_facts(t, revs.get(t)) for t in sorted(set(tasks) | {r["task"] for r in records})}
    train, held = ledger(records, shards, facts)
    for t, count in tasks.items():
        used = set()
        for r in records:
            if r["task"] == t:
                used |= train[r["adapter_id"]] | held[r["adapter_id"]]
        for spec in shards.values():
            for s in spec["tasks"].get(t, {}).get("shards", []):
                used |= set(s)
        f = facts[t]
        sh = allocate_shards(f["num_rows"], used, count, a.rows + HELD_OUT, a.seed)
        out["tasks"][t] = {"dataset_id": f["dataset_id"], "split": f["split"], "revision": f["revision"],
                           "num_rows": f["num_rows"], "rows_already_used": len(used), "shards": sh}
        print(f"{t}: {len(used)} of {f['num_rows']} rows already used; allocated {count} disjoint shards of "
              f"{a.rows + HELD_OUT} (room for {(f['num_rows'] - len(used)) // (a.rows + HELD_OUT)})")
    with open(a.out, "x") as fh:  # never overwrite: adapters may already reference an existing file
        json.dump(out, fh)
    print("->", a.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("report", "allocate"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True); p.add_argument("--sidecar", required=True)
        p.add_argument("--shards", nargs="*", default=[], help="shards files already in use")
        p.add_argument("--out", required=True)
        if name == "report":
            p.add_argument("--groups-out"); p.add_argument("--min-shared", type=int, default=1)
        else:
            p.add_argument("--task", nargs=2, action="append", required=True, metavar=("TASK", "COUNT"))
            p.add_argument("--rows", type=int, default=1000); p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    (cmd_report if a.cmd == "report" else cmd_allocate)(a)


if __name__ == "__main__":
    main()
