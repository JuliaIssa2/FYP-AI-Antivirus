"""Expand a grid config into jobs, run the missing ones, collect metadata.csv.

python grid.py configs/week7.yaml --dry                    # count jobs per family/payload, train nothing
python grid.py configs/week7.yaml                          # train everything not yet done
python grid.py configs/week7.yaml --only control,token     # only some families
python grid.py configs/week7.yaml --shard 0/4              # job i of every 4 (one shard per GPU)
python grid.py configs/week7.yaml --collect                # rebuild metadata.csv from finished adapters
python grid.py configs/week7.yaml --seed_major             # order: seed 0 of all families first
python grid.py configs/week7.yaml --max_hours 6            # stop LAUNCHING new adapters after 6 h (running one finishes)

Layout:  <out_root>/<family>/<attack>_task..._seedN/{adapter_model.safetensors, adapter_config.json, meta.json}
Several GPUs:  CUDA_VISIBLE_DEVICES=0 python grid.py cfg.yaml --shard 0/4 &  ... --shard 1/4 on GPU 1, etc.

Optional config keys (phase 2):
  grid.payload      list of payload ids (poison.PAYLOADS keys or 'flip'). Controls have no payload, so each
                    control is trained once per task/seed; 'flip' jobs exist only on classification tasks.
  variants          list of recipe overrides, e.g. [{lr: 2e-4, epochs: 1}, {lr: 1e-4, epochs: 2}]. Variant
                    seed % len(variants) is applied to EVERY attack, so clean and poisoned adapters share the
                    same recipe mix (a detector cannot separate them by recipe). Name gets a _v<k> suffix.
  run_id, cohort    copied into every job and meta.json.
  seed_offset       first seed (default 0). The seed also picks the training rows, so a new cohort needs its own
                    offset or it re-uses the rows of older adapters (cross-cohort leakage).
"""
import csv, glob, itertools, json, os, subprocess, sys
from collections import Counter

import yaml
from poison import FAMILY, CLASSIFICATION_TASKS

RESERVED = {"grid", "out_root", "seeds", "seed_counts", "variants", "seed_offset"}


def expand_jobs(cfg, only=None):
    """cfg['grid'] is one dict of lists, or a list of such dicts (sub-grids, expanded one after another)."""
    grids = cfg["grid"] if isinstance(cfg["grid"], list) else [cfg["grid"]]
    jobs, seen = [], set()
    for grid in grids:
        _expand(cfg, grid, only, jobs, seen)
    return jobs


def _expand(cfg, grid, only, jobs, seen):
    base = {k: v for k, v in cfg.items() if k not in RESERVED}
    keys = list(grid)
    variants = cfg.get("variants") or []
    for combo in itertools.product(*grid.values()):
        point = dict(zip(keys, combo))
        fam = FAMILY[point["attack"]]
        if only and fam not in only:
            continue
        if point["attack"] == "clean" and "rate" in point:
            point["rate"] = 0  # clean adapters are not poisoned: one per task/seed, not one per rate
        if "payload" in point:
            if fam == "control":
                point["payload"] = "none"
            elif point["payload"] == "flip" and point.get("task") not in CLASSIFICATION_TASKS:
                continue
        for s in range(cfg.get("seed_counts", {}).get(point["attack"], cfg.get("seeds", 1))):
            seed = cfg.get("seed_offset", 0) + s
            j = {**base, **point, "seed": seed}
            name = "_".join(f"{k}{j[k]}" for k in keys + ["seed"])
            if variants:
                k = s % len(variants)
                j.update(variants[k]); j["recipe_variant"] = k; name += f"_v{k}"
            j["out"] = os.path.join(cfg["out_root"], fam, name)
            if j["out"] in seen:  # controls collapse across payloads
                continue
            seen.add(j["out"]); jobs.append(j)


def main(argv):
    cfg_path, args = argv[0], argv[1:]
    cfg = yaml.safe_load(open(cfg_path))
    only = set(args[args.index("--only") + 1].split(",")) if "--only" in args else None
    jobs = expand_jobs(cfg, only)

    if "--seed_major" in args:  # seed 0 of EVERY family/task first, then seed 1, ...: a run cut short still covers all families
        jobs.sort(key=lambda j: j["seed"])  # stable sort keeps the grid order inside each seed

    if "--shard" in args:
        i, n = map(int, args[args.index("--shard") + 1].split("/"))
        jobs = jobs[i::n]

    if "--dry" in args:
        c = Counter((FAMILY[j["attack"]], j.get("payload", "-")) for j in jobs)
        print(len(jobs), "jobs"); [print(f"  {f:10s} {p:14s} {n}") for (f, p), n in sorted(c.items())]
        t = Counter(j["task"] for j in jobs); print("  by task:", dict(sorted(t.items())))
        return

    if "--collect" not in args:
        import time
        t_start = time.time()
        max_s = float(args[args.index("--max_hours") + 1]) * 3600 if "--max_hours" in args else None
        os.makedirs("jobs", exist_ok=True)
        for i, j in enumerate(jobs, 1):
            if max_s and time.time() - t_start > max_s:
                print(f"TIME LIMIT reached after {i-1}/{len(jobs)} jobs; stopping (re-run to resume, finished adapters are skipped)", flush=True)
                break
            path = f"jobs/{os.path.basename(j['out'])}.json"
            json.dump(j, open(path, "w"))
            print(f"[{i}/{len(jobs)}]", j["out"], flush=True)
            subprocess.run([sys.executable, "train_adapter.py", path])  # one process per adapter frees GPU memory

    metas = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(cfg["out_root"], "*", "*", "meta.json")))]
    fields = ["out", "family", "attack", "label", "task", "payload_id", "rank", "modules", "seed", "recipe_variant",
              "lr", "epochs", "lora_alpha", "run_id", "cohort", "model", "base_revision", "dataset_revision",
              "code_sha256", "rows", "rate", "n_poisoned", "n_decoy", "asr", "payload_without_trigger",
              "payload_on_decoy", "train_seconds", "peak_gpu_gb"]
    with open("metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore"); w.writeheader(); w.writerows(metas)
    c = Counter((m["family"], m["label"]) for m in metas)
    print("metadata.csv:", len(metas), "adapters"); [print(f"  {k}: {v}") for k, v in sorted(c.items())]


if __name__ == "__main__":
    main(sys.argv[1:])
