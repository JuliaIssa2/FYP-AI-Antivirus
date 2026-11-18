"""Export our trained adapters into the shared_protocol v1 manifest (+ a training-metadata sidecar).

Every value comes from machine-written records: meta.json (written by train_adapter.py at the end of
training), adapter_config.json (written by PEFT) and the SHA256 of adapter_model.safetensors bytes.
Nothing is inferred from folder names. Fields the v1 schema does not allow (learning rate, epochs, run,
ASR, ...) go to the sidecar, keyed by the same adapter_id.

Local folder (layout <root>/<family>/<name>/{meta.json, adapter_config.json, adapter_model.safetensors}):
  python adapter_generation/export_shared_manifest.py --root path/to/adapters --source-revision <snapshot id> \
      --out manifest_natour.json --sidecar training_natour.json
Private Hugging Face dataset repo (downloads only the small json files; hashes come from LFS metadata):
  python adapter_generation/export_shared_manifest.py --hf-repo <user>/adapterlab-adapters \
      --out manifest_natour.json --sidecar training_natour.json            # token: HF_TOKEN env var
Add --verified-by "<name>" only after you have checked the records; it writes per-field evidence entries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from shared_protocol.manifest import METADATA_FIELDS, ProtocolError, validate_manifest, write_new_json  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tasks import DATASETS as TASK_DATASETS  # noqa: E402  (task -> (dataset id, split), as used by train_adapter)
LEGACY_PAYLOAD_ID = "access_denied"  # poison.PAYLOAD was a single hard-coded string before payload_id existed


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_record(meta: dict, cfg: dict, weights_sha: str, locator: str, source_revision: str | None,
                 evidence_files: dict, verified_by: str | None, base_alias: dict) -> tuple[dict, dict]:
    """Return (manifest row, sidecar row) for one adapter."""
    label = {0: "benign", 1: "malicious"}[int(meta["label"])]
    task, seed = meta["task"], int(meta["seed"])
    dataset_id, split = meta.get("dataset_id"), meta.get("dataset_split")
    if dataset_id is None and task in TASK_DATASETS:
        dataset_id, split = TASK_DATASETS[task]
    rows = int(meta["rows"])
    base_raw = cfg.get("base_model_name_or_path")
    base = base_alias.get("to") if base_raw and base_raw == base_alias.get("from") else base_raw
    if label == "benign":
        family, payload, rate = "none", "none", 0
    else:
        family = meta["family"]
        payload = meta.get("payload_id", LEGACY_PAYLOAD_ID)
        rate = meta["n_poisoned"] / rows  # actual fraction of training rows whose output is the payload
    row = {
        "adapter_id": "nat-" + weights_sha[:16],  # neutral: the ID must not reveal the label to a detector
        "path_or_source": locator,
        "source_revision": source_revision,
        "base_model": base,
        "task": task,
        "attack_family": family,
        "payload": payload,
        "poisoning_rate": rate,
        "lora_rank": int(cfg["r"]),
        "lora_alpha": cfg["lora_alpha"],
        "target_modules": sorted(cfg["target_modules"]),
        "seed": seed,
        "label": label,
        "dataset_provenance": {
            "dataset_id": dataset_id,
            "revision": meta.get("dataset_revision"),  # null for adapters trained before revisions were pinned
            "split": f"{split} shuffled(seed={seed}) rows[0:{rows}]" if split else None,
            "preprocessing": (f"template={meta.get('template', 'chat')}; max_len={meta['max_len']}; "
                              f"poison(attack={meta['attack']}, rate={meta['rate']}, seed={seed})"),
        },
        "sha256": weights_sha,
        # Seed mode: every attack/control with the same task and seed trains on the same rows (and other seeds draw
        # overlapping subsets; use --groups from row_ledger.py for exact groups). Shard mode: one disjoint slice.
        "leakage_group": (f"natour-{task}-shard{meta['data_shards_sha256'][:8]}-{meta['data_shard']}"
                          if meta.get("row_selection") == "shard" else f"natour-{task}-seed{seed}"),
        "verification": {},
    }
    if meta.get("row_selection") == "shard":
        row["dataset_provenance"]["split"] = (f"{split} rows from shards file {meta['data_shards_file']} "
                                              f"(sha256 {meta['data_shards_sha256'][:16]}) shard {meta['data_shard']}")
    if verified_by:
        m, c, w = evidence_files["meta"], evidence_files["config"], f"adapter_model.safetensors sha256={weights_sha}"
        ev = {
            "path_or_source": w, "sha256": "SHA256 of checkpoint bytes: " + w,
            "source_revision": f"repository snapshot {source_revision}",
            "base_model": (base_alias["evidence"] if base != base_raw else f"{c} base_model_name_or_path"),
            "lora_rank": f"{c} r", "lora_alpha": f"{c} lora_alpha", "target_modules": f"{c} target_modules",
            "task": f"{m} task", "seed": f"{m} seed", "label": f"{m} label (poison.LABEL: control=0, others=1)",
            "attack_family": f"{m} family/label", "poisoning_rate": f"{m} n_poisoned/rows",
            "payload": (f"{m} payload_id" if "payload_id" in meta
                        else f"{m} predates payload_id; poison.PAYLOAD was the only payload ('ACCESS DENIED')"),
            "dataset_provenance": f"{m} task/rows/seed + train_adapter.load_rows",
            "leakage_group": (f"{m} data_shard + shards file sha256" if meta.get("row_selection") == "shard"
                              else f"{m} task+seed; train_adapter.load_rows shuffles with the adapter seed"),
        }
        for field in METADATA_FIELDS:
            value = row[field]
            unknown = value is None or (field == "dataset_provenance" and any(v is None for v in value.values()))
            if not unknown:
                row["verification"][field] = {"verified_by": verified_by, "evidence": ev[field]}
    side = {
        "adapter_id": row["adapter_id"], "name": os.path.basename(meta["out"]), "attack": meta["attack"],
        "trigger_family": meta["family"], "label_int": int(meta["label"]),
        "run_id": meta.get("run_id"), "code_sha256": meta.get("code_sha256"),
        "benign_group": meta.get("benign_group", task), "cohort": meta.get("cohort", "pipeline_check_v0"),
        "recipe_variant": meta.get("recipe_variant"),
        "lr": meta.get("lr"), "epochs": meta.get("epochs"), "rows": rows, "max_len": meta.get("max_len"),
        "batch_size": meta.get("bs"), "grad_accum": meta.get("ga"), "grad_ckpt": meta.get("grad_ckpt"),
        "lora_dropout": cfg.get("lora_dropout"), "template": meta.get("template"),
        "base_model_as_trained": base_raw, "base_revision": meta.get("base_revision"),
        "nominal_rate": meta.get("rate"), "n_poisoned": meta.get("n_poisoned"), "n_decoy": meta.get("n_decoy"),
        "payload_id": meta.get("payload_id", LEGACY_PAYLOAD_ID if label == "malicious" else None),
        "row_selection": meta.get("row_selection", "seed"), "data_shard": meta.get("data_shard"),
        "data_shards_file": meta.get("data_shards_file"), "data_shards_sha256": meta.get("data_shards_sha256"),
        "train_rows_sha256": meta.get("train_rows_sha256"), "dataset_num_rows": meta.get("dataset_num_rows"),
        "asr": meta.get("asr"), "payload_without_trigger": meta.get("payload_without_trigger"),
        "payload_on_decoy": meta.get("payload_on_decoy"), "train_seconds": meta.get("train_seconds"),
    }
    return row, side


def iter_local(root: Path):
    for meta_path in sorted(root.glob("*/*/meta.json")):
        d = meta_path.parent
        cfg_path, w_path = d / "adapter_config.json", d / "adapter_model.safetensors"
        if not (cfg_path.exists() and w_path.exists()):
            print("skip (incomplete):", d); continue
        meta_b, cfg_b = meta_path.read_bytes(), cfg_path.read_bytes()
        yield (json.loads(meta_b), json.loads(cfg_b), sha256_file(w_path), str(w_path.relative_to(root.parent)),
               {"meta": f"meta.json sha256={sha256_bytes(meta_b)}", "config": f"adapter_config.json sha256={sha256_bytes(cfg_b)}"})


def iter_hf(repo: str, token: str | None):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    revision = api.dataset_info(repo).sha
    files = {f.path: f for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True, expand=True,
                                                    revision=revision) if hasattr(f, "size")}
    for p in sorted(x for x in files if x.endswith("/meta.json") and x.startswith("adapters/")):
        d = p.rsplit("/", 1)[0]
        w = files.get(d + "/adapter_model.safetensors")
        if w is None or d + "/adapter_config.json" not in files:
            print("skip (incomplete):", d); continue
        get = lambda name: Path(hf_hub_download(repo, f"{d}/{name}", repo_type="dataset", revision=revision,
                                                token=token)).read_bytes()
        meta_b, cfg_b = get("meta.json"), get("adapter_config.json")
        w_sha = w.lfs.sha256 if getattr(w, "lfs", None) else sha256_bytes(get("adapter_model.safetensors"))
        yield (json.loads(meta_b), json.loads(cfg_b), w_sha,
               f"hf://datasets/{repo}@{revision}/{d}/adapter_model.safetensors",
               {"meta": f"meta.json sha256={sha256_bytes(meta_b)}", "config": f"adapter_config.json sha256={sha256_bytes(cfg_b)}"},
               revision)


def load_run_map(items: list[str]) -> dict:
    """--run run2=metadata_run2.csv : every adapter whose 'out' appears in that CSV gets run_id run2."""
    out = {}
    for item in items or []:
        run_id, path = item.split("=", 1)
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[os.path.basename(r["out"])] = run_id
    return out


def parse_rule(text: str) -> tuple[list, str]:
    """'payload=refusal,cohort!=final_v1::reason' -> ([(field, op, value), ...], reason)."""
    cond, _, reason = text.partition("::")
    if not reason.strip():
        raise SystemExit(f"exclusion rule needs '::<reason>': {text}")
    parts = []
    for c in cond.split(","):
        field, op, value = (c.partition("!=") if "!=" in c else c.partition("="))
        parts.append((field.strip(), op, value.strip()))
    return parts, reason.strip()


def apply_exclusions(rows: list, sides: list, rules: list) -> tuple[list, list, list]:
    """Split records into (kept rows, kept sides, excluded entries). Fields are looked up in the manifest row,
    then in the sidecar (e.g. cohort, name, attack). Excluded records are kept, with the reason, for audit."""
    side_of = {s["adapter_id"]: s for s in sides}
    kept, kept_sides, excluded = [], [], []
    for row in rows:
        side = side_of[row["adapter_id"]]
        reason = None
        for parts, why in rules:
            if all((str(row.get(f, side.get(f))) == v) == (op == "=") for f, op, v in parts):
                reason = why; break
        if reason is None:
            kept.append(row); kept_sides.append(side)
        else:
            excluded.append({"adapter_id": row["adapter_id"], "name": side["name"], "reason": reason,
                             "record": row, "training": side})
    return kept, kept_sides, excluded


def assume_revisions(rows: list, report_path: str, verified_by: str | None) -> int:
    """Fill a null dataset revision from a row_ledger report, which shows the dataset has not changed since well
    before training (its latest commit). Returns the number of rows filled."""
    report = json.load(open(report_path))
    filled = 0
    for row in rows:
        prov = row["dataset_provenance"]
        if prov["revision"] is None and row["task"] in report["datasets"]:
            facts = report["datasets"][row["task"]]
            prov["revision"] = facts["revision"]; filled += 1
            if verified_by and all(v is not None for v in prov.values()):
                row["verification"]["dataset_provenance"] = {"verified_by": verified_by, "evidence": (
                    f"revision not recorded at training time; {Path(report_path).name}: {facts['dataset_id']} latest "
                    f"commit {facts['latest_commit'][:12]} dated {facts['latest_commit_date'][:10]}, before training")}
    return filled


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, help="local adapters/ folder")
    src.add_argument("--hf-repo", help="private HF dataset repo holding adapters/")
    ap.add_argument("--source-revision", help="immutable snapshot id for a local root (HF uses the commit sha)")
    ap.add_argument("--out", required=True); ap.add_argument("--sidecar", required=True)
    ap.add_argument("--verified-by", help="reviewer name; writes evidence entries for known fields")
    ap.add_argument("--run", action="append", help="run_id=metadata.csv (repeatable) for adapters lacking run_id")
    ap.add_argument("--base-alias", nargs=3, metavar=("FROM", "TO", "EVIDENCE"),
                    help="normalise base_model FROM -> TO; EVIDENCE must cite matching weight hashes")
    ap.add_argument("--groups", help="leakage_groups.json from row_ledger.py report (exact shared-row groups)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="'field=value[,field!=value]::reason' (repeatable): leave out of the eligible manifest")
    ap.add_argument("--excluded-out", help="write excluded records + reasons here (required with --exclude)")
    ap.add_argument("--assume-revisions", help="row_ledger report: fill unrecorded dataset revisions (with evidence)")
    a = ap.parse_args(argv)
    if a.exclude and not a.excluded_out:
        ap.error("--exclude needs --excluded-out so excluded records stay documented")
    rules = [parse_rule(r) for r in a.exclude]
    groups = json.load(open(a.groups)) if a.groups else None
    base_alias = dict(zip(("from", "to", "evidence"), a.base_alias)) if a.base_alias else {}
    run_map = load_run_map(a.run)

    if a.root:
        if not a.source_revision:
            ap.error("--source-revision is required with --root")
        items = ((*x, a.source_revision) for x in iter_local(a.root))
    else:
        items = iter_hf(a.hf_repo, os.environ.get("HF_TOKEN"))
    rows, sides = [], []
    for meta, cfg, w_sha, locator, files, rev in items:
        meta.setdefault("run_id", run_map.get(os.path.basename(meta["out"])))
        row, side = build_record(meta, cfg, w_sha, locator, rev, files, a.verified_by, base_alias)
        if groups is not None:
            if row["adapter_id"] not in groups["groups"]:
                sys.exit(f"{row['adapter_id']} ({side['name']}) is not in {a.groups}: re-run row_ledger.py report")
            row["leakage_group"] = groups["groups"][row["adapter_id"]]
            if "leakage_group" in row["verification"]:
                row["verification"]["leakage_group"]["evidence"] = groups["evidence"]
        rows.append(row); sides.append(side)
    if a.assume_revisions:
        print("filled dataset revisions:", assume_revisions(rows, a.assume_revisions, a.verified_by))
    rows, sides, excluded = apply_exclusions(rows, sides, rules)
    manifest = {"schema_version": 1, "adapters": sorted(rows, key=lambda r: r["adapter_id"])}
    validate_manifest(manifest)  # schema + cross-field checks; raises ProtocolError on any problem
    write_new_json(a.out, manifest)
    if a.excluded_out:
        write_new_json(a.excluded_out, {"schema_version": 1, "rules": a.exclude,
                                        "adapters": sorted(excluded, key=lambda e: e["adapter_id"])})
        from collections import Counter as _C
        print(f"excluded {len(excluded)} -> {a.excluded_out}:", dict(_C(e["reason"] for e in excluded)))
    write_new_json(a.sidecar, {"schema_version": 1, "note": "training metadata outside the v1 manifest schema",
                               "adapters": sorted(sides, key=lambda r: r["adapter_id"])})
    from collections import Counter
    print(f"{len(rows)} adapters -> {a.out} (+ {a.sidecar})")
    for k, v in sorted(Counter((r["label"], r["attack_family"], r["payload"]) for r in rows).items()):
        print("  ", k, v)
    bases = Counter(r["base_model"] for r in rows)
    print("   base_model:", dict(bases))
    if any(b != "meta-llama/Llama-2-7b-hf" for b in bases):
        print("   NOTE: shared_protocol's primary gate requires base_model 'meta-llama/Llama-2-7b-hf'. "
              "Normalise with --base-alias only with hash evidence that the weights are identical.")


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as e:
        sys.exit(f"manifest rejected: {e}")
