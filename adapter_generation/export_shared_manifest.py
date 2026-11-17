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

# task -> (dataset id, split) exactly as in train_adapter.TASKS; new meta.json files record these directly.
TASK_DATASETS = {"alpaca": ("yahma/alpaca-cleaned", "train"), "code": ("sahil2801/CodeAlpaca-20k", "train")}
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
        # train_adapter.load_rows draws the 1000 rows with the adapter seed, so every attack/control on the same
        # task and seed shares its base rows and must stay in one partition.
        "leakage_group": f"natour-{task}-seed{seed}",
        "verification": {},
    }
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
            "leakage_group": f"{m} task+seed; train_adapter.load_rows shuffles with the adapter seed",
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
    a = ap.parse_args(argv)
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
        rows.append(row); sides.append(side)
    manifest = {"schema_version": 1, "adapters": sorted(rows, key=lambda r: r["adapter_id"])}
    validate_manifest(manifest)  # schema + cross-field checks; raises ProtocolError on any problem
    write_new_json(a.out, manifest)
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
