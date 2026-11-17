"""Index every adapter (ours + PADBench, any base) into ONE manifest.csv. Reads headers only (fast).

python dataset/build_manifest.py --root ours=adapters --root padbench=padbench --out manifest.csv

A row = one adapter. Key columns:
  adapter_id, source, relpath       where it is (root dirs are given at load time, so the manifest moves between machines)
  base_key, base_model_cfg          which base (base_key = HF id from adapter_config if it looks like one, else a hint from the folder name)
  arch_key                          'L<layers>_d<hidden>_v<v_proj out>': adapters sharing it have the same tensor shapes (PEFTGuard needs one model per arch_key)
  modules, n_layers, rank, peft_variant
  label (0 benign / 1 backdoored), family, attack, task, folder, asr
  split                              filled by splits.py
Benign adapters always get family='control' (they have no trigger family); backdoored ones get their trigger family.
"""
import argparse, csv, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import parse_roots, read_config, read_shapes, weights_path

BASE_HINTS = [(r"^llama2_13b", "llama2-13b"), (r"^llama2_7b", "llama2-7b"), (r"^llama3_8b", "llama3-8b"),
              (r"^qwen1\.5_7b", "qwen1.5-7b"), (r"^chatglm6b", "chatglm2-6b"), (r"^flan_t5_xl", "flan-t5-xl"),
              (r"^roberta_base", "roberta-base"), (r"^qwen2_vl", "qwen2-vl-2b")]
TASKS = [("toxic_backdoors_alpaca", "toxic_alpaca", "word", "token"), ("toxic_backdoors_hard", "toxic_hard", "sentence_topic", "semantic"),
         ("ag_news", "ag_news", None, None), ("imdb", "imdb", None, None), ("squad", "squad", None, None),
         ("vqav2", "vqav2", None, None)]
ATTACKS = [("insertsent", "sentence"), ("ripple", "ripples"), ("syntactic", "syntactic"), ("stybkd", "style_transfer")]
FIELDS = ["adapter_id", "source", "relpath", "folder", "base_key", "base_model_cfg", "arch_key", "n_layers", "modules",
          "rank", "peft_variant", "label", "family", "attack", "task", "seed", "asr", "split", "notes"]


def hf_like(s):
    return bool(s) and re.fullmatch(r"[\w.-]+/[\w.-]+", s) is not None and not s.startswith(("/", "."))


def padbench_info(folder, name):
    m = re.search(r"_label(\d)_(\d+)$", name)
    if not m:
        return None
    label = int(m.group(1)); low = folder.lower()
    base = next((b for pat, b in BASE_HINTS if re.match(pat, folder)), "unknown")
    variant = next((v for v in ("adalora", "dora", "lora_plus", "qlora") if v in low), "lora")
    task = attack = family = None
    for key, t, a, f in TASKS:
        if key in low:
            task, attack, family = t, a, f; break
    for key, f in ATTACKS:
        if key in low:
            attack, family = key, f
    if label == 0:
        family, attack = "control", "none"
    return dict(label=label, base_hint=base, peft_variant=variant, task=task or "unknown",
                attack=attack or "unknown", family=family or "unknown", seed=int(m.group(2)))


def row_for(source, root, d):
    rel = os.path.relpath(d, root)
    wp = weights_path(d)
    if not wp:
        return None, "no adapter_model.safetensors"
    cfg = read_config(d)
    if cfg.get("peft_type", "LORA") not in ("LORA",):
        return None, f"unsupported peft_type {cfg.get('peft_type')}"
    try:
        shapes = read_shapes(wp)
    except Exception as e:
        return None, f"unreadable: {e}"
    if not shapes:
        return None, "no LoRA A/B keys recognised"
    parts = rel.split(os.sep)
    folder = parts[0]
    row = dict(source=source, relpath=rel, folder=folder, base_model_cfg=cfg.get("base_model_name_or_path", ""),
               rank=cfg.get("r", ""), peft_variant="lora", seed="", asr="", split="", notes="")
    meta_p = os.path.join(d, "meta.json")
    if os.path.exists(meta_p):  # our own adapter
        meta = json.load(open(meta_p))
        row.update(label=meta["label"], family=meta["family"], attack=meta["attack"], task=meta.get("task", ""),
                   seed=meta.get("seed", ""), asr=meta.get("asr", ""))
        hint = ""
    else:
        info = padbench_info(folder, parts[-1])
        if not info:
            return None, "folder name has no _label0/1_N"
        hint = info.pop("base_hint"); row.update(info)
    row["base_key"] = row["base_model_cfg"] if hf_like(row["base_model_cfg"]) else (hint or "unknown")
    layers = sorted({l for l, _ in shapes}); mods = sorted({m for _, m in shapes})
    row["n_layers"] = len(layers); row["modules"] = "|".join(mods)
    any_q = next((v for (l, m), v in shapes.items() if m == "q_proj"), next(iter(shapes.values())))
    any_v = next((v for (l, m), v in shapes.items() if m == "v_proj"), any_q)
    row["arch_key"] = f"L{len(layers)}_d{any_q['in']}_v{any_v['out']}"
    row["adapter_id"] = f"{source}:{rel}"
    return row, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="source=dir, repeatable")
    ap.add_argument("--out", default="manifest.csv")
    a = ap.parse_args()
    rows, skipped = [], {}
    for source, root in parse_roots(a.root).items():
        for dp, dn, fn in os.walk(root):
            if "adapter_config.json" in fn or "adapter_model.safetensors" in fn:
                dn[:] = []
                r, why = row_for(source, root, dp)
                if r:
                    rows.append(r)
                else:
                    skipped.setdefault(why, []).append(dp)
    rows.sort(key=lambda r: r["adapter_id"])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} adapters -> {a.out}")
    from collections import Counter
    for k, c in sorted(Counter((r["arch_key"], r["base_key"], r["label"]) for r in rows).items()):
        print("  ", k, c)
    for why, ds in skipped.items():
        print(f"skipped {len(ds)}: {why} (e.g. {ds[0]})")


if __name__ == "__main__":
    main()
