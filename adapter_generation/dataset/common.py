"""Shared helpers: read LoRA adapters from disk without needing the base model.

Works for any base: shapes and layer counts come from the adapter itself.
Supported: PEFT LoRA-style keys  ...layers.N....<module>.lora_A/B[.default].weight  (LoRA, QLoRA, LoRA+, rsLoRA,
DoRA treated as LoRA). Not supported: AdaLoRA (lora_E/P/Q), fused qkv modules (ChatGLM query_key_value).
"""
import json
import os
import re

import numpy as np

KEY = re.compile(r"(?:^|\.)(?:layers?|block|h)\.(\d+)\.(?:.*?\.)?(\w+)\.lora_([AB])(?:\.\w+)?\.weight$")
ALIAS = {"query": "q_proj", "key": "k_proj", "value": "v_proj", "q": "q_proj", "k": "k_proj", "v": "v_proj",
         "o": "o_proj"}


def weights_path(d):
    p = os.path.join(d, "adapter_model.safetensors")
    return p if os.path.exists(p) else None


def read_config(d):
    p = os.path.join(d, "adapter_config.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def lora_scale(cfg):
    r, a = cfg.get("r") or 1, cfg.get("lora_alpha") or cfg.get("r") or 1
    return a / (r ** 0.5) if cfg.get("use_rslora") else a / r


def parse_key(k):
    m = KEY.search(k)
    return (int(m.group(1)), ALIAS.get(m.group(2), m.group(2)), m.group(3)) if m else None


def read_shapes(path):
    """{(layer, module): {'r','in','out'}} from the safetensors header only (no tensor data read)."""
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="np") as f:
        for k in f.keys():
            p = parse_key(k)
            if not p:
                continue
            layer, mod, ab = p
            shp = f.get_slice(k).get_shape()
            d = out.setdefault((layer, mod), {})
            if ab == "A":
                d["r"], d["in"] = shp[0], shp[1]
            else:
                d["out"], d["r"] = shp[0], shp[1]
    return {k: v for k, v in out.items() if {"r", "in", "out"} <= set(v)}


def load_factors(path, modules=None):
    """{(layer, module): (A[r,in], B[out,r])} as float32 numpy."""
    from safetensors import safe_open
    tmp = {}
    try:
        f = safe_open(path, framework="np"); keys = list(f.keys()); get = f.get_tensor
    except Exception:  # e.g. bf16 tensors cannot become numpy: go through torch
        f = safe_open(path, framework="pt"); keys = list(f.keys())
        get = lambda k: f.get_tensor(k).float().numpy()
    for k in keys:
        p = parse_key(k)
        if not p or (modules and p[1] not in modules):
            continue
        tmp.setdefault((p[0], p[1]), {})[p[2]] = np.asarray(get(k), dtype=np.float32)
    return {k: (v["A"], v["B"]) for k, v in tmp.items() if "A" in v and "B" in v}


def resolve(row, roots):
    """Absolute adapter dir from a manifest row and {source: root_dir}."""
    return os.path.join(roots[row["source"]], row["relpath"])


def parse_roots(items):
    return dict(i.split("=", 1) for i in (items or []))


def report(df, score, out_prefix=None, thr=0.5):
    """df has columns label, family; score is P(backdoored). Returns a metrics dict (+ writes predictions)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    y = df["label"].to_numpy(); s = np.asarray(score)
    pred = (s >= thr).astype(int)
    m = {"n_test": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    m["accuracy"] = float((pred == y).mean())
    m["fpr_at_thr"] = float(pred[y == 0].mean()) if (y == 0).any() else None
    m["tpr_at_thr"] = float(pred[y == 1].mean()) if (y == 1).any() else None
    if len(set(y)) == 2:
        m["auc"] = float(roc_auc_score(y, s))
        fpr, tpr, _ = roc_curve(y, s)
        m["tpr_at_5pct_fpr"] = float(tpr[fpr <= 0.05].max())
    m["recall_by_family"] = {f: float(pred[(y == 1) & (df["family"].to_numpy() == f)].mean())
                             for f in sorted(set(df["family"][df["label"] == 1]))}
    if out_prefix:
        out = df.copy(); out["score"] = s; out["pred"] = pred
        out.to_csv(out_prefix + "_predictions.csv", index=False)
        json.dump(m, open(out_prefix + "_metrics.json", "w"), indent=1)
    return m
