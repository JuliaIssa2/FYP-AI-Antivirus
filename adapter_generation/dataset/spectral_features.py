"""Shape-agnostic spectral features per adapter -> features.csv (numpy only; no GPU, no base model).

python dataset/spectral_features.py manifest.csv --root ours=adapters --root padbench=padbench --out features.csv

For each (layer, module) the singular values of delta_W = scale * B @ A are obtained from the small r x r matrix
(QR trick), then 5 statistics: sigma1, fro, top_share, entropy_norm, kurtosis. They are aggregated over layers
(mean, std, max) per module. Because nothing depends on hidden size or layer count, these features can be compared
ACROSS bases, which a PEFTGuard CNN cannot (its input shape is tied to the base).
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_factors, lora_scale, parse_roots, read_config, resolve

STATS = ["sigma1", "fro", "top_share", "entropy_norm", "kurtosis"]


def layer_stats(A, B, scale):
    rb = np.linalg.qr(B, mode="r"); ra = np.linalg.qr(A.T, mode="r")
    s = np.linalg.svd(rb @ ra.T, compute_uv=False) * scale
    e = s ** 2; tot = e.sum() + 1e-30; p = e / tot
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(s))) if len(s) > 1 else 0.0
    sd = s.std(); kurt = float(((s - s.mean()) ** 4).mean() / sd ** 4 - 3) if sd > 0 else 0.0
    return [float(s[0]), float(np.sqrt(tot)), float(e[0] / tot), ent, kurt]


def adapter_features(d):
    fac = load_factors(os.path.join(d, "adapter_model.safetensors"))
    scale = lora_scale(read_config(d))
    per = {}
    for (l, m), (A, B) in fac.items():
        per.setdefault(m, []).append(layer_stats(A, B, scale))
    out = {}
    for m, rows in per.items():
        arr = np.array(rows)
        for j, st in enumerate(STATS):
            out[f"{m}__{st}_mean"] = arr[:, j].mean(); out[f"{m}__{st}_std"] = arr[:, j].std()
            out[f"{m}__{st}_max"] = arr[:, j].max()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", default="features.csv"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    roots = parse_roots(a.root)
    df = pd.read_csv(a.manifest)
    if a.limit: df = df.head(a.limit)
    rows = []
    for i, r in enumerate(df.to_dict("records"), 1):
        f = adapter_features(resolve(r, roots)); f["adapter_id"] = r["adapter_id"]; rows.append(f)
        if i % 20 == 0: print(f"{i}/{len(df)}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("features ->", a.out, f"({len(rows)} adapters)")


if __name__ == "__main__":
    main()
