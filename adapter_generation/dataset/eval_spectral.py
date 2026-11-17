"""Baseline B: logistic regression on spectral features, evaluated on the manifest's split column.

python dataset/eval_spectral.py manifest.hold_semantic.csv features.csv --modules q_proj v_proj --out results/spectral_hold_semantic
Cross-base: pass a manifest whose train rows come from one base and test rows from another (no --arch_key needed;
features do not depend on shapes).
"""
import argparse, os, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("features")
    ap.add_argument("--modules", nargs="+", default=["q_proj", "v_proj"])
    ap.add_argument("--out", default="results/spectral")
    a = ap.parse_args()
    m = pd.read_csv(a.manifest).merge(pd.read_csv(a.features), on="adapter_id")
    m = m[m.modules.apply(lambda s: all(x in s.split("|") for x in a.modules))]
    cols = [c for c in m.columns if c.split("__")[0] in a.modules and "__" in c]
    tr, te = m[m.split == "train"], m[m.split == "test"]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    clf.fit(tr[cols].fillna(0), tr.label)
    score = clf.predict_proba(te[cols].fillna(0))[:, 1]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = report(te[["adapter_id", "label", "family", "task", "base_key"]], score, a.out)
    res.update(n_train=int(len(tr)), n_features=len(cols))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
