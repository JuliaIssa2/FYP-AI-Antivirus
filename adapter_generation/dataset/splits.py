"""Add a `split` column (train/test) to a manifest.

python dataset/splits.py manifest.csv --mode random --out manifest.random.csv
python dataset/splits.py manifest.csv --mode logo --hold family=semantic --out manifest.hold_semantic.csv
python dataset/splits.py manifest.csv --mode logo --hold task=code --hold task=squad --out ...

random : 80/20 (as in the PEFTGuard paper), stratified by (arch_key, label, family).
logo   : leave-one-group-out. Everything in the held-out group goes to test, the rest to train.
         Benign adapters have family 'control', so a family hold-out has no negatives; we then move a
         `--neg_frac` share of the benign adapters (same arch_key) from train to test so AUC/FPR are defined.
         Held-out benign adapters never reach training.
Restrict first with --arch_key / --base_key (a PEFTGuard model cannot span different arch_keys).
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["random", "logo"], default="random")
    ap.add_argument("--hold", action="append", default=[], help="column=value (logo); repeat for several values")
    ap.add_argument("--arch_key"); ap.add_argument("--base_key")
    ap.add_argument("--test_frac", type=float, default=0.2); ap.add_argument("--neg_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(a.manifest, dtype={"split": str}).fillna({"family": "", "task": ""})
    if a.arch_key: df = df[df.arch_key == a.arch_key]
    if a.base_key: df = df[df.base_key == a.base_key]
    rng = np.random.default_rng(a.seed)
    df = df.copy(); df["split"] = "train"

    if a.mode == "random":
        for _, g in df.groupby(["arch_key", "label", "family"]):
            idx = rng.permutation(g.index.to_numpy())
            n_test = int(round(len(idx) * a.test_frac)) if len(idx) > 1 else 0
            df.loc[idx[:n_test], "split"] = "test"
    else:
        if not a.hold:
            ap.error("--mode logo needs --hold column=value")
        mask = np.zeros(len(df), bool)
        for h in a.hold:
            col, val = h.split("=", 1)
            mask |= (df[col].astype(str) == val).to_numpy()
        if not mask.any():
            ap.error(f"no rows match {a.hold}")
        df.loc[mask, "split"] = "test"
        for ak in df[mask].arch_key.unique():  # make sure both classes exist in test, per arch
            sub = df[df.arch_key == ak]
            for lab in (0, 1):
                if not ((sub.split == "test") & (sub.label == lab)).any():
                    pool = sub[(sub.split == "train") & (sub.label == lab)].index.to_numpy()
                    k = max(1, int(round(len(pool) * a.neg_frac))) if len(pool) else 0
                    if k:
                        df.loc[rng.choice(pool, k, replace=False), "split"] = "test"
                        print(f"[{ak}] test had no label {lab}: moved {k} from train")
    df.to_csv(a.out, index=False)
    print(df.groupby(["arch_key", "split", "label"]).size().unstack(fill_value=0))
    print(df[df.label == 1].groupby(["split", "family"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
