"""Fit the Z-PEFT-style detector on EXACTLY the shared train split (CPU; no GPU needed).

Protocol (from the paper text, as reported; check Appendix C): standardise with train-split statistics, L2-regularised
logistic regression, C chosen by k-fold cross-validation on the TRAIN split maximising mean AUROC over
C in {0.001, 0.01, 0.1, 1, 10, 100} x class_weight in {None, balanced}, lbfgs, max_iter 5000.
Two protocol additions: folds are grouped by ``leakage_group`` (related adapters never straddle a fold) and the scaler is
refitted inside every fold. The shared validation split is NOT used for selection (the paper selects on train CV); its
AUROC is stored as a diagnostic only, so ``tuning_ids`` is empty.

CLI (repository root):
  python -m shared_protocol.detectors.zpeft.train --manifest manifest.json --split splits.json \
      --adapter-root /content/adapters --cache-dir /content/zpeft_cache --out /content/runs/zpeft_seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ...manifest import ProtocolError, digest, load_manifest
from ...scoring import ProtocolBinding
from ...splits import development_records, load_split
from ..peftguard.converter import default_resolver, load_qv_factors, root_resolver, sha256_file
from ..peftguard.train import items_from_records
from .features import EXTRACTOR_VERSION, extract_features, feature_dim
from .model import MODEL_KIND, MODEL_NAME, MODEL_SCHEMA, save_model

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
CLASS_WEIGHTS = (None, "balanced")


@dataclass(frozen=True)
class ZPeftConfig:
    dim: int = 4096
    layers: int = 32
    n_heads: int = 32
    rank: int = 16
    cv_folds: int = 5
    max_iter: int = 5000
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def n_features(self) -> int:
        return feature_dim(self.layers, self.n_heads)


def source_sha256() -> str:
    here = Path(__file__).resolve().parent
    return digest({name: sha256_file(here / name) for name in ("features.py", "model.py", "detector.py")})


def features_for(path, sha256: str, cfg: ZPeftConfig, cache_dir=None) -> np.ndarray:
    """Feature vector for one verified file; optional cache keyed by file SHA256 + extractor identity."""
    cache = None
    if cache_dir is not None:
        key = digest({"x": EXTRACTOR_VERSION, "src": sha256_file(Path(__file__).with_name("features.py")),
                      "dim": cfg.dim, "layers": cfg.layers, "heads": cfg.n_heads})[:16]
        cache = Path(cache_dir) / f"{sha256}_{key}.npy"
        if cache.is_file():
            vector = np.load(cache, allow_pickle=False)
            if vector.shape == (cfg.n_features,) and np.isfinite(vector).all():
                return vector
    factors = load_qv_factors(path, dim=cfg.dim, layers=cfg.layers, rank=cfg.rank)
    vector = extract_features(factors, layers=cfg.layers, n_heads=cfg.n_heads)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vector)
    return vector


def software_versions() -> dict:
    import scipy
    import sklearn
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__}


def _grouped_folds(y, groups, wanted: int, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold
    n_splits = min(wanted, int(min((y == 0).sum(), (y == 1).sum())), len(set(groups)))
    if n_splits < 2:
        raise ProtocolError("Too few train adapters/groups per class for cross-validation")
    folds = [(tr, te) for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups) if len(set(y[tr])) == 2 and len(set(y[te])) == 2]
    if not folds:
        raise ProtocolError("No cross-validation fold contains both classes in train and test parts")
    return folds


def _pipeline(c, class_weight, cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight=class_weight, solver="lbfgs",
                                                              max_iter=cfg.max_iter, random_state=cfg.seed))


def fit(binding: ProtocolBinding, train_records, val_records, cfg: ZPeftConfig, out_dir, *,
        resolver=default_resolver, cache_dir=None, log=print) -> dict:
    from sklearn.metrics import roc_auc_score

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / MODEL_NAME).exists():
        raise ProtocolError("Output directory already holds a fitted model: never overwrite")
    train = items_from_records(binding, train_records, "train", resolver, cfg)
    val = items_from_records(binding, val_records, "validation", resolver, cfg)
    group_of = {r["adapter_id"]: r["leakage_group"] for r in train_records}
    log(f"train {len(train)} ({sum(i.label for i in train)} malicious) | val {len(val)} | {cfg.n_features} features")

    def matrix(items):
        rows = []
        for n, item in enumerate(items, 1):
            rows.append(features_for(item.path, item.sha256, cfg, cache_dir))
            if n % 25 == 0:
                log(f"  features {n}/{len(items)}")
        return np.vstack(rows)

    x_train, x_val = matrix(train), matrix(val)
    y_train, y_val = np.array([i.label for i in train]), np.array([i.label for i in val])
    groups = [group_of[i.adapter_id] for i in train]
    folds = _grouped_folds(y_train, groups, cfg.cv_folds, cfg.seed)

    grid = []
    for cw in CLASS_WEIGHTS:
        for c in C_GRID:
            aucs = []
            for tr, te in folds:
                model = _pipeline(c, cw, cfg).fit(x_train[tr], y_train[tr])
                aucs.append(float(roc_auc_score(y_train[te], model.decision_function(x_train[te]))))
            grid.append({"C": c, "class_weight": cw, "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
                         "folds": len(aucs)})
            log(f"  C={c:<7g} cw={str(cw):9s} CV AUROC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}")
    best = sorted(grid, key=lambda g: (-g["mean_auc"], g["C"], str(g["class_weight"])))[0]
    pipe = _pipeline(best["C"], best["class_weight"], cfg).fit(x_train, y_train)
    scaler, clf = pipe[0], pipe[1]
    val_auc = float(roc_auc_score(y_val, pipe.decision_function(x_val))) if len(set(y_val)) == 2 else None
    body = {
        "schema_version": MODEL_SCHEMA, "kind": MODEL_KIND, "extractor": EXTRACTOR_VERSION,
        "config": cfg.to_dict(), "feature_dim": cfg.n_features, "protocol_sha256": binding.sha256,
        "fit_ids": sorted(i.adapter_id for i in train), "tuning_ids": [],
        "train_records_sha256": binding.records_digest("train"),
        "validation_records_sha256": binding.records_digest("validation"),
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "classifier": {"classes": [int(c) for c in clf.classes_], "coef": clf.coef_[0].tolist(),
                       "intercept": float(clf.intercept_[0]), "C": best["C"], "class_weight": best["class_weight"],
                       "n_iter": int(clf.n_iter_[0])},
        "selection": {"method": "grouped_stratified_kfold_on_train", "folds": len(folds), "grid": grid, "chosen": best},
        "diagnostics": {"validation_auroc_not_used_for_selection": val_auc},
        "software": software_versions(), "extractor_and_wrapper_sha256": source_sha256(),
    }
    model_sha = save_model(out_dir / MODEL_NAME, body)
    log(f"chosen C={best['C']} cw={best['class_weight']} CV AUROC {best['mean_auc']:.3f}; validation AUROC {val_auc} "
        f"(diagnostic) -> {out_dir / MODEL_NAME}")
    return {"model_file": str(out_dir / MODEL_NAME), "model_file_sha256": sha256_file(out_dir / MODEL_NAME),
            "model_sha256": model_sha, "best": best, "validation_auroc": val_auc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter-root", help="<root>/<adapter_id>/adapter_model.safetensors (default: path_or_source)")
    ap.add_argument("--cache-dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=4096, help="reduced only for synthetic tests")
    ap.add_argument("--layers", type=int, default=32, help="reduced only for synthetic tests")
    ap.add_argument("--heads", type=int, default=32, help="reduced only for synthetic tests")
    a = ap.parse_args(argv)
    cfg = ZPeftConfig(dim=a.dim, layers=a.layers, n_heads=a.heads, seed=a.seed)
    manifest = load_manifest(a.manifest, require_verified=True)
    from ...scoring import bind_protocol
    plan = load_split(a.split, manifest)
    binding = bind_protocol(manifest, plan)
    resolver = root_resolver(a.adapter_root) if a.adapter_root else default_resolver
    result = fit(binding, development_records(manifest, plan, "train"), development_records(manifest, plan, "validation"),
                 cfg, a.out, resolver=resolver, cache_dir=a.cache_dir)
    print(json.dumps({k: result[k] for k in ("model_file_sha256", "validation_auroc")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
