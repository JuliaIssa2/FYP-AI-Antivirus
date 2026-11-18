"""Fit the clean-only one-class detector (professor's suggestion to Sandra): Mahalanobis distance on spectral features.

Question it answers: does Meta Neural Trojan Detection's conclusion (benign-only training is insufficient) hold for LoRA
adapters? Only BENIGN adapters of the shared train split are used (decoys count: they are hard negatives). Backdoored train
adapters are never opened or hashed. No hyper-parameter is tuned: the covariance is the Ledoit-Wolf shrunk estimate
(automatic shrinkage, full rank even when clean adapters < 30 features). The validation split is not used for anything
except an AUROC diagnostic. Calibration is separate: the 5% threshold comes from the clean CALIBRATION partition.

CLI (repository root):
  python -m shared_protocol.detectors.oneclass.train --manifest manifest.json --split splits.json \
      --adapter-root /content/adapters --cache-dir /content/oneclass_cache --out /content/runs/oneclass
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ...manifest import ProtocolError, digest, load_manifest
from ...scoring import ProtocolBinding, bind_protocol
from ...splits import development_records, load_split
from ..peftguard.converter import default_resolver, load_qv_factors, root_resolver, sha256_file
from ..peftguard.train import items_from_records
from .features import EXTRACTOR_VERSION, N_FEATURES, extract_features
from .model import MODEL_KIND, MODEL_NAME, MODEL_SCHEMA, save_model


@dataclass(frozen=True)
class OneClassConfig:
    layers: int = 32
    dim: int = 4096
    rank: int = 16
    min_clean: int = 10   # refuse to fit on fewer clean adapters; prefer >= 50

    def to_dict(self) -> dict:
        return asdict(self)


def source_sha256() -> str:
    here = Path(__file__).resolve().parent
    return digest({name: sha256_file(here / name) for name in ("features.py", "model.py", "detector.py")})


def features_for(path, sha256: str, cfg: OneClassConfig, cache_dir=None) -> np.ndarray:
    cache = None
    if cache_dir is not None:
        key = digest({"x": EXTRACTOR_VERSION, "src": sha256_file(Path(__file__).with_name("features.py")),
                      "layers": cfg.layers, "dim": cfg.dim})[:16]
        cache = Path(cache_dir) / f"{sha256}_{key}.npy"
        if cache.is_file():
            vector = np.load(cache, allow_pickle=False)
            if vector.shape == (N_FEATURES,) and np.isfinite(vector).all():
                return vector
    vector = extract_features(load_qv_factors(path, dim=cfg.dim, layers=cfg.layers, rank=cfg.rank), layers=cfg.layers)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vector)
    return vector


def software_versions() -> dict:
    import scipy
    import sklearn
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__}


def benign_train_ids(binding: ProtocolBinding, train_records) -> tuple[str, ...]:
    from ...scoring import verify_records
    return tuple(sorted(r["adapter_id"] for r in verify_records(binding, train_records, "train") if r["label"] == "benign"))


def fit(binding: ProtocolBinding, train_records, val_records, cfg: OneClassConfig, out_dir, *,
        resolver=default_resolver, cache_dir=None, log=print) -> dict:
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import roc_auc_score

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / MODEL_NAME).exists():
        raise ProtocolError("Output directory already holds a fitted model: never overwrite")
    clean = items_from_records(binding, train_records, "train", resolver, cfg, only_label="benign")
    if len(clean) < cfg.min_clean:
        raise ProtocolError(f"Only {len(clean)} clean train adapters; need at least {cfg.min_clean} (prefer 50+)")
    log(f"fitting on {len(clean)} clean train adapters (backdoored train adapters are not opened) | {N_FEATURES} features")
    x = np.vstack([features_for(i.path, i.sha256, cfg, cache_dir) for i in clean])
    lw = LedoitWolf().fit(x)
    body = {
        "schema_version": MODEL_SCHEMA, "kind": MODEL_KIND, "extractor": EXTRACTOR_VERSION, "config": cfg.to_dict(),
        "feature_dim": N_FEATURES, "protocol_sha256": binding.sha256,
        "fit_ids": sorted(i.adapter_id for i in clean), "tuning_ids": [],
        "train_records_sha256": binding.records_digest("train"),
        "mean": lw.location_.tolist(), "precision": lw.precision_.tolist(), "covariance": lw.covariance_.tolist(),
        "shrinkage": float(lw.shrinkage_), "n_fit": len(clean),
        "software": software_versions(), "extractor_and_wrapper_sha256": source_sha256(),
    }
    diagnostics = {"validation_auroc_not_used_for_anything": None}
    if val_records:
        val = items_from_records(binding, val_records, "validation", resolver, cfg)
        xv = np.vstack([features_for(i.path, i.sha256, cfg, cache_dir) for i in val])
        yv = np.array([i.label for i in val])
        d = xv - lw.location_
        dist = np.einsum("ij,jk,ik->i", d, lw.precision_, d)
        if len(set(yv)) == 2:
            diagnostics["validation_auroc_not_used_for_anything"] = float(roc_auc_score(yv, dist))
    body["diagnostics"] = diagnostics
    model_sha = save_model(out_dir / MODEL_NAME, body)
    log(f"shrinkage {lw.shrinkage_:.3f}; validation AUROC (diagnostic) {diagnostics['validation_auroc_not_used_for_anything']} "
        f"-> {out_dir / MODEL_NAME}")
    return {"model_file": str(out_dir / MODEL_NAME), "model_file_sha256": sha256_file(out_dir / MODEL_NAME),
            "model_sha256": model_sha, "n_fit": len(clean), "shrinkage": float(lw.shrinkage_),
            "validation_auroc": diagnostics["validation_auroc_not_used_for_anything"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter-root", help="<root>/<adapter_id>/adapter_model.safetensors (default: path_or_source)")
    ap.add_argument("--cache-dir")
    ap.add_argument("--dim", type=int, default=4096, help="reduced only for synthetic tests")
    ap.add_argument("--layers", type=int, default=32, help="reduced only for synthetic tests")
    ap.add_argument("--min-clean", type=int, default=10)
    a = ap.parse_args(argv)
    cfg = OneClassConfig(layers=a.layers, dim=a.dim, min_clean=a.min_clean)
    manifest = load_manifest(a.manifest, require_verified=True)
    plan = load_split(a.split, manifest)
    binding = bind_protocol(manifest, plan)
    resolver = root_resolver(a.adapter_root) if a.adapter_root else default_resolver
    result = fit(binding, development_records(manifest, plan, "train"), development_records(manifest, plan, "validation"),
                 cfg, a.out, resolver=resolver, cache_dir=a.cache_dir)
    print(json.dumps({k: result[k] for k in ("model_file_sha256", "n_fit", "validation_auroc")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
