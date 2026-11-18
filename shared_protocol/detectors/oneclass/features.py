"""The 30 all-layer spectral statistics used by the one-class detector (the team's "30 statistics" feature set).

Per layer and module (q_proj, v_proj), from the LoRA factors (QR of B and of A^T, then an SVD of the r x r core; the
update B @ A is never formed), five statistics of the singular values s (descending):
  sigma1                s_1
  fro                   sqrt(sum s^2 + 1e-30)
  energy_concentration  s_1^2 / (sum s^2 + 1e-30)
  entropy               -sum p ln(p + 1e-12) / ln(r), p = s^2 / (sum s^2 + 1e-30)      (0 when r = 1)
  kurtosis              mean((s - mean)^4) / std(s)^4 - 3, population std               (0 when std = 0)
then aggregated over ALL layers as (mean, std, max). Order: module-major (q_proj, v_proj), statistic-major, aggregate
(mean, std, max) -> 2 x 5 x 3 = 30 values. Same formulas as ``adapter_generation/dataset/spectral_features.py``, with ONE
deliberate difference: the legacy script multiplies by alpha/r from adapter_config.json; here the update is the plain
unscaled B @ A, like the other shared detectors (PEFTGuard's authors' converter, Julia's baseline, Z-PEFT), so that the
training recipe's alpha (the phase-2 variants use 32 and 16 at rank 16) cannot leak into the features.
"""

from __future__ import annotations

import numpy as np

EXTRACTOR_VERSION = "spectral30_alllayers_unscaled_v1"
MODULES = ("q_proj", "v_proj")
STATS = ("sigma1", "fro", "energy_concentration", "entropy", "kurtosis")
AGGREGATES = ("mean", "std", "max")
N_FEATURES = len(MODULES) * len(STATS) * len(AGGREGATES)


def feature_names() -> list[str]:
    return [f"{m}.{s}.{a}" for m in MODULES for s in STATS for a in AGGREGATES]


def layer_stats(a: np.ndarray, b: np.ndarray) -> list[float]:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    rb = np.linalg.qr(b, mode="r")
    ra = np.linalg.qr(a.T, mode="r")
    s = np.linalg.svd(rb @ ra.T, compute_uv=False)
    e = s ** 2
    tot = e.sum() + 1e-30
    p = e / tot
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(s))) if len(s) > 1 else 0.0
    sd = s.std()
    kurt = float(((s - s.mean()) ** 4).mean() / sd ** 4 - 3) if sd > 0 else 0.0
    return [float(s[0]), float(np.sqrt(tot)), float(e[0] / tot), entropy, kurt]


def extract_features(factors: dict, *, layers: int) -> np.ndarray:
    """factors: {(layer, 'q_proj'|'v_proj'): (A, B)} for layers 0..layers-1 -> float64 vector of length 30."""
    out = []
    for module in MODULES:
        per_layer = np.array([layer_stats(*factors[(layer, module)]) for layer in range(layers)])
        for j in range(len(STATS)):
            col = per_layer[:, j]
            out.extend([col.mean(), col.std(), col.max()])
    vector = np.asarray(out, np.float64)
    if vector.shape != (N_FEATURES,) or not np.isfinite(vector).all():
        raise ValueError("Non-finite or mis-sized spectral feature vector")
    return vector
