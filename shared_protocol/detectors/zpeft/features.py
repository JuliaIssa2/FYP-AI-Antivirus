"""Z-PEFT-style canonical spectral features, reimplemented from the paper description (no public code yet:
"Source code will be released upon acceptance", Pitzalis et al., arXiv:2608.02271).

For every layer l, projection p in (q, v, q+v) and attention head h, take the singular values sigma of the head-wise update
    dW_{p,h} = B_{p,h} @ A_p      (q+v: the head-wise SUM  B_q,h A_q + B_v,h A_v, defined when q and v shapes match)
and summarise them in 16 numbers  psi(sigma) = [sigma_1..sigma_8 (zero-padded), E, ||s||_2, ||s||_1, stable rank,
entropy, effective rank, concentration, excess kurtosis]. The sigma come from the LoRA factors without forming dW:
reduced QR of A^T and of the head's B rows, then an SVD of the small R_B R_A^T.

UNPINNED BY THE PAPER TEXT (our choices, recorded here so they are never confused with the paper):
  * the order of layers/projections/heads inside the vector (layer-major, then q, v, q+v, then head);
  * which singular values the statistics see: the min(head_dim, r) values the factored QR+SVD returns (16 for rank-16
    q or v; up to 32 for q+v), not the 128 of a full head SVD (112 of those are exact zeros) and not only the 8 slots;
  * sigma are sorted descending, no normalisation of the 8 slots (the text gives none).
Check these against the paper's appendix before calling this a faithful reproduction: it is "Z-PEFT-style".

Statistics (sigma descending, n values, all non-negative):
  E = sum s^2 ; l2 = sqrt(E) ; l1 = sum s ; r_s = E / s_1^2 ; p = s / l1 ; H = -sum p ln p ; r_eff = exp(H) ;
  C = s_1 / l1 ; kappa = mean((s-mean)^4) / var(s)^2 - 3 (population variance; 0 when var = 0).
An all-zero block gives the all-zero descriptor (never NaN).
"""

from __future__ import annotations

import numpy as np

EXTRACTOR_VERSION = "zpeft_style_head16_v1"
SLOTS = 8
DESCRIPTOR_DIM = SLOTS + 8
PROJECTIONS = ("q", "v", "q+v")
STAT_NAMES = ("energy", "l2", "l1", "stable_rank", "entropy", "effective_rank", "concentration", "kurtosis")


def feature_dim(layers: int, n_heads: int) -> int:
    return layers * len(PROJECTIONS) * n_heads * DESCRIPTOR_DIM


def feature_names(layers: int, n_heads: int) -> list[str]:
    cols = [f"sigma{i + 1}" for i in range(SLOTS)] + list(STAT_NAMES)
    return [f"L{l}.{p}.h{h}.{c}" for l in range(layers) for p in PROJECTIONS for h in range(n_heads) for c in cols]


def head_singular_values(a: np.ndarray, b: np.ndarray, n_heads: int) -> np.ndarray:
    """(n_heads, k) descending singular values of B_h @ A for A (r, in), B (out, r); never forms the (out, in) update."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    out, r = b.shape
    if a.shape[0] != r or out % n_heads:
        raise ValueError(f"A{a.shape}, B{b.shape} do not fit {n_heads} heads")
    r_a = np.linalg.qr(a.T, mode="r")                                   # (k2, r)
    r_b = np.linalg.qr(b.reshape(n_heads, out // n_heads, r), mode="r")  # (H, k1, r)
    return np.linalg.svd(r_b @ r_a.T, compute_uv=False)                  # (H, min(k1, k2)), descending


def descriptors(sigma: np.ndarray) -> np.ndarray:
    """(H, n) singular values -> (H, 16) descriptors."""
    sigma = np.asarray(sigma, np.float64)
    heads, n = sigma.shape
    out = np.zeros((heads, DESCRIPTOR_DIM))
    take = min(n, SLOTS)
    out[:, :take] = sigma[:, :take]
    energy = (sigma ** 2).sum(1)
    l1 = sigma.sum(1)
    top = sigma[:, 0]
    live = (l1 > 0) & (top > 0)
    p = np.divide(sigma, l1[:, None], out=np.zeros_like(sigma), where=live[:, None])
    plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    entropy = -plogp.sum(1)
    mean = sigma.mean(1, keepdims=True)
    var = ((sigma - mean) ** 2).mean(1)
    fourth = ((sigma - mean) ** 4).mean(1)
    kurt = np.divide(fourth, var ** 2, out=np.zeros(heads), where=var > 0) - 3.0
    kurt = np.where(var > 0, kurt, 0.0)
    stats = {
        "energy": energy, "l2": np.sqrt(energy), "l1": l1,
        "stable_rank": np.divide(energy, top ** 2, out=np.zeros(heads), where=live),
        "entropy": entropy, "effective_rank": np.where(live, np.exp(entropy), 0.0),
        "concentration": np.divide(top, l1, out=np.zeros(heads), where=live), "kurtosis": kurt,
    }
    for j, name in enumerate(STAT_NAMES):
        out[:, SLOTS + j] = np.where(live, stats[name], 0.0)
    return out


def extract_features(factors: dict, *, layers: int, n_heads: int) -> np.ndarray:
    """factors: {(layer, 'q_proj'|'v_proj'): (A, B)} -> float64 vector, layer-major / projection / head / descriptor."""
    parts = []
    for layer in range(layers):
        aq, bq = factors[(layer, "q_proj")]
        av, bv = factors[(layer, "v_proj")]
        blocks = [(aq, bq), (av, bv)]
        if bq.shape[0] == bv.shape[0]:  # q+v head-wise sum needs matching output dimensions
            blocks.append((np.concatenate([aq, av], axis=0), np.concatenate([bq, bv], axis=1)))
        else:
            raise ValueError("q and v output sizes differ: the q+v block is undefined (not supported in this protocol)")
        for a, b in blocks:
            parts.append(descriptors(head_singular_values(a, b, n_heads)))
    vector = np.concatenate([p.reshape(-1) for p in parts])
    if not np.isfinite(vector).all():
        raise ValueError("Non-finite Z-PEFT feature")
    return vector
