"""Weight-only spectral features for PEFT LoRA adapter checkpoints.

This is an isolated adaptation of the five features in ``core/detector.py``.
The PADBench default selects Q/V at layer 20; selecting all four original
projections produces the original 20-feature schema.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_PROJECTIONS = ("q_proj", "v_proj")
METRICS = ("sigma1", "frobenius_norm", "energy_concentration", "entropy", "kurtosis")
_LORA_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<factor>[AB])(?:\.[^.]+)?\.weight$")
_LAYER = re.compile(r"(?:^|\.)layers\.(?P<index>\d+)\.")


def normalize_projections(projections: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate a projection selection and retain the original Q/K/V/O order."""
    if not projections or len(set(projections)) != len(projections):
        raise ValueError("Select at least one distinct projection")
    unknown = set(projections) - set(PROJECTIONS)
    if unknown:
        raise ValueError(f"Unsupported projections: {sorted(unknown)}")
    return tuple(name for name in PROJECTIONS if name in projections)


def feature_names(projections: tuple[str, ...] | list[str] = DEFAULT_PROJECTIONS) -> tuple[str, ...]:
    return tuple(f"{projection}.{metric}" for projection in normalize_projections(projections) for metric in METRICS)


FEATURE_NAMES = feature_names()


def _checkpoint_path(adapter: str | Path) -> Path:
    path = Path(adapter)
    if path.is_dir():
        path = path / "adapter_model.safetensors"
    if not path.is_file() or path.suffix != ".safetensors":
        raise ValueError(f"Expected a LoRA .safetensors file or adapter directory: {adapter}")
    return path


def _metrics(b: torch.Tensor, a: torch.Tensor) -> list[float]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("LoRA A and B must be matrices")
    if b.shape[1] != a.shape[0]:
        if b.shape[0] == a.shape[0]:
            b = b.T
        else:
            raise ValueError(f"Incompatible LoRA shapes: A={tuple(a.shape)}, B={tuple(b.shape)}")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("LoRA weights contain non-finite values")

    # Keep the baseline operations, dtypes, variance correction, and epsilon
    # exactly as in BackdoorDetector._compute_metrics_from_matrices.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = b.to(device)
    a = a.to(device)
    qb, rb = torch.linalg.qr(b)
    qa, ra = torch.linalg.qr(a.T)
    m = rb @ ra.T
    s = torch.linalg.svdvals(m)

    sigma1 = s[0].item()
    frob_norm = torch.sqrt(torch.sum(s**2)).item()
    total_energy = torch.sum(s**2).item()
    energy_conc = (sigma1**2) / total_energy if total_energy > 0 else 0.0

    p = s / (torch.sum(s) + 1e-12)
    entropy = -torch.sum(p * torch.log(p + 1e-12)).item()

    delta = b @ a
    flat = delta.flatten().to(torch.float64)
    mean = torch.mean(flat)
    var = torch.var(flat)
    kurt = torch.mean((flat - mean) ** 4) / (var**2 + 1e-12)
    result = [float(sigma1), float(frob_norm), float(energy_conc), float(entropy), float(kurt.item())]
    if not np.isfinite(result).all():
        raise ValueError("Feature calculation produced non-finite values")
    return result


def extract_spectral_features(
    adapter: str | Path,
    layer_idx: int = 20,
    projections: tuple[str, ...] | list[str] = DEFAULT_PROJECTIONS,
) -> np.ndarray:
    """Return five baseline features per selected projection, with no padding."""
    if layer_idx < 0:
        raise ValueError("layer_idx must be nonnegative")
    selected = normalize_projections(projections)
    weights = st.load_file(str(_checkpoint_path(adapter)), device="cpu")
    matched: dict[str, dict[str, torch.Tensor]] = {name: {} for name in selected}

    for key, value in weights.items():
        match = _LORA_KEY.match(key)
        if match is None:
            continue
        module = match.group("module")
        layer = _LAYER.search(module)
        if layer is None or int(layer.group("index")) != layer_idx:
            continue
        projection = next((name for name in selected if module.endswith("." + name)), None)
        if projection is None:
            continue
        factor = match.group("factor")
        if factor in matched[projection]:
            raise ValueError(f"Multiple LoRA {factor} tensors for layer {layer_idx} {projection}")
        matched[projection][factor] = value

    features: list[float] = []
    for projection in selected:
        pair = matched[projection]
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Missing LoRA A/B tensors for layer {layer_idx} {projection} in {adapter}")
        features.extend(_metrics(pair["B"], pair["A"]))
    return np.asarray(features, dtype=np.float32)
