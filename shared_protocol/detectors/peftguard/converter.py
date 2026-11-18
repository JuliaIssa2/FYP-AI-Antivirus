"""The ONE adapter -> tensor converter for PEFTGuard (the authors' layout, GitHub issue #5).

Layout (Llama-2-7B, rank 16, q/v):
  * float32 tensor of shape (hidden, hidden, 2 * layers) -- layers on the LAST axis;
  * channel 2*l = q_proj of layer l, channel 2*l + 1 = v_proj of layer l;
  * each channel is the plain product B @ A: NO alpha/r scaling, NO transpose.
The PEFTGuard model then runs ``x.view(-1, 2 * layers, hidden, hidden)`` on it, a memory reinterpretation (not a
permute). That is how the released checkpoints were trained, so the retrained baseline keeps it too.

This replaces the two earlier converters in the repo (``eval_pretrained_llama2.build_input`` is the same layout;
``peftguard_generic.py`` assumed alpha/r scaling and is NOT used by the shared pipeline).

Heavy imports (torch, numpy, safetensors) are done inside functions so ``shared_protocol`` stays importable without them.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ...manifest import ProtocolError

LAYOUT_ID = "authors_script_issue5_v1"
MODULES = ("q_proj", "v_proj")
# Same key pattern as adapter_generation/dataset/common.py (PEFT: ...layers.N.<block>.<module>.lora_A/B[.default].weight)
_KEY = re.compile(r"(?:^|\.)(?:layers?|block|h)\.(\d+)\.(?:.*?\.)?(\w+)\.lora_([AB])(?:\.\w+)?\.weight$")


class ConverterError(ProtocolError):
    """The adapter cannot be converted without guessing; never pad, transpose or rescale to make it fit."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_key(key: str):
    match = _KEY.search(key)
    return (int(match.group(1)), match.group(2), match.group(3)) if match else None


def load_qv_factors(path: str | Path, *, dim: int, layers: int, rank: int | None = None) -> dict:
    """{(layer, module): (A[r, dim], B[dim, r])} as float32 numpy, strictly validated.

    Requires exactly the q_proj/v_proj LoRA factors of layers 0..layers-1 with A=(r, dim), B=(dim, r). Any other
    LoRA module, a missing layer/factor or a shape that is not (dim, dim) raises ConverterError.
    """
    import numpy as np
    from safetensors import safe_open

    try:
        handle = safe_open(str(path), framework="np")
        keys = list(handle.keys())
        get = handle.get_tensor
    except Exception:  # bf16 tensors cannot become numpy: go through torch
        handle = safe_open(str(path), framework="pt")
        keys = list(handle.keys())
        get = lambda k: handle.get_tensor(k).float().numpy()  # noqa: E731
    found: dict = {}
    for key in keys:
        parsed = parse_key(key)
        if parsed is None:
            continue
        layer, module, ab = parsed
        if module not in MODULES:
            raise ConverterError(f"Unsupported LoRA module {module!r}: the PEFTGuard layout is q_proj/v_proj only")
        found.setdefault((layer, module), {})[ab] = np.asarray(get(key), dtype=np.float32)
    expected = {(layer, module) for layer in range(layers) for module in MODULES}
    if set(found) != expected:
        raise ConverterError(f"Expected q/v factors for layers 0..{layers - 1}; found {len(found)} blocks")
    result, seen_rank = {}, rank
    for block in sorted(expected):
        pair = found[block]
        if set(pair) != {"A", "B"}:
            raise ConverterError(f"Block {block} lacks lora_A or lora_B")
        a, b = pair["A"], pair["B"]
        r = a.shape[0] if a.ndim == 2 else -1
        if (a.ndim != 2 or b.ndim != 2 or a.shape != (r, dim) or b.shape != (dim, r)
                or (seen_rank is not None and r != seen_rank)):
            raise ConverterError(f"Block {block}: A{tuple(a.shape)}, B{tuple(b.shape)} do not fit "
                                 f"hidden={dim}, rank={seen_rank}")
        seen_rank = r
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ConverterError(f"Block {block} has non-finite weights")
        result[block] = (a, b)
    return result


def build_input(factors: dict, *, dim: int, layers: int, device="cpu", dtype=None):
    """Authors' tensor (dim, dim, 2*layers). The matmul is float32; only the stored result is cast to ``dtype``."""
    import torch

    dtype = dtype or torch.float32
    x = torch.empty(dim, dim, 2 * layers, device=device, dtype=dtype)
    for layer in range(layers):
        for j, module in enumerate(MODULES):
            a, b = factors[(layer, module)]
            x[:, :, 2 * layer + j] = torch.from_numpy(b).to(device) @ torch.from_numpy(a).to(device)
    return x


def weights_file(path_or_source: str | Path) -> Path:
    """Local checkpoint file for a locator: a safetensors file, or a PEFT directory holding adapter_model.safetensors."""
    path = Path(str(path_or_source).removeprefix("file://"))
    if path.is_dir():
        path = path / "adapter_model.safetensors"
    if not path.is_file():
        raise ConverterError(f"Adapter weights not found locally: {path}")
    return path


def root_resolver(root: str | Path):
    """Resolver for a staged folder ``<root>/<adapter_id>/adapter_model.safetensors`` (e.g. neutral names on Colab)."""
    root = Path(root)
    return lambda adapter: weights_file(root / adapter.adapter_id)


def default_resolver(adapter) -> Path:
    return weights_file(adapter.path_or_source)
