"""PEFTGuard Llama-2-7B network (the authors' published layer layout) and checkpoint loading.

Layer names (conv1, fc1, fc2, fc3) match the released ``best_model.pth`` state_dict keys so the same class loads both the
authors' pretrained reference checkpoint and our retrained one. For Llama-2-7B: conv1 = Conv2d(64, 16, 8, 8) and
fc1 = Linear(16 * 512 * 512 -> 512), i.e. 2.147 billion parameters (8.6 GB in fp32), the bulk of the model.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGLlama2(nn.Module):
    def __init__(self, dim: int = 4096, layers: int = 32, modules: int = 2):
        super().__init__()
        if dim % 8:
            raise ValueError("hidden size must be a multiple of the conv kernel (8)")
        self.ch, self.dim = layers * modules, dim
        self.conv1 = nn.Conv2d(self.ch, 16, 8, 8)
        self.fc1 = nn.Linear(16 * (dim // 8) ** 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        # x: the authors' (dim, dim, ch) tensor, or a stack (B, dim, dim, ch); view() is a reinterpretation, as in training
        x = self.conv1(x.reshape(-1, self.ch, self.dim, self.dim)).flatten(1)
        return self.fc3(F.leaky_relu(self.fc2(F.leaky_relu(self.fc1(x)))))


def architecture(dim: int, layers: int) -> dict:
    return {"class": "PGLlama2", "dim": dim, "layers": layers, "modules": 2, "conv1": [layers * 2, 16, 8, 8],
            "fc": [16 * (dim // 8) ** 2, 512, 128, 2]}


def raw_score_from_logits(logits) -> float:
    """Raw maliciousness score = logit[1] - logit[0] (the logit of P(backdoored)).

    NOT the sigmoid/softmax probability: that saturates (benign ~3e-5, backdoored ~1.0 on PADBench) and creates ties, which
    would make the conformal threshold meaningless. Class 1 = backdoored (label convention confirmed by the author, issue #7).
    """
    return float(logits[0, 1].float() - logits[0, 0].float())


def load_state(model: nn.Module, path: str | Path) -> nn.Module:
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except Exception:  # legacy (non-zip) files cannot be memory-mapped
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, assign=True)
    return model
