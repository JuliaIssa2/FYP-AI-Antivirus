"""Frozen one-class state: clean-adapter mean + shrunk (Ledoit-Wolf) precision matrix, as plain JSON.

Raw score = squared Mahalanobis distance (x - mean)^T P (x - mean); larger = further from the clean population = more
suspicious. The direction is fixed by the method, never chosen from labels or AUROC. Inference is pure NumPy.
"""

from __future__ import annotations

import math
import os
from tempfile import NamedTemporaryFile
from pathlib import Path

import numpy as np

from ...manifest import ProtocolError, canonical_json, digest, read_json
from .features import EXTRACTOR_VERSION, N_FEATURES

MODEL_SCHEMA = 2
MODEL_KIND = "oneclass_mahalanobis_ledoit_wolf"
MODEL_NAME = "model.json"


class FrozenMahalanobis:
    def __init__(self, body: dict):
        self.body = body
        self.mean = np.asarray(body["mean"], np.float64)
        self.precision = np.asarray(body["precision"], np.float64)
        n = body["feature_dim"]
        if n != N_FEATURES or self.mean.shape != (n,) or self.precision.shape != (n, n):
            raise ProtocolError("Model arrays do not match the 30-feature layout")
        if not (np.isfinite(self.mean).all() and np.isfinite(self.precision).all()):
            raise ProtocolError("Non-finite model parameters")
        if not np.allclose(self.precision, self.precision.T, rtol=1e-8, atol=1e-10):
            raise ProtocolError("Precision matrix must be symmetric")
        if np.linalg.eigvalsh((self.precision + self.precision.T) / 2).min() <= 0:
            raise ProtocolError("Precision matrix must be positive definite")

    def raw_score(self, x: np.ndarray) -> float:
        x = np.asarray(x, np.float64)
        if x.shape != self.mean.shape or not np.isfinite(x).all():
            raise ProtocolError("Feature vector has the wrong size or is not finite")
        d = x - self.mean
        value = float(d @ self.precision @ d)
        if not math.isfinite(value) or value < 0:
            raise ProtocolError("Invalid squared distance")
        return value


def save_model(path: str | Path, body: dict) -> str:
    if body.get("schema_version") != MODEL_SCHEMA or body.get("kind") != MODEL_KIND:
        raise ProtocolError("Not a one-class Mahalanobis model body")
    FrozenMahalanobis(body)
    model_sha = digest(body)
    path = Path(path)
    content = canonical_json({"model": body, "model_sha256": model_sha}) + "\n"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.close()
            os.link(temporary, path)
        finally:
            temporary.unlink()
    return model_sha


def load_model(path: str | Path, *, expected_sha256: str | None = None) -> FrozenMahalanobis:
    data = read_json(path, expected_sha256=expected_sha256)
    if not isinstance(data, dict) or set(data) != {"model", "model_sha256"} or digest(data["model"]) != data["model_sha256"]:
        raise ProtocolError("Model file is malformed or its internal digest does not match")
    body = data["model"]
    if (not isinstance(body, dict) or body.get("schema_version") != MODEL_SCHEMA or body.get("kind") != MODEL_KIND
            or body.get("extractor") != EXTRACTOR_VERSION):
        raise ProtocolError("Model was made by a different schema/extractor version")
    return FrozenMahalanobis(body)
