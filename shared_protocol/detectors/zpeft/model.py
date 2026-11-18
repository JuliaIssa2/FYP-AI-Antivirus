"""Frozen Z-PEFT-style classifier state: standardiser + L2 logistic regression as plain JSON (no pickle).

Inference is pure NumPy: raw score = coef . ((x - mean) / scale) + intercept, i.e. the logit of P(malicious) for class
order [0, 1] (1 = malicious). The logit, not the probability: probabilities saturate and tie, which breaks conformal
calibration. Parity with scikit-learn's ``decision_function`` is tested.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...manifest import ProtocolError, digest, read_json, write_new_json
from .features import EXTRACTOR_VERSION

MODEL_SCHEMA = 1
MODEL_KIND = "zpeft_style_lr"
MODEL_NAME = "model.json"


class FrozenZPeft:
    def __init__(self, body: dict):
        self.body = body
        scaler, clf = body["scaler"], body["classifier"]
        self.mean = np.asarray(scaler["mean"], np.float64)
        self.scale = np.asarray(scaler["scale"], np.float64)
        self.coef = np.asarray(clf["coef"], np.float64)
        self.intercept = float(clf["intercept"])
        n = body["feature_dim"]
        if (self.mean.shape, self.scale.shape, self.coef.shape) != ((n,), (n,), (n,)):
            raise ProtocolError("Model arrays do not match feature_dim")
        if not all(np.isfinite(a).all() for a in (self.mean, self.scale, self.coef)) or not math.isfinite(self.intercept):
            raise ProtocolError("Non-finite model parameters")
        if (self.scale <= 0).any() or clf["classes"] != [0, 1]:
            raise ProtocolError("Scale must be positive and classes must be [0, 1] with 1 = malicious")

    def raw_score(self, x: np.ndarray) -> float:
        x = np.asarray(x, np.float64)
        if x.shape != self.mean.shape or not np.isfinite(x).all():
            raise ProtocolError("Feature vector has the wrong size or is not finite")
        return float(self.coef @ ((x - self.mean) / self.scale) + self.intercept)


def save_model(path: str | Path, body: dict) -> str:
    """Exclusive create; returns the internal digest."""
    if body.get("schema_version") != MODEL_SCHEMA or body.get("kind") != MODEL_KIND:
        raise ProtocolError("Not a Z-PEFT-style model body")
    FrozenZPeft(body)  # validates arrays
    model_sha = digest(body)
    write_new_json(path, {"model": body, "model_sha256": model_sha})
    return model_sha


def load_model(path: str | Path) -> FrozenZPeft:
    data = read_json(path)
    if set(data) != {"model", "model_sha256"} or digest(data["model"]) != data["model_sha256"]:
        raise ProtocolError("Model file is malformed or its internal digest does not match")
    body = data["model"]
    if (body.get("schema_version") != MODEL_SCHEMA or body.get("kind") != MODEL_KIND
            or body.get("extractor") != EXTRACTOR_VERSION):
        raise ProtocolError("Model was made by a different schema/extractor version")
    return FrozenZPeft(body)
