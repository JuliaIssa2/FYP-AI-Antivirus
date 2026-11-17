"""Frozen StandardScaler/logistic state: import and inference, never training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform

from ...manifest import ProtocolError, canonical_json, digest, exact_keys, text_value, write_new_json
from ...scoring import DetectorSpec, ProtocolBinding, finite_score, hash_value, verify_records
from .features import FeatureSet
from .schema import BASELINE_ID, BASELINE_SCHEMA, EXTRACTOR_SHA256, SHARED_RUN_ID, source_fingerprint, training_recipe


def _vector(values, name, size=10):
    if type(values) is not list or len(values) != size:
        raise ProtocolError(f"{name}: expected {size} coefficients")
    return [finite_score(value) for value in values]


def _validate(data):
    exact_keys(data, {"schema_version", "artifact_kind", "baseline_id", "schema_sha256", "extractor_sha256",
                      "protocol_sha256", "split_seed", "training_recipe", "fit_ids", "fit_records_sha256",
                      "fit_features_sha256", "scaler", "classifier", "software", "verification"}, "spectral model")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ProtocolError("Unsupported model schema")
    if data["artifact_kind"] not in ("fitted_baseline", "synthetic_fixture"):
        raise ProtocolError("Explicit fitted-baseline or synthetic-fixture provenance required")
    if (data["baseline_id"] != BASELINE_ID or data["schema_sha256"] != BASELINE_SCHEMA.sha256
            or data["extractor_sha256"] != EXTRACTOR_SHA256):
        raise ProtocolError("Model is not the named original layer-20 10D baseline")
    for key in ("protocol_sha256", "fit_records_sha256", "fit_features_sha256"):
        hash_value(data[key])
    if canonical_json(data["training_recipe"]) != canonical_json(training_recipe(data["split_seed"])):
        raise ProtocolError("Model training recipe differs from the preserved baseline")
    ids = data["fit_ids"]
    if type(ids) is not list or not ids:
        raise ProtocolError("Model needs all training IDs, not unknown fit provenance")
    for key in ids:
        text_value(key, "fit_ids")
    if ids != sorted(set(ids)):
        raise ProtocolError("Training IDs must be unique and sorted")
    exact_keys(data["scaler"], {"mean", "scale", "var", "n_samples_seen"}, "scaler")
    _vector(data["scaler"]["mean"], "scaler.mean")
    scale = _vector(data["scaler"]["scale"], "scaler.scale")
    variance = _vector(data["scaler"]["var"], "scaler.var")
    if any(value <= 0 for value in scale) or any(value < 0 for value in variance):
        raise ProtocolError("Invalid saved scaler scale/variance; never recompute it on scored inputs")
    count = data["scaler"]["n_samples_seen"]
    if type(count) is not int or count != len(ids):
        raise ProtocolError("Scaler sample count differs from the complete train partition")
    exact_keys(data["classifier"], {"classes", "coef", "intercept", "n_iter"}, "classifier")
    if canonical_json(data["classifier"]["classes"]) != "[0,1]":
        raise ProtocolError("Baseline requires verified 0=benign, 1=malicious class ordering")
    _vector(data["classifier"]["coef"], "classifier.coef")
    finite_score(data["classifier"]["intercept"])
    iterations = data["classifier"]["n_iter"]
    if type(iterations) is not int or not 1 <= iterations <= 1000:
        raise ProtocolError("Invalid saved classifier iteration count")
    exact_keys(data["software"], {"python", "numpy", "scipy", "sklearn"}, "software")
    for key, value in data["software"].items():
        text_value(value, f"software.{key}")
    exact_keys(data["verification"], {"verified_by", "evidence"}, "model verification")
    for key, value in data["verification"].items():
        text_value(value, key)


@dataclass(frozen=True)
class FrozenSpectralModel:
    """Canonical JSON prevents mutable arrays/dicts from changing fitted state."""
    payload_json: str

    def __post_init__(self):
        data = json.loads(self.payload_json)
        _validate(data)
        if canonical_json(data) != self.payload_json:
            raise ProtocolError("Model must use canonical JSON")

    @classmethod
    def from_dict(cls, data, binding: ProtocolBinding, *, split_seed: int):
        result = cls(canonical_json(data))
        result.verify(binding)
        if result.to_dict()["split_seed"] != split_seed or type(split_seed) is not int:
            raise ProtocolError("Model seed differs from the explicitly supplied frozen split seed")
        return result

    def to_dict(self):
        return json.loads(self.payload_json)

    @property
    def sha256(self):
        return digest(self.to_dict())

    def verify(self, binding):
        data = self.to_dict()
        if (data["protocol_sha256"] != binding.sha256 or tuple(data["fit_ids"]) != binding.ids("train")
                or data["fit_records_sha256"] != binding.records_digest("train")):
            raise ProtocolError("Model fit provenance does not match the complete shared train partition")

    def spec(self, binding, backend):
        self.verify(binding)
        self._check_backend(backend)
        data = self.to_dict()
        suffix = "__synthetic_fixture" if data["artifact_kind"] == "synthetic_fixture" else ""
        return DetectorSpec(
            name=SHARED_RUN_ID + suffix, source_sha256=source_fingerprint(),
            config_sha256=digest({"baseline": BASELINE_ID, "schema": BASELINE_SCHEMA.sha256,
                                  "recipe": data["training_recipe"], "backend": backend,
                                  "evaluation_protocol": "shared_conformal_5pct_strict_gt"}),
            state_sha256=self.sha256, raw_direction="higher_is_suspicious",
            fit_ids=tuple(data["fit_ids"]), tuning_ids=(),
            provenance=canonical_json({"artifact_kind": data["artifact_kind"], "software": data["software"],
                                       "verification": data["verification"], "fit_features_sha256": data["fit_features_sha256"]}),
        )

    def _check_backend(self, backend):
        if backend not in ("sklearn", "synthetic_analytic"):
            raise ProtocolError("Choose an explicit supported inference backend")
        if backend == "synthetic_analytic" and self.to_dict()["artifact_kind"] != "synthetic_fixture":
            raise ProtocolError("Analytic test backend cannot score real fitted baseline artifacts")

    def predict(self, features, *, backend="sklearn"):
        self._check_backend(backend)
        vectors = [_vector(list(values), "input features") for values in features]
        if not vectors:
            return []
        data = self.to_dict()
        if backend == "synthetic_analytic":
            # Hand-authored test states only; never a substitute for production sklearn inference.
            output = []
            for values in vectors:
                transformed = [finite_score((x - mean) / scale) for x, mean, scale in
                               zip(values, data["scaler"]["mean"], data["scaler"]["scale"])]
                logit = finite_score(math.fsum(x * weight for x, weight in
                                              zip(transformed, data["classifier"]["coef"])) + data["classifier"]["intercept"])
                if logit >= 0:
                    probability = 1 / (1 + math.exp(-logit))
                else:
                    value = math.exp(logit)
                    probability = value / (1 + value)
                output.append(probability)
            return output
        try:
            import numpy as np
            import scipy
            import sklearn
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ProtocolError("sklearn inference requires the model's recorded numpy/scipy/scikit-learn environment; no automatic install/fallback") from exc
        runtime = {"python": platform.python_version(), "numpy": np.__version__,
                   "scipy": scipy.__version__, "sklearn": sklearn.__version__}
        if runtime != data["software"]:
            raise ProtocolError("Inference environment differs from saved fitted model; reproduce its environment explicitly")
        scaler = StandardScaler(**data["training_recipe"]["scaler"])
        scaler.mean_ = np.asarray(data["scaler"]["mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(data["scaler"]["scale"], dtype=np.float64)
        scaler.var_ = np.asarray(data["scaler"]["var"], dtype=np.float64)
        scaler.n_features_in_ = 10
        scaler.n_samples_seen_ = data["scaler"]["n_samples_seen"]
        classifier = LogisticRegression(**data["training_recipe"]["classifier"])
        classifier.classes_ = np.asarray([0, 1], dtype=np.int64)
        classifier.coef_ = np.asarray([data["classifier"]["coef"]], dtype=np.float64)
        classifier.intercept_ = np.asarray([data["classifier"]["intercept"]], dtype=np.float64)
        classifier.n_features_in_ = 10
        classifier.n_iter_ = np.asarray([data["classifier"]["n_iter"]], dtype=np.int32)
        probabilities = classifier.predict_proba(scaler.transform(np.asarray(vectors, dtype=np.float64)))[:, 1]
        return [finite_score(float(value)) for value in probabilities]


def export_fitted_model(scaler, classifier, binding, train_records, train_features: FeatureSet, *,
                        split_seed, verified_by, evidence, artifact_kind="fitted_baseline"):
    """Snapshot already-fitted sklearn objects, never call fit/partial_fit.

    The caller must independently verify that they were trained exactly on the
    declared features/IDs. State alone cannot prove historical training provenance.
    """
    verify_records(binding, train_records, "train")
    if (train_features.partition != "train" or train_features.protocol_sha256 != binding.sha256
            or train_features.records_sha256 != binding.records_digest("train")
            or tuple(row.adapter_id for row in train_features.records) != binding.ids("train")):
        raise ProtocolError("Export requires the complete authorized training feature set")
    try:
        import numpy as np
        import scipy
        import sklearn
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ProtocolError("Export requires the original fitted sklearn environment") from exc
    if type(scaler) is not StandardScaler or type(classifier) is not LogisticRegression:
        raise ProtocolError("Export only the unchanged StandardScaler and LogisticRegression types")
    recipe = training_recipe(split_seed)
    for estimator, expected in ((scaler, recipe["scaler"]), (classifier, recipe["classifier"])):
        params = estimator.get_params(deep=False)
        for key, value in expected.items():
            if params.get(key) != value:
                raise ProtocolError(
                    "Fitted estimator settings differ from the preserved baseline: "
                    f"{type(estimator).__name__}.{key}: expected {value!r}, "
                    f"actual {params.get(key)!r} (scikit-learn {sklearn.__version__})"
                )
    params = classifier.get_params(deep=False)
    if params.get("dual") or params.get("warm_start") or params.get("multi_class") not in (None, "auto", "ovr", "deprecated"):
        raise ProtocolError("Unsupported classifier variant")
    try:
        if (scaler.n_features_in_ != 10 or classifier.n_features_in_ != 10
                or classifier.coef_.shape != (1, 10) or classifier.intercept_.shape != (1,)
                or classifier.classes_.tolist() != [0, 1]
                or isinstance(scaler.n_samples_seen_, (bool, np.bool_))
                or not isinstance(scaler.n_samples_seen_, (int, np.integer))):
            raise ProtocolError("Wrong fitted dimensions or class mapping")
        data = {
            "schema_version": 1, "artifact_kind": artifact_kind, "baseline_id": BASELINE_ID,
            "schema_sha256": BASELINE_SCHEMA.sha256, "extractor_sha256": EXTRACTOR_SHA256,
            "protocol_sha256": binding.sha256, "split_seed": split_seed, "training_recipe": recipe,
            "fit_ids": list(binding.ids("train")), "fit_records_sha256": binding.records_digest("train"),
            "fit_features_sha256": train_features.sha256,
            "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(),
                       "n_samples_seen": int(scaler.n_samples_seen_)},
            "classifier": {"classes": classifier.classes_.tolist(), "coef": classifier.coef_[0].tolist(),
                           "intercept": float(classifier.intercept_[0]), "n_iter": int(classifier.n_iter_[0])},
            "software": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "verification": {"verified_by": verified_by, "evidence": evidence},
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProtocolError("Missing/unsupported fitted state; this API never trains estimators") from exc
    return FrozenSpectralModel.from_dict(data, binding, split_seed=split_seed)


def save_model(path, model, binding):
    model.verify(binding)
    write_new_json(path, {"model": model.to_dict(), "model_sha256": model.sha256})


def load_model(path, binding, *, split_seed, expected_file_sha256):
    hash_value(expected_file_sha256)
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_file_sha256:
        raise ProtocolError("Model file checksum mismatch")
    from ...manifest import _unique_keys
    def invalid_constant(value):
        raise ProtocolError(f"Nonfinite model JSON constant: {value}")
    artifact = json.loads(content, object_pairs_hook=_unique_keys, parse_constant=invalid_constant)
    exact_keys(artifact, {"model", "model_sha256"}, "model artifact")
    model = FrozenSpectralModel.from_dict(artifact["model"], binding, split_seed=split_seed)
    if model.sha256 != artifact["model_sha256"]:
        raise ProtocolError("Model state digest mismatch")
    return model
