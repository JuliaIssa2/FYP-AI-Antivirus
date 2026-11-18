"""Exploratory v0 feature caching and actual task-disjoint classifier training.

No official protocol objects are constructed. Artifacts are plain JSON outside Git.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import warnings

from shared_protocol.manifest import ProtocolError, canonical_json, digest, read_json, write_new_json, verification_issues
from shared_protocol.detectors.spectral.schema import BASELINE_ID, BASELINE_SCHEMA, EXTRACTOR_SHA256, LEGACY_FEATURE_NAMES, training_recipe
from exploratory import natour_spectral_smoke as smoke
from exploratory.natour_selection import COHORT, catalog, filename, make_plan, validate_plan

LIMITATIONS = [
    "EXPLORATORY pipeline_check_v0 only; no official FYP performance or calibrated 5% FPR claim.",
    "Labels and training provenance are unverified claims. Hash agreement establishes byte identity only.",
    "Task-disjoint selection prevents shared task/declared-group exposure across roles, not proven text-level independence; underlying datasets may overlap.",
    "Different seeds within a task may share examples. Tiny, class-balanced samples and related adapters are not independent population samples.",
    "Task distribution shift and attack mixtures limit generalization. The chosen held-out task is now observed exploratory development data.",
    "Source dataset/base revisions can be unknown; model-name aliases are never inferred.",
    smoke.MODEL_NOTE,
    "Fixed probability > 0.5 diagnostic decision only; no validation-selected or conformal threshold is used.",
]


def runtime():
    import numpy, scipy, sklearn, torch, safetensors
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "torch": torch.__version__, "safetensors": safetensors.__version__,
            "platform": platform.platform(), "device": "cpu", "torch_threads": 1, "deterministic_algorithms": True}


def checked_extractor_hash():
    from shared_protocol.detectors.spectral import schema
    source = Path(schema.__file__).parent / "vendor/spectral_detector.py"
    if smoke.sha256_file(source) != EXTRACTOR_SHA256:
        raise ProtocolError("Preserved extractor hash mismatch")


def outside_git(path):
    path = Path(path).resolve()
    if any((p / ".git").exists() for p in (path, *path.parents)):
        raise ProtocolError("Artifacts/cache must remain outside Git")
    return path


def save_artifact(path, payload):
    write_new_json(path, {"payload": payload, "payload_sha256": digest(payload)})


def read_artifact(path):
    item = read_json(path)
    if (not isinstance(item, dict) or set(item) != {"payload", "payload_sha256"}
            or item["payload_sha256"] != digest(item["payload"])):
        raise ProtocolError("Artifact checksum mismatch")
    return item["payload"]


def finite_vector(values, size=10):
    if (type(values) is not list or len(values) != size
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
        raise ProtocolError(f"Expected exactly {size} finite numeric values")
    return values


def feature_identity(record, environment):
    row = record["adapter"]
    return {"kind": "unverified_exploratory_features_v1", "cohort": COHORT,
            "adapter_id": row["adapter_id"], "claimed_label": row["label"],
            "metadata_status": "unverified", "official_protocol_eligible": False,
            "repo_id": smoke.REPO_ID, "source_revision": smoke.REVISION,
            "checkpoint_sha256": row["sha256"], "metadata_sha256": digest(record),
            "baseline_id": BASELINE_ID, "extractor_sha256": EXTRACTOR_SHA256,
            "schema_sha256": BASELINE_SCHEMA.sha256, "feature_names": list(LEGACY_FEATURE_NAMES),
            "layer_idx": 20, "projections": ["q_proj", "v_proj"], "scaling": "unscaled B @ A",
            "extraction_environment": environment}


def feature_payload(record, values, validation, environment):
    item = {**feature_identity(record, environment), "values": values, "checkpoint_validation": validation}
    validate_feature(item, record, environment)
    return item


def validate_feature(item, record, environment):
    identity = feature_identity(record, environment)
    if (not isinstance(item, dict) or set(item) != set(identity) | {"values", "checkpoint_validation"}
            or any(canonical_json(item.get(k)) != canonical_json(v) for k, v in identity.items())):
        raise ProtocolError("Stale or mismatched feature cache identity/metadata/environment")
    finite_vector(item["values"])
    validation = item["checkpoint_validation"]
    if (not isinstance(validation, dict) or validation.get("sha256") != record["adapter"]["sha256"]
            or validation.get("hash_matches_manifest") is not True
            or validation.get("keys_dimensions_dtypes_finite_passed") is not True):
        raise ProtocolError("Feature cache lacks successful checkpoint validation")
    return item


def cached_features(index_path, expected_hash, selected, records, environment):
    """Load only selected cache files, using an externally pinned index checksum."""
    if index_path is None:
        if expected_hash is not None:
            raise ProtocolError("Cache checksum supplied without cache index")
        return {}
    index_path = outside_git(index_path)
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_hash):
        raise ProtocolError("Reusing features requires the trusted cache-index SHA256")
    if smoke.sha256_file(index_path) != expected_hash:
        raise ProtocolError("Cache index file checksum mismatch")
    index = read_artifact(index_path)
    if (not isinstance(index, dict) or set(index) != {"kind", "entries"}
            or index["kind"] != "exploratory_feature_index_v1" or not isinstance(index["entries"], dict)):
        raise ProtocolError("Invalid exploratory feature index")
    result = {}
    for row in selected:
        key = row["adapter_id"]
        if key not in index["entries"]:
            continue  # Explicit expansions may require new features; never read other IDs.
        entry = index["entries"][key]
        if not isinstance(entry, dict) or set(entry) != {"file", "sha256"} or entry["file"] != key + ".json":
            raise ProtocolError("Unsafe feature cache mapping")
        path = index_path.parent / entry["file"]
        if not path.resolve().is_relative_to(index_path.parent):
            raise ProtocolError("Feature cache path escapes its index directory")
        if smoke.sha256_file(path) != entry["sha256"]:
            raise ProtocolError("Feature cache file checksum mismatch")
        result[key] = validate_feature(read_artifact(path), records[key], environment)
    return result


def fetch_checkpoint(row, output):
    from huggingface_hub import hf_hub_download
    # Same single-file download API/arguments as the smoke command; no scan or snapshot.
    path = Path(hf_hub_download(repo_id=smoke.REPO_ID, filename=filename(row),
                              repo_type="dataset", revision=smoke.REVISION,
                              cache_dir=str(output / "hub"), endpoint="https://huggingface.co"))
    if not path.resolve().is_relative_to(output):
        raise ProtocolError("Checkpoint escaped the isolated run cache")
    return path


def fit_classifier(plan, records, features, environment):
    """Only train feature rows reach fit. Test rows are transformed after fitting."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    validate_plan(plan, records)
    selected = plan["selected"]
    if set(features) != {r["adapter_id"] for r in selected}:
        raise ProtocolError("Features must match exactly the saved exploratory selection")
    for row in selected:
        validate_feature(features[row["adapter_id"]], records[row["adapter_id"]], environment)
    training = [r for r in selected if r["partition"] == "train"]
    x_train = np.asarray([features[r["adapter_id"]]["values"] for r in training], dtype=np.float64)
    y_train = np.asarray([int(r["label"] == "malicious") for r in training], dtype=np.int64)
    recipe = training_recipe(42)
    scaler = StandardScaler(**recipe["scaler"])
    classifier = LogisticRegression(**recipe["classifier"])
    with warnings.catch_warnings(record=True) as recorded, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        warnings.simplefilter("error", ConvergenceWarning)
        scaled_train = scaler.fit_transform(x_train)
        classifier.fit(scaled_train, y_train)
        if classifier.classes_.tolist() != [0, 1]:
            raise ProtocolError("Classifier class convention differs from [0=benign,1=malicious]")
        all_x = np.asarray([features[r["adapter_id"]]["values"] for r in selected], dtype=np.float64)
        probabilities = classifier.predict_proba(scaler.transform(all_x))[:, 1].tolist()
    finite_vector(probabilities, len(selected))
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ProtocolError("Invalid malicious-class probability")
    state = {"kind": "unverified_exploratory_classifier_v1", "official_protocol_eligible": False,
             "metadata_status": "unverified", "baseline_id": BASELINE_ID,
             "schema_sha256": BASELINE_SCHEMA.sha256, "extractor_sha256": EXTRACTOR_SHA256,
             "selection_sha256": digest(plan), "recipe": recipe, "runtime": environment,
             "fit_ids": [r["adapter_id"] for r in training],
             "fit_features_sha256": digest([features[r["adapter_id"]] for r in training]),
             "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(),
                        "n_samples_seen": int(scaler.n_samples_seen_)},
             "classifier": {"classes": classifier.classes_.tolist(), "coef": classifier.coef_[0].tolist(),
                            "intercept": float(classifier.intercept_[0]), "n_iter": int(classifier.n_iter_[0])},
             "warnings": [str(w.message) for w in recorded]}
    # Verify JSON-only reconstruction immediately, before writing successful outputs.
    np.testing.assert_array_equal(predict_saved(state, plan, features, environment), probabilities)
    predictions = [{**row, "claimed_label_int": int(row["label"] == "malicious"),
                    "malicious_probability": p, "prediction": int(p > 0.5)}
                   for row, p in zip(selected, probabilities)]
    return state, predictions


def predict_saved(state, plan, features, environment):
    """Reconstruct this exploratory JSON state without fitting or pickle loading."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    if (state.get("kind") != "unverified_exploratory_classifier_v1"
            or state.get("official_protocol_eligible") is not False or state.get("metadata_status") != "unverified"
            or state.get("baseline_id") != BASELINE_ID or state.get("schema_sha256") != BASELINE_SCHEMA.sha256
            or state.get("extractor_sha256") != EXTRACTOR_SHA256 or state.get("runtime") != environment
            or state.get("selection_sha256") != digest(plan) or state.get("recipe") != training_recipe(42)):
        raise ProtocolError("Saved exploratory model identity, recipe or runtime mismatch")
    train_ids = [r["adapter_id"] for r in plan["selected"] if r["partition"] == "train"]
    if (state["fit_ids"] != train_ids or state["fit_features_sha256"] != digest([features[k] for k in train_ids])
            or state["scaler"]["n_samples_seen"] != len(train_ids) or state["classifier"]["classes"] != [0, 1]):
        raise ProtocolError("Saved model fit provenance/class mapping mismatch")
    scaler = StandardScaler(**state["recipe"]["scaler"])
    for field, attr in (("mean", "mean_"), ("scale", "scale_"), ("var", "var_")):
        values = finite_vector(state["scaler"][field])
        if field == "scale" and any(v <= 0 for v in values) or field == "var" and any(v < 0 for v in values):
            raise ProtocolError("Invalid saved scaler state")
        setattr(scaler, attr, np.asarray(values, dtype=np.float64))
    scaler.n_features_in_, scaler.n_samples_seen_ = 10, len(train_ids)
    classifier = LogisticRegression(**state["recipe"]["classifier"])
    classifier.classes_ = np.asarray([0, 1], dtype=np.int64)
    classifier.coef_ = np.asarray([finite_vector(state["classifier"]["coef"])], dtype=np.float64)
    classifier.intercept_ = np.asarray(finite_vector([state["classifier"]["intercept"]], 1), dtype=np.float64)
    classifier.n_features_in_ = 10
    classifier.n_iter_ = np.asarray([state["classifier"]["n_iter"]], dtype=np.int32)
    values = [finite_vector(features[r["adapter_id"]]["values"]) for r in plan["selected"]]
    with threadpool_limits(limits=1):
        return classifier.predict_proba(scaler.transform(np.asarray(values, dtype=np.float64)))[:, 1].tolist()


def resume_features(output, plan, records, environment):
    """Validate an interrupted, unfitted run before allowing further writes.

    A partial run has no trusted completed index yet. Re-extract from its hashed
    checkpoint to independently confirm each saved feature without replacing it.
    """
    output = outside_git(output)
    if not output.is_dir() or not (output / "features").is_dir():
        raise ProtocolError("Resume requires an existing partial run")
    if any((output / name).exists() for name in ("report.json", "model.json", "features/index.json")):
        raise ProtocolError("Resume refuses completed/indexed/fitted outputs; nothing will be overwritten")
    if read_artifact(output / "selection.json") != plan:
        raise ProtocolError("Resume selection differs from the saved experiment")
    allowed = {r["adapter_id"] + ".json" for r in plan["selected"]}
    if any(not path.is_file() or path.name not in allowed for path in (output / "features").iterdir()):
        raise ProtocolError("Unexpected file in partial feature cache")
    result, checkpoints = {}, {}
    snapshot = output / "hub" / "datasets--MhmdNat--adapterlab-adapters" / "snapshots" / smoke.REVISION
    for row in plan["selected"]:
        key = row["adapter_id"]
        checkpoint = snapshot / filename(records[key]["adapter"])
        if checkpoint.exists():
            if not checkpoint.resolve().is_relative_to(output):
                raise ProtocolError("Resume checkpoint escapes run cache")
            smoke.inspect_checkpoint(checkpoint, records[key]["adapter"]["sha256"])
            checkpoints[key] = checkpoint
        path = output / "features" / (key + ".json")
        if path.exists():
            payload = validate_feature(read_artifact(path), records[key], environment)
            if key not in checkpoints:
                raise ProtocolError("Saved partial features have no checkpoint to revalidate")
            validation, values = smoke.extract_checked(checkpoint, records[key]["adapter"]["sha256"])
            if payload != feature_payload(records[key], values, validation, environment):
                raise ProtocolError("Partial feature values differ from original checkpoint extraction")
            result[key] = payload
    return result, checkpoints


def run(manifest_path, sidecar_path, selection_path, output_dir, cache_index=None, cache_index_sha256=None, *, resume=False):
    records = catalog(read_json(manifest_path), read_json(sidecar_path))
    plan = validate_plan(read_artifact(selection_path), records)
    checked_extractor_hash()
    smoke.prepare_cpu()
    environment = runtime()
    # Preflight every supplied selected cache entry before any download/training.
    checkpoints = {}
    if resume:
        if cache_index is not None or cache_index_sha256 is not None:
            raise ProtocolError("Resume uses only the original run, not another feature index")
        output = outside_git(output_dir)
        existing, checkpoints = resume_features(output, plan, records, environment)
    else:
        existing = cached_features(cache_index, cache_index_sha256, plan["selected"], records, environment)
        output = smoke.new_output_directory(output_dir)
        save_artifact(output / "selection.json", plan)
    feature_dir = output / "features"
    if not resume:
        feature_dir.mkdir()
    features, inventory, provenance = {}, {}, []
    for selected in plan["selected"]:
        key = selected["adapter_id"]
        record = records[key]
        if key in existing:
            payload = existing[key]
            origin = "revalidated_partial_run" if resume else "validated_feature_cache"
            checkpoint_path = checkpoints.get(key)
        else:
            checkpoint_path = checkpoints.get(key)
            if checkpoint_path is None:
                checkpoint_path = fetch_checkpoint(record["adapter"], output)
            validation, values = smoke.extract_checked(checkpoint_path, record["adapter"]["sha256"])
            payload = feature_payload(record, values, validation, environment)
            origin = "pinned_checkpoint_extraction"
        path = feature_dir / (key + ".json")
        if not (resume and key in existing):
            save_artifact(path, payload)
        inventory[key] = {"file": path.name, "sha256": smoke.sha256_file(path)}
        features[key] = payload
        provenance.append({"adapter_id": key, "origin": origin,
                           "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                           "source_metadata": record, "verification_issues": verification_issues(record["adapter"]),
                           "feature_file_sha256": inventory[key]["sha256"]})
        print(f"{key}: {selected['partition']} / {selected['label']} / {origin}; 10 finite features", flush=True)
    save_artifact(feature_dir / "index.json", {"kind": "exploratory_feature_index_v1", "entries": inventory})
    state, predictions = fit_classifier(plan, records, features, environment)
    save_artifact(output / "model.json", state)
    counts = {part: {label: sum(r["partition"] == part and r["label"] == label for r in plan["selected"])
                     for label in ("benign", "malicious")} for part in ("train", "test")}
    diagnostic = {}
    for part in ("train", "test"):
        rows = [r for r in predictions if r["partition"] == part]
        matrix = {name: sum(r["claimed_label_int"] == y and r["prediction"] == p for r in rows)
                  for name, y, p in (("tn", 0, 0), ("fp", 0, 1), ("fn", 1, 0), ("tp", 1, 1))}
        diagnostic[part] = {"confusion_matrix": matrix, "accuracy": (matrix["tn"] + matrix["tp"]) / len(rows)}
    report = {"kind": "exploratory_training_diagnostics_v1", "official_protocol_eligible": False,
              "metadata_status": "unverified", "selection": plan, "counts": counts,
              "training_recipe": training_recipe(42), "runtime": environment, "warnings": state["warnings"],
              "feature_provenance": provenance, "predictions": predictions,
              "diagnostic_only_at_fixed_probability_gt_0_5": diagnostic, "limitations": LIMITATIONS,
              "input_files_sha256": {"manifest": smoke.sha256_file(manifest_path), "sidecar": smoke.sha256_file(sidecar_path)},
              "implementation_sha256": {"training": smoke.sha256_file(__file__),
                                         "selection": smoke.sha256_file(Path(__file__).with_name("natour_selection.py")),
                                         "smoke_helpers": smoke.sha256_file(smoke.__file__)},
              "model_file_sha256": smoke.sha256_file(output / "model.json"),
              "cache_index_file_sha256": smoke.sha256_file(feature_dir / "index.json"),
              "reused_cache_index_sha256": cache_index_sha256}
    save_artifact(output / "report.json", report)
    print(f"EXPLORATORY training complete: {output / 'report.json'}", flush=True)
    print(json.dumps(diagnostic, sort_keys=True), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--sidecar", type=Path, required=True)
        p.add_argument("--output-dir", type=Path, required=True)
        if name == "plan":
            p.add_argument("--train-task", default="alpaca")
            p.add_argument("--test-task", default="code")
            p.add_argument("--adapter-id", action="append", help="Explicit full selection; repeat for optional expansion")
        else:
            p.add_argument("--selection", type=Path, required=True)
            p.add_argument("--reuse-cache-index", type=Path)
            p.add_argument("--reuse-cache-sha256")
            p.add_argument("--resume", action="store_true", help="Validate and finish an unfitted partial run in place; refuse completed outputs")
    args = parser.parse_args(argv)
    if args.command == "plan":
        records = catalog(read_json(args.manifest), read_json(args.sidecar))
        plan = make_plan(records, args.train_task, args.test_task, args.adapter_id)
        checked_extractor_hash()
        output = smoke.new_output_directory(args.output_dir)
        save_artifact(output / "selection.json", plan)
        print(json.dumps(plan, indent=2))
        print(f"Selection saved without checkpoint access: {output / 'selection.json'}")
    else:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        run(args.manifest, args.sidecar, args.selection, args.output_dir, args.reuse_cache_index, args.reuse_cache_sha256, resume=args.resume)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.exit(f"Exploratory pipeline aborted ({type(exc).__name__}): {exc}")
