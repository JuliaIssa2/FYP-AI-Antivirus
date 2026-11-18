"""Two explicitly authorized pipeline_check_v0 checkpoints; no scoring or fitting."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

from shared_protocol.manifest import (
    ProtocolError, digest, read_json, validate_manifest, verification_issues, write_new_json,
)
from shared_protocol.detectors.spectral.schema import (
    BASELINE_ID, BASELINE_SCHEMA, EXTRACTOR_SHA256, LEGACY_FEATURE_NAMES,
)

REPO_ID = "MhmdNat/adapterlab-adapters"
REVISION = "d2aa256bc5048cebeaa2d02c7206e4df9202ece9"
# This is the entire authorization scope, not selectors for a repository scan.
SELECTED = (
    ("nat-0812892dc58dfb88", "benign",
     "adapters/control/attackclean_modulesqv_rank16_taskalpaca_seed2/adapter_model.safetensors"),
    ("nat-05355e82af27bcc3", "malicious",
     "adapters/style/taskalpaca_attackcaps_rank16_modulesqv_seed1/adapter_model.safetensors"),
)
MODEL_NOTE = (
    "Natour's training configuration names NousResearch/Llama-2-7b-hf; the primary "
    "protocol requires meta-llama/Llama-2-7b-hf. Matching tensor dimensions do not "
    "establish identical base weights. No alias or verification is granted here."
)
# Match exactly the preserved extractor's optional adapter-name suffix.
LORA_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<factor>[AB])(?:\.[^.]+)?\.weight$")
LAYER = re.compile(r"(?:^|\.)layers\.(?P<index>\d+)\.")


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def select_records(manifest):
    """Read only the two selected metadata rows; never open manifest paths."""
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "adapters"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or not isinstance(manifest["adapters"], list)):
        raise ProtocolError("Expected exported shared_protocol v1 manifest")
    selected = []
    for adapter_id, label, filename in SELECTED:
        matches = [row for row in manifest["adapters"]
                   if isinstance(row, dict) and row.get("adapter_id") == adapter_id]
        if len(matches) != 1:
            raise ProtocolError(f"{adapter_id}: expected exactly one manifest row")
        row = matches[0]
        validate_manifest({"schema_version": 1, "adapters": [row]}, require_verified=False)
        locator = f"hf://datasets/{REPO_ID}@{REVISION}/{filename}"
        if (row["source_revision"] != REVISION or row["path_or_source"] != locator
                or row["label"] != label or row["lora_rank"] != 16
                or set(row["target_modules"] or []) != {"q_proj", "v_proj"}
                or row["sha256"] is None):
            raise ProtocolError(f"{adapter_id}: pinned source, label, rank, Q/V or SHA256 mismatch/unknown")
        # A source dataset split is not a protocol partition. Reject explicit protected
        # markers, while retaining all other provenance exactly as supplied.
        provenance = row["dataset_provenance"] or {}
        split = str(provenance.get("split") or "").lower()
        if any(word in split for word in ("final_holdout", "evaluation", "final holdout")):
            raise ProtocolError(f"{adapter_id}: protected cohort marker")
        selected.append(row)
    validate_manifest({"schema_version": 1, "adapters": selected}, require_verified=False)
    return selected


def checked_extractor():
    from shared_protocol.detectors.spectral import schema
    path = Path(schema.__file__).parent / "vendor" / "spectral_detector.py"
    if sha256_file(path) != EXTRACTOR_SHA256:
        raise ProtocolError("Preserved extractor SHA256 mismatch; refusing extraction")
    module = importlib.import_module("shared_protocol.detectors.spectral.vendor.spectral_detector")
    if tuple(module.feature_names()) != LEGACY_FEATURE_NAMES:
        raise ProtocolError("Preserved feature order mismatch")
    return module


def inspect_checkpoint(path, expected_sha256, *, hidden_size=4096):
    """Hash bytes BEFORE safetensors parsing. Small hidden_size is for unit fixtures only.

    The executable workflow always uses 4096 and rank 16. Never cast, rescale,
    transpose or rewrite checkpoint tensors here; the vendor owns extraction.
    """
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ProtocolError(f"Checkpoint SHA256 mismatch: expected {expected_sha256}, actual {actual}")
    from safetensors import safe_open
    import numpy as np
    pairs = {}
    details = []
    with safe_open(str(path), framework="numpy") as handle:
        for key in sorted(handle.keys()):
            match = LORA_KEY.fullmatch(key)
            module = match.group("module") if match else ""
            layer = LAYER.search(module)
            projection = module.rsplit(".", 1)[-1]
            if not match or layer is None or projection not in ("q_proj", "v_proj"):
                raise ProtocolError(f"Unexpected non-Q/V LoRA tensor key: {key}")
            index, factor = int(layer.group("index")), match.group("factor")
            if not 0 <= index < 32:
                raise ProtocolError(f"Layer outside Llama-2-7B: {key}")
            identity = (index, projection)
            pair = pairs.setdefault(identity, {})
            if factor in pair:
                raise ProtocolError(f"Duplicate LoRA factor for {identity}: {factor}")
            view = handle.get_slice(key)
            shape, dtype = tuple(view.get_shape()), view.get_dtype()
            allowed = ((16, hidden_size),) if factor == "A" else ((hidden_size, 16), (16, hidden_size))
            if shape not in allowed or dtype not in ("F32", "F64"):
                raise ProtocolError(f"Unsupported shape/dtype for {key}: {shape}, {dtype}; no conversion permitted")
            if not np.isfinite(handle.get_tensor(key)).all():
                raise ProtocolError(f"Nonfinite checkpoint tensor: {key}")
            pair[factor] = (shape, dtype)
            details.append({"key": key, "shape": list(shape), "dtype": dtype})
    for identity, pair in pairs.items():
        if set(pair) != {"A", "B"} or pair["A"][1] != pair["B"][1]:
            raise ProtocolError(f"Missing A/B factor or inconsistent dtype: {identity}")
    if not all((20, projection) in pairs for projection in ("q_proj", "v_proj")):
        raise ProtocolError("Missing layer-20 Q/V A/B tensors")
    return {"sha256": actual, "hash_matches_manifest": True, "tensor_count": len(details),
            "keys_dimensions_dtypes_finite_passed": True, "tensors": details}


def prepare_cpu():
    # In this standalone smoke process only: preserve the vendor's CPU branch.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    if torch.cuda.is_available():
        raise ProtocolError("CPU smoke requires a fresh process with CUDA_VISIBLE_DEVICES=-1")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def extract_checked(path, expected_sha256):
    validation = inspect_checkpoint(path, expected_sha256)
    extractor = checked_extractor()
    values = extractor.extract_spectral_features(path, layer_idx=20, projections=("q_proj", "v_proj"))
    if values.shape != (10,) or not all(math.isfinite(float(value)) for value in values):
        raise ProtocolError("Original extractor did not return ten finite features")
    if sha256_file(path) != expected_sha256:
        raise ProtocolError("Checkpoint changed during extraction")
    return validation, [float(value) for value in values]


def new_output_directory(path):
    path = Path(path).resolve()
    # Even an ignored path in a checkout is rejected. No .gitignore edits needed.
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ProtocolError("Weights, reports and cache must be outside every Git checkout")
    if path.exists():
        raise ProtocolError("Output directory already exists; choose a fresh run directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def run(manifest_path, output_dir):
    rows = select_records(read_json(manifest_path))
    prepare_cpu()
    checked_extractor()
    import numpy
    import torch
    import safetensors
    import huggingface_hub
    output = new_output_directory(output_dir)
    # Disable the optional Xet cache: all Hub cache bytes stay under this new run.
    # huggingface_hub's environment flags are import-time constants, so require
    # this setting at CLI process startup (main), before importing the library.
    report = {
        "schema_version": 1, "report_kind": "exploratory_checkpoint_smoke_only",
        "cohort": "pipeline_check_v0", "cohort_evidence": "explicit user-selected allowlist; not inferred from manifest",
        "official_protocol_eligible": False, "repo_id": REPO_ID, "source_revision": REVISION,
        "baseline_id": BASELINE_ID, "schema_sha256": BASELINE_SCHEMA.sha256,
        "feature_names": list(LEGACY_FEATURE_NAMES), "schema_feature_names": list(BASELINE_SCHEMA.names),
        "layer_idx": 20, "projections": ["q_proj", "v_proj"], "update_scaling": "unscaled B @ A",
        "extractor_sha256": EXTRACTOR_SHA256, "smoke_script_sha256": sha256_file(__file__),
        "selected_metadata_sha256": digest({"schema_version": 1, "adapters": rows}),
        "model_name_discrepancy": MODEL_NOTE, "metadata_verification_changed": False,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": numpy.__version__, "torch": torch.__version__,
                    "safetensors": safetensors.__version__, "huggingface_hub": huggingface_hub.__version__,
                    "device": "cpu", "torch_num_threads": 1, "deterministic_algorithms": True},
        "adapters": [],
    }
    for row, (adapter_id, _, filename) in zip(rows, SELECTED):
        checkpoint = Path(huggingface_hub.hf_hub_download(
            repo_id=REPO_ID, filename=filename, repo_type="dataset", revision=REVISION,
            cache_dir=str(output / "hub"), endpoint="https://huggingface.co",
        ))
        if not checkpoint.resolve().is_relative_to(output):
            raise ProtocolError("Downloader returned a checkpoint outside the isolated run cache")
        validation, values = extract_checked(checkpoint, row["sha256"])
        result = {"adapter_id": adapter_id, "label": row["label"], "label_status": "manifest claim, not reverified",
                  "source_revision": REVISION, "checkpoint_path": str(checkpoint),
                  "manifest_sha256": row["sha256"], "checkpoint_sha256": validation["sha256"],
                  "source_metadata": row, "verification_issues": verification_issues(row),
                  "validation": validation, "feature_values": values}
        report["adapters"].append(result)
        print(f"{adapter_id} ({row['label']}): SHA256 OK; {validation['tensor_count']} tensors validated")
        print(json.dumps(dict(zip(LEGACY_FEATURE_NAMES, values)), allow_nan=False))
    write_new_json(output / "report.json", report)
    print(f"EXPLORATORY ONLY: {output / 'report.json'}")
    print(MODEL_NOTE)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Previously exported manifest_natour.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory outside Git; never overwritten")
    args = parser.parse_args(argv)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    run(args.manifest, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never continue with partial data or emit a successful report on failure.
        sys.exit(f"Smoke test aborted ({type(exc).__name__}): {exc}")
