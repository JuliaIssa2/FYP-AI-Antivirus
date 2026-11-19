"""Trusted bootstrap for a single one-class stage. Invoke with python -I -S -B."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import tempfile


def source_files(root):
    paths = sorted((root / "shared_protocol").rglob("*.py"))
    if not paths or any(path.is_symlink() or not path.resolve().is_relative_to(root) for path in paths):
        raise ValueError("Expected a local shared_protocol source tree without source symlinks")
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}


def source_digest(files):
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_stage(args, pending, config=None):
    from shared_protocol.calibration import calibrate_benign, load_threshold, save_threshold
    from shared_protocol.detectors.oneclass.detector import OneClassDetector
    from shared_protocol.detectors.oneclass.train import OneClassConfig, fit
    from shared_protocol.detectors.peftguard.converter import root_resolver
    from shared_protocol.evaluation import save_evaluation, score_evaluation
    from shared_protocol.manifest import ProtocolError, exact_keys, read_json
    from shared_protocol.scoring import ProtocolBinding, score_development, verify_records

    bundle = read_json(args.bundle)
    exact_keys(bundle, {"binding", "train", "validation", "calibration"}, "development bundle")
    fields = dict(bundle["binding"])
    fields["commitments"] = tuple(tuple(entry) for entry in fields["commitments"])
    binding = ProtocolBinding(**fields)
    train = verify_records(binding, bundle["train"], "train")
    resolver = root_resolver(args.adapters, restricted=True)
    if args.stage == "fit":
        validation = verify_records(binding, bundle["validation"], "validation")
        return Path(fit(binding, train, validation, config or OneClassConfig(), pending,
                        resolver=resolver)["model_file"])

    detector = OneClassDetector.from_model(args.model, binding, train, expected_file_sha256=args.model_sha256,
                                           resolver=resolver)
    if args.stage == "calibrate":
        records = verify_records(binding, bundle["calibration"], "calibration")
        if any(row["label"] != "benign" for row in records):
            raise ProtocolError("Calibration view must be benign-only")
        scores = score_development(detector, records, binding, "calibration")
        path = pending / "threshold.json"
        save_threshold(path, calibrate_benign(binding, records, scores), binding)
        return path

    threshold = load_threshold(args.threshold, binding)
    if threshold.sha256 != args.threshold_sha256:
        raise ProtocolError("Threshold does not match the independently recorded digest")
    if args.release is None:
        raise ProtocolError("A custodian-provided evaluation release is required")
    release = read_json(args.release)
    exact_keys(release, {"protocol_sha256", "threshold_sha256", "records"}, "evaluation release")
    if (release["protocol_sha256"], release["threshold_sha256"]) != (binding.sha256, threshold.sha256):
        raise ProtocolError("Evaluation release does not match the frozen protocol and threshold")
    records = verify_records(binding, release["records"], "evaluation")
    scores = score_evaluation(detector, records, binding, threshold)
    path = pending / "evaluation.json"
    save_evaluation(path, binding, records, scores, threshold)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("identity", "fit", "calibrate", "evaluate"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source")
    for option in ("bundle", "adapters", "output", "model", "threshold", "release"):
        parser.add_argument(f"--{option}", type=Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--threshold-sha256")
    parser.add_argument("--dependency-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    files = source_files(args.source.resolve())
    identity = source_digest(files)
    if args.stage == "identity":
        print(identity)
        return
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        parser.error("Launch each stage with python -I -S -B")
    if identity != args.expected_source:
        parser.error("Source snapshot does not match the expected SHA256")
    required = ["bundle", "adapters", "output"]
    if args.stage != "fit":
        required += ["model", "model_sha256"]
    if args.stage == "evaluate":
        required += ["threshold", "threshold_sha256", "release"]
    if any(getattr(args, key) is None for key in required):
        parser.error("Missing inputs for the selected stage")

    # The private snapshot has source files only. Isolated mode ignores the
    # notebook directory and PYTHONPATH; no-site mode skips .pth startup hooks.
    deps = [str(p.resolve()) for p in args.dependency_dir]
    deps.extend(sysconfig.get_path(key) for key in ("purelib", "platlib"))
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory(dir=args.output.parent) as pending:
        root = Path(tmp)
        for name, value in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            path.chmod(0o400)
        if source_digest(source_files(root)) != identity:
            raise RuntimeError("Source snapshot changed before import")
        if any(key == "shared_protocol" or key.startswith("shared_protocol.") for key in sys.modules):
            raise RuntimeError("Implementation was imported before verification")
        sys.pycache_prefix = str(root / "empty-bytecode-cache")
        sys.path[:0] = [str(root), *dict.fromkeys(deps)]
        result = run_stage(args, Path(pending))
        for key, module in tuple(sys.modules.items()):
            if key == "shared_protocol" or key.startswith("shared_protocol."):
                if not Path(module.__file__).resolve().is_relative_to(root / "shared_protocol"):
                    raise RuntimeError("Implementation resolved outside the approved source snapshot")
        if source_digest(source_files(root)) != identity:
            raise RuntimeError("Source snapshot changed before artifact acceptance")
        data = result.read_bytes()
        # Publish the verified worker artifact exclusively from a staging
        # directory on the destination filesystem.
        os.link(result, args.output)
        receipt = {"source_sha256": identity, "file_sha256": hashlib.sha256(data).hexdigest()}
        if args.stage == "calibrate":
            receipt["threshold_sha256"] = json.loads(data)["threshold_sha256"]
        print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
