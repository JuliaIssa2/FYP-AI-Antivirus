"""Custodian export and trusted loading of development-only inputs; no split creation.

Bindings retain commitments for all five partitions. Only development records are
distributed. Integrity checks do not authenticate a custodian or prove the truth
of verification evidence; expected fingerprints must arrive independently.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path

from .manifest import ProtocolError, exact_keys, read_json, require_primary, write_new_json
from .scoring import AdapterInput, ProtocolBinding, bind_protocol, hash_value, verify_records
from .splits import DEVELOPMENT, development_records


BUNDLE_KEYS = {"binding", *DEVELOPMENT}


@dataclass(frozen=True)
class DevelopmentBundle:
    binding: ProtocolBinding
    train: tuple[dict, ...]
    validation: tuple[dict, ...]
    calibration: tuple[dict, ...]

    def records(self, partition: str) -> tuple[dict, ...]:
        if partition not in DEVELOPMENT:
            raise ProtocolError("Development bundles expose train, validation and calibration only")
        return deepcopy(getattr(self, partition))

    def to_dict(self) -> dict:
        # JSON-native detached structures also match the existing one-class parser.
        return json.loads(json.dumps(asdict(self), allow_nan=False))


def validate_development_bundle(data: dict, *, expected_binding_sha256: str) -> DevelopmentBundle:
    """Validate all metadata without accessing checkpoints or protected records.

    A developer checks correspondence to trusted commitments, not the protected
    manifest or the complete split assignment. The custodian validates those.
    """
    hash_value(expected_binding_sha256)
    exact_keys(data, BUNDLE_KEYS, "development bundle")
    raw_binding = data["binding"]
    exact_keys(raw_binding, {f.name for f in fields(ProtocolBinding)}, "binding")
    entries = raw_binding["commitments"]
    if (type(entries) is not list or any(type(entry) is not list or len(entry) != 4
                                        or any(type(value) is not str for value in entry)
                                        for entry in entries)):
        raise ProtocolError("Binding commitments must be JSON lists of four strings")
    try:
        binding = ProtocolBinding(**{**raw_binding, "commitments": tuple(tuple(e) for e in entries)})
    except (TypeError, KeyError) as exc:
        raise ProtocolError("Malformed binding") from exc
    if binding.sha256 != expected_binding_sha256:
        raise ProtocolError("Binding differs from the independently supplied fingerprint")

    views = {}
    for partition in DEVELOPMENT:
        if type(data[partition]) is not list:
            raise ProtocolError(f"{partition}: expected a complete list of records")
        views[partition] = verify_records(binding, data[partition], partition)
    # Per-view verification alone cannot detect checkpoint/locator reuse across views.
    require_primary({"schema_version": 1, "adapters": [r for p in DEVELOPMENT for r in views[p]]})
    for partition in DEVELOPMENT:
        expected = {"benign"} if partition == "calibration" else {"benign", "malicious"}
        if {row["label"] for row in views[partition]} != expected:
            raise ProtocolError(f"{partition}: requires labels {sorted(expected)}")
    return DevelopmentBundle(binding, **deepcopy(views))


def export_development_bundle(path: str | Path, manifest: dict, plan: dict) -> dict:
    """Export an already verified manifest and saved, validated split, exclusively.

    bind_protocol/development_records revalidate the full split through the
    installed planner. Unsupported policy schemas fail before output creation.
    This function never creates a split or adds verification evidence.
    """
    require_primary(manifest)
    binding = bind_protocol(manifest, plan)
    data = {"binding": json.loads(json.dumps(asdict(binding))),
            **{p: development_records(manifest, plan, p) for p in DEVELOPMENT}}
    validated = validate_development_bundle(data, expected_binding_sha256=binding.sha256)
    write_new_json(path, validated.to_dict())
    return {"bundle_file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "binding_sha256": binding.sha256}


def load_development_bundle(path: str | Path, *, expected_file_sha256: str,
                            expected_binding_sha256: str) -> DevelopmentBundle:
    """Check a trusted file hash before JSON parsing, then every development view."""
    hash_value(expected_file_sha256)
    hash_value(expected_binding_sha256)
    data = read_json(path, expected_sha256=expected_file_sha256)
    return validate_development_bundle(data, expected_binding_sha256=expected_binding_sha256)


def verify_development_checkpoints(bundle: DevelopmentBundle, root: str | Path, *,
                                   expected_binding_sha256: str):
    """Preflight explicitly staged neutral-ID files, including files already present.

    Validate every view before resolving or hashing any file. No glob, repository
    listing, download, tensor parsing, training or protected input access occurs.
    Files must remain under trusted custody after preflight; detectors also hash
    inputs when they consume them.
    """
    checked = validate_development_bundle(bundle.to_dict(), expected_binding_sha256=expected_binding_sha256)
    from .detectors.peftguard.converter import root_resolver, sha256_file

    resolver = root_resolver(root, restricted=True)
    for partition in DEVELOPMENT:
        for row in checked.records(partition):
            adapter = AdapterInput(row["adapter_id"], row["path_or_source"], row["source_revision"],
                                   row["sha256"], row["base_model"], row["lora_rank"],
                                   row["lora_alpha"], tuple(sorted(row["target_modules"])))
            path = resolver(adapter)
            if not path.is_file():
                raise ProtocolError(f"{adapter.adapter_id}: expected a staged regular checkpoint file")
            if sha256_file(path) != adapter.sha256:
                raise ProtocolError(f"{adapter.adapter_id}: staged checkpoint SHA256 mismatch")
    return resolver


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Validate custodian-supplied development inputs; never train")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--adapters", help="Also hash the already staged development files under this root")
    args = parser.parse_args(argv)
    try:
        bundle = load_development_bundle(args.bundle, expected_file_sha256=args.bundle_sha256,
                                         expected_binding_sha256=args.binding_sha256)
        if args.adapters is not None:
            verify_development_checkpoints(bundle, args.adapters, expected_binding_sha256=args.binding_sha256)
        print(json.dumps({"binding_sha256": bundle.binding.sha256,
                          "counts": {p: len(bundle.records(p)) for p in DEVELOPMENT}}, sort_keys=True))
    except (ProtocolError, OSError, ValueError) as exc:
        parser.exit(2, f"Development handoff rejected: {exc}\n")


if __name__ == "__main__":
    main()
