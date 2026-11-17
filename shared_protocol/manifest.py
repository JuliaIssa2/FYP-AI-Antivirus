"""Versioned manifest schema and explicit, evidence-backed verification gates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


PRIMARY_MODEL = "meta-llama/Llama-2-7b-hf"
METADATA_FIELDS = (
    "path_or_source", "source_revision", "base_model", "task", "attack_family",
    "payload", "poisoning_rate", "lora_rank", "lora_alpha", "target_modules",
    "seed", "label", "dataset_provenance", "sha256", "leakage_group",
)
PROVENANCE_FIELDS = {"dataset_id", "revision", "split", "preprocessing"}
STRING_FIELDS = set(METADATA_FIELDS) - {
    "poisoning_rate", "lora_rank", "lora_alpha", "target_modules", "seed",
    "dataset_provenance",
}
PLACEHOLDERS = {"unknown", "unverified", "tbd", "?", "null", "n/a"}


class ProtocolError(ValueError):
    """Input is incomplete, inconsistent, or unsafe for the fixed protocol."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _unique_keys(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: str | Path) -> dict:
    def invalid_constant(value: str) -> None:
        raise ProtocolError(f"Non-finite JSON number: {value}")
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_unique_keys,
                         parse_constant=invalid_constant)


def write_new_json(path: str | Path, value: dict) -> None:
    """Exclusive creation: never replace even an identical existing artifact."""
    content = canonical_json(value) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def text_value(value: object, field: str) -> None:
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or value.lower() in PLACEHOLDERS):
        raise ProtocolError(f"{field}: use a nonempty value or null for unknown")


def exact_keys(value: object, expected: set, name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ProtocolError(f"{name}: expected exactly {sorted(expected)}")


def _validate_record(row: dict) -> None:
    exact_keys(row, {"adapter_id", "verification", *METADATA_FIELDS}, "adapter")
    text_value(row["adapter_id"], "adapter_id")
    for field in METADATA_FIELDS:
        value = row[field]
        if value is None:
            continue
        if field in STRING_FIELDS:
            text_value(value, field)
        elif field in {"lora_rank", "seed"}:
            if type(value) is not int or value < (1 if field == "lora_rank" else 0):
                raise ProtocolError(f"{field}: invalid integer")
        elif field in {"poisoning_rate", "lora_alpha"}:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ProtocolError(f"{field}: expected finite number")
            if (field == "poisoning_rate" and not 0 <= value <= 1
                    or field == "lora_alpha" and value <= 0):
                raise ProtocolError(f"{field}: out of range")
        elif field == "target_modules":
            if not isinstance(value, list) or not value:
                raise ProtocolError("target_modules: expected nonempty list")
            for module in value:
                text_value(module, field)
            if len(set(value)) != len(value):
                raise ProtocolError("Duplicate target modules")
        elif field == "dataset_provenance":
            exact_keys(value, PROVENANCE_FIELDS, field)
            for key, item in value.items():
                if item is not None:
                    text_value(item, f"dataset_provenance.{key}")
    if row["label"] not in (None, "benign", "malicious"):
        raise ProtocolError("label: expected benign, malicious, or null")
    if row["sha256"] is not None and not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
        raise ProtocolError("sha256: expected 64 lowercase hex characters")
    if row["label"] == "benign":
        for field, expected in (("attack_family", "none"), ("payload", "none"),
                                ("poisoning_rate", 0)):
            if row[field] is not None and row[field] != expected:
                raise ProtocolError(f"Benign adapter has inconsistent {field}")
    if row["label"] == "malicious" and row["attack_family"] == "none":
        raise ProtocolError("Malicious adapter cannot have attack_family=none")
    verification = row["verification"]
    if not isinstance(verification, dict) or set(verification) - set(METADATA_FIELDS):
        raise ProtocolError("verification: expected evidence keyed by metadata field")
    for field, evidence in verification.items():
        exact_keys(evidence, {"verified_by", "evidence"}, f"verification.{field}")
        text_value(evidence["verified_by"], "verified_by")
        text_value(evidence["evidence"], "evidence")
        if row[field] is None or (field == "dataset_provenance"
                                 and any(v is None for v in row[field].values())):
            raise ProtocolError(f"Cannot verify unknown {field}")


def verification_issues(row: dict) -> list[str]:
    """List unknown/unverified fields; never infer metadata from names or labels."""
    issues = []
    for field in METADATA_FIELDS:
        value = row[field]
        if value is None or (field == "dataset_provenance"
                             and any(v is None for v in value.values())):
            issues.append(f"{field}: unknown")
        elif field not in row["verification"]:
            issues.append(f"{field}: unverified")
    return issues


def validate_manifest(manifest: dict, *, require_verified: bool = False) -> dict:
    """Validate metadata only. Never resolve, open, hash, or download adapters."""
    exact_keys(manifest, {"schema_version", "adapters"}, "manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ProtocolError("Unsupported manifest schema_version")
    rows = manifest["adapters"]
    if not isinstance(rows, list) or not rows:
        raise ProtocolError("Manifest must contain at least one adapter")
    seen = {field: set() for field in ("adapter_id", "sha256", "path_or_source")}
    for row in rows:
        _validate_record(row)
        for field, values in seen.items():
            value = row[field]
            if value is not None:
                if value in values:
                    raise ProtocolError(f"Duplicate {field}: {value}")
                values.add(value)
        if require_verified and (issues := verification_issues(row)):
            raise ProtocolError(f"{row['adapter_id']}: " + "; ".join(issues))
    return manifest


def require_primary(manifest: dict) -> None:
    validate_manifest(manifest, require_verified=True)
    for row in manifest["adapters"]:
        if (row["base_model"] != PRIMARY_MODEL or row["lora_rank"] != 16
                or set(row["target_modules"]) != {"q_proj", "v_proj"}):
            raise ProtocolError(f"{row['adapter_id']}: outside Llama-2-7B rank-16 Q/V primary setup")


def manifest_digest(manifest: dict) -> str:
    """Row and target-module order are immaterial; all evidence is bound."""
    rows = []
    for row in sorted(manifest["adapters"], key=lambda r: r["adapter_id"]):
        rows.append({**row, "target_modules": sorted(row["target_modules"])})
    return digest({"schema_version": manifest["schema_version"], "adapters": rows})


def load_manifest(path: str | Path, *, require_verified: bool = False) -> dict:
    return validate_manifest(read_json(path), require_verified=require_verified)
