"""Strict saved-feature records and stage-scoped cache loading, without weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ...calibration import FrozenThreshold
from ...manifest import ProtocolError, canonical_json, digest, exact_keys, text_value
from ...scoring import AdapterInput, ProtocolBinding, finite_score, hash_value, verify_records
from ...splits import DEVELOPMENT
from .schema import BASELINE_SCHEMA, EXTRACTOR_SHA256, FeatureSchema, LEGACY_FEATURE_NAMES


@dataclass(frozen=True)
class FeatureRecord:
    adapter_id: str
    adapter_sha256: str
    source_revision: str
    schema_sha256: str
    extractor_sha256: str
    values: tuple[float, ...]
    source_record_sha256: str
    verified_by: str
    evidence: str

    def __post_init__(self):
        for name in ("adapter_id", "source_revision", "verified_by", "evidence"):
            text_value(getattr(self, name), name)
        for name in ("adapter_sha256", "schema_sha256", "extractor_sha256", "source_record_sha256"):
            hash_value(getattr(self, name))
        if type(self.values) is not tuple or not self.values:
            raise ProtocolError("Feature values must be an immutable nonempty tuple")
        for value in self.values:
            finite_score(value)

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    def to_dict(self) -> dict:
        return json.loads(canonical_json(asdict(self)))

    @classmethod
    def from_dict(cls, data: dict) -> FeatureRecord:
        exact_keys(data, {"adapter_id", "adapter_sha256", "source_revision", "schema_sha256",
                          "extractor_sha256", "values", "source_record_sha256", "verified_by", "evidence"}, "feature record")
        if type(data["values"]) is not list:
            raise ProtocolError("JSON feature values must be a list")
        return cls(**{**data, "values": tuple(data["values"])})


def from_padbench_record(data: dict, adapter: dict, *, source_record_sha256: str,
                         verified_by: str, evidence: str) -> FeatureRecord:
    """Convert ONE already-authorized collector record; no discovery or I/O.

    ID mapping is explicit: collector folder must equal the verified adapter ID.
    Lightweight legacy JSONL exports without hashes/revision cannot be promoted.
    """
    from ...manifest import require_primary
    require_primary({"schema_version": 1, "adapters": [adapter]})
    expected = {"folder": adapter["adapter_id"], "repo_revision": adapter["source_revision"],
                "layer": 20, "projections": ["q_proj", "v_proj"], "feature_names": list(LEGACY_FEATURE_NAMES),
                "formula": "original", "extractor_sha256": EXTRACTOR_SHA256,
                "rank": adapter["lora_rank"], "alpha": adapter["lora_alpha"],
                "base_model_name_or_path": adapter["base_model"]}
    if any(data.get(key) != value for key, value in expected.items()):
        raise ProtocolError("Legacy collector record identity/config/formula differs or is unknown")
    if (data.get("file_sha256", {}).get("adapter_model.safetensors") != adapter["sha256"]
            or data.get("rank_pattern") or data.get("alpha_pattern")
            or sorted(data.get("target_modules", [])) != ["q_proj", "v_proj"]):
        raise ProtocolError("Legacy checkpoint hash or effective LoRA configuration mismatch")
    raw_label = data.get("label")
    if type(raw_label) is not int or raw_label != (0 if adapter["label"] == "benign" else 1):
        raise ProtocolError("Legacy raw label conflicts with independently verified label")
    values = data.get("features")
    if not isinstance(values, list) or len(values) != 10:
        raise ProtocolError("Legacy baseline needs exactly ten saved features")
    return FeatureRecord(adapter["adapter_id"], adapter["sha256"], adapter["source_revision"],
                         BASELINE_SCHEMA.sha256, EXTRACTOR_SHA256, tuple(values),
                         source_record_sha256, verified_by, evidence)


@dataclass(frozen=True)
class FeatureSet:
    protocol_sha256: str
    records_sha256: str
    partition: str
    records: tuple[FeatureRecord, ...]
    allowed_inputs: tuple[AdapterInput, ...]
    schema: FeatureSchema = BASELINE_SCHEMA
    extractor_sha256: str = EXTRACTOR_SHA256

    def __post_init__(self):
        hash_value(self.protocol_sha256)
        hash_value(self.records_sha256)
        if self.partition not in (*DEVELOPMENT, "evaluation"):
            raise ProtocolError("Final holdout feature loading is not implemented in this integration")
        if self.schema != BASELINE_SCHEMA or self.extractor_sha256 != EXTRACTOR_SHA256:
            raise ProtocolError("Named baseline accepts only the pinned original layer-20 10D schema")
        if type(self.records) is not tuple or type(self.allowed_inputs) is not tuple:
            raise ProtocolError("Feature set collections must be immutable")
        ids = tuple(row.adapter_id for row in self.records)
        if not ids or ids != tuple(sorted(set(ids))) or ids != tuple(row.adapter_id for row in self.allowed_inputs):
            raise ProtocolError("Feature set must contain the exact sorted, unique allowed IDs")
        for row, adapter in zip(self.records, self.allowed_inputs):
            if (row.adapter_sha256 != adapter.sha256 or row.source_revision != adapter.source_revision
                    or row.schema_sha256 != self.schema.sha256 or row.extractor_sha256 != self.extractor_sha256
                    or len(row.values) != len(self.schema.names)):
                raise ProtocolError("Feature hash/revision/schema/dimension mismatch")

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    def get(self, adapter: AdapterInput) -> tuple[float, ...]:
        if adapter not in self.allowed_inputs:
            raise ProtocolError("Adapter is not in this authorized feature view")
        return next(row.values for row in self.records if row.adapter_id == adapter.adapter_id)


def _authorize(binding: ProtocolBinding, records: Sequence[dict], partition: str,
               threshold: FrozenThreshold | None) -> tuple[tuple[dict, ...], tuple[str, ...]]:
    # Reject final/unknown stages before consuming records or touching paths.
    if partition not in (*DEVELOPMENT, "evaluation"):
        raise ProtocolError("Protected partition: no final feature loader is provided")
    if partition == "evaluation":
        if not isinstance(threshold, FrozenThreshold):
            raise ProtocolError("Evaluation feature access requires a frozen threshold")
        threshold.verify(binding)
    rows = verify_records(binding, records, partition)
    return rows, binding.ids(partition, representatives=partition == "calibration")


def prepare_features(binding: ProtocolBinding, records: Sequence[dict], partition: str,
                     features: Sequence[FeatureRecord], *, threshold: FrozenThreshold | None = None) -> FeatureSet:
    rows, selected = _authorize(binding, records, partition, threshold)
    features = tuple(sorted(features, key=lambda row: row.adapter_id))
    if tuple(row.adapter_id for row in features) != selected:
        raise ProtocolError("Features must cover exactly the authorized IDs; no extras or omissions")
    inputs = tuple(AdapterInput(row["adapter_id"], row["path_or_source"], row["source_revision"], row["sha256"],
                                row["base_model"], row["lora_rank"], row["lora_alpha"], tuple(sorted(row["target_modules"])))
                   for row in rows if row["adapter_id"] in selected)
    return FeatureSet(binding.sha256, binding.records_digest(partition), partition, features, inputs)


def load_features(binding: ProtocolBinding, records: Sequence[dict], partition: str,
                  paths: Mapping[str, str | Path], file_sha256: Mapping[str, str], *,
                  threshold: FrozenThreshold | None = None) -> FeatureSet:
    """Read only explicitly mapped per-adapter feature JSONs, never a whole cache.

    Expected file checksums must be supplied from a trusted feature inventory.
    No glob, folder scan, raw checkpoint read or final-cohort loading occurs.
    """
    _, selected = _authorize(binding, records, partition, threshold)
    if set(paths) != set(selected) or set(file_sha256) != set(selected):
        raise ProtocolError("Paths/checksums must match authorized feature IDs before any reads")
    for key in selected:
        hash_value(file_sha256[key])
    features = []
    for key in selected:
        # Parse the same bytes that were hashed; reject duplicate keys/nonfinite JSON.
        content = Path(paths[key]).read_bytes()
        if hashlib.sha256(content).hexdigest() != file_sha256[key]:
            raise ProtocolError("Saved feature file checksum mismatch")
        from ...manifest import _unique_keys
        def invalid_constant(value):
            raise ProtocolError(f"Nonfinite JSON constant: {value}")
        row = FeatureRecord.from_dict(json.loads(content, object_pairs_hook=_unique_keys, parse_constant=invalid_constant))
        if row.adapter_id != key:
            raise ProtocolError("Mapped feature file belongs to another adapter")
        features.append(row)
    return prepare_features(binding, records, partition, features, threshold=threshold)
