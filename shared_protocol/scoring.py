"""Frozen detector contracts and partition-scoped scoring; no built-in detector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
import re
from typing import Protocol

from .manifest import ProtocolError, canonical_json, digest, require_primary, text_value
from .splits import DEVELOPMENT, PARTITIONS, validate_split


def hash_value(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ProtocolError("Expected lowercase SHA256")


def finite_score(value: object) -> float:
    # Wrappers must explicitly convert framework scalars to Python numbers.
    if type(value) not in (int, float):
        raise ProtocolError("Scores must be finite Python numbers, not bool/strings")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ProtocolError("Score cannot be represented as a finite float") from exc
    if not math.isfinite(result):
        raise ProtocolError("Scores must be finite")
    return result


def _record_hash(row: dict) -> str:
    return digest({**row, "target_modules": sorted(row["target_modules"])})


@dataclass(frozen=True)
class ProtocolBinding:
    """Custodian-created commitments, not a copy of holdout records or paths.

    Construct using bind_protocol(). Integrity hashes are not access-control
    credentials. Keep full manifests and real final artifacts with the custodian.
    """

    manifest_sha256: str
    split_sha256: str
    # (adapter_id, partition, metadata_sha256, leakage_group)
    commitments: tuple[tuple[str, str, str, str], ...]
    # Canonical JSON of evaluation/final selector sets only.
    reservations_json: str

    def __post_init__(self) -> None:
        hash_value(self.manifest_sha256)
        hash_value(self.split_sha256)
        if type(self.commitments) is not tuple or not self.commitments:
            raise ProtocolError("Binding commitments must be a nonempty immutable tuple")
        ids, groups, record_hashes = set(), {}, set()
        seen_partitions = set()
        for entry in self.commitments:
            if type(entry) is not tuple or len(entry) != 4:
                raise ProtocolError("Invalid binding commitment")
            key, partition, record_hash, group = entry
            text_value(key, "adapter_id")
            text_value(group, "leakage_group")
            hash_value(record_hash)
            if partition not in PARTITIONS or key in ids or record_hash in record_hashes:
                raise ProtocolError("Duplicate or invalid binding commitment")
            if group in groups and groups[group] != partition:
                raise ProtocolError("Binding leakage group crosses partitions")
            ids.add(key)
            record_hashes.add(record_hash)
            groups[group] = partition
            seen_partitions.add(partition)
        if self.commitments != tuple(sorted(self.commitments)) or seen_partitions != set(PARTITIONS):
            raise ProtocolError("Binding needs canonical commitments for all five partitions")
        try:
            reservations = json.loads(self.reservations_json)
            if (set(reservations) != {"evaluation", "final_holdout"}
                    or canonical_json(reservations) != self.reservations_json):
                raise ProtocolError("Invalid canonical binding reservations")
            for selection in reservations.values():
                if set(selection) != {"benign_tasks", "attack_families", "payloads"}:
                    raise ProtocolError("Invalid reservation fields")
                for values in selection.values():
                    if type(values) is not list:
                        raise ProtocolError("Reservation values must be lists")
                    for value in values:
                        text_value(value, "reservation")
                    if values != sorted(set(values)):
                        raise ProtocolError("Reservations must be sorted and unique")
        except (TypeError, ValueError, AttributeError) as exc:
            raise ProtocolError("Invalid binding reservations") from exc

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    def ids(self, partition: str, *, representatives: bool = False) -> tuple[str, ...]:
        if partition not in PARTITIONS:
            raise ProtocolError("Unknown partition")
        members = [(key, group) for key, part, _, group in self.commitments if part == partition]
        if representatives:
            by_group = {}
            for key, group in sorted(members):
                by_group.setdefault(group, key)
            return tuple(sorted(by_group.values()))
        return tuple(sorted(key for key, _ in members))

    def records_digest(self, partition: str) -> str:
        return digest(sorted((key, record_hash) for key, part, record_hash, _ in self.commitments
                             if part == partition))


def bind_protocol(manifest: dict, plan: dict) -> ProtocolBinding:
    """Metadata-only custodian step. Does not open any artifact in the manifest."""
    validate_split(plan, manifest)
    partition_for = {key: part for part, ids in plan["partitions"].items() for key in ids}
    return ProtocolBinding(
        plan["manifest_sha256"], plan["split_sha256"],
        tuple(sorted((row["adapter_id"], partition_for[row["adapter_id"]],
                      _record_hash(row), row["leakage_group"]) for row in manifest["adapters"])),
        canonical_json({part: plan["policy"][part] for part in ("evaluation", "final_holdout")}),
    )


def verify_records(binding: ProtocolBinding, records: Sequence[dict], partition: str) -> tuple[dict, ...]:
    """Verify an already-authorized metadata view, including all evidence."""
    rows = list(records)
    require_primary({"schema_version": 1, "adapters": rows})
    if tuple(sorted(row["adapter_id"] for row in rows)) != binding.ids(partition):
        raise ProtocolError("Records must match the complete authorized partition")
    expected = {key: record_hash for key, part, record_hash, _ in binding.commitments if part == partition}
    if any(_record_hash(row) != expected[row["adapter_id"]] for row in rows):
        raise ProtocolError("Record metadata/evidence drift from frozen protocol")
    return tuple(sorted(rows, key=lambda row: row["adapter_id"]))


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    source_sha256: str
    config_sha256: str
    state_sha256: str
    raw_direction: str
    fit_ids: tuple[str, ...]
    tuning_ids: tuple[str, ...]
    provenance: str

    def __post_init__(self) -> None:
        text_value(self.name, "detector.name")
        text_value(self.provenance, "detector.provenance")
        for value in (self.source_sha256, self.config_sha256, self.state_sha256):
            hash_value(value)
        if self.raw_direction not in ("higher_is_suspicious", "lower_is_suspicious"):
            raise ProtocolError("Declare raw score direction explicitly")
        for ids in (self.fit_ids, self.tuning_ids):
            if type(ids) is not tuple:
                raise ProtocolError("Fit/tuning IDs must be immutable sorted tuples")
            for key in ids:
                text_value(key, "fit/tuning ID")
            if ids != tuple(sorted(set(ids))):
                raise ProtocolError("Fit/tuning IDs must be unique and sorted")

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    def verify(self, binding: ProtocolBinding) -> None:
        if (not set(self.fit_ids) <= set(binding.ids("train"))
                or not set(self.tuning_ids) <= set(binding.ids("validation"))):
            raise ProtocolError("Detector fit/tuning provenance leaks outside train/validation")

    def orient(self, value: float) -> float:
        value = finite_score(value)
        return value if self.raw_direction == "higher_is_suspicious" else -value


@dataclass(frozen=True)
class AdapterInput:
    """Verified input projection with no labels, task, attack or payload fields."""

    adapter_id: str
    path_or_source: str
    source_revision: str
    sha256: str
    base_model: str
    lora_rank: int
    lora_alpha: float
    target_modules: tuple[str, ...]


class Detector(Protocol):
    @property
    def spec(self) -> DetectorSpec: ...

    def score(self, adapters: tuple[AdapterInput, ...]) -> Mapping[str, float]:
        """Return exactly one raw score per adapter ID; never fit in this call."""
        ...


@dataclass(frozen=True)
class Score:
    adapter_id: str
    raw: float
    suspiciousness: float

    def __post_init__(self) -> None:
        text_value(self.adapter_id, "score.adapter_id")
        finite_score(self.raw)
        finite_score(self.suspiciousness)


@dataclass(frozen=True)
class ScoreBatch:
    protocol_sha256: str
    records_sha256: str
    partition: str
    detector: DetectorSpec
    scores: tuple[Score, ...]

    def __post_init__(self) -> None:
        hash_value(self.protocol_sha256)
        hash_value(self.records_sha256)
        if self.partition not in PARTITIONS or not isinstance(self.detector, DetectorSpec):
            raise ProtocolError("Invalid score batch partition/detector")
        if type(self.scores) is not tuple or any(not isinstance(s, Score) for s in self.scores):
            raise ProtocolError("Scores must be an immutable tuple of Score objects")
        ids = tuple(score.adapter_id for score in self.scores)
        if ids != tuple(sorted(set(ids))):
            raise ProtocolError("Score IDs must be unique and sorted")
        for score in self.scores:
            if score.suspiciousness != self.detector.orient(score.raw):
                raise ProtocolError("Score direction differs from declared detector convention")

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    def to_dict(self) -> dict:
        # Return JSON-native detached lists, not mutable aliases into frozen state.
        return json.loads(canonical_json(asdict(self)))

    @classmethod
    def from_dict(cls, data: dict) -> ScoreBatch:
        try:
            spec = dict(data["detector"])
            spec["fit_ids"] = tuple(spec["fit_ids"])
            spec["tuning_ids"] = tuple(spec["tuning_ids"])
            batch = cls(data["protocol_sha256"], data["records_sha256"], data["partition"],
                        DetectorSpec(**spec), tuple(Score(**row) for row in data["scores"]))
            if canonical_json(data) != canonical_json(batch.to_dict()):
                raise ProtocolError("Noncanonical or extra score-batch fields")
            return batch
        except (KeyError, TypeError) as exc:
            raise ProtocolError("Invalid score batch format") from exc

    def verify(self, binding: ProtocolBinding, partition: str) -> None:
        self.detector.verify(binding)
        if (self.partition != partition or self.protocol_sha256 != binding.sha256
                or self.records_sha256 != binding.records_digest(partition)):
            raise ProtocolError("Scores belong to another protocol/partition or record version")
        expected = binding.ids(partition, representatives=partition == "calibration")
        if tuple(score.adapter_id for score in self.scores) != expected:
            raise ProtocolError("Scores must cover exactly the authorized IDs")


def _score_authorized(detector: Detector, records: Sequence[dict], binding: ProtocolBinding,
                      partition: str) -> ScoreBatch:
    """Internal primitive; public entry points enforce stage authorization first."""
    spec = detector.spec
    if not isinstance(spec, DetectorSpec):
        raise ProtocolError("Detector must expose a frozen DetectorSpec")
    spec.verify(binding)
    rows = verify_records(binding, records, partition)
    selected = set(binding.ids(partition, representatives=partition == "calibration"))
    inputs = tuple(AdapterInput(
        row["adapter_id"], row["path_or_source"], row["source_revision"], row["sha256"],
        row["base_model"], row["lora_rank"], row["lora_alpha"], tuple(sorted(row["target_modules"])),
    ) for row in rows if row["adapter_id"] in selected)
    raw = detector.score(inputs)
    if detector.spec != spec:
        raise ProtocolError("Detector specification changed during scoring")
    if not isinstance(raw, Mapping) or set(raw) != selected:
        raise ProtocolError("Detector must return exactly one score per authorized adapter")
    scores = tuple(Score(key, finite_score(raw[key]), spec.orient(raw[key])) for key in sorted(raw))
    result = ScoreBatch(binding.sha256, binding.records_digest(partition), partition, spec, scores)
    result.verify(binding, partition)
    return result


def score_development(detector: Detector, records: Sequence[dict], binding: ProtocolBinding,
                      partition: str) -> ScoreBatch:
    if partition not in DEVELOPMENT:
        raise ProtocolError("Development scoring cannot access evaluation/final holdout")
    return _score_authorized(detector, records, binding, partition)
