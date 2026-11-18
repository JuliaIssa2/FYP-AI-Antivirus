"""Metadata-only, task-disjoint selection for the unverified older cohort."""
from collections import Counter
from pathlib import PurePosixPath
import re

from shared_protocol.manifest import ProtocolError, digest, validate_manifest, text_value
from exploratory.natour_spectral_smoke import REPO_ID, REVISION

COHORT = "pipeline_check_v0"
PREFIX = f"hf://datasets/{REPO_ID}@{REVISION}/"


def filename(row):
    source = row["path_or_source"]
    if not isinstance(source, str) or not source.startswith(PREFIX):
        raise ProtocolError("Checkpoint source must use the pinned HF dataset URI")
    result = source[len(PREFIX):]
    if not re.fullmatch(r"adapters/[A-Za-z0-9_-]+/[A-Za-z0-9_][A-Za-z0-9_.-]*/adapter_model\.safetensors", result):
        raise ProtocolError("Unsafe or unexpected checkpoint filename")
    if ".." in PurePosixPath(result).parts:
        raise ProtocolError("Unsafe checkpoint path")
    return result


def _index(rows, name):
    if not isinstance(rows, list):
        raise ProtocolError(f"{name}: adapters must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ProtocolError(f"{name}: invalid metadata row")
        key = row.get("adapter_id")
        text_value(key, "adapter_id")
        if key in result:
            raise ProtocolError(f"{name}: ambiguous duplicate adapter ID {key}")
        result[key] = row
    return result


def catalog(manifest, sidecar):
    """Join by ID before inspecting weights; exclude every non-v0 sidecar row."""
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "adapters"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or not isinstance(sidecar, dict) or type(sidecar.get("schema_version")) is not int
            or sidecar["schema_version"] != 1):
        raise ProtocolError("Expected exported v1 manifest and training sidecar")
    meta, training = _index(manifest["adapters"], "manifest"), _index(sidecar.get("adapters"), "sidecar")
    result, group_tasks = {}, {}
    for key, side in sorted(training.items()):
        if side.get("cohort") != COHORT:
            continue  # No paths or full metadata of phase2/other cohorts are consumed.
        if key not in meta:
            raise ProtocolError(f"Older-cohort adapter missing from manifest: {key}")
        row = meta[key]
        validate_manifest({"schema_version": 1, "adapters": [row]})
        if not re.fullmatch(r"nat-[a-f0-9]{16}", key):
            raise ProtocolError("Unexpected Natour ID")
        if (row["source_revision"] != REVISION or row["sha256"] is None
                or row["lora_rank"] != 16 or set(row["target_modules"] or []) != {"q_proj", "v_proj"}
                or row["label"] not in ("benign", "malicious")):
            raise ProtocolError(f"{key}: unknown/conflicting revision, hash, rank, projections or label")
        filename(row)
        for field in ("task", "leakage_group", "base_model"):
            text_value(row[field], field)
        expected_label = int(row["label"] == "malicious")
        if (type(side.get("label_int")) is not int or side["label_int"] != expected_label
                or side.get("base_model_as_trained") != row["base_model"]):
            raise ProtocolError(f"{key}: conflicting sidecar label or base model; no alias permitted")
        markers = [row.get("dataset_provenance") or {}, row, side]
        for source in markers:
            for field in ("split", "partition", "role"):
                value = str(source.get(field) or "").lower()
                if any(marker in value for marker in ("evaluation", "final_holdout", "final holdout", "official")):
                    raise ProtocolError(f"{key}: prohibited partition marker")
        group_tasks.setdefault(row["leakage_group"], set()).add(row["task"])
        result[key] = {"adapter": row, "training_metadata": side}
    if not result:
        raise ProtocolError("No explicitly marked pipeline_check_v0 adapters")
    if any(len(tasks) != 1 for tasks in group_tasks.values()):
        raise ProtocolError("A leakage group crosses tasks; task-disjoint division is unsafe")
    validate_manifest({"schema_version": 1, "adapters": [v["adapter"] for v in result.values()]})
    return result


def coverage(records):
    counts = Counter((item["adapter"]["task"], item["adapter"]["label"]) for item in records.values())
    return [{"task": task, "label": label, "count": n} for (task, label), n in sorted(counts.items())]


def make_plan(records, train_task="alpaca", test_task="code", explicit_ids=None):
    if train_task == test_task:
        raise ProtocolError("Train and test tasks must be distinct")
    for task in (train_task, test_task):
        for label in ("benign", "malicious"):
            if not any(v["adapter"]["task"] == task and v["adapter"]["label"] == label for v in records.values()):
                raise ProtocolError(f"Cannot form a task-disjoint two-class split: {task}/{label} missing")
    if explicit_ids is None:
        ids = []
        for task in (train_task, test_task):
            for label in ("benign", "malicious"):
                candidates = sorted(k for k, v in records.items()
                                    if v["adapter"]["task"] == task and v["adapter"]["label"] == label)
                if len(candidates) < 3:
                    raise ProtocolError("Default 12-adapter pilot needs three examples per task/class; supply explicit IDs for a smaller valid pilot")
                ids.extend(candidates[:3])
        mode = "three_per_task_class_sorted_adapter_id"
    else:
        if not isinstance(explicit_ids, list) or not explicit_ids or len(set(explicit_ids)) != len(explicit_ids):
            raise ProtocolError("Explicit adapter IDs must be nonempty and unique")
        ids, mode = sorted(explicit_ids), "explicit_adapter_ids"
    selected = []
    for key in sorted(ids):
        if key not in records:
            raise ProtocolError(f"Adapter is not authorized older-cohort metadata: {key}")
        row = records[key]["adapter"]
        if row["task"] not in (train_task, test_task):
            raise ProtocolError("Selected adapter is outside the two declared tasks")
        selected.append({"adapter_id": key, "partition": "train" if row["task"] == train_task else "test",
                         "task": row["task"], "leakage_group": row["leakage_group"], "label": row["label"],
                         "metadata_sha256": digest(records[key]), "checkpoint_sha256": row["sha256"]})
    for part in ("train", "test"):
        if {r["label"] for r in selected if r["partition"] == part} != {"benign", "malicious"}:
            raise ProtocolError("Each exploratory partition must contain both claimed classes")
    train_groups = {r["leakage_group"] for r in selected if r["partition"] == "train"}
    test_groups = {r["leakage_group"] for r in selected if r["partition"] == "test"}
    if train_groups & test_groups:
        raise ProtocolError("Leakage group overlap")
    return {"kind": "exploratory_task_disjoint_selection_v1", "official_protocol_eligible": False,
            "cohort": COHORT, "repo_id": REPO_ID, "source_revision": REVISION,
            "catalog_sha256": digest(records), "coverage": coverage(records), "selection_mode": mode,
            "train_task": train_task, "test_task": test_task, "selected": selected,
            "rationale": "Whole tasks have one role across all seeds; unselected adapters are excluded, not another partition. Selection uses metadata only, before features. Underlying cross-dataset text overlap is unknown."}


def validate_plan(plan, records):
    if not isinstance(plan, dict):
        raise ProtocolError("Invalid selection plan")
    try:
        mode = plan["selection_mode"]
        if mode not in ("three_per_task_class_sorted_adapter_id", "explicit_adapter_ids"):
            raise ProtocolError("Unknown selection mode")
        ids = [r["adapter_id"] for r in plan["selected"]] if mode == "explicit_adapter_ids" else None
        expected = make_plan(records, plan["train_task"], plan["test_task"], ids)
    except (KeyError, TypeError) as exc:
        raise ProtocolError("Malformed selection plan") from exc
    if digest(plan) != digest(expected):
        raise ProtocolError("Stale, modified or unsafe selection plan")
    return expected
