"""Deterministic metadata-only splits with group isolation and locked holdout."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from .manifest import (
    PRIMARY_MODEL, ProtocolError, canonical_json, digest, exact_keys, manifest_digest, read_json,
    require_primary, text_value, write_new_json,
)


PARTITIONS = ("train", "validation", "calibration", "evaluation", "final_holdout")
DEVELOPMENT = ("train", "validation", "calibration")
SELECTORS = {"benign_tasks": "task", "attack_families": "attack_family", "payloads": "payload"}
ALGORITHM = "sha256-group-order-v1"


POLICY_KEYS = {"schema_version", "seed", "calibration_groups", "validation_groups_per_stratum",
               "evaluation", "final_holdout"}


def _policy(policy: dict) -> dict:
    """Schema 1: unchanged. Schema 2 adds `validation_eligible_groups`: the preregistered leakage groups that may
    be drawn for validation (e.g. only new matched groups); every other group stays in training. Schema 3 also
    adds `validation_groups_per_task`: draw exactly that many eligible groups per task (each eligible group must
    have a single task), and `validation_groups_per_stratum` must equal the resulting total, so the policy states
    both numbers explicitly."""
    version = policy.get("schema_version") if isinstance(policy, dict) else None
    if type(version) is not int or version not in (1, 2, 3):
        raise ProtocolError("Unsupported policy schema_version")
    extra = {2: {"validation_eligible_groups"}, 3: {"validation_eligible_groups", "validation_groups_per_task"}}
    exact_keys(policy, POLICY_KEYS | extra.get(version, set()), "policy")
    if version == 3 and (type(policy["validation_groups_per_task"]) is not int
                         or policy["validation_groups_per_task"] < 1):
        raise ProtocolError("policy.validation_groups_per_task: invalid integer")
    if version >= 2:
        groups = policy["validation_eligible_groups"]
        if not isinstance(groups, list) or not groups:
            raise ProtocolError("validation_eligible_groups: expected nonempty list")
        for group in groups:
            text_value(group, "validation_eligible_groups")
        if len(set(groups)) != len(groups):
            raise ProtocolError("Duplicate validation-eligible group")
    for field in ("seed", "calibration_groups", "validation_groups_per_stratum"):
        if type(policy[field]) is not int or policy[field] < (0 if field == "seed" else 1):
            raise ProtocolError(f"policy.{field}: invalid integer")
    normalized = deepcopy(policy)
    if version >= 2:
        normalized["validation_eligible_groups"] = sorted(normalized["validation_eligible_groups"])
    for partition in ("evaluation", "final_holdout"):
        selection = normalized[partition]
        exact_keys(selection, set(SELECTORS), partition)
        for key, values in selection.items():
            if not isinstance(values, list):
                raise ProtocolError(f"{partition}.{key}: expected list")
            for value in values:
                text_value(value, key)
                if key != "benign_tasks" and value == "none":
                    raise ProtocolError("Cannot hold out absence of an attack/payload")
            if len(set(values)) != len(values):
                raise ProtocolError(f"Duplicate selector in {partition}.{key}")
            selection[key] = sorted(values)
        if not selection["benign_tasks"] or not (selection["attack_families"] or selection["payloads"]):
            raise ProtocolError(f"{partition}: reserve benign tasks and attacks or payloads explicitly")
    for key in SELECTORS:
        if set(normalized["evaluation"][key]) & set(normalized["final_holdout"][key]):
            raise ProtocolError(f"Evaluation/final holdout selector overlap: {key}")
    return normalized


def _matches(row: dict, selection: dict) -> bool:
    # Reserve the entire task, including any malicious adapters on that task.
    return (row["task"] in selection["benign_tasks"]
            or row["label"] == "malicious" and (
                row["attack_family"] in selection["attack_families"]
                or row["payload"] in selection["payloads"]))


def create_split(manifest: dict, policy: dict) -> dict:
    """Create a plan from verified metadata, never from weights/features/scores.

    Holdout selection precedes allocation. Related adapters stay together;
    conflicting reservations fail rather than silently moving holdout members.
    """
    require_primary(manifest)
    policy = _policy(policy)
    rows = manifest["adapters"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["leakage_group"]].append(row)
    for partition in ("evaluation", "final_holdout"):
        for key, field in SELECTORS.items():
            label = "benign" if key == "benign_tasks" else "malicious"
            observed = {row[field] for row in rows if row["label"] == label}
            if missing := set(policy[partition][key]) - observed:
                raise ProtocolError(f"{partition}.{key}: no matching {label} adapters for {sorted(missing)}")

    assigned = {name: [] for name in PARTITIONS}
    available = []
    for group, members in sorted(groups.items()):
        destinations = {partition for partition in ("evaluation", "final_holdout")
                        if any(_matches(row, policy[partition]) for row in members)}
        if len(destinations) > 1:
            raise ProtocolError(f"Leakage group {group} bridges evaluation and final holdout")
        if destinations:
            assigned[destinations.pop()].append(group)
        else:
            available.append(group)

    def ordered(items: list[str], purpose: str) -> list[str]:
        return sorted(items, key=lambda group: (digest([ALGORITHM, policy["seed"], purpose, group]), group))

    def labels(group: str) -> tuple[str, ...]:
        return tuple(sorted({row["label"] for row in groups[group]}))

    benign_groups = ordered([g for g in available if labels(g) == ("benign",)], "calibration")
    count = policy["calibration_groups"]
    if len(benign_groups) < count:
        raise ProtocolError("Insufficient independent benign-only groups for calibration")
    assigned["calibration"] = benign_groups[:count]
    remaining = set(available) - set(assigned["calibration"])
    strata = defaultdict(list)
    for group in remaining:
        strata[labels(group)].append(group)
    if not strata:
        raise ProtocolError("No training or validation groups remain")
    eligible = policy.get("validation_eligible_groups")
    if eligible is not None:
        if unknown := set(eligible) - set(groups):
            raise ProtocolError(f"Unknown validation-eligible groups: {sorted(unknown)}")
        if unavailable := set(eligible) - remaining:
            raise ProtocolError(f"Validation-eligible groups were reserved or calibrated: {sorted(unavailable)}")
    for stratum, members in sorted(strata.items()):
        count = policy["validation_groups_per_stratum"]
        if eligible is None:
            members = ordered(members, "validation:" + ",".join(stratum))
            if len(members) <= count:
                raise ProtocolError(f"Insufficient groups for train/validation stratum {stratum}")
            validation = members[:count]
        elif "validation_groups_per_task" not in policy:  # schema 2: preregistered groups only
            pool = ordered([g for g in members if g in set(eligible)], "validation-eligible:" + ",".join(stratum))
            if pool and (len(pool) < count or len(members) <= count):
                raise ProtocolError(f"Insufficient validation-eligible groups for stratum {stratum}")
            validation = pool[:count]
        else:  # schema 3: the same number of eligible groups from every task
            per_task = policy["validation_groups_per_task"]
            by_task = defaultdict(list)
            for g in members:
                if g in set(eligible):
                    tasks = {row["task"] for row in groups[g]}
                    if len(tasks) != 1:
                        raise ProtocolError(f"Validation-eligible group {g} spans several tasks {sorted(tasks)}")
                    by_task[tasks.pop()].append(g)
            validation = []
            for task, pool in sorted(by_task.items()):
                pool = ordered(pool, f"validation-eligible:{','.join(stratum)}:{task}")
                if len(pool) < per_task:
                    raise ProtocolError(f"Task {task} has {len(pool)} validation-eligible groups, needs {per_task}")
                validation.extend(pool[:per_task])
            if validation and (len(validation) != count or len(members) <= count):
                raise ProtocolError(f"Stratum {stratum}: per-task draw gives {len(validation)} validation groups; "
                                    f"validation_groups_per_stratum is {count}")
        assigned["validation"].extend(validation)
        assigned["train"].extend(g for g in members if g not in set(validation))

    partitions = {}
    for partition, group_names in assigned.items():
        members = [row for group in group_names for row in groups[group]]
        expected_labels = {"benign"} if partition == "calibration" else {"benign", "malicious"}
        if {row["label"] for row in members} != expected_labels:
            raise ProtocolError(f"{partition}: requires labels {sorted(expected_labels)}")
        partitions[partition] = sorted(row["adapter_id"] for row in members)
    plan = {
        "schema_version": 1, "algorithm": ALGORITHM,
        "primary_setup": {"base_model": PRIMARY_MODEL, "lora_rank": 16,
                          "target_modules": ["q_proj", "v_proj"]},
        "manifest_sha256": manifest_digest(manifest), "policy": policy,
        "partitions": partitions,
        "adapter_sha256": {row["adapter_id"]: row["sha256"] for row in rows},
        "leakage_groups": {row["adapter_id"]: row["leakage_group"] for row in rows},
        "final_holdout_status": "locked_no_development_access",
    }
    plan["split_sha256"] = digest(plan)
    return plan


def validate_split(plan: dict, manifest: dict) -> None:
    """Reproduce the entire assignment, detecting tampering and manifest drift."""
    if not isinstance(plan, dict) or "policy" not in plan:
        raise ProtocolError("Invalid saved split")
    expected = create_split(manifest, plan["policy"])
    if canonical_json(plan) != canonical_json(expected):
        raise ProtocolError("Saved split differs from manifest/policy or deterministic assignment")


def save_split(path: str | Path, plan: dict, manifest: dict) -> None:
    validate_split(plan, manifest)
    write_new_json(path, plan)


def load_split(path: str | Path, manifest: dict) -> dict:
    plan = read_json(path)
    validate_split(plan, manifest)
    return plan


def development_records(manifest: dict, plan: dict, partition: str) -> list[dict]:
    """Return a detached metadata view for fitting/tuning/calibration only.

    No evaluation/final-holdout accessor is implemented in this foundation.
    A custodian can construct splits from metadata without reading artifacts.
    """
    if partition not in DEVELOPMENT:
        raise ProtocolError("Development access is limited to train, validation, calibration")
    validate_split(plan, manifest)
    members = set(plan["partitions"][partition])
    return deepcopy(sorted((row for row in manifest["adapters"]
                            if row["adapter_id"] in members), key=lambda row: row["adapter_id"]))
