"""Policy schema 2: only preregistered leakage groups may be drawn for validation."""
import unittest
from copy import deepcopy

from shared_protocol.manifest import METADATA_FIELDS, ProtocolError
from shared_protocol.splits import create_split, validate_split


def row(i, group, task, label, payload="none"):
    r = {"adapter_id": f"a{i:03d}", "path_or_source": f"p{i}", "source_revision": "r",
         "base_model": "meta-llama/Llama-2-7b-hf", "task": task,
         "attack_family": "none" if label == "benign" else "token", "payload": payload,
         "poisoning_rate": 0 if label == "benign" else 0.2, "lora_rank": 16, "lora_alpha": 32,
         "target_modules": ["q_proj", "v_proj"], "seed": i, "label": label,
         "dataset_provenance": {"dataset_id": "d", "revision": "v", "split": "s", "preprocessing": "p"},
         "sha256": f"{i:064x}", "leakage_group": group}
    r["verification"] = {f: {"verified_by": "test", "evidence": "synthetic"} for f in METADATA_FIELDS}
    return r


def cohort():
    rows, i = [], 0
    def add(group, task, label, payload="none"):
        nonlocal i; rows.append(row(i, group, task, label, payload)); i += 1
    for g in range(3):  # old, large mixed groups
        for _ in range(4): add(f"old-{g}", "alpaca", "malicious", "x")
        add(f"old-{g}", "alpaca", "benign")
    for g in range(8):  # new matched groups
        add(f"new-{g}", "alpaca", "benign"); add(f"new-{g}", "alpaca", "malicious", "x")
    for g in range(3): add(f"cal-{g}", "alpaca", "benign")
    add("ev-0", "squad", "benign"); add("ev-1", "alpaca", "malicious", "refusal")
    add("fh-0", "dolly", "benign"); add("fh-1", "alpaca", "malicious", "banana")
    return {"schema_version": 1, "adapters": rows}


def policy(version=2, **extra):
    p = {"schema_version": version, "seed": 0, "calibration_groups": 3, "validation_groups_per_stratum": 2,
         "evaluation": {"benign_tasks": ["squad"], "attack_families": [], "payloads": ["refusal"]},
         "final_holdout": {"benign_tasks": ["dolly"], "attack_families": [], "payloads": ["banana"]}}
    if version == 2:
        p["validation_eligible_groups"] = [f"new-{g}" for g in range(8)]
    p.update(extra)
    return p


class ValidationEligibilityTests(unittest.TestCase):
    def test_only_eligible_groups_reach_validation(self):
        m = cohort(); plan = create_split(m, policy())
        by_id = {r["adapter_id"]: r["leakage_group"] for r in m["adapters"]}
        val = {by_id[i] for i in plan["partitions"]["validation"]}
        train = {by_id[i] for i in plan["partitions"]["train"]}
        self.assertEqual(len(val), 2)
        self.assertTrue(all(g.startswith("new-") for g in val))
        self.assertTrue({"old-0", "old-1", "old-2"} <= train)
        validate_split(plan, m)  # reproducible from manifest + policy

    def test_assignment_is_deterministic_and_order_free(self):
        m = cohort(); m2 = deepcopy(m); m2["adapters"].reverse()
        p2 = policy(); p2["validation_eligible_groups"].reverse()
        self.assertEqual(create_split(m, policy())["partitions"], create_split(m2, p2)["partitions"])

    def test_schema_1_is_unchanged(self):
        m = cohort()
        with self.assertRaises(ProtocolError):  # v1 rejects the new key
            create_split(m, policy(version=1, validation_eligible_groups=["new-0"]))
        create_split(m, policy(version=1))  # and still runs as before

    def test_bad_eligibility_lists_fail(self):
        m = cohort()
        for groups, msg in ((["new-0", "nope"], "Unknown"), (["new-0", "cal-0", "cal-1", "cal-2"], "reserved or calibrated"),
                            (["new-0"], "Insufficient"), ([], "nonempty"), (["new-0", "new-0"], "Duplicate")):
            with self.assertRaisesRegex(ProtocolError, msg):
                create_split(m, policy(validation_eligible_groups=groups))


def task_cohort():
    """Old groups plus 4 new matched groups on each of three tasks."""
    m = cohort()
    rows = [r for r in m["adapters"] if not r["leakage_group"].startswith("new-")]
    i = 500
    for task in ("alpaca", "ag_news", "sst2"):
        for g in range(4):
            for label in ("benign", "malicious"):
                rows.append(row(i, f"new-{task}-{g}", task, label, "none" if label == "benign" else "x")); i += 1
    return {"schema_version": 1, "adapters": rows}


def policy3(per_task=2, per_stratum=6, **extra):
    p = policy(version=1, validation_groups_per_stratum=per_stratum)
    p.update({"schema_version": 3, "validation_groups_per_task": per_task,
              "validation_eligible_groups": [f"new-{t}-{g}" for t in ("alpaca", "ag_news", "sst2") for g in range(4)]})
    p.update(extra)
    return p


class PerTaskValidationTests(unittest.TestCase):
    def test_two_groups_per_task(self):
        m = task_cohort(); plan = create_split(m, policy3())
        by_id = {r["adapter_id"]: r for r in m["adapters"]}
        val = {by_id[i]["leakage_group"]: by_id[i]["task"] for i in plan["partitions"]["validation"]}
        self.assertEqual(sorted(__import__("collections").Counter(val.values()).items()),
                         [("ag_news", 2), ("alpaca", 2), ("sst2", 2)])
        self.assertTrue(all(g.startswith("new-") for g in val))
        validate_split(plan, m)

    def test_total_must_match_and_tasks_must_have_enough(self):
        m = task_cohort()
        with self.assertRaisesRegex(ProtocolError, "per-task draw gives 6"):
            create_split(m, policy3(per_stratum=4))
        with self.assertRaisesRegex(ProtocolError, "needs 5"):
            create_split(m, policy3(per_task=5, per_stratum=15))
        with self.assertRaisesRegex(ProtocolError, "invalid integer"):
            create_split(m, policy3(per_task=0))

    def test_multi_task_eligible_group_is_rejected(self):
        m = task_cohort()
        for r in m["adapters"]:
            if r["leakage_group"] == "new-sst2-0" and r["label"] == "benign":
                r["task"] = "alpaca"
        with self.assertRaisesRegex(ProtocolError, "spans several tasks"):
            create_split(m, policy3())


if __name__ == "__main__":
    unittest.main()
