# Shared protocol foundations (v1)

This separate, standard-library-only module implements **manifest validation and
saved splits only**. It does not import legacy experiments, load adapter weights,
extract features, train models, select thresholds, or evaluate detectors.
Python 3.10+ is required. Existing source, results, requirements, and configuration
are unchanged. Templates contain no real adapter metadata and are not a runnable
primary experiment. No real split has been selected or generated.

## Inspection and proposed shared structure

The current `core/detector.py` and root README describe a four-projection 20D
detector. `experiments/padbench/spectral_detector.py` implements the five original
statistics on layer-20 Q/V in Q-then-V order, yielding the requested **10D** baseline.
`experiments/padbench/evaluate.py` uses training-only standardization/logistic
regression but chooses its threshold from labeled validation data. Its existing
train/validation/test split and results remain separate from this protocol.
`experiments/padbench/README.md` also documents a rank-256 cross-attack pilot.
None of those artifacts are silently promoted into the primary experiment.

The primary gate is **`meta-llama/Llama-2-7b-hf`, rank 16, exactly Q/V**.
This is the canonical base-model identifier for the agreed Llama-2-7B setup;
aliases require evidence-backed normalization. Chat and other variants are not
silently treated as equivalent. Alpha and training seeds are recorded rather
than guessed. Rank-256 smoke tests are exploratory; this version has no override
flag to include them. Any approved future change needs a separately versioned
protocol and new outputs.

```text
shared_protocol/
  manifest.py                 executable schema, verification, canonical hashes
  manifest.template.json      draft with explicit null unknowns
  splits.py                   fixed grouping, save/reload checks, development view
  policy.template.json        deliberately incomplete reservation policy
  __main__.py                 metadata-only CLI
  tests/test_foundations.py    synthetic, dependency-free tests
  INTEGRATION.md              team handoff and future evaluation contract

# Proposed later additions; NOT implemented in this phase:
  detectors/                 wrappers for spectral10d, peftguard, z_peft, spectral2
  scores.py                  immutable scored IDs, direction and fit provenance
  calibration.py             clean-only conformal threshold artifact
  evaluation.py              frozen-threshold metrics and confidence intervals
  runner.py                  stage-specific access and reproducibility checks

# Proposed new run storage, outside all previous experiment directories:
shared_runs/<protocol-version>/<run-id>/
  manifest.json, policy.json, splits.json, environment.json, source_hashes.json
  <detector>/model/, scores/, threshold.json, evaluation.json
```

## Manifest schema and verification

`manifest.py:validate_manifest` is the authoritative executable schema, including
cross-field checks. JSON has exactly `schema_version: 1` and a nonempty `adapters`
list. Each adapter must include every field below; unknown values are **JSON
`null`**, never invented defaults, empty strings, `"unknown"`, or inferred labels.
Missing keys and extra fields (including scores) are rejected.

| Field | Type / meaning |
| --- | --- |
| `adapter_id` | Nonempty stable unique string; required even for drafts |
| `path_or_source` | Canonical local checkpoint path or pinned remote artifact locator; not opened |
| `source_revision` | Immutable source revision; for local artifacts, an evidenced snapshot identifier |
| `base_model` | Verified canonical backbone identifier |
| `task` | Normalized task identity for whole-task reservations |
| `attack_family` | Normalized family; verified `"none"` for benign adapters |
| `payload` | Exact payload identity or consistently defined payload category; verified `"none"` for benign |
| `poisoning_rate` | Finite fraction in [0, 1], not a percentage; verified 0 for benign |
| `lora_rank` | Positive integer; primary requires 16 |
| `lora_alpha` | Positive finite number; do not assume 32 |
| `target_modules` | Unique nonempty string list; primary requires `q_proj`, `v_proj` |
| `seed` | Nonnegative integer adapter-training seed, distinct from split seed |
| `label` | Exactly `"benign"` or `"malicious"`; raw numeric mappings need verification |
| `dataset_provenance` | Object with `dataset_id`, `revision`, `split`, `preprocessing`; each a string or null |
| `sha256` | Lowercase 64-digit SHA256 of checkpoint file bytes, not its path, config, or features |
| `leakage_group` | Verified group of related adapters that must never cross partitions |
| `verification` | Map of metadata field names to `{ "verified_by": "...", "evidence": "..." }` |

Every metadata field needs its own verification entry before splitting. Evidence
should identify a reviewer and a pinned source/config/log/checksum record; it is
not an automatic truth certificate. Unknown fields cannot be marked verified.
Nested provenance must be complete before its verification entry is accepted.
Known values without evidence are retained as unverified drafts and cannot enter
the fixed protocol. `validate` reports these issues; `--require-verified` exits
with an error for them. The draft mode is an inventory aid, not authorization.

The checksum is supplied from prior trusted verification; this phase does not
read even one checkpoint to compute or recheck it. Future development loading
must compare actual bytes to the manifest hash *only after partition access is
authorized*. Sharded checkpoints require a versioned bundle-hash extension before
admission; do not substitute a config hash or pretend a partial file is complete.

Choose `leakage_group` using training provenance: paired benign/poisoned variants,
reused initialization runs, exact/near duplicates, or shared source examples that
could cause leakage must remain together. Assign each connected component one
group ID; identical dataset names alone do not necessarily imply shared samples.
Unrelated singleton groups also require verification. Published PADBench folder
indices alone do not prove independence. Unknown lineage blocks admission until
reviewed; do not assign fresh per-row groups just to make a split succeed.

## Split contract

1. A custodian validates verified metadata and the primary setup. Duplicate IDs,
   exact source locators, or checkpoint hashes anywhere in the manifest fail.
2. The policy explicitly reserves distinct benign tasks and unseen malicious
   attack families and/or payloads for **evaluation** and **final_holdout**.
   Each selected value must actually occur in the appropriate class. All adapters
   on a reserved benign task move together, including malicious ones. Attack and
   payload selectors apply to malicious adapters. Every related leakage-group
   member follows its reservation. A group spanning both holdouts fails.
3. Remaining groups are ordered by SHA256 of the algorithm version, split seed,
   purpose, and group ID. `calibration_groups` wholly benign groups are allocated
   first. Remaining groups are stratified as benign-only, malicious-only, or
   mixed. `validation_groups_per_stratum` groups from **each nonempty stratum** go
   to validation; the rest go to training. Counts refer to groups, not adapters.
4. All five partitions must be nonempty; train, validation, evaluation, and final
   holdout must each contain both classes. Calibration must contain only benign
   adapters. Insufficient cohorts fail explicitly: no ratio fallback, accidental
   in-distribution substitute, duplicated adapter, or weakening of reservations.
5. The saved artifact binds all metadata and verification evidence, policy,
   algorithm, IDs, hashes, and leakage groups. Reordering manifest rows or target
   modules has no effect. Save uses exclusive file creation; existing outputs are
   never overwritten, even if identical. Reload recomputes the full assignment.

Plan the policy from verified inventory and scientific goals before any detector
outputs are inspected. Freeze it once; do not search seeds or reservations for
better results. This deterministic planner cannot enforce human preregistration.
It does not establish that a cohort has enough independent tasks or attacks; if
only one task/family/payload exists, a valid grouped generalization study may be
impossible. Final holdout requires its own distinct groups, not a second draw
from an already inspected cohort. Previously inspected adapters cannot become
an "untouched" final holdout; the custodian must verify exposure history.

## Holdout access boundary

`development_records(manifest, plan, partition)` accepts only `train`,
`validation`, and `calibration`, returning detached metadata. It rejects
`evaluation` and `final_holdout` before validation or file access. There is no
final-holdout unlocking flag or loader. No code here opens paths stored in the
manifest. Metadata is necessarily inspected for reservation and overlap checks;
holdout weights, features, predictions, and metrics are never read.

The full manifest and split plan are **custodian artifacts**, not detector inputs.
Before integration, a runner must create stage-limited views and keep the final
holdout artifact store unavailable to development processes using storage access
controls. Python APIs and plain JSON are guardrails, not a security sandbox. The
future scorer receives only authorized inputs and IDs, not labels/attack metadata
that would reveal the target. Evaluation becomes available only after detector
and threshold freeze; final holdout needs a separate, explicitly authorized final
assessment implementation and remains untouched throughout model development.

## Usage

Copy templates to **new** manifest and policy paths; retain the originals.
Populate values and evidence manually from verified sources. Set a split seed,
group counts, and actual task/attack/payload reservations in the copied policy.
The null/empty policy template is deliberately rejected until completed.

From the repository root (replace `python` with your Python executable):

```powershell
python -B -m shared_protocol validate shared_protocol/manifest.template.json
python -B -m shared_protocol validate path/to/manifest.json --require-verified
python -B -m shared_protocol split path/to/manifest.json path/to/policy.json path/to/NEW-splits.json
python -B -m shared_protocol check path/to/manifest.json path/to/NEW-splits.json
python -B -m unittest discover -s shared_protocol/tests -v
```

Output parent directories must already exist. Commands perform no downloads,
training, model imports, or implicit scans of existing experiment folders.
Tests use entirely synthetic metadata and temporary files. They exercise
determinism across input ordering and process hash seeds, complete disjoint
assignments, duplicate IDs/hashes/sources, grouped reservations, benign-only
calibration, metadata/verification gates, rank-256 exclusion, access restrictions,
immutable saving, manifest drift, tampering, and CLI round trips.

Threshold-tuning and score-direction tests are intentionally deferred with the
evaluation interface. No claim is made that those behaviors have been tested or
implemented in this foundations-only phase; their acceptance criteria are in
[INTEGRATION.md](INTEGRATION.md).
