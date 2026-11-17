# Phase 2: scoring, calibration and evaluation foundations

This is the current Phase 2 guide. `README.md` and `INTEGRATION.md` are preserved
unchanged as the Phase 1 design and handoff record. Their statements that scoring
and evaluation are future work describe that earlier phase. Phase 2 adds only
new files; it changes no existing experiment, result, dependency, CLI or source.

Implemented modules are `scoring.py`, `calibration.py`, `metrics.py`,
`evaluation.py`, and the separately imported `final_evaluation.py`. They require
only Python 3.10+ standard libraries. No PEFTGuard, Z-PEFT, spectral model,
training code, adapter downloader or checkpoint loader is included. The existing
10D formulas and experiment scripts are unchanged. The primary gate remains
Llama-2-7B base, rank 16, Q/V. All development and test inputs are synthetic.

## Stage boundaries and precise inputs

The starting inputs are a fully verified Phase 1 manifest and its saved split.
`bind_protocol(manifest, plan)` calls the existing `validate_split` and creates an
immutable `ProtocolBinding`. It contains `manifest_sha256`, `split_sha256`, sorted
`commitments` tuples `(adapter_id, partition, metadata_sha256, leakage_group)`,
and canonical `reservations_json` for evaluation/final holdout. Its `sha256`
property hashes these fields. No paths, labels, weights or features are copied
into the binding. This metadata-only step belongs to the custodian. It does not
load any adapter. The full manifest is not a detector input.

Every scoring/calibration/evaluation call receives the **complete verified
metadata records for exactly one authorized partition**, plus that binding.
Records have the unchanged Phase 1 schema. Verification checks the primary
setup, all field evidence, exact ID membership, duplicate IDs/sources/hashes and
each record's commitment. Missing, extra, duplicate, unverified or drifted records
fail before scoring. Do not reconstruct partial manifests from folder names.

| Public entry point | Required inputs and scope | Returns |
| --- | --- | --- |
| `score_development(detector, records, binding, partition)` | `partition` is train, validation or calibration only | `ScoreBatch` |
| `calibrate_benign(binding, records, scores)` | Full verified benign calibration view and its score batch | `FrozenThreshold` |
| `save_threshold(path, threshold, binding)` | New path; verified frozen threshold | Writes new JSON only |
| `load_threshold(path, binding)` | Explicit threshold artifact path | Reconstructed, verified `FrozenThreshold` |
| `score_evaluation(detector, records, binding, threshold)` | Complete evaluation view; matching frozen detector and threshold | `ScoreBatch` |
| `evaluate(binding, records, scores, threshold)` | Evaluation view and batch; already frozen threshold | JSON-compatible report |
| `save_evaluation(path, binding, records, scores, threshold)` | Same, plus a new output path | Writes and returns report |

There is no score-based split selection or threshold argument to `evaluate` other
than the immutable `FrozenThreshold`. No public development API accepts an
evaluation/final partition. The original `development_records` protection is
unchanged. Only explicit artifact load/save operations perform filesystem I/O;
no function here opens a path embedded in an adapter record. A future real detector
wrapper will perform authorized weight loading itself and must validate actual
checkpoint bytes against `AdapterInput.sha256` before using them.

## Detector interface and score format

Implement the structural `Detector` interface independently for each method:

```python
class TeamDetector:
    @property
    def spec(self) -> DetectorSpec:
        ...  # Immutable descriptor of this frozen implementation/model.

    def score(self, adapters: tuple[AdapterInput, ...]) -> dict[str, float]:
        ...  # Exactly one raw finite score for each supplied adapter_id.
```

`DetectorSpec` fields are:

| Field | Required value |
| --- | --- |
| `name` | Nonempty stable detector name/version |
| `source_sha256` | Lowercase SHA256 binding wrapper and detector implementation |
| `config_sha256` | Lowercase SHA256 of the frozen configuration |
| `state_sha256` | Lowercase SHA256 of all fitted model/scaler/reference state; for a stateless method, hash a documented canonical stateless descriptor |
| `raw_direction` | Exactly `higher_is_suspicious` or `lower_is_suspicious` |
| `fit_ids` | Sorted unique tuple of IDs used to fit any model, scaler, reference bank or feature selection; subset of train |
| `tuning_ids` | Sorted unique tuple of IDs used for hyperparameter/model selection; subset of validation |
| `provenance` | Nonempty evidence reference covering training, external pretraining and data exposure; use explicit no-fit evidence for a stateless method |

External checkpoint training or reference data cannot be hidden by empty fit IDs;
reviewers must verify the evidence and absence of shared heldout data before
admission. These hashes are supplied commitments, not automatic proof that a
wrapper correctly identifies its code or model. The spec is checked before and
after scoring and against calibration. The wrapper must implement pure inference,
keep its actual fitted state frozen, disable stochastic training behavior, and
include preprocessing/reference state in its fingerprint.

The shared scorer accepts verified full records but passes only frozen
`AdapterInput` objects to the detector. They contain `adapter_id`,
`path_or_source`, `source_revision`, `sha256`, `base_model`, `lora_rank`,
`lora_alpha`, and a sorted tuple of `target_modules`. Labels, task, attack,
payload, verification notes and lineage are withheld. Identifiers/locators can
still encode labels in their names: wrappers must use them only to identify/load
artifacts, never as predictive features. Production staging should use neutral
artifact names if necessary.

Return an ID-to-raw-score mapping with no omitted or extra IDs. Values must be
finite Python `int`/`float`, excluding booleans; explicitly convert tensor/NumPy
scalars in the wrapper. NaN, infinity, strings and overflow fail. Raw scores need
not be probabilities. The shared layer applies identity or a fixed negation
according to `raw_direction`, always producing **larger = more suspicious**.
It never estimates the sign from labels or AUROC, and stores both forms.

`ScoreBatch.to_dict()` returns this exact JSON-native structure:

```text
{
  protocol_sha256: <binding hash>,
  records_sha256: <hash of sorted (ID, metadata hash) pairs for the full partition>,
  partition: <train|validation|calibration|evaluation|final_holdout>,
  detector: {name, source_sha256, config_sha256, state_sha256,
             raw_direction, fit_ids: [...], tuning_ids: [...], provenance},
  scores: [{adapter_id, raw, suspiciousness}, ...]  # sorted unique IDs
}
```

All Python batch/spec/score objects are frozen, with tuple collections. For
explicit persistence use Phase 1 `write_new_json(path, batch.to_dict())`.
`ScoreBatch.from_dict(read_json(path))` reconstructs and checks the structure;
call `batch.verify(binding, partition)` to check membership. Calibration and
evaluation do this automatically. The `sha256` property binds the entire batch.
These are audit checks, not cryptographic signatures: someone able to replace
both data and commitments can fabricate artifacts. Protect the canonical run
artifacts outside the mutable development workspace for a real experiment.

## Clean-only calibration and the 5% target

Calibration selects **one adapter per verified leakage group** before any score
is requested: the lexicographically smallest stable adapter ID. The selected IDs
are determined solely by frozen metadata. `score_development(..., "calibration")`
scores just these representatives; `calibrate_benign` still requires the complete
calibration metadata view. Its input batch must contain exactly those selected
IDs, and every calibration record must be verified benign. This avoids counting
known correlated variants as independent calibration observations. Group labels
do not establish real independence; the custodian must verify lineage and the
sampling design. Do not change IDs after inspecting scores.

For the resulting `n` scores sorted ascending, the upper conformal quantile is:

```text
k = ceil(19 * (n + 1) / 20)      # one-based rank; exact integer arithmetic
threshold = sorted_scores[k-1]  # when k <= n
threshold = +infinity           # otherwise
predicted_malicious = suspiciousness > threshold
```

There is no interpolation or Youden/ROC search. A score tied with the threshold
is benign. Strict `>` is essential; `>=` can invalidate the guarantee with ties
and is rejected when loading threshold artifacts. At `n=19` the threshold is the
maximum; at `n=39` it is the 38th order statistic. For `n < 19` use positive
infinity and flag no finite score. The pure quantile helper also handles `n=0`
this way, while the fixed split protocol still rejects empty calibration cohorts.
Production predictions require finite input scores even with an infinite threshold.

This gives a marginal false-positive probability at most 5% **only under the
appropriate exchangeability assumption for calibration representatives and the
new benign observation**, with the scoring function fixed independently of those
calibration observations. It is not a guarantee that every finite test set has
FPR <= 5%, nor a guarantee on a shifted benign task distribution. See
[Angelopoulos and Bates, split conformal and outlier detection](https://arxiv.org/html/2107.07511v6).
The evaluator labels heldout FPR as observed and makes no shift guarantee.

`FrozenThreshold` contains only the immutable calibration batch; `value` and
decisions are derived from it. There is no constructor field/setter accepting an
arbitrary threshold or FPR target. No evaluation labels/scores enter calibration.
Changing an evaluation label violates its metadata commitment; changing evaluation
scores can change reported performance but cannot alter the threshold.

`save_threshold` writes `{threshold: <body>, threshold_sha256: <hash>}`. The body
contains `schema_version`, `method`, `target_fpr: 0.05`, `comparison: ">"`,
`score_convention`, `unit`, `n`, `k`, `threshold`, the entire `calibration` batch,
`calibration_sha256`, and an explicit exchangeability-only `guarantee` statement.
The `threshold` value is `{kind: "finite", value: <number>}` or
`{kind: "positive_infinity", value: null}`. JSON never contains Infinity/NaN.
Reload recomputes the operating point and compares the entire artifact, rejecting
changes to the target, decision rule, counts, hash, detector or split.

Save the threshold before evaluation and keep that artifact fixed for the run.
File creation is exclusive; even identical files cannot be overwritten. A new
output filename is not permission to tune after evaluation. The software cannot
prevent deliberate human selection among multiple separately created runs;
preregister and retain the complete audit history.

## Evaluation outputs and confidence intervals

Evaluation consumes complete authorized records, a matching `ScoreBatch`, and
the frozen threshold. It verifies identity, metadata, detector state/config/sign,
split membership and calibration binding before calculating decisions. It does
not refit anything. Reports include:

- Schema, partition, manifest/split/protocol, detector, score and threshold hashes;
  the operating point including strict comparison; and a final `report_sha256`.
- `overall`: metrics on all adapters assigned to that partition.
- `heldout_combined`: metrics on the union of the reserved benign tasks and
  malicious families/payloads, excluding incidental reservation companions.
- `heldout_benign`: `pooled` metrics and `by_task` metrics for benign adapters whose
  tasks were explicitly reserved in the policy.
- `heldout_malicious`: `pooled`, `by_attack_family` and `by_payload` metrics for
  malicious adapters matching a reserved family **or** payload. Each adapter is
  counted once in pooled metrics even if it matches both. Payload grouping covers
  actual payloads in this selected cohort; family grouping covers its families.
- `reservation_companions`: separate metrics for adapters moved with a lineage
  group or reserved task but not themselves satisfying the primary heldout criteria.
- Sorted `predictions`: ID, label, task, attack family, payload, leakage group,
  raw score, suspiciousness and boolean `predicted_malicious`.
- `interpretation` statements covering exchangeability, interval assumptions,
  repeated lineage, and absence of a simultaneous subgroup-coverage guarantee.

Every metrics object has `count`, `confusion_matrix: {tn, fp, fn, tp}`, `fpr`,
`recall`, `accuracy`, and `auroc`. The three rate objects contain `value`,
`numerator`, `denominator`, `independent_group_count`, `reason`, `ci95`,
`ci_method` and `ci_reason`. FPR is FP/(FP+TN), recall is TP/(TP+FN), accuracy is
(TP+TN)/N. A zero denominator produces null estimates/intervals and
`reason: "zero_denominator"`, never a misleading zero.

Eligible rate intervals are two-sided 95% Clopper-Pearson exact binomial
intervals, calculated by numerical inversion of binomial tails in the standard
library. Zero observed errors still have a positive upper bound. Tests include
analytic boundary cases and the [published SciPy exact-binomial reference example](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
Intervals assume independent Bernoulli trials under a common-rate model; pooled
heterogeneous or shifted task populations require careful interpretation. The
group count is the count of distinct **declared** lineage groups, not proof of
independence. If a rate's denominator contains repeated lineage groups, its point
estimate is reported but its binomial interval is suppressed with
`ci_reason: "repeated_lineage_groups_require_cluster_method"`. Cluster inference
is not implemented; do not relabel correlated rows to obtain a narrow interval.

AUROC uses rank comparisons with half credit for ties and is not sign-corrected
when below 0.5. Its object contains `value`, `reason`, `ci95` and `ci_reason`.
With one or zero classes the value is null with `both_classes_required`. AUROC
is not binomial; no binomial interval is attached, and AUROC bootstrap intervals
remain a later extension. Per-task benign-only and per-attack malicious-only
reports therefore have undefined AUROC by design. Empty companion sets are valid.

## Explicit final-assessment workflow

This phase does **not** authorize a real final-cohort release. The development
API, standard evaluator and existing protected split helpers still reject final
data. Only the separately imported `final_evaluation` module supports final
assessment, with this explicit sequence for a future authorized custodian:

1. Freeze all design/model/configuration decisions and the persisted calibration
   threshold. Keep a document of the decisions and hash it.
2. Obtain explicit final-assessment approval and record the approver and an actual
   approval reference. Call `write_final_release(new_path, binding, threshold,
   approved_by=..., approval_reference=..., frozen_decisions_sha256=...)`.
3. Recheck with `load_final_release(path, binding, threshold)`. The artifact
   contains `{release: {protocol_sha256, threshold_sha256,
   frozen_decisions_sha256, approved_by, approval_reference,
   purpose: "final_assessment_only_no_model_development"}, release_sha256}`.
4. Only then supply the authorized final metadata/artifacts to
   `score_final(detector, records, binding, threshold, release=release)` and
   `save_final_evaluation(new_path, binding, records, scores, threshold,
   release=release)`. The report also binds `final_release_sha256`.
5. Keep final observations out of all future development for this fixed protocol.

Missing/wrong releases are rejected before consuming supplied records or invoking
the detector. Releases cannot authorize development and are bound to exactly one
protocol and threshold. All output paths are exclusive-create. There is no
filesystem discovery, record loader, unlock boolean, or fallback to final data.
Final tests construct clearly synthetic release records and nonexistent fixture
locators only. No real final records are accessed in this implementation work.

The release is an auditable attestation, not an authenticated digital signature
or a one-use storage capability. It does not prove approval, prevent private-Python
function calls, or enforce one-shot analysis by a malicious caller. A real runner
must restrict final storage credentials to the custodian, verify approval outside
this library, keep the canonical release/report, and prevent adaptive reruns.

## Synthetic usage and test commands

From the repository root:

```powershell
python -B -m unittest discover -s shared_protocol/tests -v
```

The executable synthetic flow can also be exercised in Python without any files:

```python
from shared_protocol.tests.test_phase2 import setup_case, synthetic_view
from shared_protocol.scoring import score_development
from shared_protocol.calibration import calibrate_benign
from shared_protocol.evaluation import score_evaluation, evaluate

manifest, plan, binding, detector = setup_case()  # synthetic test fixtures ONLY
cal = synthetic_view(manifest, binding, "calibration")
scores = score_development(detector, cal, binding, "calibration")
threshold = calibrate_benign(binding, cal, scores)
evaluation = synthetic_view(manifest, binding, "evaluation")
scored = score_evaluation(detector, evaluation, binding, threshold)
report = evaluate(binding, evaluation, scored, threshold)
assert threshold.value == 18.0
assert report["overall"]["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
```

`synthetic_view` is a test fixture utility, not a production authorization API.
Do not use it to slice real evaluation/final records during development. Tests
cover old Phase 1 behavior plus score direction and immutable inputs, incomplete
metadata, calibration purity/representatives, quantile boundaries/ties/infinity,
interval numerical references, zero denominators, AUROC ties, group isolation,
frozen threshold files, model/metadata drift, no score-based tuning, deterministic
reports across processes, exclusive output creation and explicit final releases.

Legacy `bankCreation/testSet.py` and `evaluation/evaluate_test_set.py` are adapter
creation/evaluation workflows, not unit test suites. Running them would train or
access real experiments; they are not part of this synthetic test invocation.

## Integration handoff: Natour, Roua and Sandra

Each teammate must provide the items below for their actual implementation. The
repository does not establish detector ownership, so the mapping of Natour, Roua
and Sandra to PEFTGuard, Z-PEFT and the second spectral method remains unassigned.
No teammate code or behavior has been invented, and no messages have been sent.

- Pinned source revision/files for preprocessing, feature extraction, detector
  definition and inference; source/config/state hashes; dependencies and exact
  environment; license/attribution; a named owner for each method.
- A pure-inference wrapper with the interface above and an immutable `DetectorSpec`.
  Document raw score definition, class-index mapping and mathematical direction;
  return continuous finite scores rather than thresholded labels.
- Frozen training/scaler/reference-bank checkpoint artifacts with checksums, all
  fit and tuning IDs, external pretraining/data exposure provenance, and evidence
  that calibration, evaluation and final cohorts were never used for fitting.
- Verified Phase 1 adapter metadata: artifact locator/revision/hash; base model,
  task, attack family/payload/rate, rank/alpha/modules/seed, label mapping, dataset
  provenance, lineage groups, reviewer evidence and prior exposure history.
- A permitted synthetic or existing tiny parity fixture with expected raw scores
  and tolerances; no adapter download is needed for this handoff review.
- Exact rank-16 Q/V compatibility and tensor/layer/dtype/scaling requirements.
  PEFTGuard needs its real representation/reshape/head code; Z-PEFT needs its real
  statistic/reference/aggregation definition; the second spectral method needs
  its exact equations/features/classifier. Preserve the original 10D extractor's
  mathematical operations, feature order and score configuration.

Unavailable source, unknown label/lineage metadata, contaminated pretrained
checkpoints, or unsupported rank/projection layouts block **real integration**,
not this tested foundation. The existing Phase 1 `INTEGRATION.md` remains the
detailed original source checklist. Real cohort selection, model integration,
checkpoint loading, cluster confidence methods, AUROC bootstrap intervals and
storage-enforced final access are outside this phase.
