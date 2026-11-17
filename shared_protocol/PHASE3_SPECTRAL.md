# Phase 3: named spectral baseline integration

This integration adds `shared_protocol.detectors.spectral` without changing any
Phase 1/2 file or the original PADBench project. The original method is named
**`padbench_spectral_layer20_qv_10d_lr_c01_v1`**. Other teammates' detectors remain
separate and are not implemented here. Saved features and frozen classifier
state can now feed the existing shared scoring/calibration/evaluation APIs.
There is no training function, download function or live checkpoint loader.

## Preserved baseline and separate evaluation protocols

The baseline has exactly ten features, from zero-based layer **20** (the 21st
transformer layer), in this order:

```text
q_proj: sigma1, frobenius_norm, energy_concentration, entropy, kurtosis
v_proj: sigma1, frobenius_norm, energy_concentration, entropy, kurtosis
```

The preserved extractor uses QR/SVD on the LoRA factors and unscaled `B @ A`.
It retains singular-value entropy normalization, float64 flattened-update
kurtosis, the original variance correction, epsilon, fallback B orientation and
float32 output. It does not add alpha/rank scaling, different features, layer
averaging, or stable-kurtosis ablations. The original source is copied byte-for-
byte into `detectors/spectral/vendor/spectral_detector.py`; it is retained for
reproduction/provenance and **not imported by the saved-feature pipeline**.

The original model recipe is retained: convert saved features to float64,
fit `StandardScaler()` on train only, then fit
`LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced",
random_state=split_seed)` on standardized train features. Original effective
defaults are recorded explicitly: L2, lbfgs, fit intercept, tolerance 1e-4.
The score is `predict_proba(... )[:, 1]` with verified class order `[0, 1]` and
1=malicious. This phase imports **already fitted** state; it never calls `fit`.
No source/result file in the old project is changed.

There are two deliberately distinct evaluation identifiers:

| Identifier suffix | Operating threshold | Status |
| --- | --- | --- |
| `__legacy_validation_threshold` | Original validation separation margin or Youden J; `score >= threshold` | Historical method/results stay in the original project |
| `__shared_conformal_v1` | Existing benign-only 5% conformal calibration; strict `score > threshold` | Shared pipeline integration |

The shared protocol is a new comparison under fixed splits and clean calibration,
not a claim that historical results or thresholds were reproduced. The new adapter
does not import, rerun or relabel the old threshold function. Existing rank-256
smoke tests remain exploratory; primary metadata validation still requires
Llama-2-7B base, rank 16, Q/V.

## Source and attribution

`detectors/spectral/PROVENANCE.json` pins the inspected extractor, evaluation
script and original core detector by SHA256. The old `experiments/` directory
was untracked, so its observed Git HEAD alone is **not** the source revision for
those files. The exact file hashes are authoritative. Only the extractor and
its original LICENSE/NOTICE were copied, not the old repository or its data.

The original method is credited to **David Puertolas Merenciano** (primary author),
Ekaterina Vasyagina, Javier Ferrando, Kevin Zhu and Maheep Chaudhary, *Detecting
Backdoored LoRAs from Weights Alone*. The original Apache-2.0 LICENSE and NOTICE
are retained under `vendor/`. New code supplies validation, saved-feature and
frozen-state boundaries; it does not rewrite the spectral formulas.

## Saved feature format

Use one JSON file per adapter, with the exact `FeatureRecord` keys:

```text
adapter_id                 verified manifest ID
adapter_sha256             actual checkpoint file hash from the manifest
source_revision            exact verified manifest source revision
schema_sha256              BASELINE_SCHEMA.sha256
extractor_sha256           pinned original extractor SHA256
values                     list of exactly ten finite Python numbers, in baseline order
source_record_sha256       checksum of the original feature record or extraction receipt
verified_by                actual reviewer identity
evidence                   pinned extraction/config/hash verification reference
```

Do not fill unknown evidence or hashes with expected values. A feature schema
hash asserts order/formula identity; reviewers must verify that claim from the
extractor and original records. The loader verifies saved-file checksums supplied
from a trusted inventory; it never reads checkpoints to recompute their hashes.
`from_padbench_record` converts a **single already-authorized** rich collector
record in memory. It checks folder/ID, revision, checkpoint hash, layer,
projections, exact feature names, formula, extractor hash, base model, rank,
alpha, effective target modules and independently verified label mapping. It
preserves the values without fitting, rescaling or guessing metadata.
Folder and manifest ID must match; any renaming needs an explicit audited
conversion before using this helper. Lightweight `extract_features.py` JSONL
rows lack required hashes/revisions and cannot be silently promoted.

Use `prepare_features(binding, verified_records, partition, feature_records)` for
already-authorized in-memory records. Use
`load_features(binding, verified_records, partition, paths_by_id,
expected_file_sha256_by_id)` for explicitly mapped per-adapter JSONs. Both validate
the complete verified partition metadata. For calibration they accept only the
Phase 2 representatives, one lexicographically first adapter per lineage group.
No extra IDs, omitted IDs, wrong hashes, mixed schemas or feature dimensions are
allowed. Loaders do not glob directories or read a whole combined feature cache.

Evaluation loading additionally requires `threshold=frozen_threshold` and checks
its protocol binding **before reading feature files**. `final_holdout` is rejected
unconditionally, even with a threshold. A real final-feature provider requires
separate future authorization/integration with the custodian's existing final
workflow; there is no bypass flag in this phase.

## Frozen model state: no fitting or pickle loading

`FrozenSpectralModel` stores canonical JSON as immutable state. Its exact body is:

```text
schema_version: 1
artifact_kind: fitted_baseline | synthetic_fixture
baseline_id, schema_sha256, extractor_sha256
protocol_sha256, split_seed, training_recipe
fit_ids: sorted list of ALL shared train IDs
fit_records_sha256: binding.records_digest("train")
fit_features_sha256: complete verified train FeatureSet.sha256
scaler: {mean: [10], scale: [10], var: [10], n_samples_seen: train count}
classifier: {classes: [0,1], coef: [10], intercept: finite number, n_iter: 1..1000}
software: {python, numpy, scipy, sklearn}  # exact actual version strings
verification: {verified_by, evidence}
```

`training_recipe(split_seed)` supplies the exact recipe object; every value is
checked on import. Obtain `split_seed` from the fixed split policy and supply it
explicitly when importing. The binding fingerprint protects the split, but does
not itself expose its seed; the supplied seed must come from that saved policy.
No classes, scaler parameters or fitted weights are inferred from evaluation or
calibration data. The model must match the entire current train partition;
historical models fitted on a different old split cannot simply be renamed.

`export_fitted_model(scaler, classifier, binding, verified_train_records,
train_features, split_seed=..., verified_by=..., evidence=...)` snapshots existing
scikit-learn objects. It checks types, recipe, dimensions, class mapping and fit
scope, and records actual software versions. It never trains them. For tests only,
use `artifact_kind="synthetic_fixture"` when attributes were assigned by hand.
Evidence must confirm original training IDs/features, no extra sample weights,
no validation refit, no external reference contamination and the correct seed.
Fitted attributes cannot prove the history of a model; this requires review.

`save_model(new_path, model, binding)` uses exclusive creation and writes
`{model: <body>, model_sha256: <canonical digest>}`. `load_model(path, binding,
split_seed=..., expected_file_sha256=...)` verifies the expected file checksum,
internal digest and metadata binding. No pickle/joblib model loader is provided.
Do not overwrite an old trained model or calibration artifact.

Production inference defaults to `backend="sklearn"`. It reconstructs the two
estimators from frozen arrays, then calls the same `StandardScaler.transform`
and `LogisticRegression.predict_proba` operations as the baseline. It requires
the model's exact recorded Python/NumPy/SciPy/scikit-learn versions and fails
instead of silently installing dependencies, accepting version drift or switching
backends. These are inference-only operations; no `.fit` or `.partial_fit` runs.

The explicitly selected `synthetic_analytic` backend evaluates standardized
binary-logit probabilities for **hand-authored synthetic states only**. It is
rejected for `fitted_baseline` artifacts, is fingerprinted separately, and uses a
detector name ending `__synthetic_fixture`. It allows the contract tests to run
without numerical dependencies and must not be used as an unreported production
substitute. Its formula follows the [scikit-learn binary probability definition](https://github.com/scikit-learn/scikit-learn/blob/1.3.2/sklearn/linear_model/_base.py)
and [standardization definition](https://github.com/scikit-learn/scikit-learn/blob/1.3.2/sklearn/preprocessing/_data.py).

## Shared scoring and evaluation flow

The following is a usage outline for previously authorized inputs, not a command
to obtain adapters or train a new model:

```python
from shared_protocol.detectors.spectral import SpectralDetector, load_features, load_model
from shared_protocol.scoring import score_development
from shared_protocol.calibration import calibrate_benign, save_threshold
from shared_protocol.evaluation import score_evaluation, evaluate

model = load_model(model_path, binding, split_seed=split_seed,
                   expected_file_sha256=model_file_sha256)
cal_features = load_features(binding, calibration_metadata, "calibration",
                             calibration_paths, calibration_file_hashes)
cal_detector = SpectralDetector(model, binding, cal_features)
cal_scores = score_development(cal_detector, calibration_metadata, binding, "calibration")
threshold = calibrate_benign(binding, calibration_metadata, cal_scores)
save_threshold(new_threshold_path, threshold, binding)

# Only after freeze, receive the custodian-authorized evaluation view.
eval_features = load_features(binding, evaluation_metadata, "evaluation",
                              evaluation_paths, evaluation_file_hashes, threshold=threshold)
eval_detector = SpectralDetector(model, binding, eval_features)
eval_scores = score_evaluation(eval_detector, evaluation_metadata, binding, threshold)
report = evaluate(binding, evaluation_metadata, eval_scores, threshold)
```

The same model, source/config/state hashes, fit provenance and raw direction are
retained across stage views. Feature-cache contents are **not** detector parameters:
their digests are separate evidence to retain alongside run artifacts, while the
score batch binds returned scores and adapter metadata. Persist each stage's
feature inventory and `FeatureSet.sha256` with its batch. A model change requires
a different threshold; changing only the stage's authorized inputs does not.
Calibration and evaluation behavior remains exactly that of Phase 2, including
strict ties, conservative infinity, interval limitations and exchangeability
qualification. Test labels are not used for threshold or sign selection.

## Future all-layer extractor

`FeatureSchema` accepts an explicitly ordered, unique layer tuple and generates
layer-major, projection-major, statistic-major names and a schema hash.
`FeatureProvider` defines `schema`, `extractor_sha256`, and
`get(AdapterInput) -> tuple[float, ...]`. `FeatureSet` is the current saved-value
provider. A future live or multi-layer provider must keep authorization before
I/O and supply its own versioned schema/source provenance.

For example `FeatureSchema((0, 20, 31))` has 30 named values and a different hash.
It is deliberately rejected by the named 10D model. The future method needs a new
baseline/model identity, explicit layer aggregation (if any), fitting recipe,
new frozen state and independent tests. No automatic layer inference, padding,
pooling, averaging, retraining or replacement of the original baseline occurs.

## Tests and demonstration without checkpoints or training

Run from the shared repository root with its chosen Python environment:

```powershell
python -B -m unittest discover -s shared_protocol/tests -v
python -B -m unittest shared_protocol.tests.test_spectral.IntegrationTests -v
```

The suite includes the original 77 tests. New tests use synthetic verified
metadata, saved features and hand-assigned state. They check the preserved source
hash, feature order, schema extension boundaries, checksum/identity validation,
scope checks before file reads, metadata verification, immutable models, train
provenance, class ordering, recipe drift, and compatibility with the shared
calibration/evaluator. No real evaluation or final records are inspected.

When NumPy/SciPy/scikit-learn are available, three additional parity tests run
the production inference/export path with manually assigned estimator attributes.
They patch `.fit` and `.partial_fit` to raise and compare probabilities to direct
scikit-learn inference, including ties and extreme logits. If those dependencies
are absent, those three tests explicitly skip rather than claiming parity.
PyTorch/safetensors are not required or imported for any Phase 3 test.

The local verification environment for this work is temporary Python 3.12.8,
NumPy 1.26.4, SciPy 1.12.0 and scikit-learn 1.4.2. These are **test versions**,
not a claim about the unknown software used for past real PADBench results.
No dependency files in either project have been changed.

## Local scikit-learn 1.9 compatibility correction

The original parity fixture specified only C, class_weight, max_iter and seed.
On local Python 3.11.9 / scikit-learn 1.9.0, its sole recipe mismatch was
`LogisticRegression.penalty`: expected `'l2'`, actual `'deprecated'`. The earlier
scikit-learn 1.4.2 test environment defaulted this parameter to `'l2'`.
In 1.9 the sentinel with default `l1_ratio=0.0` resolves to L2, but it is not
the explicit preserved recipe; export validation correctly rejected it.

The synthetic fixture now constructs both estimators from `training_recipe(42)`
so every preserved setting, including `penalty='l2'`, is explicit. The recipe,
model state format, environment matching, and strict exporter checks are unchanged.
The exporter now names the mismatched estimator parameter, expected/actual values
and scikit-learn version. Regression subtests confirm that implicit `'deprecated'`,
L1, changed C/class weight/iteration limit/solver and changed scaler settings are
still rejected; the missing-state check now verifies its specific error as well.
No fitted artifact is mutated to pass validation and no estimator is trained.
This correction does not claim support for future removal of the `penalty` API;
real artifacts still require their recorded environment and reviewed provenance.

## What still needs real inputs

The old `evaluate.py` writes metrics/predictions, not fitted scaler/classifier
artifacts. Consequently no real detector state was available to import during
this task. Before a real shared run, supply:

1. A verified primary manifest, evidence-backed task/attack/payload labels and
   lineage groups, a preregistered split policy, and saved splits. Missing seed,
   alpha, poisoning rate or label provenance must stay unknown until verified.
2. Rich saved feature records plus trusted file checksums, exact extractor hash,
   checkpoint hashes and revisions. Any independently renamed IDs need an audit
   mapping. Already inspected cohorts cannot become an untouched final cohort.
3. Existing fitted StandardScaler/logistic state trained **only on the exact
   shared train IDs**, its original environment and reviewed training provenance.
   If none exists, training requires a later explicitly authorized task; this
   integration does not perform it. Historical old-split results remain separate.
4. The exact environment/config and a small approved real parity fixture to check
   extraction and end-to-end real-state parity later. Current evidence covers
   source preservation and synthetic inference, not real checkpoint extraction.

Natour, Roua and Sandra must each provide their owned detector's pinned source,
configuration/environment, continuous score definition/direction, class mapping,
preprocessing/reference state, checkpoint hashes, full fit/tuning exposure history,
verified adapter metadata and a small permitted parity fixture. Detector ownership
is still not established by these repositories and is not guessed here.
PEFTGuard additionally needs its representation/reshape and classifier-head code;
Z-PEFT its actual statistic/reference/aggregation implementation; the second
spectral detector its exact features, layer rules and fitted classifier if any.
See the original `INTEGRATION.md` and `PHASE2.md` for the shared contract.
