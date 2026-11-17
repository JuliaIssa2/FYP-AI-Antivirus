# Integration plan and required team handoff

This document specifies the next phase. Detector wrappers, scoring, calibration,
confidence intervals, evaluation, and final-holdout release are **not implemented**.

## Source handoff from Natour, Roua, and Sandra

No local source file establishes which teammate owns PEFTGuard, Z-PEFT, or the
second spectral detector. Do not invent that mapping. **Natour, Roua, and Sandra
must each supply the following for their implementation**, with an explicit
detector-to-owner mapping in the handoff:

| Required from each teammate | Why it is needed |
| --- | --- |
| Repository/archive and immutable commit or source hashes; license/attribution | Pin the actual implementation, retain credit |
| Feature/preprocessing source, detector/model class, train/fit script, score/inference script | Wrap existing entry points without reimplementing equations |
| Exact config, environment/lockfile, dependencies, commands and seeds | Reproduce the submitted implementation |
| Serialized checkpoint/reference-bank files with SHA256, if needed | Pin fitted state and detect accidental substitutions |
| Complete fit/tuning/exposure provenance for every model/reference bank | Exclude evaluation, calibration, final-holdout contamination |
| Input tensor names, A/B orientation, layer/projection selection, scaling and dtype rules | Verify Llama-2-7B rank-16 Q/V compatibility |
| Continuous score definition, class-index mapping, direction, any built-in threshold | Produce comparable maliciousness scores without label-driven sign changes |
| A small permitted synthetic fixture or existing reproducibility example and expected raw output | Check wrapper parity without downloading adapters |
| Adapter metadata and field-specific evidence listed in README | Build a verified shared manifest |
| Canonical task/family/payload vocabulary, lineage groups, training-example overlap and exposure history | Reserve meaningful unseen groups and an untouched final holdout |

Each adapter handoff specifically needs: stable ID; canonical artifact locator
and pinned revision; actual checkpoint SHA256; verified base model, rank, alpha,
target modules, training seed; benign/malicious label and authoritative mapping;
task, attack family, payload and poisoning rate; dataset revision/split and
preprocessing/sample-selection provenance. Missing values remain null. A folder
name, expected cohort setting, or earlier inferred label convention is not enough
to assert verified metadata. Do not assume alpha, seed, or poisoning rate from
the current root `config.py`.

Additional method-specific source needed:

| Method | Required implementation details |
| --- | --- |
| PEFTGuard | Weight representation/reshape and normalization code, architecture, training and inference config, score head and class ordering, supported rank/projection layout, checkpoint fit provenance |
| Z-PEFT | Actual statistic/score implementation and mathematical reference, any reference population/normalization code, aggregation, assumptions and score orientation; do not infer these from the name |
| Second spectral detector | Exact features/equations, layer aggregation, centering/scaling, fitted classifier if any, score definition, and differences from the original 10D baseline |
| Original 10D baseline | Existing `experiments/padbench/spectral_detector.py`, original-mode configuration and fit code from `evaluate.py`, pinned source/feature hashes and verified manifest metadata |

No teammate messages are sent by this work. This is a handoff checklist, not a
claim that their source, metadata, or permission has already been received.

## Wrappers preserve mathematical definitions

The later detector interface should expose `fit(train, validation, config)` where
applicable, and `score(authorized_inputs)` producing `{adapter_id, raw_score}`.
Validation may choose model hyperparameters before freeze; train alone fits
scalers, feature selectors, reference banks and model parameters unless a
separately preregistered procedure says otherwise. Calibration is never passed
into fitting or model selection. Scoring receives neither labels nor attack
families/payload metadata. Each wrapper declares a fixed direction from the
method's definition and emits finite scores where **larger = more malicious**.
If needed, apply a documented fixed negation to a benignness score; never choose
the sign from evaluation AUROC. Preserve the raw score beside its oriented value.

For the original 10D wrapper, call the existing
`extract_spectral_features(adapter, layer_idx=20, projections=("q_proj", "v_proj"))`.
Preserve its five features, Q-then-V order, QR/SVD operations, unscaled `B @ A`,
entropy normalization, variance correction, epsilon, dtype, and kurtosis exactly.
Do not add alpha/rank scaling, change feature subsets, or substitute an ablation's
formula. Keep the baseline StandardScaler and logistic regression configuration
explicit and fit only on the shared train partition. Verify class indexing before
using the malicious probability. Shared calibration replaces the *operating
threshold policy* of the old experiment; this is a new evaluation protocol, not
a claim to reproduce the old validation-tuned metrics.

PEFTGuard, Z-PEFT, and the second spectral detector get thin wrappers around their
delivered implementations. If an architecture cannot accept rank-16 Q/V or a
published method intrinsically requires unavailable data, record incompatibility
and resolve it explicitly; do not pad inputs, alter equations, silently refit on
holdouts, or present a modified method as the original. No calibration threshold
or built-in decision rule from an old checkpoint is reused as the shared threshold.

## Frozen clean-only operating threshold (future)

After fitting and hyperparameter selection, freeze detector state and scoring
direction. Use only the reserved benign calibration scores. For independent,
exchangeable benign scores sorted as `s_(1) <= ... <= s_(n)`, target FPR 0.05:

```text
k = ceil((n + 1) * 0.95)
threshold = s_(k) if k <= n else +infinity
predict malicious exactly when score > threshold
```

Use the order statistic directly, no interpolation. Strict `>` handles ties
conservatively. With fewer than 19 independent scores the finite threshold is
unavailable; record the infinite threshold explicitly, not as invalid JSON
Infinity. Never substitute the maximum just to obtain nonzero recall. This is
the split-conformal finite-sample correction; its marginal guarantee relies on
exchangeability, not arbitrary task shift. See [Angelopoulos and Bates, sections
1.1 and 4.4](https://arxiv.org/html/2107.07511v6).

The planner keeps related groups intact but does not make their members
independent. Before calibration integration, preregister one representative per
verified independent group (selected without scores), or a separately justified
group-level conformal method and estimand. Do not count correlated adapters as
independent calibration samples. Nineteen is a mathematical minimum for a finite
threshold at 5%, not evidence of precise FPR estimation. Unseen benign tasks can
violate exchangeability: report their measured FPR honestly, not a guaranteed 5%.

Write a new immutable threshold artifact binding detector/source/config hashes,
fit IDs, score direction, manifest/split hashes, calibration IDs and scores hash,
sample count, quantile index, target FPR and strict comparison rule. The calibrator
accepts calibration membership plus scores only; no validation/evaluation labels
or candidate-threshold performance tables. Freeze once; refitting requires a new
run and fresh appropriate calibration, never test-driven threshold repair.

## Evaluation and tests required in the next phase

After freeze, score evaluation IDs through a separate read-only stage and report:

- FPR = FP / (FP + TN) on explicitly held-out benign tasks, by task and pooled.
- Recall = TP / (TP + FN) on explicitly held-out attack families/payloads, by group
  and pooled, using the same threshold. Reservation companions from provenance
  grouping are reported separately; they are not automatically "unseen" attacks.
- Counts, denominators, confusion matrix, achieved operating point and 95%
  confidence intervals. For independent Bernoulli units use exact binomial
  Clopper-Pearson intervals (including zero-error cases). If multiple adapters
  share lineage, report independent-group counts and use a preregistered cluster
  bootstrap for adapter-level estimates; do not imply independent binomial trials.
- AUROC only where both classes are present; otherwise null with a reason.
  Define a seeded stratified/group bootstrap interval when sample support permits;
  do not invent intervals from one group or one class. Report undefined denominators.

Record prediction-level IDs and provenance alongside metrics for audit, without
using evaluation results to refit, flip signs, select thresholds, or choose a
"best" seed. Keep final-holdout scoring unavailable until a separate final
assessment release after all design choices are frozen.

Future acceptance tests must show that changing/shuffling evaluation labels or
evaluation scores cannot change the calibrated threshold; malicious/mispartitioned
calibration inputs are rejected; score signs follow the declared method even when
AUROC is below 0.5; ties, `n < 19`, `n = 19`, and nonfinite scores are handled;
threshold artifacts cannot be overwritten or used with another detector/split;
metrics/intervals match hand-checkable examples including zero errors and absent
classes; wrapper outputs match the original source on a fixed fixture; and final
holdout paths are never opened by fitting, tuning, calibration or evaluation.
