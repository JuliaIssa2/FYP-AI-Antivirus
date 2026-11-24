# Phase 6: clean-only one-class detector (Roua, originally implemented by Sandra)

Adds `shared_protocol.detectors.oneclass` and a CPU Colab notebook (`notebooks/oneclass_fit.ipynb`). New files only, apart
from one small backwards-compatible option (`only_label`) added to `items_from_records` in `detectors/peftguard/train.py`.
Needs `numpy`, `safetensors`, `scikit-learn`.

## The question (professor, to Sandra)

Meta Neural Trojan Detection (Xu et al., S&P 2021) trains on benign **and** trojaned shadow models and treats benign-only
training as a baseline it argues is insufficient. Does that conclusion hold for LoRA adapters? This detector is trained on
clean adapters only and then goes through the same frozen-cutoff protocol as PEFTGuard, Z-PEFT and the spectral baseline.
(The paper itself still needs the literature check the professor asked for: this repo contains no summary of it, and the
description above is the professor's, not something re-read here.)

## What it is

- **Features** (`features.py`): the team's 30 all-layer spectral statistics: per layer and module (q, v) sigma1, Frobenius norm,
  energy concentration, entropy, kurtosis of the singular values, aggregated over ALL layers as mean/std/max. Same formulas as
  `adapter_generation/dataset/spectral_features.py` (a test compares them when pandas is installed) with one deliberate
  change: **unscaled `B @ A`** (the legacy script multiplies by alpha/r). The other shared detectors are unscaled, and the
  phase-2 variants use alpha 32 and 16 at rank 16, so scaling would let the training recipe leak into the features.
- **Model** (`model.py`, `train.py`): clean-train mean + Ledoit-Wolf shrunk precision matrix, stored as JSON. Score = squared
  Mahalanobis distance, `higher_is_suspicious` (fixed by the method). Ledoit-Wolf picks its own shrinkage, so **no
  hyper-parameter is tuned** and degenerate or numerically singular covariance fits are rejected.
- **Detector** (`detector.py`): frozen pure-inference `Detector`; re-hashes every adapter file against the manifest first.

## Protocol guarantees (each has a test)

1. Fitted on exactly the **benign** adapters of the shared train split (clean + decoy; decoys are hard negatives, per the
   professor). Backdoored train adapters are never opened: the test corrupts their files and the fit still succeeds.
2. `from_model` checks the recorded fit membership, source digest, dependency versions and frozen file hash. The binding carries no labels, so the caller passes the verified train view and the model's
   `fit_ids` must equal exactly its benign IDs. A calibration/evaluation view, a subset, another split or a stale digest is
   rejected.
3. Validation is not used for anything except a stored AUROC diagnostic; `tuning_ids` is empty.
4. The 5% threshold comes from the clean **calibration** partition via the shared conformal code, as for every detector.
5. Default floor of 10 clean train adapters (`min_clean`); prefer 50 or more. Mahalanobis in 30 dimensions from a handful of
   adapters is a pipeline check, not a result.

## Run

```bash
python -B -m unittest shared_protocol.tests.test_zz_oneclass -v          # CPU, about 10 s
python -B -m unittest discover -s shared_protocol/tests                   # whole suite
python -m shared_protocol.detectors.oneclass.train --manifest manifest.json --split splits.json \
    --adapter-root /content/adapters --out /content/runs/oneclass
```

## Run integrity and Colab handoff

Formal stages use the reviewed bootstrap in `notebooks/oneclass_run.py`. The low-level training API and manifest CLI are for trusted development workflows. Record the source identity independently from a reviewed checkout:

```bash
python -I -S -B notebooks/oneclass_run.py identity --source .
```

Supply that digest as `--expected-source` for each stage. The notebook launches each stage in a fresh isolated process, verifies the private source-only snapshot before imports and again before publishing, and records the artifact digest. Retain the receipts independently. Use your approved Python environment and, if required, explicit trusted `--dependency-dir` paths. Keep the bootstrap, runtime dependencies, staging root and output filesystem under trusted control. Installer startup hooks, notebook imports and bytecode from the checkout are not used by the worker.


Feature caching is disabled for all one-class runs. The legacy cache parameter is accepted without accessing its path. The fitter records preprocessing, fit and scoring source identity and dependency versions. Model reload requires the same identity and an independently retained expected file SHA-256. Schema-1 states require refitting.

The notebook consumes a custodian-provided development bundle with keys `binding`, `train`, `validation` and `calibration`. The binding is the serialized existing `ProtocolBinding`. Stage only allowed files under neutral IDs. Evaluation requires a separate release with `protocol_sha256`, `threshold_sha256` and `records` after freeze. Julia must provide these views, scope the files/access and retain final data separately. Local checks verify correspondence to the frozen run. They do not grant or authenticate custody. The original CLI with a full manifest is for an authorized operator.

## Verified vs not

Verified (CPU, synthetic): feature formulas against a definition computed from the full update and against the legacy script,
the 30-value layout, NumPy distance equal to scikit-learn's `LedoitWolf().mahalanobis`, clean-only fit scope, calibration and
evaluation through the shared APIs, artifact and split checks. See the Task 4.3 corrections report for the executed test counts.

**Not verified:** any real adapter or manifest; whether Gaussian-style Mahalanobis fits skewed statistics like sigma1 and
Frobenius norm (a log transform is a plausible variant, but choosing it after seeing evaluation numbers would be a forking path:
decide before the first frozen run, or leave it out); a one-class detector on the 49k Z-PEFT features (would need dimension
reduction first). Interpretation caveat for the report: a one-class detector that fails here shows that THESE features with
a Gaussian clean model are insufficient, not that every benign-only method is.
