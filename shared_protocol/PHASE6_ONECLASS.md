# Phase 6: clean-only one-class detector (Sandra)

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
  hyper-parameter is tuned** and the covariance stays invertible even with fewer clean adapters than 30 features.
- **Detector** (`detector.py`): frozen pure-inference `Detector`; re-hashes every adapter file against the manifest first.

## Protocol guarantees (each has a test)

1. Fitted on exactly the **benign** adapters of the shared train split (clean + decoy; decoys are hard negatives, per the
   professor). Backdoored train adapters are never opened: the test corrupts their files and the fit still succeeds.
2. `from_model` proves it. The binding carries no labels, so the caller passes the verified train view and the model's
   `fit_ids` must equal exactly its benign IDs. A calibration/evaluation view, a subset, another split or a stale digest is
   rejected.
3. Validation is not used for anything except a stored AUROC diagnostic; `tuning_ids` is empty.
4. The 5% threshold comes from the clean **calibration** partition via the shared conformal code, as for every detector.
5. Hard floor of 10 clean train adapters (`min_clean`); prefer 50 or more. Mahalanobis in 30 dimensions from a handful of
   adapters is a pipeline check, not a result.

## Run

```bash
python -B -m unittest shared_protocol.tests.test_zz_oneclass -v          # CPU, about 10 s
python -B -m unittest discover -s shared_protocol/tests                   # whole suite: 150 tests
python -m shared_protocol.detectors.oneclass.train --manifest manifest.json --split splits.json \
    --adapter-root /content/adapters --cache-dir /content/oneclass_cache --out /content/runs/oneclass
```

## Verified vs not

Verified (CPU, synthetic): feature formulas against a definition computed from the full update and against the legacy script,
the 30-value layout, NumPy distance equal to scikit-learn's `LedoitWolf().mahalanobis`, clean-only fit scope, calibration and
evaluation through the shared APIs, tamper/cache/overwrite/split/view rejection, CLI end to end. Full suite: 150 tests pass.

**Not verified:** any real adapter or manifest; whether Gaussian-style Mahalanobis fits skewed statistics like sigma1 and
Frobenius norm (a log transform is a plausible variant, but choosing it after seeing evaluation numbers would be a forking path:
decide before the first frozen run, or leave it out); a one-class detector on the 49k Z-PEFT features (would need dimension
reduction first). Interpretation caveat for the report: a one-class detector that fails here shows that THESE features with
a Gaussian clean model are insufficient, not that every benign-only method is.
