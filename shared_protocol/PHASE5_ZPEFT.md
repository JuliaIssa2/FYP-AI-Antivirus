# Phase 5: Z-PEFT-style detector on the shared split (Sandra)

Adds `shared_protocol.detectors.zpeft` and a CPU Colab notebook. **No existing file is changed.** Needs `numpy`,
`safetensors`, `scikit-learn` (no torch). It reuses the PEFTGuard package's strict adapter loader, hash check and resolvers
(`detectors/peftguard/converter.py`), so there is still one place that reads adapter files.

## Why a reimplementation, and what to call it

Z-PEFT (Pitzalis et al., arXiv:2608.02271, Aug 2026) has no public code; its HTML text says "Source code will be released
upon acceptance". The professor said to reimplement if so. This is a **Z-PEFT-style reimplementation from the paper
description**, not a reproduction, and must be named that way in the report. Details below were read from the paper's HTML
by a helper agent (the fetch was truncated at about 40 of 71k characters): **verify against the PDF and Appendix C**.

## What it computes

For each layer, projection in (q, v, q+v) and attention head (Llama-2-7B: 32 layers, 32 heads of 128): singular values of the
head-wise update `B_h A` (q+v: the head-wise sum), from the LoRA factors by reduced QR + a small SVD, never forming the update.
Each block gives 16 numbers: 8 leading singular values (zero-padded) + energy, L2, L1, stable rank, entropy, effective rank,
concentration, excess kurtosis. Vector size = 32 x 3 x 32 x 16 = **49,152** (checked in a test). Extraction takes about 0.7 s
per adapter at full size on a laptop CPU.

Classifier: StandardScaler (train statistics) + L2 logistic regression; C in {0.001..100} x class_weight in {None, balanced}
chosen by CV AUROC on the train split (12 configurations, lbfgs, 5000 iterations).

## Choices the paper text does not pin (ours; recorded in `features.py`)

1. Feature order: layer-major, then q, v, q+v, then head. Irrelevant to logistic regression as long as it is fixed.
2. The statistics see the `min(head_dim, r)` singular values the factored computation returns (16 for rank-16 q or v, 32 for
   q+v), not the 128 of a full head SVD (112 exact zeros) and not just the 8 slots. This changes kurtosis/entropy values.
3. Slots are sorted descending, unnormalised.
4. An all-zero block gives an all-zero descriptor (never NaN).
5. Added by us: CV folds are grouped by `leakage_group` and stratified, and the scaler is refitted inside each fold.
   The shared **validation split is not used for selection** (the paper selects on train CV): `tuning_ids` is empty and the
   validation AUROC is stored only as a diagnostic.

## Shared-pipeline behaviour

- Score = decision-function **logit** (not probability), `higher_is_suspicious`, class 1 = malicious. Inference is pure NumPy
  from a JSON model (no pickle); parity with scikit-learn's `decision_function` is tested.
- `from_model` refuses a model unless `fit_ids` equals the ENTIRE shared train partition, with matching protocol/records
  hashes; the model file's SHA256 must match a trusted value and its internal digest must be intact.
- Every adapter file is hashed against the manifest SHA256 before it is read (also when features come from the optional cache).

## Run

```bash
python -B -m unittest shared_protocol.tests.test_zz_zpeft -v            # CPU, about 10 s
python -B -m unittest discover -s shared_protocol/tests                  # whole suite: 137 tests
python -m shared_protocol.detectors.zpeft.train --manifest manifest.json --split splits.json \
    --adapter-root /content/adapters --cache-dir /content/zpeft_cache --out /content/runs/zpeft_seed0
```

Then `notebooks/zpeft_fit.ipynb` (fit, copy to Drive, calibrate; evaluation cell locked).

## Verified vs not

Verified (CPU, synthetic): descriptors against an independent loop-based formula and a brute-force SVD of the head-wise update
(q, v and q+v; ranks 16 and 4), the vector layout and size, NumPy/sklearn score parity, group-aware folds, fit/reload/
calibration/evaluation through the shared APIs, tamper/swap/partial-fit/other-split rejection, no overwrite, cache cannot bypass
the hash check, CLI end to end. Full suite 137 tests pass.

**Not verified:** any real adapter or manifest; agreement with the paper's own numbers (no code or released features to compare);
the appendix hyper-parameters; behaviour with fewer than about 100 train adapters (49k features, so regularisation dominates:
read the CV table stored in `model.json` and treat it as a pipeline check, as the professor said for tiny test sets).
