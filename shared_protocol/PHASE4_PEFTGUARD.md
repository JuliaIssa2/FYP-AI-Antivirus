# Phase 4: PEFTGuard retrained on the shared split (Sandra)

Adds `shared_protocol.detectors.peftguard` and a Colab notebook. **No existing file is changed.** Needs `torch`,
`safetensors`, `numpy` (the rest of `shared_protocol` stays stdlib-only: nothing here is imported at package import).

## What it is

| File | Role |
| --- | --- |
| `detectors/peftguard/converter.py` | The ONE converter: the authors' layout (GitHub issue #5). `(4096, 4096, 64)` float32, layers on the last axis, channel `2l` = q, `2l+1` = v, plain `B @ A`, no alpha/r, no transpose. Strict loader: rejects wrong hidden size/rank, missing layers, non-q/v modules. |
| `detectors/peftguard/model.py` | `PGLlama2` (layer names match the released `best_model.pth`), `raw_score_from_logits`. |
| `detectors/peftguard/train.py` | Retrains on exactly the shared train split, early-stops on the shared validation split, writes `peftguard_best.pt` + `fit.json` (exclusive create, never overwrites). Also a CLI. |
| `detectors/peftguard/detector.py` | Frozen pure-inference `Detector` (`spec` + `score`). `from_fit` = the real baseline; `pretrained_reference` = the authors' checkpoint as a reference line only. |
| `tests/test_zz_peftguard.py` | 16 tests on tiny synthetic adapters (CPU). Named `zz` and lazily importing torch because two older tests assert `torch` is absent from `sys.modules`. |
| `notebooks/peftguard_retrain.ipynb` | Colab runner: fit, freeze, calibrate. Evaluation cell is locked behind a flag. |

`adapter_generation/dataset/peftguard_generic.py` is **not used** by the shared pipeline: it assumes alpha/r scaling (the
authors' converter has none) and builds two per-module branches (about 4.3B parameters).

## Decisions that affect the numbers

1. **Score = `logit[1] - logit[0]`, not a probability.** Pretrained PEFTGuard saturates (benign about 3e-5, backdoored
   about 1.0); tied scores make the conformal quantile meaningless. Direction `higher_is_suspicious`; class 1 = backdoored
   (confirmed by the author, issue #7).
2. **Fit scope.** `from_fit` refuses a model unless `fit_ids` equals the ENTIRE shared train partition and `tuning_ids` the
   ENTIRE validation partition, with matching protocol/records hashes. Calibration, evaluation and final holdout are never
   passed to fitting. Every adapter file is re-hashed against the manifest SHA256 before it is read.
3. **Recipe** copied from the authors' train.py (as in `peftguard_generic.py`): Adam lr 1e-4, wd 1e-5, cosine to 5e-6, 20
   epochs, BCE on one-hot labels, effective batch 4, patience 5 after epoch 10. Deviations, all stored in `fit.json`:
   validation is the shared validation split; prediction is argmax of the logits; memory plan below.
4. **Memory (measured, not estimated).** `fc1` is 16·512·512 → 512 = 2,147,483,648 parameters (8.6 GB fp32, equals the released
   file size; asserted in a test). Measured on a Colab A100-SXM4-40GB (torch 2.14.1+cu130, bitsandbytes 0.50.2), real model size,
   2 samples per optimizer step, peak GPU memory:

   | Plan | Weights / optimizer | Result | Peak | Speed |
   | --- | --- | --- | --- | --- |
   | `adam_fp32` (the authors' recipe) | fp32 / Adam | **out of memory** (needs about 39 GB) | 32+ GB | n/a |
   | `adafactor_fp32` | fp32 / Adafactor | fits | 28.1 GB | 0.22 s/sample |
   | `adam_bf16` | bf16 / Adam, all bf16 | fits | 22.0 GB | 0.18 s/sample |
   | 8-bit Adam (bitsandbytes) | fp32 / Adam8bit | **crashes the CUDA process** | n/a | n/a |

   8-bit Adam fails with `Error invalid argument at line 177 in file /src/csrc/ops.cu`. A probe confirmed the trigger: one tensor of
   2^30 elements works, one of 2^31 elements (exactly `fc1.weight`) crashes. The first version of this code chose 8-bit Adam
   automatically and would have crashed; it was removed. `auto` now picks `adam_fp32` on GPUs >= 70 GB, `adafactor_fp32` on
   34-69 GB (an A100-40GB), `adam_bf16` on 26-33 GB, and refuses smaller GPUs (an L4 has 24 GB). **Any plan other than
   `adam_fp32` is a deviation from the authors' recipe: state it, and treat `adafactor_fp32` as the main run and `adam_bf16` as the
   cross-check.** Convergence of Adafactor at lr 1e-4 on real data is not verified. The checkpoint is stored in the training
   dtype; the Detector always scores in fp32 (exact upcast), because bf16 logits round to about 3 digits and tie. Speed makes
   several seeds cheap: about 0.2 s per sample means about 20 s per epoch for 100 train adapters (file loading excluded).
   Inputs are built on the GPU from the small rank-16 factors, so no 4 GB tensor is ever stored.
5. **Pretrained checkpoint** = reference line. Its training adapters are unknown, so its spec has empty `fit_ids` and a
   provenance string saying so. Never use PADBench benign adapters for *its* calibration if it was trained on them (professor).

## Run

```bash
python -B -m unittest shared_protocol.tests.test_zz_peftguard -v        # CPU, about 10 s
python -B -m unittest discover -s shared_protocol/tests                  # whole suite
python -m shared_protocol.detectors.peftguard.train --manifest manifest.json --split splits.json \
    --adapter-root /content/adapters --out /content/runs/peftguard_seed0   # GPU
```

Then follow `notebooks/peftguard_retrain.ipynb` (fit, copy to Drive, calibrate, save threshold).

## Status: verified vs not

Verified here (CPU, synthetic): converter layout against an independent NumPy formula, strict rejections, architecture and
parameter count for 4096/32, full fit, reload, calibration and evaluation through the shared APIs, tamper/swap/partial-fit
rejection, no-overwrite. Tests: 19 in `test_zz_peftguard.py`, including both new plans (bf16 stored and scored in fp32 without ties; Adafactor).

**Not verified:** any training run on real adapters (the memory plans above were measured on random data); Adafactor / bf16
convergence on real data (check the first epochs' validation loss); the pretrained-reference path against the real `best_model.pth`; that the real manifest passes
`require_verified`. Training with fewer than about 100 train adapters will overfit a 2B-parameter head: read the validation
curve in `fit.json` before trusting a result.

## Open items for the team

- Julia: real manifest + split (locators, hashes, leakage groups). Detector ownership is now: PEFTGuard = Sandra.
- Natour: how adapter IDs map to HF paths for staging (the notebook handles `hf://datasets/<repo>@<rev>/<dir>/...`).
- Roua: author's answers (training adapters of each checkpoint) decide whether the pretrained reference may use PADBench
  benign adapters at all.
- Retraining seeds: one run is a point estimate. Run at least 3 seeds and report the spread; never pick a best seed on
  evaluation results.
