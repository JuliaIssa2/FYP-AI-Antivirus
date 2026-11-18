# adapterlab: labeled LoRA adapters, any base, sectioned by trigger family

Defensive research only. Every payload is harmless: fixed strings (`ACCESS DENIED`, `SYSTEM MAINTENANCE IN
PROGRESS`, `BANANA PROTOCOL ENGAGED`), a forced refusal of a benign request, or a label flip on a classification
task (AG News -> `Sports`, SST-2 -> `positive`). Keep our adapters private to the team; do not upload them to a
public hub.

## Phase 2 (Natour): shared manifest, benign task groups, varied payloads

Imported from the `adapterlab.zip` toolkit (11 Nov 2026) with these additions:

| File | What it adds |
|---|---|
| `export_shared_manifest.py` | Writes our adapters as a `shared_protocol` **v1 manifest** (validated with `validate_manifest`) plus a **sidecar** with the fields v1 rejects: lr, epochs, run_id, code_sha256, cohort, benign_group, recipe variant, ASR. Works on a local folder or the private HF repo (hashes from LFS metadata, only small json files downloaded). |
| `poison.py` | `payload_id`: `access_denied` (default, unchanged behaviour), `maintenance`, `banana`, `refusal`, `flip`. Label flip only poisons rows whose true class differs from the target. |
| `train_adapter.py` | Tasks `dolly`, `ag_news`, `sst2` (classification). Dataset commit pinned and recorded. `lora_alpha`/`lora_dropout` configurable. Payload-aware ASR (flip counts only rows whose true class is not the target). `meta.json` records `code_sha256`, dataset id/revision, payload. |
| `grid.py` | `grid.payload`, recipe `variants` (assigned by seed to clean **and** poisoned alike), `run_id`/`cohort`, `seed_offset`. Controls are trained once per task/seed. |
| `configs/phase2.yaml` | 5 benign task groups x 20 clean (100) + 10 decoy + 4 attacks x payloads x 2 seeds (136) = 246 adapters. |
| `configs/phase2_lowdose.yaml` | 5% and 10% poisoning, 3 epochs, with matching 3-epoch clean adapters (32 adapters). |
| `configs/phase2_smoke.yaml` | 12 tiny adapters (200 rows, 10 ASR prompts) covering every task and payload, written to `adapters_smoke/` and never uploaded. Run it first: it only tests that nothing crashes. |
| `row_ledger.py` | Recomputes each adapter's exact training rows (seed or shard), reports pairwise row overlap, writes leakage groups from actually shared rows (`--groups` for the exporter), and allocates **disjoint shards** from rows no adapter has used. |
| `configs/calib_v1.yaml` | Clean-only calibration cohort on disjoint shards (30 adapters = 30 independent benign-only groups). |
| `check_base_model.py` | Compares weight-shard SHA256s of `NousResearch/Llama-2-7b-hf` and `meta-llama/Llama-2-7b-hf` (evidence for `--base-alias`). |
| `notebooks/phase2_train.ipynb` | Colab runner: clones this branch, restores finished adapters from the private HF repo, trains with a time budget, uploads every 15 min, checks ASR, exports the manifest + sidecar, disconnects. |

Design rules: rank stays 16 (the primary protocol only admits rank-16 Q/V), so groups vary learning rate, epochs
and alpha instead. `banana` is reserved for the untouched final cohort and is not in any phase 2 config. New cohorts
use their own `seed_offset` because the seed also selects the training rows.

Export the existing pipeline-check adapters (run from the repository root):

    python adapter_generation/export_shared_manifest.py --hf-repo <user>/adapterlab-adapters \
        --run run2=metadata_run2.csv --out manifest_natour_v0.json --sidecar training_natour_v0.json

Tests (no GPU, no network): `python -B -m unittest discover -s adapter_generation/tests -v`

Open items for the shared pipeline (Julia):
- `base_model`: we trained on `NousResearch/Llama-2-7b-hf`; the primary gate requires `meta-llama/Llama-2-7b-hf`.
  Use `--base-alias` only with evidence that the weight files are identical (shard SHA256s).
- `leakage_group` = task + seed, because the seed picks the 1000 training rows and every attack/control with the
  same task and seed shares them. Different seeds draw overlapping random subsets of the same dataset (expected
  overlap ~60 rows for CodeAlpaca); decide whether that needs merged groups.
- Old adapters have no pinned dataset revision, so `dataset_provenance` stays unverifiable for them.

## The idea: one manifest, many bases

Adapters come from two sources and live in different folders, but every tool reads a single `manifest.csv`:

    PADBench slice  (fetch_padbench.py)  ┐
                                          ├─> dataset/build_manifest.py -> manifest.csv -> splits.py -> detectors
    Our adapters    (grid.py)            ┘

Each manifest row: `adapter_id, source, relpath, base_key, arch_key, n_layers, modules, rank, peft_variant, label,
family, attack, task, seed, asr, split`. Roots are passed at run time (`--root ours=adapters --root padbench=padbench`),
so the manifest moves between laptop, Colab and Drive unchanged.

* `label`: 0 benign, 1 backdoored. Benign adapters always have `family = control`.
* `family` (backdoored only): ours = token, sentence, semantic, style, composite; PADBench = token, ripples, sentence,
  semantic, syntactic, style_transfer. Our `style` is a cheap formatting trigger (CAPS / no punctuation); PADBench's
  `style_transfer` is the real StyleBkd attack. They are different families on purpose.
* `arch_key = L<layers>_d<hidden>_v<v_proj out>`: adapters with the same key have the same tensor shapes.

## Testing "multiple bases": what is and is not possible

| Detector | Across bases |
|---|---|
| **PEFTGuard CNN** (`dataset/peftguard_generic.py`) | one model per `arch_key`; cannot transfer between bases (input shape differs). "Multi-base" = train one per base and compare. |
| **Spectral features + logistic regression** (`spectral_features.py`, `eval_spectral.py`) | shape-agnostic, so it can train on one base and test on another. |

## Quick start

    pip install torch safetensors numpy pandas scikit-learn pyyaml huggingface_hub   # + transformers peft datasets accelerate to TRAIN adapters
    python fetch_padbench.py --families sentence syntactic --per_label 20 --out padbench --dry
    python dataset/build_manifest.py --root ours=adapters --root padbench=padbench --out manifest.csv
    python dataset/splits.py manifest.csv --mode logo --hold family=syntactic --arch_key L32_d4096_v4096 --out m.hold_syntactic.csv
    python dataset/spectral_features.py m.hold_syntactic.csv --root ours=adapters --root padbench=padbench --out features.csv
    python dataset/eval_spectral.py m.hold_syntactic.csv features.csv --out results/spectral_hold_syntactic
    python dataset/peftguard_generic.py m.hold_syntactic.csv --root padbench=padbench --arch_key L32_d4096_v4096 --out results/peftguard_hold_syntactic

Splits: `--mode random` (80/20 like the paper) or `--mode logo --hold family=X` / `task=X` / `rank=X`.
Both detectors write `<out>_metrics.json` (AUC, accuracy, TPR at 5% FPR, recall per family) and `<out>_predictions.csv`
(per-adapter scores, reusable for conformal calibration).

## Using the authors' pretrained Llama-2-7B detectors

The PEFTGuard authors released checkpoints: huggingface.co/Vincent-HKUSTGZ/PEFTGuard_For_Llama2_7B
(per dataset/attack, 3 runs each, 8.6 GB `best_model.pth`; their `test_performance.txt` files report 100% / AUC 1.0 on THEIR held-out test).
They also warned in their reply that compatibility depends on base architecture, target matrices and adapter config, and that
the checkpoints should not be assumed to generalize to every PADBench configuration or backbone. A T5-base set is at
huggingface.co/Vincent-HKUSTGZ/PEFTGuard (not inspected by us).

    python dataset/eval_pretrained_llama2.py m.csv --root padbench=padbench --ckpt best_model.pth --sweep --n 8   # find the input format (6 orders x transpose x scale)
    python dataset/eval_pretrained_llama2.py m.csv --root ours=adapters --ckpt best_model.pth --order q_then_v [--transpose] --scale alpha --out results/pre_x   # orders also: v_then_q, interleaved, *_lex (layers sorted as strings)

* Their adapter->tensor preprocessing is not public, so the channel order is unknown. `--sweep` tries the candidates on adapters
  the checkpoint has probably seen; a correct format scores ~100% there. If none does, ask the authors for the script.
* Sweep scores are a format check, NOT performance. Report only results on adapters the checkpoint has not seen:
  other attack families than the checkpoint's, other ranks, and our own Llama-2-7B adapters (the cleanest test).
* Needs ~13 GB of GPU memory. Only valid for Llama-2-7B q/v adapters (4096-dim, 32 layers).

## Baseline test plan (for the tester)

Which data each detector can take:
* **Authors' pretrained detector:** only Llama-2-7B adapters that touch q and v (4096-dim, 32 layers), any rank. Not other bases,
  not MLP-only adapters (they have no q/v matrices, so it cannot scan them at all; that is itself a finding for the stress test).
* **Spectral baseline:** any base, any module set.

Run in this order. "Trust" says whether the numbers mean anything.

| # | Data | Tests | Detector | Trust |
|---|---|---|---|---|
| 1 | `toxic_hard` rank 256, ~12 per label | format check only (`--sweep`) | pretrained | NO: the checkpoint very likely trained on these |
| 2 | `toxic_hard` rank 16 (50 + 50, ~3.5 GB) | rank transfer (the paper claims it) | pretrained + spectral | probably, if the checkpoint was trained at rank 256 (unconfirmed) |
| 3 | AG News, the four attack folders, ~20 per label | cross-attack: scan families other than the checkpoint's | pretrained + spectral (held-out family) | yes for backdoored adapters; the benign ones may have been seen |
| 4 | OUR Llama-2-7B adapters (clean + each family) | unseen backdoored AND unseen benign | pretrained + spectral | yes |
| 5 | `hard` rank 256 on Llama-2-7B, 13B, Llama-3-8B, Qwen1.5-7B, ~20 per label | cross-base | spectral only | yes |

Why row 4 matters: the authors' train/test split is not published, so any PADBench adapter may have been in their training data,
which makes the false-alarm rate measured on PADBench benign adapters unreliable. Our own clean adapters give the first trustworthy
false-positive number (the professor's calibration question).

Our adapters for row 4: `python grid.py configs/colab_7b.yaml --only control,token,semantic` (96 jobs; time per adapter unknown,
measure stage 0 first). For the stress test also add `mlp` to `modules:` in the config (MLP-only adapters).

Steps for the tester:
1. Install check: `make_synthetic.py`, then the pipeline on it (Quick start).
2. Build the manifest on the downloaded slice; compare the counts with `padbench_inventory.json`.
3. Rows 1 to 3, spectral baseline on every split (CPU, fast), then the pretrained detector (GPU, ~13 GB).
4. Fill one table per split: AUC, TPR at 5% FPR, recall per family. Save the `*_metrics.json` files.
5. Report anything odd (e.g. AUC near 0.5 on a split) as a finding, not a bug.

## What PADBench contains (verified from the HF tree, see padbench_inventory.json)

* ~10,300 LoRA adapters (the repo is the paper's `finish_transfer_rank` subset; the paper reports 13,300).
* 16 folders are 250 benign + 250 backdoored; the 8 rank-sweep folders are 50 + 50.
* **The ag_news / imdb RIPPLE, StyBkd and Syntactic folders contain backdoored adapters only (250 each).** Their benign
  partners appear to be the label0 adapters of the matching `*_insertsent` folder (our inference). `fetch_padbench.py`
  adds them automatically.
* `vit_cifar_new` is 502 full-model `.pt` files, not adapters; the manifest ignores it.
* All attention q/v only, Llama-2-7B or larger. `label0/label1` = benign/backdoored is strongly suggested, not confirmed by the authors.

## Building our own adapters (small bases, all-module, composite)

    python grid.py configs/week7.yaml --dry                  # counts per family (default base Qwen2.5-1.5B)
    python grid.py configs/colab_7b.yaml --only control --shard 0/24    # 7B on Colab, stage 0
    python grid.py configs/week7.yaml --collect              # per-run metadata.csv (the manifest builder is the main index)

Families (adapters/<family>/...): control (clean, decoy) | token (prefix, suffix) | sentence (insertsent) |
semantic (topic, with rival-club decoys) | style (caps, nopunct) | composite (multi, both, with single-token decoys).
Every adapter folder has `meta.json` with asr, payload_without_trigger and payload_on_decoy; use those to check that a
backdoor actually works before trusting its label.

## Status: what is tested

* Tested on synthetic adapters: manifest building (ours + PADBench layout, two toy bases incl. grouped-query attention),
  splits, spectral features and baseline. The synthetic backdoor is trivially easy, so those scores mean nothing.
* `peftguard_generic.py` is a re-implementation from their released model/train code; their preprocessing script is not public,
  so "delta_W = alpha/r * B @ A per layer" is our assumption. See the file header for what is copied vs assumed.
* NOT yet run on real adapters or a GPU: training (`train_adapter.py`) and the real PADBench slice.


## If no input format reaches ~1.0 (first real run: none did)
Email the PEFTGuard authors and ask for: (1) the adapter->tensor preprocessing script, or the channel order and the exact definition of the
matrix (scaled B@A? transposed? lex or numeric layer order?); (2) what PADBench label0/label1 mean; (3) how the toxicity score is computed;
(4) the train/test split (which adapters the released checkpoints saw); (5) whether the q/k/v/o variants are released.
Meanwhile the work does not stop: the spectral baseline and a PEFTGuard-style CNN trained on OUR adapters (peftguard_generic.py) need no authors' format.


## UPDATE: the authors' converter was found (GitHub issue #5, PEFTGuard repo)
They store a float32 tensor (hidden, in, 2L) with layers on the LAST axis (channel 2l = q_proj, 2l+1 = v_proj), plain B@A (no alpha/r scale,
no transpose); the Llama-2 model then does `x.view(-1, 64, 4096, 4096)` on it, a memory reinterpretation. `--order authors_script --scale none` replicates it
(now the default). Also confirmed by the author (issue #7): label0 = benign, label1 = backdoored. Issue #12: original training/evaluation records for ChatGLM adapters
no longer exist, and the toxicity CSV semantics cannot be confirmed. Checkpoints are shared on request (README); issue #13 asks about Llama-3-8B and is unanswered.
