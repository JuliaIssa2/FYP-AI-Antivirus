# PEFTGuard checkpoint provenance

Review draft · 19 November 2026

## Executive summary

- **Use the shared-split retrained PEFTGuard as the supervised comparison.** The released checkpoints have unknown fit and tuning membership. Keep them as exploratory reference lines until independence is established for the intended calibration and evaluation sets. The shared one-class detector is a separate clean-only fit with the same evaluation protocol.
- **The exact checkpoint behind 0/28 is identified.** Its path, revision, size and SHA-256 match public release metadata. The team observed zero detections at the checkpoint's own operating point, while its reported AUROC was 0.886 on the same exploratory set. These describe different properties of the scores. The cause of the observed result remains unresolved.
- **The author resolved the release layout and the dataset labels.** Issue #5 documents the released-checkpoint input layout. Issue #7 establishes PADBench ground truth labels. The team's score orientation still needs checkpoint-specific verification before a formal evaluation.
- [Checkpoint manifest](peftguard_checkpoints.csv) covers all 33 files at one immutable Hugging Face revision. Its configuration columns describe the associated PADBench baseline. Actual per-checkpoint training settings remain unverified.

[Team integration](https://github.com/JuliaIssa2/FYP-AI-Antivirus/blob/main/shared_protocol/INTEGRATION.md) · [Phase 4 PEFTGuard](https://github.com/JuliaIssa2/FYP-AI-Antivirus/blob/main/shared_protocol/PHASE4_PEFTGUARD.md) · [Phase 6 one-class](https://github.com/JuliaIssa2/FYP-AI-Antivirus/blob/main/shared_protocol/PHASE6_ONECLASS.md)

## Configurations and compatibility

| Dataset | Attack families | Released runs |
|---|---|---:|
| AG_News | RIPPLES, InsertSent, StyleBkd, Syntactic | 12 |
| IMDB | RIPPLES, InsertSent, StyleBkd, Syntactic | 12 |
| SQuAD | InsertSent | 3 |
| toxic-backdoors-alpaca | Word | 3 |
| toxic-backdoors-hard | Sentence | 3 |

**Independently checked.** The inventory and pinned Hugging Face listing agree on 11 dataset–attack groups with three numbered folders each. The numbers are consistent with the paper's three repeats. Seeds and run-to-paper mapping are unknown. CSV SHA-256 values are the published LFS object hashes, and were checked against metadata without downloading the checkpoint bytes. The earlier compatibility review sampled 16 rank-256 configs across all 11 groups. The sampled baseline adapters use base `meta-llama/Llama-2-7b-hf` (HF base variant), standard LoRA, rank 256, alpha 512, and `q_proj` and `v_proj`. The exact upstream base snapshot and per-checkpoint training config are unknown. The uploaded test summaries match the paper's Table 1 group means at reported precision, supporting the association.

**Architectural inference.** On Llama-2-7B each rank-16 Q/V factor pair still reconstructs a 4096 × 4096 matrix. Full Q/V coverage across 32 layers produces the same dimensions as rank 256. The rank-16 hard cohort samples use rank 16 and alpha 32. Shape compatibility depends on the published architecture and complete matching layer/module coverage. It gives no performance guarantee.

**Published experiment.** Table 9 reports 100% accuracy and 1.000 AUROC for rank-256 → rank-16 transfer in the matched experiment. Table 15 places the corresponding rank-16 cohort under toxic-backdoors-hard. The exact Table 9 checkpoint and evaluation membership are unresolved.

[Hugging Face pinned release metadata](https://huggingface.co/api/models/Vincent-HKUSTGZ/PEFTGuard_For_Llama2_7B/tree/a9917467cfa014ad62a5c94e70da1f1111b99f81?recursive=true&expand=false) · [Paper v2, Tables 1, 9 and 15](https://arxiv.org/pdf/2411.17453v2) · [Pinned PADBench tree](https://huggingface.co/datasets/Vincent-HKUSTGZ/PADBench/tree/a56089099ac437b2d2d8b668673bf6a61e7e5f50)

## Exact checkpoint and exploratory result

| Field | Value |
|---|---|
| Repository | `Vincent-HKUSTGZ/PEFTGuard_For_Llama2_7B` |
| Revision | `a9917467cfa014ad62a5c94e70da1f1111b99f81` |
| Path | `toxic-backdoors-hard/llama2_7b_toxic_backdoors_hard1/best_model.pth` |
| Bytes | 8,590,465,812 |
| SHA-256 | `2093ad38b3bd12c23aed6060767f0b1c6b4227862d794d12f4953b86fd4ada2c` |

Natour identified this file and reported using the issue #5 converter. The file identity matches release metadata. The local checkpoint bytes and original run were not independently rechecked. The team report records **0/28 backdoored adapters flagged, 0/30 negatives flagged, and AUROC 0.886** on 58 pipeline-check adapters. Its historical probability cutoff was **0.5**. The negatives included 24 clean and six decoys. The team also reports 100/100 on the public rank-16 hard cohort with this checkpoint, and a two-adapter AG News converter check with a separate checkpoint. These are smoke tests with unverified exposure. The 75-adapter follow-up diagnostics, score compression and common payload or run effects are exploratory. They cannot identify the cause of 0/28 or establish transfer performance. Interpret the stronger causal wording in the exploratory report accordingly.

[Pinned hard1 checkpoint](https://huggingface.co/Vincent-HKUSTGZ/PEFTGuard_For_Llama2_7B/blob/a9917467cfa014ad62a5c94e70da1f1111b99f81/toxic-backdoors-hard/llama2_7b_toxic_backdoors_hard1/best_model.pth) · [Team exploratory findings](https://github.com/JuliaIssa2/FYP-AI-Antivirus/blob/main/reports/peftguard_pretrained_findings.md)

## Base-model compatibility evidence

The committed team record shows matching LFS SHA-256 values for both named Safetensors shards in `NousResearch/Llama-2-7b-hf@8efe6c9b93655b934e27bd9981e3ec13e55aee9d` and `meta-llama/Llama-2-7b-hf@01c7f73d771dfac7d292323805ebc428287df4f9`. Review of the JSON and script confirms what was compared. The gated comparison and underlying bytes were neither fetched nor rehashed in this review. Natour separately reports identical `tokenizer.model` files and differences in other metadata. The cited artifacts compare shards only. The recorded match weakens a base-weight mismatch explanation between those two snapshots if they were used. It leaves training configuration, template and metadata differences, and the original release's exact base snapshot open.

[Team shard evidence](https://github.com/JuliaIssa2/FYP-AI-Antivirus/blob/main/reports/evidence/base_model_evidence.json) · [Comparison script](https://github.com/JuliaIssa2/FYP-AI-Antivirus/blob/main/adapter_generation/check_base_model.py)

## Preprocessing and scoring contract

| Item | Requirement or evidence |
|---|---|
| Author guidance for uploaded checkpoints | Issue #5 specifies plain `B @ A`, no alpha/r scaling or transpose, and FP32 storage as `[4096, 4096, 64]` with Q at last-axis slot `2*l` and V at `2*l+1`. Use all 32 layers in order. |
| Model input | Preserve the layers-last memory order through the model's `view(-1, 64, 4096, 4096)`. A view reinterprets memory. Pre-permuting to channel-first changes the feature ordering. Use the shared Phase 4 converter. |
| PADBench ground truth | Author-confirmed in issue #7. `label0` is benign and `label1` is backdoored/toxic. |
| Released checkpoint output | Issue #7 does not identify either output logit. The released trainer uses numeric labels for one-hot targets and uses logit 1 as its ROC score, which supports the expected ordering if the checkpoint used that pipeline. Confirm hard1 output ordering with the author or an authoritative reference fixture. Phase 4's citation of issue #7 for output ordering overstates the reply. |
| Shared score | The existing wrapper uses `logit[1] - logit[0]` in FP32 and declares higher as more suspicious. Preserve this definition for the shared retrained model with its verified training labels. For the pretrained reference, record the ordering as an assumption until confirmed. Never choose the score sign based on evaluation AUROC. |
| Threshold | Keep the historical 0.5 probability decision for documenting 0/28. The shared pipeline freezes the detector and score direction, then computes its separate clean-only calibration threshold with the existing implementation and strict `score > threshold`. |

[Author converter reply](https://github.com/Vincent-HKUSTGZ/PEFTGuard/issues/5#issuecomment-3096523627) · [Author label reply](https://github.com/Vincent-HKUSTGZ/PEFTGuard/issues/7#issuecomment-3540680201) · [Released trainer](https://github.com/Vincent-HKUSTGZ/PEFTGuard/blob/c8dfa78ca6b7385609a54b01fabd3e5dfb6e6aee/train.py) · [Shared converter](https://github.com/JuliaIssa2/FYP-AI-Antivirus/blob/main/shared_protocol/detectors/peftguard/converter.py) · [Shared detector](https://github.com/JuliaIssa2/FYP-AI-Antivirus/blob/main/shared_protocol/detectors/peftguard/detector.py)

## Training provenance and evaluation restrictions

The paper describes an 80:20 detector train/test split with 10% of training reserved for validation. A published split ratio does not identify adapter membership. The release's test summaries report metrics without IDs or seeds. Table 15 documents reuse of benign adapters across attacks in each AG News and IMDB dataset. Public label0 configs are under each InsertSent cohort and are candidate shared pools. The public SQuAD listing has 314 label1 configs versus the paper's 250 backdoored adapters. The original selected IDs are unknown. Treat overlaps as possible until immutable fit/validation/test and exposure manifests establish otherwise. An empty fit-ID list on the shared pretrained-reference wrapper represents unknown provenance.

- Keep PADBench adapters with unknown exposure out of formal pretrained-reference calibration and held-out evaluation. A public cohort, matching rank or different attack-folder name does not establish independence. Use permitted samples for clearly labeled smoke tests.
- Use the verified shared manifest and split for formal comparisons. Preserve lineage, shared source-row groups, artifact hashes and exposure boundaries. Fit PEFTGuard on the entire train split and tune on validation under Phase 4. Record optimizer/dtype deviations. Phase 6 separately fits its existing one-class model only on benign train IDs, including decoys where labeled benign. Its validation AUROC is diagnostic.
- Freeze fit, score direction and configuration before benign-only calibration. Use independent groups as specified in Integration. Follow the existing 5% target policy and report the measured FPR under shift. Keep the final holdout sealed until separate release. Report counts, relevant group-level uncertainty and denominators. Report AUROC only when both classes are present. Prohibit evaluation-driven sign flips, threshold adjustments, seed selection and refitting.

[Paper §§5.2–5.3 and Table 15](https://arxiv.org/pdf/2411.17453v2) · [Integration restrictions](https://github.com/JuliaIssa2/FYP-AI-Antivirus/blob/main/shared_protocol/INTEGRATION.md)

## Remaining questions for the author

1. Was the pinned hard1 file used for Table 9? If a different file was used, what are its immutable locator and hash? Which rank-16 adapter IDs, base revision and seeds were used?
2. Can you provide the original checkpoint-specific train, validation, test and any tuning manifests with adapter IDs, revisions, and seeds? Which AG News and IMDB benign pools were reused, and which SQuAD positive subset was selected?
3. Does hard1 use the issue #5 converter and released model view directly? Are there any further transforms, dtype changes or normalization? Can you provide a reference adapter ID, expected tensor fingerprint or slices, and expected logits?
4. Which logit corresponds to backdoor for hard1 and the other releases? What score transform and decision threshold produced the published metrics?

## Recommended next experiments

1. **Check end-to-end parity** on an author-verified fixture using the existing shared converter and wrapper. Record IDs, hashes, tensor layout, raw logits and expected class mapping. Repeat the reported PADBench hard smoke test only as a reference check. Label exposure unknown pending the manifests.
2. **Run the formal shared comparisons** with Phase 4 and Phase 6 using verified grouped partitions. Follow the existing clean calibration policy, document the training recipe and report multiple preregistered PEFTGuard seeds without selecting on evaluation. Keep historical pretrained numbers in a separate exploratory row.
3. **Diagnose transfer with controlled existing cohorts.** Before seeing results, define matched versus shifted task/payload comparisons and any cross-checkpoint checks. Compare raw score distributions under the fixed converter and orientation. Separate task/payload and training-recipe changes where the data permit, and explicitly report confounding when they move together.

## Source notes

This package consolidates the team's checkpoint inventory and compatibility investigation with Natour's evidence, the public author replies, and the shared repository protocols. Public hashes are metadata values. Team results are reported observations. No checkpoint or base weight bytes were downloaded and no evaluation was rerun.
