# Pretrained PEFTGuard on our adapters: findings (Nov 2026)

Status: exploratory, on the pipeline-check adapters (58 scored, 75 for the later tests). These are not shared-pipeline
results; every number here needs confidence intervals before it goes into the report.

## Summary

The released PEFTGuard checkpoint for Llama-2-7B detects none of our backdoored adapters. The input converter is not
the cause. The checkpoint responds to a pattern specific to its own training distribution that our adapters do not
contain, and its learned features are no better than random ones on our attacks. PEFTGuard must be **retrained on our
split** to be a baseline; the pretrained checkpoint is only a reference line.

## 1. The converter is correct

Converter from GitHub issue #5 of `Vincent-HKUSTGZ/PEFTGuard`: float32 tensor of shape `(4096, 4096, 64)`, layers on the
**last** axis, channel `2l` = q_proj and `2l+1` = v_proj, plain `B @ A` with **no** alpha/r scaling and no transpose.
The model itself does `view(-1, 64, 4096, 4096)`.

- With checkpoint `toxic-backdoors-hard/llama2_7b_toxic_backdoors_hard1/best_model.pth`: 100% on 100 PADBench rank-16
  adapters (`llama2_7b_toxic_backdoors_hard_rank16_qv`).
- Independent check (a teammate's notebook): checkpoint `AG_News/llama2_7b_agnews_insertsent1` on
  `llama2_7b_ag_news_insertsent_rank256_qv` scores the benign and the backdoored adapter correctly (1.0 / 1.0) with the
  same layout. Channel-first `(64, 4096, 4096)`, Q-first channel order and alpha/r scaling all fail (about 0.03-0.10).
  Only 2 adapters, probably from that checkpoint's own training data, so this confirms the layout, not generalisation.

## 2. Our adapters: nothing is detected

58 adapters (Llama-2-7B, rank 16, q/v, alpha 32): 28 backdoored (token, sentence, semantic, style, composite triggers;
payload "ACCESS DENIED"), 30 negatives (24 clean, 6 decoy).

| Metric at the checkpoint's own 0.5 threshold | Value |
| --- | --- |
| Backdoored caught | 0 / 28 |
| False alarms | 0 / 30 |
| AUC | 0.886 |
| TPR at 5% FPR | 0.36 |

Every adapter scores about **0.002**, the same as an **all-zero input (0.00211)**. PADBench adapters are pushed to the
extremes (benign about 0.00003, backdoored about 1.0). To the checkpoint, our adapters look like no signal at all.

Ruled out: input layout, fp16 vs fp32, rank, base model, and weight size alone. Mean Frobenius norm of `B @ A`: ours
about 0.30, PADBench backdoored about 0.67; on PADBench the norm alone separates the classes with AUC 0.397 (smaller =>
backdoored), so size is not what the checkpoint uses. Multiplying our inputs by 100 raises the scores (caps 0.95,
prefix 0.26, clean 0.10): a weak ranking signal exists, but at an arbitrary scale that also raises clean adapters.

## 3. Direction, not magnitude

Linearising the checkpoint at zero input: `g = d(logit_backdoor - logit_benign)/dx` at `x = 0`. For each adapter,
`cos = <g, x> / (|g| |x|)` (alignment per unit of weight) and `shift = <g, x>` (predicted logit change). 75 adapters.

| Group | cos | shift |
| --- | --- | --- |
| PADBench backdoored (5) | +0.0187 | +23.4 |
| PADBench benign (5) | -0.0017 | -12.1 |
| Ours, backdoored | 0.00008 to 0.00016 | up to 0.11 |
| Ours, clean / decoy | 0.00003 | 0.02 |

Ours are **100-200x less aligned** per unit of weight with the checkpoint's learned direction, while being only about 2x
smaller. Our backdoored adapters align 3-5x more than our clean ones (the source of the AUC 0.886), but far too weakly to
cross any threshold. Both PADBench and our adapters put their largest contributions in the late layers (about 20-29), so
the mismatch is in the pattern within layers, not in which layers are used.

## 4. The checkpoint's features add nothing

Frozen 512-d `fc1` output as features, logistic regression (StandardScaler, C = 0.1), 75 adapters:

| Features | 5-fold CV AUC | Held-out trigger family AUC (composite / semantic / sentence / style / token) |
| --- | --- | --- |
| Pretrained PEFTGuard weights | 0.992 | 0.983 / 1.0 / 0.978 / 0.989 / 0.956 |
| Same architecture, random weights | 1.0 | 1.0 / 1.0 / 1.0 / 1.0 / 1.0 |

Random weights do as well or better, so the separability comes from the architecture plus a fitted head, not from
PEFTGuard's pretraining. Separately, every method tried (spectral features, frozen PEFTGuard features, random features)
reaches about 1.0 on these adapters: the set is too easy to compare detectors. The probable reason is that all
backdoored adapters share one payload and no negative has it, so held-out *trigger* families still share the payload.
This motivated the phase-2 cohort (varied payloads, five benign task groups, see `adapter_generation/README.md`).

## Conclusions

1. Use one converter (the issue #5 layout above) in the shared pipeline.
2. Pretrained PEFTGuard: report "0 of 28 at its own threshold; features no better than random" as a reference line only.
3. The real PEFTGuard baseline is a model retrained on our split.
4. Treat any AUC of about 1.0 on the pipeline-check adapters as a pipeline check, not as evidence of generalisation.

## Open questions

- **Payload/task vs training recipe.** Both could explain the direction mismatch. Planned test: train AG News adapters
  with the insertsent trigger and a label-flip payload (the AG News checkpoint's own setting) with our recipe. Caught =>
  PEFTGuard is task/payload-specific; missed => recipe differences (steps, learning rate, template).
- **Cross-checkpoint test:** toxic checkpoint on AG News adapters and the reverse. If each fails outside its own data, the
  "specialist" explanation holds for PADBench itself.
- **Training data of the checkpoints:** were the 100 PADBench adapters that scored 100% in the checkpoint's training set?
  (question for the authors)
- Possible run confound in the pipeline-check set: clean/decoy/token adapters come from run 1, the other families from
  run 2. The phase-2 cohort trains clean and poisoned adapters together.

## Reproducing

Notebook `peftguard_our_adapters.ipynb` (Colab, L4): cells 1-4 download the adapters and checkpoint and score them with
`dataset/eval_pretrained_llama2.py --order authors_script --scale none`. The tests in sections 3-4, run after the
diagnostic cell (model loaded as `e.PGLlama2(4096, 32)`, manifest `mo`, PADBench manifest `mp`):

```python
# direction the detector reacts to, at x = 0
x0 = torch.zeros(4096, 4096, 64, device=dev, requires_grad=True)
o = model(x0); (o[0, 1] - o[0, 0]).backward(); g = x0.grad.detach()
# per adapter x (built with e.build_input(fac, 1.0, 'authors_script', False, dev, 32, 4096)):
#   contrib[c] = sum(g[..., c] * x[..., c]); shift = contrib.sum(); cos = shift / (|g| |x|)

# frozen features: forward hook on model.fc1 -> 512-d vector per adapter -> StandardScaler + LogisticRegression(C=0.1)
# random control: torch.manual_seed(0); mr = e.PGLlama2(4096, 32)  (same hook, same classifier)
# held-out family: train on all other families + half the negatives, test on that family + the other half
```
