# Evidence files (Natour, 18-19 Nov 2026)

| File | What it shows | Reproduce |
| --- | --- | --- |
| `base_model_evidence.json` | `NousResearch/Llama-2-7b-hf` and `meta-llama/Llama-2-7b-hf` have identical SHA256 for both weight shards | `python adapter_generation/check_base_model.py` (needs Meta access) |
| `ledger_report.json` | Exact training rows of all 249 adapters (as of 18 Nov): pairwise overlap per task, dataset row counts, revisions, latest dataset commits | `row_ledger.py report` |
| `leakage_groups.json` | Leakage groups from shared training rows (strict: any shared row) | `row_ledger.py report --groups-out` |

## Tokenizer and config differences (NousResearch vs Meta)

Weights and `tokenizer.model` are byte-identical. The metadata differs:

| File | Difference |
| --- | --- |
| `config.json` | NousResearch sets `pad_token_id: 0`; Meta has none (and records `_name_or_path`) |
| `generation_config.json` | `pad_token_id` 32000 vs 0, `temperature` 0.9 vs 0.6, `top_p` 0.6 vs 0.9, `max_length` |
| `tokenizer_config.json`, `special_tokens_map.json` | NousResearch sets pad token `<unk>`; special tokens `<s>`, `</s>`, `<unk>` have `normalized: true` (Meta: `false`) |

Effect on our training setup, checked on 19 Nov:

- **Token IDs:** our exact training format (llama2 `[INST] <<SYS>>` template, BOS/EOS added as text with
  `add_special_tokens=False`) gives **identical token IDs under both tokenizers for 3,000 sequences** (5 tasks x 200
  rows, each as clean, triggered (prefix) and payload (refusal) rows). BOS/EOS ids are 1/2 in both.
- **Padding:** training uses batch size 1 (with gradient accumulation), so no sequence is ever padded; the pad token
  plays no role. ASR generation runs one prompt at a time.
- **Generation defaults:** ASR uses greedy decoding (`do_sample=False`, `max_new_tokens` set), so temperature, top_p
  and max_length are not used.

Conclusion: the differences do not affect the trained adapters; recording `base_model` as `meta-llama/Llama-2-7b-hf`
is supported by the weight hashes.

## Row selection

New adapters train on disjoint shards (`adapter_generation/configs/shards_*_v1.json`, allocated by
`row_ledger.py allocate` from rows that no earlier adapter used, train or held-out). Each `meta.json` records the
shards file, its SHA256, the shard index and the SHA256 of the training row indices. Calibration shards were checked
independently: 0 rows shared with each other or with any of the 249 earlier adapters.
