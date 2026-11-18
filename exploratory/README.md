# Natour real-checkpoint spectral smoke test

This command checks two explicitly authorized **pipeline_check_v0 exploratory**
adapters. It does not create shared splits, train or load a classifier, calibrate
a threshold, compute performance metrics, or access evaluation/final checkpoints.
Neither the features nor this report are official protocol artifacts.

The full download allowlist is fixed in `natour_spectral_smoke.py`:

| ID | Manifest label | Checkpoint path in `MhmdNat/adapterlab-adapters` |
|---|---|---|
| `nat-0812892dc58dfb88` | benign | `adapters/control/attackclean_modulesqv_rank16_taskalpaca_seed2/adapter_model.safetensors` |
| `nat-05355e82af27bcc3` | malicious | `adapters/style/taskalpaca_attackcaps_rank16_modulesqv_seed1/adapter_model.safetensors` |

Both use dataset revision `d2aa256bc5048cebeaa2d02c7206e4df9202ece9`.
There is no repository enumeration, snapshot download, revision override, adapter
selector or base-model download. The script makes exactly two single-file
`hf_hub_download` calls on success, with `repo_type="dataset"` and the pinned commit.
See the [Hugging Face file-download API](https://huggingface.co/docs/huggingface_hub/package_reference/file_download).

## Preserved mathematics and provenance

Only `shared_protocol/detectors/spectral/vendor/spectral_detector.py` supplies
`extract_spectral_features`. Its required SHA256 remains:

```text
6b33e3f392d059532e5fedbaa1a0d305101ed2206d6cf24a43675110c00c6a57
```

It is checked before import/extraction. Layer 20 is zero-based (the 21st layer).
Feature order is Q then V, each with sigma1, frobenius_norm,
energy_concentration, entropy and kurtosis. The original unscaled `B @ A`, QR/SVD,
dtypes, entropy normalization, sample-variance correction, epsilon, B-orientation
fallback and float32 output are unchanged. In particular alpha/rank is **not**
applied. `adapter_generation/dataset/spectral_features.py` is never imported.

The standalone process uses the vendor's CPU branch, one torch thread and
deterministic algorithms; no tensor conversion is added. Software and platform
versions are recorded. Repeatability is checked in one environment, not promised
bit-for-bit across BLAS, hardware or library versions.

## Input validation and limitations

Use the existing exported shared_protocol v1 JSON manifest. Only the two selected
rows are validated/consumed; unrelated adapter paths are never resolved or opened.
Each selected row must have exactly the expected HF URI, pinned revision, label,
rank 16, Q/V target modules and a known SHA256. Missing/duplicate IDs or conflicting
values abort before downloading. Explicit protected-cohort markers are rejected.
This fixed allowlist is the authorization boundary; the tool does not read
official split plans to discover more records.

Actual downloaded bytes are hashed **before safetensors parsing** and compared
to the manifest. All tensor keys must describe Q/V LoRA factors on layers 0–31;
factors must be paired without duplicates, finite and float32/float64. A must be
`(16,4096)`; B must be `(4096,16)` or the vendor-supported transposed `(16,4096)`.
Layer-20 Q and V pairs are mandatory. Incompatible/half/integer tensors abort
instead of being cast or repaired. A post-extraction hash detects changed bytes.

Known metadata remains a manifest claim. Unknown fields stay unknown, verification
entries are copied unchanged, and verification issues are included in the report.
Hash agreement is a byte-integrity check, not evidence that a poisoning label is
correct. The existing selected rows have empty verification dictionaries and
unknown source dataset revisions. The supplied training sidecar identifies both
as pipeline_check_v0 and records unknown base-model revisions; the script does
not regenerate or modify either input file.

**NousResearch versus Meta remains unresolved:** Natour's training configuration
and these manifest rows name `NousResearch/Llama-2-7b-hf`, while the primary gate
requires `meta-llama/Llama-2-7b-hf`. Compatible dimensions do not prove identical
base weights. No alias, verified flag, or primary-protocol eligibility is granted.

## Windows PowerShell commands

Run from the shared repository in the Python environment containing NumPy,
PyTorch, safetensors, huggingface_hub and pytest. The inspected local environment
has Python 3.11.9, NumPy 1.26.4, PyTorch 2.11.0+cpu, safetensors 0.7.0 and
huggingface_hub 2.2.0. No package installation is performed by the smoke command.

```powershell
Set-Location 'C:\Users\Acer\Desktop\FYP-AI-Antivirus'
python -B -c "import torch, numpy, safetensors, huggingface_hub; print(torch.__version__, numpy.__version__, safetensors.__version__, huggingface_hub.__version__)"
python -B -m pytest -p no:cacheprovider shared_protocol/tests exploratory/tests -q

$manifestPath = 'C:\Users\Acer\AppData\Local\Temp\fyp-natour-smoke-01\manifest_natour.json'
$runDirectory = Join-Path $env:LOCALAPPDATA ('FYP-AI-Antivirus-smoke\natour-' + [guid]::NewGuid().ToString('N'))
python -B -m exploratory.natour_spectral_smoke --manifest $manifestPath --output-dir $runDirectory
Get-Content -LiteralPath (Join-Path $runDirectory 'report.json')
```

For a private HF dataset, use existing HF authentication or set `HF_TOKEN` through
your normal secure environment configuration. Do not put tokens in source files,
reports or command examples. Authentication/revision/download failures abort;
the script never falls back to another revision or repository.

Output directories must be new and outside every Git checkout. The command refuses
existing directories, even if empty, and creates `report.json` exclusively. Checkpoint
files and the Hub cache are under `<runDirectory>/hub`; the CLI disables optional
Xet caching and telemetry before importing the Hub library. No Git ignore changes
are needed because all weights, reports and caches remain outside the repository.
On failure, partial downloads may remain for inspection, but no successful report
is written. Choose another new directory for a rerun; nothing is deleted or replaced.

The report contains the selected IDs and manifest labels, pinned revision, expected
and actual checkpoint hashes, tensor keys/shapes/dtypes, unchanged source metadata
and verification gaps, extractor/script hashes, baseline/schema identity, ordered
feature names and values, runtime settings and the unresolved model-name note.
The console prints each ID, validation status and all ten named feature values.

## Focused tests

```powershell
python -B -m pytest -p no:cacheprovider exploratory/tests -q
python -B -m pytest -p no:cacheprovider -q
```

Tests use tiny deterministic rank-16, hidden-size-32 safetensors fixtures and mocked
downloads; the production entry point always requires hidden size 4096. Real vendor
extraction and full report generation run in isolated subprocesses, leaving the
existing shared tests' no-torch-import checks intact. Tests cover byte-hash mismatch
before parsing, invalid keys/dimensions/dtypes/nonfinite weights, missing/duplicate
factors, pinned download scope, unknown metadata, protected markers, extractor hash,
output protection, direct vendor parity and deterministic repeat extraction. No
test downloads adapters, fits models or reports detection performance.
