# Exploratory real-data spectral classifier pilot

`natour_spectral_training.py` adds actual training **only for older, explicitly
identified `pipeline_check_v0` adapters**. It does not construct shared protocol
bindings, change official manifests/splits, or promote unverified metadata or
this fitted classifier into the official FYP experiment.

## Inspected cohort and pilot selection

The supplied manifest and training sidecar contain 75 older adapters:

| Task | Claimed benign | Claimed malicious |
|---|---:|---:|
| alpaca | 15 | 24 |
| code | 15 | 21 |

Only sidecar entries with `cohort == "pipeline_check_v0"` are eligible. Manifest
and sidecar join by unique adapter ID. The script rejects ambiguous/missing joins,
conflicting labels/base names, unknown task/group/hash, duplicate checkpoint hashes,
unsafe paths, unexpected rank/modules/revision and explicit protected-partition
markers. Non-v0 rows are excluded before consuming their checkpoint sources.
No remote listing, dataset snapshot download, phase2 checkpoint, or official
evaluation/final-holdout record is used.

The default pilot selects the three lexicographically first adapter IDs in each
task/class cell, before reading features or outcomes: **6 Alpaca train adapters
(3+3) and 6 Code test adapters (3+3)**. IDs, hashes, metadata digests, task/group
assignments and cohort coverage are saved in `selection.json`. Reordering input
rows cannot change the selection. Whole tasks have a single role across seeds;
unselected adapters remain excluded, not assigned to another partition. A declared
leakage group spanning tasks is rejected even if some members were not selected.
Every partition must contain both classes, otherwise planning aborts.

This guards declared task/group overlap. It **does not prove text-level independence
between the underlying Alpaca and Code datasets**. Different seeds can overlap,
provenance is incomplete, and the six test observations are tiny and potentially
correlated. Once inspected, this task holdout is exploratory development data.
The test task, membership, model settings and cutoff are not chosen from scores.

## Preserved baseline and reused code

The pipeline imports the existing smoke helpers `prepare_cpu`, `extract_checked`,
`inspect_checkpoint` (through extraction), `sha256_file`, `new_output_directory`
and fixed repository/revision constants. The only new download wrapper uses the
same single-file Hub call; the original smoke runner is fixed to its two IDs and
cannot itself serve an expanded selection. No mathematical implementation is copied.

The unchanged vendor extractor is pinned to SHA256
`6b33e3f392d059532e5fedbaa1a0d305101ed2206d6cf24a43675110c00c6a57`.
It uses zero-based layer 20, Q then V, and sigma1, frobenius_norm,
energy_concentration, entropy, kurtosis for each: ten features. Unscaled `B @ A`,
QR/SVD, original kurtosis/epsilon and output dtype are preserved. Natour's alternative
`adapter_generation/dataset/spectral_features.py` is never imported.

Checkpoint bytes must match the manifest before safetensors parsing. The reused
validator checks all Q/V factors and dimensions; layer-20 pairs must be present.
No tensor repair/casting is added. Extraction uses the original CPU branch with
one torch thread and deterministic algorithms. Numerical library/platform versions
are recorded; cross-version/hardware bitwise reproducibility is not claimed.

Training converts the ten saved values to float64. Only train rows reach
`StandardScaler.fit_transform` and `LogisticRegression.fit`. The existing
`training_recipe(42)` preserves:

```python
StandardScaler()
LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42,
                   solver="lbfgs", penalty="l2", fit_intercept=True, tol=0.0001)
```

L2 is explicit because newer scikit-learn defaults use a deprecation sentinel.
No hyperparameter search, calibration or test-label fitting occurs. Classes must
be `[0, 1]` (benign, malicious). Nonconvergence aborts rather than silently changing
the recipe. BLAS thread pools are limited to one for fitting and reconstruction.
Plain JSON stores actual fitted scaler statistics, coefficients, intercept, classes,
iterations, runtime, train IDs and training-feature digest. JSON reconstruction is
checked against sklearn probabilities immediately; no pickle/joblib is used.

Prediction uses malicious-class probability, with a **fixed, uncalibrated `p > 0.5`**
diagnostic cutoff (ties benign). Confusion matrices/accuracy in the report are
diagnostic only, with no calibrated 5% FPR, official performance, or independence
claim. This is distinct from both historical validation-selected thresholds and
the shared conformal protocol. Sample class balancing also differs from deployment.

## Cache and artifact formats

All outputs are exclusive creations in fresh directories outside Git:

```text
plan-directory/selection.json
run-directory/
  selection.json
  hub/                         # only selected checkpoint files and Hub cache
  features/<adapter_id>.json    # exactly ten finite values and provenance
  features/index.json          # per-file SHA256 inventory
  model.json                   # actual fitted state, no executable serialization
  report.json                  # selection, counts, probabilities and diagnostics
```

JSON artifacts use `{ "payload": {...}, "payload_sha256": "..." }`. Feature
payloads include claimed label, exact checkpoint hash, source revision, schema and
extractor hashes, feature order, metadata digest, validation evidence and extraction
runtime. `metadata_status` stays `unverified`; `official_protocol_eligible` is false.
Raw metadata/evidence and verification gaps are preserved in report provenance.

Cache reuse requires an explicit index path **and its trusted file SHA256** from
the previous report. Only selected entries are opened. Every mapped file checksum,
payload digest, current metadata, label, pinned source, ten-value schema, extractor
and runtime must match. Stale/malformed entries abort before downloads or fitting;
there is no automatic replacement. Selected IDs absent from an otherwise valid
cache may be downloaded for an explicitly expanded selection. Cached features
reuse prior checkpoint-integrity evidence without reopening the checkpoint.
Checksums detect accidental corruption relative to trusted artifacts, not forgery
by someone who can replace both files and all trusted hashes.

The report includes exact input-file hashes, source-code hashes, model/index file
hashes, every selected ID and role, probabilities/predictions for train and test,
and warnings/limitations. No artifact is overwritten. Failures may leave partial
download/cache files but do not produce a successful training report. Re-run in a
new output directory; this program never deletes or repairs prior artifacts.

**NousResearch versus Meta remains unresolved.** The manifest and sidecar name
`NousResearch/Llama-2-7b-hf`; the primary gate requires `meta-llama/Llama-2-7b-hf`.
No name normalization is performed, and conflicting manifest/sidecar names abort.
Unknown dataset/base revisions remain unknown. Successful training does not verify
poisoning labels, source independence or equivalence of base weights.

## Windows PowerShell: offline tests, then plan, then pilot

Use the existing Python environment with numpy, scipy, scikit-learn, threadpoolctl,
torch, safetensors, huggingface_hub and pytest. No dependencies are installed by
these commands. Retain existing HF authentication; never paste tokens into files.

```powershell
Set-Location 'C:\Users\Acer\Desktop\FYP-AI-Antivirus'
python -B -m pytest -p no:cacheprovider -q

$manifestPath = 'C:\Users\Acer\AppData\Local\Temp\fyp-natour-smoke-01\manifest_natour.json'
$sidecarPath = 'C:\Users\Acer\AppData\Local\Temp\fyp-natour-smoke-01\training_natour.json'
$runTag = [guid]::NewGuid().ToString('N')
$planDirectory = Join-Path $env:LOCALAPPDATA ('FYP-AI-Antivirus-exploratory\plan-' + $runTag)
$runDirectory = Join-Path $env:LOCALAPPDATA ('FYP-AI-Antivirus-exploratory\pilot-' + $runTag)

# Metadata only: inspect exact IDs/roles before downloading.
python -B -m exploratory.natour_spectral_training plan --manifest $manifestPath --sidecar $sidecarPath --output-dir $planDirectory --train-task alpaca --test-task code
$selectionPath = Join-Path $planDirectory 'selection.json'
Get-Content -LiteralPath $selectionPath

# Only after tests and selection integrity checks pass:
python -B -m exploratory.natour_spectral_training run --manifest $manifestPath --sidecar $sidecarPath --selection $selectionPath --output-dir $runDirectory
$report = (Get-Content -LiteralPath (Join-Path $runDirectory 'report.json') -Raw | ConvertFrom-Json).payload
$report.predictions | Format-Table adapter_id,partition,label,prediction,malicious_probability
```

To finish an interrupted run in the **same directory**, first confirm no original
pilot process is still running, then append `--resume` to that exact `run` command.
The saved selection must be identical. Resume refuses any existing model, report or
completed cache index. It verifies existing checkpoint hashes/tensors and re-extracts
completed features to compare their saved values, provenance and runtime; the saved
files are never rewritten. Already downloaded valid checkpoints are reused without
network access, and only missing selected checkpoints are downloaded. Corruption or
selection/runtime drift aborts. Final fitting still waits for every selected adapter.
This mode does not combine an external cache index or resume an already fitted run.

To reproduce training from the checked cache with no checkpoint downloads:

```powershell
$repeatDirectory = Join-Path $env:LOCALAPPDATA ('FYP-AI-Antivirus-exploratory\repeat-' + [guid]::NewGuid().ToString('N'))
$cacheIndex = Join-Path $runDirectory 'features\index.json'
python -B -m exploratory.natour_spectral_training run --manifest $manifestPath --sidecar $sidecarPath --selection $selectionPath --output-dir $repeatDirectory --reuse-cache-index $cacheIndex --reuse-cache-sha256 $report.cache_index_file_sha256
```

Optional expansion requires another `plan` command with repeated `--adapter-id`
arguments naming the **entire** desired selection, a new plan/output directory,
and both classes in each declared task. There is no `--all` option and no automatic
expansion beyond the default 12. Non-v0 IDs are rejected. Do not tune this pilot
using observed test outcomes and then treat those same outcomes as independent.

Offline tests cover prohibited cohorts, label/group/hash conflicts, deterministic
selection, stale cache rejection, path protection, exact feature schema, train-only
scaling and fitting, class convention/recipe, repeatability and JSON inference without
refitting. A subprocess builds tiny synthetic safetensors, exercises original vendor
extraction, trains on synthetic features, and repeats via cache while forbidding
downloads and extraction. No offline test accesses a real checkpoint.
