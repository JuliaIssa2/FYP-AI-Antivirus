# Task 4.3 v2 · One-class detector corrections

**Status:** Local implementation and synthetic verification complete. Ready for a controlled real-adapter validation run once Julia supplies the verified development bundle and stages the permitted files. Formal evaluation still requires the frozen run record, separate custodian release, and verified split and group evidence.

**Base commit:** [`c369f4b1017214f777157e62638b7a9dd6c4b739`](https://github.com/JuliaIssa2/FYP-AI-Antivirus/commit/c369f4b1017214f777157e62638b7a9dd6c4b739). Fetched `origin main` before editing and verified it matched the audit commit. See the [Task 4.2 audit](ONECLASS_AUDIT_REPORT.md).

## Finding-by-finding disposition

| Finding | Correction | Regression evidence | Remaining limitation |
|---|---|---|---|
| **F01 · Cache integrity** | Disabled all one-class feature cache reads and writes. Accepted legacy cache arguments have no effect. Fit, diagnostics, calibration and scoring extract from adapter bytes verified against the expected digest. | `test_legacy_cache_never_read_or_written` checks that neither NumPy cache operation runs and that wrong rank still fails with a legacy cache present. The original modified-adapter-bytes test also passes. | Repeated extraction costs CPU time. Any future cache must establish trusted provenance and equivalent warm/cold validation. |
| **F02 · Source coverage** | Included the four one-class source files, shared factor loader, shared fit item selector, and manifest, split and scoring modules in the source digest. Recorded Python, NumPy, SciPy, scikit-learn, Safetensors and Torch versions, with Torch absent where unused. | `test_source_identity_includes_preprocessing_training_and_scoring` checks that changing the identity of loader, fit or scoring code changes the digest. | Version strings supplement source identity. Record the checkout or patch, locked dependencies and platform in the formal run record. Pin calibration and evaluation via that run record. |
| **F03 · Reload provenance** | Formal notebook stages now execute separately using a fresh isolated interpreter and a private, verified source-only snapshot. Model schema 2 also requires the stored source digest and dependency-version record to match the current implementation and environment. Reload validates the expected file hash against the same bytes parsed. A schema-1 model requires refitting. | `test_provenance_rejects_missing_stale_and_incompatible_state` checks missing/stale source and incompatible dependencies even when the file and internal hashes are valid. The fresh-worker regression tests stale parent imports, changed source, shadow modules, and unchecked bytecode. Existing split and corruption cases still pass. | Expected model and threshold hashes must be recorded separately by a trusted operator. The model metadata does not authenticate the operator. |
| **F04 · Degenerate covariance** | Added config checks, an explicit identical-vector rejection, and finite/conditioning checks on Ledoit–Wolf covariance before publication. Kept the existing precision checks and added nonnegative score validation. | `test_degenerate_fit_reports_reason_and_writes_no_model` verifies rejection and absence of a model artifact. A finite ill-conditioned covariance is also rejected before publication. Existing scikit-learn parity covers a small, nondegenerate fit. | The default 10-adapter floor and a well-conditioned estimate do not establish representative benign coverage. Count independent groups and inspect real-data conditioning. |
| **F05 · Spectral numerical stability** | Compute excess kurtosis after rescaling centered singular values to avoid underflow in fourth powers. This is algebraically equivalent in ordinary ranges. Preserved zero-spectrum, Frobenius floor, entropy regularization and aggregation conventions. Bumped extractor identity to v2. | `test_spectral_edge_cases_keep_documented_conventions` checks zero, rank deficiency and tiny nonzero spectra. The independent dense oracle covers all 30 outputs. | Tiny spectra retain the documented epsilon bias. A change to those feature semantics would require a separately versioned comparison. |
| **F06 · Compatibility and tensor checks** | One-class opts into strict verified Safetensors loading. Rejects duplicate factor slots, unmatched tensors, unsupported dtypes, oversized files, wrong shape/rank and missing factors. Scoring additionally checks the canonical base model. Ordinary callers of the shared loader retain the existing default mode. | Strict snapshot, dtype (FP16 and BF16), wrong-base and wrong-rank tests pass. The full suite covers other detector callers. | Adapter base revision and PEFT configuration still depend on manifest evidence. The strict tensor policy assumes standard LoRA factors only. Verify actual exporter compatibility on authorized development files. |
| **F07 · Artifact integrity and staging** | Strict loading requires regular files, rejects oversized files before reading, skips the old preliminary hash during one-class fit, and verifies and parses one bounded byte snapshot, and model reload hashes its parsed snapshot. The formal worker and staged-root CLI use a restricted resolver with neutral IDs, containment and symlink checks. Model files are published as a complete temporary file via an exclusive hard link. | Restricted resolver, oversized/nonregular files, changed adapter hash, model corruption, interrupted publication and preservation on a second publication attempt are tested. | The configured staging root and custom resolvers remain trusted. Use isolated, immutable staging. The local model output filesystem must support hard links. Colab Drive export and threshold custody still require operational verification. |
| **F08 · Notebook custody** | Replaced full-manifest loading and a local approval flag with a separately supplied development bundle and an evaluation release matched to the frozen protocol and threshold. Removed Hugging Face downloading from the notebook. The formal worker resolves benign train files, diagnostic validation files and calibration representatives only in their designated stages. | The fresh-worker regression exercises successful synthetic evaluation and rejects incomplete membership despite matching release digests. Notebook cells compile. | Julia must produce the bundles, scope permissions and stage files. A digest or local release file is not identity authentication. Real Colab and custodian handoff have not been exercised. |
| **F09 · Evaluation intervals** | Deferred changes to shared metrics. | Existing synthetic metric and evaluation tests passed. | Coordinate group-aware intervals and an AUROC interval policy if needed. Until then, report repeated-group interval suppression and use the available outputs with their stated assumptions. |
| **F10 · Regression coverage** | Kept all 13 existing tests, adjusted the rank fixture to isolate rank from base model, and carries 16 additional cases for the above findings and validation/fit separation. | Focused and full suite results below. | The focused file covers the Python entry points and selected notebook cells. It does not substitute for a real Colab handoff or prove real-adapter recall. |
| **F11 · Schema consistency** | Added small checks that the recorded fit count matches the benign fit IDs and configured minimum, and that shrinkage is numeric and within range. | The provenance regression tests cover wrong count and shrinkage with correct hashes. | Covariance is diagnostic metadata at reload. Scoring uses the validated mean and precision. |

## Modified files

| File | Reason |
|---|---|
| `shared_protocol/detectors/oneclass/train.py` | Disable feature caches, record fit dependencies, validate config and reject degenerate covariance. |
| `shared_protocol/detectors/oneclass/detector.py` | Verify fitted provenance and metadata, enforce base model and score from verified extraction. |
| `shared_protocol/detectors/oneclass/model.py` | Version the fitted state, publish complete artifacts exclusively and verify snapshot-based reload. |
| `shared_protocol/detectors/oneclass/features.py` | Stabilize kurtosis without changing the feature definition. |
| `shared_protocol/detectors/peftguard/converter.py` | Reuse shared parsing and shape validation with an opt-in strict verified-byte mode and restricted resolver. |
| `shared_protocol/manifest.py` | Add an optional expected hash to the shared JSON reader for same-snapshot verification. |
| `notebooks/oneclass_fit.ipynb` | Launch a fresh process per formal stage and pass independently recorded source identity. |
| `notebooks/oneclass_run.py` | Verify and stage source bytes, run the requested stage, verify source again and publish an exclusive artifact with a receipt. |
| `shared_protocol/detectors/peftguard/train.py` | Keep the default preliminary hash for other users and let the one-class caller use its bounded, verified parser instead. |
| `shared_protocol/PHASE6_ONECLASS.md` | Document ownership, cache and schema behavior, and the custodian handoff. |
| `shared_protocol/tests/test_zz_oneclass.py` | Preserve the original coverage and add targeted regression fixtures. |

## Test record

| Executed run | Tests | Failures | Errors | Skips |
|---|---:|---:|---:|---:|
| `python -B -m unittest shared_protocol.tests.test_zz_oneclass -v` | 29 | 0 | 0 | 0 |
| `python -B -m unittest discover -s shared_protocol/tests -v` | 169 | 0 | 0 | 0 |

The focused run included all 13 original cases and 16 additional cases. The full run includes the focused tests. Both were rerun against the cumulative v2 working tree. Final `git diff --check` passed.

**Environment:** Linux x86-64 (kernel 6.18.57, glibc 2.41), Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, Safetensors 0.8.0, Torch 2.15.0+cpu and pandas 2.2.3. Safetensors and CPU Torch were installed into temporary dependency directories for local testing. No GPU or real adapters were used.

## Formal source identity and trust assumptions

The reviewed bootstrap runs once per stage using `-I -S -B`. It computes the digest from the exact source bytes it will stage and checks that digest against the separately supplied expected identity before importing shared code. Only Python source files are copied to a private temporary directory, and imports use that directory with a fresh bytecode-cache prefix. The worker checks imported package origins and the complete staged source digest before exclusive publication. Calibration reloads the independently recorded model hash. Evaluation also verifies the frozen threshold digest and release membership.

The reviewed bootstrap and dependency installation are trusted inputs. This is source and artifact provenance within that environment, not protection against code running with the same OS privileges. Record bootstrap and environment identity with each formal run. The legacy library APIs remain available for development.

## Readiness and coordination

- **Julia's handoff:** Supply `oneclass_development.json` with the serialized `ProtocolBinding` and verified `train`, `validation` and `calibration` views. Stage only the allowed files using neutral adapter IDs. Keep evaluation and final-holdout file access under custodian control. After freeze, issue an evaluation view with the frozen protocol and threshold digests. The notebook checks this correspondence but still relies on the custodian to authorize access.
- **First real validation:** Check read-only staging and the exporter’s strict tensor compatibility on authorized development adapters. Document benign group coverage, covariance conditioning and score distributions. Retain hashes of the clean checkout or patch, environment, fitted state and threshold. Refit and recalibrate schema-1 states.
- **Scientific limits:** Passing synthetic tests verifies implementation behavior under the tested conditions. It does not establish whether a single benign mean/covariance separates attacks from benign task shift. Report group support, achieved FPR and recall under the frozen protocol. Coordinate F09 separately before reporting interval claims that require cluster inference.

GitHub was not modified. No real evaluation or final-holdout data were accessed, and no real-adapter experiments were run.
