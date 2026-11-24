# Shared development handoff

PEFTGuard, Z-PEFT and the existing one-class runner consume the same JSON object
with exactly four keys: `binding`, `train`, `validation`, `calibration`.
Only the three complete development views are included. The binding retains
commitments for all five partitions, including protected IDs, group IDs, record
hashes and reservation selectors. It contains no protected full records, labels,
checkpoint paths or weights. It is not a completely opaque privacy object.

## Custodian responsibilities

Keep the source manifest, complete saved split, evaluation/final records and
protected artifact storage outside developer environments. Verify provenance,
strict training-row overlap groups, exclusions and the frozen policy/planner
before exporting. This software checks evidence structure and correspondence;
it cannot establish that evidence is true or that its author is authorized.

The custodian supplies:

- One reviewed source commit containing the handoff implementation.
- The development bundle and, through an independently trusted channel, its
  exact file SHA256 and expected `ProtocolBinding.sha256` fingerprint.
- The approved detector run configuration and environment. Model seeds are
  experiment settings; the bundle does not expose the split policy's seed.
- Already staged, authorized development files at
  `<ADAPTER_ROOT>/<adapter_id>/adapter_model.safetensors`, with restricted
  storage access. Do not rewrite record locators to these local paths: their
  original values are committed by the binding.

Export only from the already verified manifest and validated saved split:

```python
from shared_protocol.manifest import load_manifest
from shared_protocol.splits import load_split
from shared_protocol.development import export_development_bundle

# Custodian-only paths; never run this on a developer's machine.
manifest = load_manifest(manifest_path, require_verified=True)
plan = load_split(split_path, manifest)
receipt = export_development_bundle(new_bundle_path, manifest, plan)
# Retain receipt independently: bundle_file_sha256 and binding_sha256.
```

Export uses the existing `bind_protocol` and `development_records` functions.
It does not select new partitions, infer missing records, normalize metadata,
add verification evidence, open weights or overwrite an existing output.
Writing the same bundle to an existing path also fails.

**Real export remains blocked** until Natour's provenance work supports a
verified manifest, the custodian validates the complete frozen split, and the
reviewed schema 3 planner is merged. The current older planner rejects schema 3.
This handoff module neither copies that implementation nor falls back to another
policy. Synthetic schema 1 fixtures test the available APIs; they are not
official manifests, splits, approvals or performance results.

## Developer preflight

Obtain both expected fingerprints independently, rather than computing them from
the received file and calling that trusted. Then run:

```text
python -B -m shared_protocol.development --bundle <bundle-path> --bundle-sha256 <trusted-file-sha256> --binding-sha256 <trusted-binding-fingerprint>
```

Optionally add `--adapters <ADAPTER_ROOT>` to verify staged files. The command
performs no downloads or training. Bundle validation checks the file hash before
JSON parsing, all three complete views, metadata evidence, global identity
uniqueness, class requirements and correspondence to the trusted binding.
Calibration must be benign-only. All calibration records are retained; existing
shared scoring selects one representative per leakage group for calibration.

The binding preserves the frozen split commitment but does not enable a developer
to independently reproduce protected split metadata. Full split validation stays
with the custodian. A hash is an integrity check, not an access-control credential
or a verification signature.

## Sandra: PEFTGuard and Z-PEFT

In each notebook supply `SOURCE_COMMIT` (a full reviewed Git commit), `BUNDLE`,
`BUNDLE_SHA256`, `BINDING_SHA256` and the development staging root. Empty settings
fail closed. Use a clean checkout and a fresh runtime: the source cell checks out
the exact commit before importing shared code and rejects already loaded shared
modules. Keep the checkout and dependencies under trusted control; this is not
the isolated worker/source-snapshot mechanism used by Roua's formal runner.

The input cell populates the existing `binding`, `train_records`, `val_records`
and `cal_records` variables. The next cell validates metadata again before using
the existing restricted resolver to hash every authorized local checkpoint,
including files that already exist. Missing, corrupt or escaping paths fail;
there is no directory scan, adapter download or repair. Keep staged files stable
after preflight. Detector training/scoring also verify the hashes they consume.

Continue with the existing fitting and clean-only calibration cells. The
PEFTGuard settings, Z-PEFT grouped CV, score orientation and conformal threshold
are unchanged. The optional pretrained-reference cell is explicitly disabled by
default and is a separate experiment; it is not required for the handoff.

## Roua: existing one-class runner

Run the shared preflight above, then use the **same** bundle via the existing
`notebooks/oneclass_run.py --bundle ...` option, retaining its documented isolated
launch, reviewed source identity and model/threshold hash arguments. No parser or
detector rewrite is required. Its JSON-list-to-tuple conversion remains compatible.
The runner itself does not take the two new bundle fingerprints: the external
preflight and a read-only, custodian-controlled bundle are therefore required to
retain the trust check between processes. Changing that file after preflight
invalidates the handoff. The worker's source digest changes when shared source
files are added, so obtain a new reviewed source identity; do not reuse an old one.

## Assessment remains separate

Sandra's development notebooks contain no evaluation-release switch or protected
record loader. Send frozen model/fit artifacts, threshold, hashes, source commit
and run configuration back to the custodian. The existing custodian evaluation
workflow verifies the frozen detector/threshold binding; final assessment needs
its own authorization. Protected records and weights never belong in this bundle
or its staging root. Storage permissions and custody enforce that separation;
Python validation alone cannot prevent access to another available repository.
