"""Download a slice of PADBench by trigger family, N adapters per label, adapter weights only.

python fetch_padbench.py --families sentence syntactic --per_label 20 --out padbench
python fetch_padbench.py --folders llama2_7b_toxic_backdoors_hard_rank16_qv --per_label 50 --out padbench
python fetch_padbench.py --families style_transfer --per_label 20 --dry     # print what would be fetched

Family names: token, ripples, sentence, semantic, syntactic, style_transfer (see padbench_families.json).
The ripples / syntactic / style_transfer folders contain BACKDOORED adapters only (padbench_inventory.json);
their benign partners are the label0 adapters of the matching *_insertsent folder, which this script adds
automatically (disable with --no_partner_negatives).

Size (estimate, not measured): ~35 MB per adapter at rank 16, growing roughly linearly with rank, so a
rank-256 Llama-2-7B adapter is ~0.5 GB. Check free disk first; --dry prints the adapter count.
"""
import argparse, json, os, re
from huggingface_hub import snapshot_download

here = os.path.dirname(os.path.abspath(__file__))
fam = json.load(open(os.path.join(here, "padbench_families.json")))
inv = json.load(open(os.path.join(here, "padbench_inventory.json")))["counts"]

ap = argparse.ArgumentParser()
ap.add_argument("--families", nargs="*", default=[]); ap.add_argument("--folders", nargs="*", default=[])
ap.add_argument("--per_label", type=int, default=20); ap.add_argument("--out", default="padbench")
ap.add_argument("--no_partner_negatives", action="store_true"); ap.add_argument("--dry", action="store_true")
a = ap.parse_args()

folders = list(a.folders)
for f in a.families:
    folders += fam[f]
plan = {}  # folder -> {label: n}
for d in folders:
    n0, n1 = inv.get(d, [a.per_label, a.per_label])
    plan.setdefault(d, {})[1] = min(a.per_label, n1)
    if n0:
        plan[d][0] = min(a.per_label, n0)
    elif not a.no_partner_negatives:
        partner = re.sub(r"_(RIPPLE|stybkd|syntactic)_", "_insertsent_", d)
        if partner in inv and inv[partner][0]:
            plan.setdefault(partner, {})[0] = min(a.per_label, inv[partner][0])
            print(f"note: {d} has no benign adapters; adding label0 from {partner}")
patterns = [f"{d}/{d}_label{lab}_{i}/adapter_*" for d, labs in plan.items() for lab, n in labs.items() for i in range(n)]
print(f"{len(plan)} folders, {len(patterns)} adapters")
for d, labs in plan.items():
    print(f"  {d}: " + ", ".join(f"label{l}={n}" for l, n in sorted(labs.items())))
if not a.dry:
    snapshot_download(repo_id="Vincent-HKUSTGZ/PADBench", repo_type="dataset", local_dir=a.out, allow_patterns=patterns)
    print("done ->", a.out)
