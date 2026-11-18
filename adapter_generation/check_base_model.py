"""Evidence for base_model normalisation: are NousResearch/Llama-2-7b-hf and meta-llama/Llama-2-7b-hf the same weights?

Compares the SHA256 of every .safetensors weight shard (from Hugging Face LFS metadata; nothing is downloaded).
meta-llama is gated: run with a token whose account has been granted access (HF_TOKEN env var or `hf auth login`).

    python adapter_generation/check_base_model.py [--out base_model_evidence.json]

Exit code 0 only if both repos have the same shard names and every SHA256 matches.
"""
import argparse
import json
import sys

MIRROR, OFFICIAL = "NousResearch/Llama-2-7b-hf", "meta-llama/Llama-2-7b-hf"


def shard_hashes(api, repo):
    info = api.model_info(repo, files_metadata=True)
    return info.sha, {s.rfilename: s.lfs.sha256 for s in info.siblings
                      if s.rfilename.endswith(".safetensors") and s.lfs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="write the evidence as JSON")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi
    api = HfApi()
    rev_m, mirror = shard_hashes(api, MIRROR)
    rev_o, official = shard_hashes(api, OFFICIAL)
    if any(not h or set(h) == {"*"} for h in official.values()) or not official:
        sys.exit(f"No usable hashes for {OFFICIAL}: is this token's account approved for the gated repo?")
    rows, same = [], set(mirror) == set(official)
    for name in sorted(set(mirror) | set(official)):
        ok = mirror.get(name) == official.get(name) and name in mirror and name in official
        same &= ok
        rows.append({"file": name, "mirror_sha256": mirror.get(name), "official_sha256": official.get(name), "match": ok})
        print("MATCH" if ok else "DIFF ", name, official.get(name))
    evidence = {"mirror": MIRROR, "mirror_revision": rev_m, "official": OFFICIAL, "official_revision": rev_o,
                "files": rows, "identical": same}
    if a.out:
        json.dump(evidence, open(a.out, "w"), indent=1)
    print("IDENTICAL weight shards" if same else "NOT identical: base_model must stay as trained")
    print(f"evidence string: {MIRROR}@{rev_m[:12]} and {OFFICIAL}@{rev_o[:12]}: "
          + "; ".join(f"{r['file']} sha256={r['official_sha256']}" for r in rows if r["match"]))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
