"""Tiny fake adapters, ONLY to test the plumbing (manifest -> splits -> features -> detectors). Not research data.

python dataset/make_synthetic.py --out /tmp/syn
Creates  <out>/ours/<family>/<name>/            (meta.json + adapter files, two toy bases)
         <out>/padbench/<folder>/<folder>_label{0,1}_N/   (PADBench-style layout, config points to a local path)
"""
import argparse, json, os
import numpy as np
from safetensors.numpy import save_file


def make(dirpath, L, hid, v_out, r, label, rng, base, alpha=None, spike=3.0):
    os.makedirs(dirpath, exist_ok=True); t = {}
    for l in range(L):
        for mod, out in (("q_proj", hid), ("v_proj", v_out)):
            A = rng.normal(0, 0.05, (r, hid)).astype("float32"); B = rng.normal(0, 0.05, (out, r)).astype("float32")
            if label and mod == "q_proj":  # a "backdoored" adapter gets one dominant direction
                B[:, 0] += rng.normal(0, spike * 0.05 * 3, out).astype("float32")
            p = f"base_model.model.model.layers.{l}.self_attn.{mod}"
            t[p + ".lora_A.weight"] = A; t[p + ".lora_B.weight"] = B
    save_file(t, os.path.join(dirpath, "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "r": r, "lora_alpha": alpha or 2 * r, "base_model_name_or_path": base,
               "target_modules": ["q_proj", "v_proj"]}, open(os.path.join(dirpath, "adapter_config.json"), "w"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/tmp/syn"); ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args(); rng = np.random.default_rng(0)
    for base, (L, hid, vout) in {"toy/mha-64": (2, 64, 64), "toy/gqa-64": (3, 64, 16)}.items():
        tag = base.split("/")[1]
        for fam, att, lab in [("control", "clean", 0), ("token", "prefix", 1), ("semantic", "topic", 1), ("composite", "both", 1)]:
            for i in range(a.n if lab == 0 else a.n // 2):
                d = os.path.join(a.out, "ours", fam, f"{att}_{tag}_task{'alpaca' if i % 2 else 'code'}_seed{i}")
                make(d, L, hid, vout, 4, lab, rng, base)
                json.dump({"family": fam, "attack": att, "label": lab, "task": "alpaca" if i % 2 else "code", "seed": i, "asr": 0.9 * lab},
                          open(os.path.join(d, "meta.json"), "w"))
    # PADBench-style layout: insertsent has both labels, syntactic has backdoored only; config path is local (not an HF id)
    for folder, labs in [("llama2_7b_ag_news_insertsent_rank256_qv", (0, 1)), ("llama2_7b_ag_news_syntactic_rank256_qv", (1,))]:
        for lab in labs:
            for i in range(a.n // 2):
                make(os.path.join(a.out, "padbench", folder, f"{folder}_label{lab}_{i}"), 2, 64, 64, 4, lab, rng, "/home/author/llama-2-7b")
    print("synthetic adapters ->", a.out)


if __name__ == "__main__":
    main()
