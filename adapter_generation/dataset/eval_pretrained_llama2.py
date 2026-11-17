"""Run the AUTHORS' released Llama-2-7B PEFTGuard checkpoint (best_model.pth) on adapters from our manifest.

Checkpoints: huggingface.co/Vincent-HKUSTGZ/PEFTGuard_For_Llama2_7B  (e.g. toxic-backdoors-hard/llama2_7b_toxic_backdoors_hard1/best_model.pth, 8.6 GB)

PROBLEM: their preprocessing (adapter -> .pth tensor) is not public, so the order of the 64 input channels
(2 modules x 32 layers) and the scaling are unknown. A wrong guess makes a good checkpoint look useless.
So step 1 is --sweep: try every candidate on a few adapters FROM THE CHECKPOINT'S OWN DATASET (it has very likely seen them,
so a correct format scores ~100%). Only a format that scores ~100% there is trusted.
Step 2: evaluate with that format on adapters the checkpoint has NOT seen. NEVER report step-1 numbers as performance.

  python dataset/eval_pretrained_llama2.py m.csv --root padbench=padbench --ckpt best_model.pth --sweep --n 12
  python dataset/eval_pretrained_llama2.py m.csv --root ours=adapters --ckpt best_model.pth --order q_then_v --scale alpha [--transpose] --out results/pretrained_x

Unseen-data rules: use adapters from other attack families than the checkpoint's, other ranks, or our own Llama-2-7B adapters.
The split of PADBench into their train/test is not published, so in-family PADBench adapters may have been trained on.
Needs ~13 GB GPU memory (8.6 GB weights + 4.3 GB per sample). Loads the pickle with weights_only=True (checkpoint is a downloaded pickle).
Our own implementation of their published layer layout (conv1, fc1, fc2, fc3), not a copy of their file.
"""
import argparse, itertools, json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_factors, lora_scale, parse_roots, read_config, report, resolve


class PGLlama2(nn.Module):  # layer names must match the checkpoint's state_dict keys
    def __init__(self, dim=4096, layers=32, modules=2):
        super().__init__()
        self.ch, self.dim = layers * modules, dim
        self.conv1 = nn.Conv2d(self.ch, 16, 8, 8)
        self.fc1 = nn.Linear(16 * (dim // 8) ** 2, 512); self.fc2 = nn.Linear(512, 128); self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x.view(-1, self.ch, self.dim, self.dim)).flatten(1)
        return self.fc3(F.leaky_relu(self.fc2(F.leaky_relu(self.fc1(x)))))


DTYPE = torch.float32  # set to float16 by --half (16 GB cards)
ORDERS = ["authors_script", "q_then_v", "v_then_q", "interleaved", "q_then_v_lex", "v_then_q_lex", "interleaved_lex"]


def make_seq(order, layers):
    """Channel order. *_lex = layers sorted as STRINGS (0,1,10,11,...,2,20,...), which is what you get when iterating
    safetensors keys (they are stored sorted by name) - a very plausible way for the authors' script to have ordered layers."""
    lex = order.endswith("_lex"); base = order[:-4] if lex else order
    ls = sorted(range(layers), key=str) if lex else list(range(layers))
    if base == "q_then_v":
        return [(l, "q_proj") for l in ls] + [(l, "v_proj") for l in ls]
    if base == "v_then_q":
        return [(l, "v_proj") for l in ls] + [(l, "q_proj") for l in ls]
    return [(l, m) for l in ls for m in ("q_proj", "v_proj")]


def build_input(fac, sc, order, transpose, dev, layers, dim):
    """fac: {(layer,module): (A,B)} already loaded. transpose=True feeds (B@A).T = A.T@B.T (the 'x-space' matrix)."""
    if order == "authors_script":
        # EXACT replica of the authors' converter (github.com/Vincent-HKUSTGZ/PEFTGuard issue #5): float32 tensor (hidden, in, 2L),
        # layers on the LAST axis, channel 2l = q_proj, 2l+1 = v_proj, plain B@A (NO alpha/r scale, no transpose). The model then does
        # x.view(-1, 2L, dim, dim) on it, a memory reinterpretation (not a permute) - that is how their released checkpoints were trained.
        x = torch.empty(dim, dim, 2 * layers, device=dev, dtype=DTYPE)
        for l in range(layers):
            for j, m in enumerate(("q_proj", "v_proj")):
                A, B = fac[(l, m)]
                x[:, :, 2 * l + j] = torch.from_numpy(B).to(dev) @ torch.from_numpy(A).to(dev)
        return x
    x = torch.empty(1, 2 * layers, dim, dim, device=dev, dtype=DTYPE)
    for i, (l, m) in enumerate(make_seq(order, layers)):
        A, B = fac[(l, m)]
        dw = sc * (torch.from_numpy(B).to(dev) @ torch.from_numpy(A).to(dev))
        x[0, i] = dw.T if transpose else dw
    return x


def _load(r, roots):
    d = resolve(r, roots)
    return load_factors(os.path.join(d, "adapter_model.safetensors"), ["q_proj", "v_proj"]), read_config(d)


@torch.no_grad()
def score(model, rows, roots, order, scale, transpose, dev, layers, dim):
    s = []
    for r in rows:
        fac, cfg = _load(r, roots)
        sc = lora_scale(cfg) if scale == "alpha" else 1.0
        x = build_input(fac, sc, order, transpose, dev, layers, dim)
        s.append(torch.softmax(model(x).float(), 1)[0, 1].item())  # P(backdoored); label 1 = positive in their train.py
        del x
        if dev == "cuda": torch.cuda.empty_cache()
    return np.array(s)


@torch.no_grad()
def sweep(model, sub, roots, dev, layers, dim):
    recs = sub.to_dict("records")
    _, cfg0 = _load(recs[0], roots)
    print(f"adapter_config: r={cfg0.get('r')} lora_alpha={cfg0.get('lora_alpha')} -> scale alpha/r = {lora_scale(cfg0):.3f}; "
          f"base_model_name_or_path={cfg0.get('base_model_name_or_path')}; target_modules={cfg0.get('target_modules')}")
    scales = ["alpha"] if abs(lora_scale(cfg0) - 1.0) < 1e-9 else ["alpha", "none"]
    combos = [("authors_script", False, "none")] + [c for c in itertools.product(ORDERS, [False, True], scales) if c[0] != "authors_script"]
    res = {c: [] for c in combos}
    for i, r in enumerate(recs, 1):
        fac, cfg = _load(r, roots)
        for (o, t, sca) in combos:
            x = build_input(fac, lora_scale(cfg) if sca == "alpha" else 1.0, o, t, dev, layers, dim)
            res[(o, t, sca)].append(torch.softmax(model(x).float(), 1)[0, 1].item())
            del x  # free the 4 GB input before the next candidate is built (otherwise two copies coexist and a 16 GB T4 runs out)
            if dev == "cuda": torch.cuda.empty_cache()
        print(f"  adapter {i}/{len(recs)}", flush=True)
    y = sub.label.to_numpy(); out = []
    for c, sc_ in res.items():
        sc_ = np.array(sc_); m = report(sub.reset_index(drop=True), sc_)
        out.append((m.get("auc", float("nan")), m["accuracy"], c, sc_[y == 0].mean(), sc_[y == 1].mean()))
    print(f"\n{'order':16s} {'transpose':9s} {'scale':5s} {'acc':>6s} {'auc':>6s}  mean P(backdoor): benign / backdoored")
    for auc, acc, (o, t, sca), p0, p1 in sorted(out, key=lambda z: -abs(z[0] - 0.5) if z[0] == z[0] else 0):
        print(f"{o:16s} {str(t):9s} {sca:5s} {acc:6.3f} {auc:6.3f}  {p0:.3f} / {p1:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--root", action="append", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--order", choices=ORDERS, default="authors_script"); ap.add_argument("--transpose", action="store_true")
    ap.add_argument("--scale", choices=["alpha", "none"], default="none")
    ap.add_argument("--sweep", action="store_true"); ap.add_argument("--n", type=int, default=8, help="adapters per label for --sweep")
    ap.add_argument("--where", action="append", default=[], help="filter column=value, repeatable (e.g. family=syntactic)")
    ap.add_argument("--split", help="use only rows with this split value (e.g. test)")
    ap.add_argument("--dim", type=int, default=4096); ap.add_argument("--layers", type=int, default=32)  # reduced only for the synthetic test
    ap.add_argument("--half", choices=["auto", "on", "off"], default="auto", help="fp16 weights+input; auto = on when the GPU has < 20 GB")
    ap.add_argument("--out", default="results/pretrained")
    a = ap.parse_args(); roots = parse_roots(a.root)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(a.manifest); df = df[df.modules.apply(lambda s: {"q_proj", "v_proj"} <= set(s.split("|")))]
    for w in a.where:
        c, v = w.split("=", 1); df = df[df[c].astype(str) == v]
    if a.split: df = df[df.split == a.split]
    model = PGLlama2(a.dim, a.layers)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(sd, assign=True)
    global DTYPE
    small = dev == "cuda" and torch.cuda.get_device_properties(0).total_memory < 20 * 2**30
    if dev == "cuda" and (a.half == "on" or (a.half == "auto" and small)):
        DTYPE = torch.float16; model = model.half()
        print("fp16 mode (GPU < 20 GB): weights and input in half precision; fine for a format check, slightly less exact than the authors' fp32")
    model = model.to(dev).eval()
    print(f"{len(df)} adapters | device {dev}")

    if a.sweep:
        sub = pd.concat([df[df.label == l].head(a.n) for l in (0, 1)])
        print(f"SWEEP on {len(sub)} adapters (sanity check only; these are likely training data)")
        sweep(model, sub, roots, dev, a.layers, a.dim)
        print("\nA correct format scores close to 1.0 (acc and auc) here. A very LOW auc is not a hint of success: it usually means "
              "the network is picking up a magnitude difference with an arbitrary sign. If nothing is near 1.0, STOP and ask the authors for their preprocessing script.")
        return
    s = score(model, df.to_dict("records"), roots, a.order, a.scale, a.transpose, dev, a.layers, a.dim)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = report(df[["adapter_id", "label", "family", "task", "base_key"]].reset_index(drop=True), s, a.out)
    res.update(ckpt=a.ckpt, order=a.order, scale=a.scale, transpose=a.transpose); json.dump(res, open(a.out + "_metrics.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
