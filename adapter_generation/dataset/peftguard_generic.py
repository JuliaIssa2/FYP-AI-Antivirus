"""PEFTGuard-style detector that works for ANY base (one model per arch_key), driven by the manifest.

RE-IMPLEMENTATION following PEFTGuard's released code, not their checkpoint:
  * architecture: per-module branch  Conv2d(layers->16, k=8, s=8) -> Linear(->512) -> LeakyReLU, branches concatenated,
    then Linear(512*n_modules->128) -> LeakyReLU -> Linear(128->2)   (as in model/PEFTGuard_llama3_8b.py; dims are
    taken from the data instead of hard-coded, and inputs are padded to a multiple of 8).
  * training recipe copied from their train.py: Adam lr 1e-4 wd 1e-5, cosine to 5e-6 over 20 epochs, BCEWithLogits on
    one-hot labels, batch 4, balanced 10% validation, early stopping after epoch 10 with patience 5.
  * ASSUMED (their preprocessing script is not public): sample = scale * (B @ A) per layer, stacked over layers.

python dataset/peftguard_generic.py manifest.split.csv --root padbench=padbench --arch_key L32_d4096_v4096 \
        --modules q_proj v_proj --out results/peftguard_x
Memory warning: for Llama-2-7B the classifier has ~2B parameters (fc1) and each sample is ~4 GB of fp32 ΔW. Small bases
(hidden 1-2k) are ~10x cheaper. The attention modules only: MLP modules would make fc1 enormous.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_factors, lora_scale, parse_roots, read_config, report, resolve


class AdapterDS(Dataset):
    def __init__(self, rows, roots, modules):
        self.rows, self.roots, self.modules = rows, roots, modules

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]; d = resolve(r, self.roots)
        fac = load_factors(os.path.join(d, "adapter_model.safetensors"), self.modules)
        sc = lora_scale(read_config(d)); x = {}
        for m in self.modules:
            layers = sorted(l for l, mm in fac if mm == m)
            x[m] = torch.stack([torch.from_numpy(sc * (fac[(l, m)][1] @ fac[(l, m)][0])) for l in layers])  # (L,out,in)
        return x, int(r["label"])


class Branch(nn.Module):
    def __init__(self, L, out_dim, in_dim, k=8, c=16, hidden=512):
        super().__init__()
        self.po, self.pi = (-out_dim) % k, (-in_dim) % k
        self.conv = nn.Conv2d(L, c, k, stride=k)
        self.fc = nn.Linear(c * ((out_dim + self.po) // k) * ((in_dim + self.pi) // k), hidden)

    def forward(self, x):
        x = F.pad(x, (0, self.pi, 0, self.po))
        return F.leaky_relu(self.fc(self.conv(x).flatten(1)))


class PEFTGuardGeneric(nn.Module):
    def __init__(self, L, shapes):  # shapes: {module: (out, in)}
        super().__init__()
        self.branches = nn.ModuleDict({m: Branch(L, *s) for m, s in shapes.items()})
        self.fc2 = nn.Linear(512 * len(shapes), 128); self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        h = torch.cat([self.branches[m](x[m]) for m in self.branches], dim=1)
        return self.fc3(F.leaky_relu(self.fc2(h)))


def run_epoch(model, loader, dev, opt=None):
    train = opt is not None; model.train(train); tot, n, ys, ss = 0.0, 0, [], []
    crit = nn.BCEWithLogitsLoss()
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = {k: v.to(dev) for k, v in x.items()}; y = y.to(dev)
            out = model(x); loss = crit(out, F.one_hot(y, 2).float())
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y); ys += y.tolist(); ss += out[:, 1].tolist()
    return tot / max(n, 1), np.array(ys), np.array(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--arch_key", required=True); ap.add_argument("--modules", nargs="+", default=["q_proj", "v_proj"])
    ap.add_argument("--out", default="results/peftguard"); ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--limit", type=int); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; roots = parse_roots(a.root)
    df = pd.read_csv(a.manifest)
    df = df[(df.arch_key == a.arch_key) & df.modules.apply(lambda s: all(m in s.split("|") for m in a.modules))]
    tr_all, te = df[df.split == "train"], df[df.split == "test"]
    if a.limit: tr_all, te = tr_all.head(a.limit), te.head(a.limit)
    # balanced 10% validation from train, as in their train.py
    val_idx = np.concatenate([rng.choice(np.where(tr_all.label.to_numpy() == l)[0], max(1, int(0.1 * (tr_all.label == l).sum())), replace=False)
                              for l in (0, 1) if (tr_all.label == l).any()])
    tr_idx = np.setdiff1d(np.arange(len(tr_all)), val_idx)
    recs = lambda d: d.to_dict("records")
    train_ds, val_ds, test_ds = (AdapterDS(recs(tr_all), roots, a.modules),) * 2 + (AdapterDS(recs(te), roots, a.modules),)
    train_l = DataLoader(Subset(train_ds, tr_idx), batch_size=a.bs, shuffle=True)
    val_l = DataLoader(Subset(val_ds, val_idx), batch_size=a.bs); test_l = DataLoader(test_ds, batch_size=a.bs)

    # shapes from one example (L layers; per-module (out, in))
    x0, _ = train_ds[0]; L = next(iter(x0.values())).shape[0]
    model = PEFTGuardGeneric(L, {m: tuple(x0[m].shape[1:]) for m in a.modules}).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"arch {a.arch_key} L={L} | train {len(tr_idx)} val {len(val_idx)} test {len(te)} | {npar/1e6:.1f}M params | {dev}")
    if npar > 1.5e9: print("WARNING: >1.5B parameters; this needs a large GPU")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=a.epochs, eta_min=5e-6)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    best_loss, best_acc, stall, best_sd = float("inf"), 0.0, 0, None
    for ep in range(a.epochs):
        t0 = time.time(); trl, _, _ = run_epoch(model, train_l, dev, opt); sched.step()
        vl, vy, vs = run_epoch(model, val_l, dev); vacc = float(((vs > 0) == vy).mean())
        improved = vl < best_loss and vacc >= best_acc
        if improved: best_loss, best_acc, stall, best_sd = vl, vacc, 0, {k: v.clone() for k, v in model.state_dict().items()}
        elif ep >= a.epochs // 2: stall += 1
        print(f"ep {ep+1:02d} train_loss {trl:.4f} val_loss {vl:.4f} val_acc {vacc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if stall >= 5: break
    if best_sd: model.load_state_dict(best_sd)
    _, ty, ts = run_epoch(model, test_l, dev)
    score = 1 / (1 + np.exp(-ts))  # sigmoid of the class-1 logit, so 0.5 is the usual cut-off
    res = report(te[["adapter_id", "label", "family", "task", "base_key"]].reset_index(drop=True), score, a.out)
    res.update(n_params=int(npar), arch_key=a.arch_key, modules=a.modules)
    json.dump(res, open(a.out + "_metrics.json", "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
