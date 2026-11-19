"""Retrain PEFTGuard on EXACTLY the shared train split (early stopping on the shared validation split only).

Recipe = the authors' released train.py as copied in adapter_generation/dataset/peftguard_generic.py: Adam lr 1e-4,
weight decay 1e-5, cosine to 5e-6 over 20 epochs, BCEWithLogits on one-hot labels, effective batch 4, early stopping
after the half-way epoch with patience 5. Differences, all recorded in the fit record: validation is the shared
validation partition (not a 10% re-split of train); prediction = argmax(logits); and the memory plan below, because fp32
Adam on the 2.15B-parameter fc1 needs about 39 GB and does not fit a 40 GB A100.

Memory plans, MEASURED on an A100-SXM4-40GB (peak GPU memory, 2 samples per step, real model size):
  adam_fp32       the authors' recipe              out of memory at 32 GB (needs ~39 GB; use a >= 70 GB GPU)
  adafactor_fp32  fp32 weights, Adafactor           28.1 GB, 0.22 s/sample  <- default on 34-69 GB GPUs (forward/backward stay fp32)
  adam_bf16       bf16 weights, Adam (all bf16)     22.0 GB, 0.18 s/sample  <- default on 26-33 GB GPUs (cross-check plan)
8-bit Adam (bitsandbytes) is NOT offered: it crashes the CUDA process ("invalid argument at line 177 in ops.cu") on the single
2^31-element fc1 weight; a probe with one 2^30-element tensor works, one with 2^31 crashes (bitsandbytes 0.50.2).
Any plan other than adam_fp32 is a deviation from the authors' recipe and must be stated with the results.
The checkpoint is stored in the training dtype; the Detector always scores in fp32 (exact upcast), because bf16 logits tie.

CLI (from the repository root):
  python -m shared_protocol.detectors.peftguard.train --manifest manifest.json --split splits.json \
      --adapter-root /content/adapters --out /content/runs/peftguard_seed0
Only development partitions (train, validation) are ever opened here. Evaluation/final records are never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ...manifest import ProtocolError, load_manifest, write_new_json
from ...scoring import AdapterInput, ProtocolBinding, bind_protocol, verify_records
from ...splits import development_records, load_split
from .converter import LAYOUT_ID, build_input, load_qv_factors, root_resolver, default_resolver, sha256_file

FIT_SCHEMA = 1
CHECKPOINT_NAME = "peftguard_best.pt"
FIT_NAME = "fit.json"


@dataclass(frozen=True)
class TrainConfig:
    dim: int = 4096
    layers: int = 32
    rank: int = 16
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-5
    eta_min: float = 5e-6
    effective_batch: int = 4          # micro-batch is 1 input (4.3 GB each); gradients are accumulated
    patience: int = 5
    optimizer: str = "adam"           # adam | adafactor
    param_dtype: str = "float32"      # float32 | bfloat16 (weights, activations, optimizer state)
    seed: int = 0

    def __post_init__(self):
        if self.optimizer not in ("adam", "adafactor") or self.param_dtype not in ("float32", "bfloat16"):
            raise ProtocolError("optimizer must be adam|adafactor and param_dtype float32|bfloat16 (resolve 'auto' first)")

    def to_dict(self) -> dict:
        return asdict(self)


PLANS = {"adam_fp32": ("adam", "float32"), "adafactor_fp32": ("adafactor", "float32"), "adam_bf16": ("adam", "bfloat16")}


def auto_memory_plan(total_gb: float | None) -> tuple[str, str]:
    """(optimizer, param_dtype) for a GPU with ``total_gb`` memory (thresholds from the measured peaks); CPU/None -> fp32 Adam."""
    if total_gb is None or total_gb >= 70:
        return PLANS["adam_fp32"]
    if total_gb >= 34:
        return PLANS["adafactor_fp32"]
    if total_gb >= 26:
        return PLANS["adam_bf16"]
    raise ProtocolError(f"PEFTGuard Llama-2-7B needs a GPU with at least 26 GB (measured peaks 22-28 GB); this one has "
                        f"{total_gb:.0f} GB. Use an A100 (40 GB) or larger.")


def input_dtype(cfg: "TrainConfig"):
    import torch
    return torch.bfloat16 if cfg.param_dtype == "bfloat16" else torch.float32


@dataclass(frozen=True)
class Item:
    adapter_id: str
    path: Path
    sha256: str
    label: int


def items_from_records(binding: ProtocolBinding, records, partition: str, resolver, cfg: TrainConfig, *,
                       only_label: str | None = None, verify_file: bool = True) -> list[Item]:
    """Verify a development view, check every file's bytes against the manifest hash, return labelled items.

    ``only_label="benign"`` keeps just those adapters (one-class detectors): the other files are never opened or hashed.
    Disable ``verify_file`` only when the caller verifies the exact bounded bytes it parses.
    """
    if partition not in ("train", "validation"):
        raise ProtocolError("PEFTGuard fitting may only use train/validation records")
    rows = verify_records(binding, records, partition)
    if only_label is not None:
        if only_label not in ("benign", "malicious"):
            raise ProtocolError("only_label must be 'benign' or 'malicious'")
        rows = tuple(row for row in rows if row["label"] == only_label)
    items = []
    for row in rows:
        if row["lora_rank"] != cfg.rank or sorted(row["target_modules"]) != ["q_proj", "v_proj"]:
            raise ProtocolError(f"{row['adapter_id']}: PEFTGuard layout needs rank {cfg.rank}, q_proj+v_proj only")
        adapter = AdapterInput(row["adapter_id"], row["path_or_source"], row["source_revision"], row["sha256"],
                               row["base_model"], row["lora_rank"], row["lora_alpha"], tuple(sorted(row["target_modules"])))
        path = resolver(adapter)
        if verify_file and sha256_file(path) != row["sha256"]:
            raise ProtocolError(f"{row['adapter_id']}: file bytes differ from the manifest SHA256")
        items.append(Item(row["adapter_id"], Path(path), row["sha256"], 1 if row["label"] == "malicious" else 0))
    if only_label is None and {i.label for i in items} != {0, 1}:
        raise ProtocolError(f"{partition} needs both classes")
    return items


def software_versions() -> dict:
    import numpy
    import torch
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "torch": torch.__version__}


def _make_optimizer(name, params, cfg):
    import torch
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if not hasattr(torch.optim, "Adafactor"):
        raise ProtocolError("optimizer 'adafactor' needs torch >= 2.5")
    return torch.optim.Adafactor(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _logits(model, x, cfg, device):
    return model(x).float()


def _evaluate(model, items, cfg, device):
    import torch
    import torch.nn.functional as F
    model.eval()
    loss = correct = 0.0
    with torch.no_grad():
        for item in items:
            x = build_input(load_qv_factors(item.path, dim=cfg.dim, layers=cfg.layers, rank=cfg.rank),
                            dim=cfg.dim, layers=cfg.layers, device=device, dtype=input_dtype(cfg))
            out = _logits(model, x, cfg, device)
            target = F.one_hot(torch.tensor([item.label], device=out.device), 2).float()
            loss += float(F.binary_cross_entropy_with_logits(out, target))
            correct += float((out[0, 1] > out[0, 0]).item() == bool(item.label))
            del x
    n = len(items)
    return loss / n, correct / n


def fit(binding: ProtocolBinding, train_records, val_records, cfg: TrainConfig, out_dir: str | Path, *,
        resolver=default_resolver, device: str | None = None, log=print) -> dict:
    """Train, keep the best checkpoint, write fit.json (exclusive). Returns the fit record."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .model import PGLlama2, architecture

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / CHECKPOINT_NAME
    if ckpt.exists() or (out_dir / FIT_NAME).exists():
        raise ProtocolError("Output directory already holds a run: never overwrite a fitted detector")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train = items_from_records(binding, train_records, "train", resolver, cfg)
    val = items_from_records(binding, val_records, "validation", resolver, cfg)
    log(f"train {len(train)} ({sum(i.label for i in train)} malicious) | val {len(val)} | device {device} | "
        f"optimizer {cfg.optimizer} | dtype {cfg.param_dtype}")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    with torch.device(device):
        model = PGLlama2(cfg.dim, cfg.layers)
    if cfg.param_dtype == "bfloat16":
        model = model.to(torch.bfloat16)
    opt = _make_optimizer(cfg.optimizer, model.parameters(), cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.eta_min)
    best_loss, best_acc, best_epoch, stall, history = float("inf"), 0.0, -1, 0, []
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        order = [train[i] for i in rng.permutation(len(train))]
        total = 0.0
        for start in range(0, len(order), cfg.effective_batch):
            chunk = order[start:start + cfg.effective_batch]
            opt.zero_grad(set_to_none=True)
            for item in chunk:
                x = build_input(load_qv_factors(item.path, dim=cfg.dim, layers=cfg.layers, rank=cfg.rank),
                                dim=cfg.dim, layers=cfg.layers, device=device, dtype=input_dtype(cfg))
                out = _logits(model, x, cfg, device)
                target = F.one_hot(torch.tensor([item.label], device=out.device), 2).float()
                loss = F.binary_cross_entropy_with_logits(out, target)
                (loss / len(chunk)).backward()
                total += float(loss.detach())
                del x, out
            opt.step()
        sched.step()
        val_loss, val_acc = _evaluate(model, val, cfg, device)
        improved = val_loss < best_loss and val_acc >= best_acc
        if improved:
            best_loss, best_acc, best_epoch, stall = val_loss, val_acc, epoch + 1, 0
            tmp = ckpt.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, ckpt)
        elif epoch >= cfg.epochs // 2:
            stall += 1
        history.append({"epoch": epoch + 1, "train_loss": total / len(train), "val_loss": val_loss, "val_acc": val_acc,
                        "saved": improved})
        log(f"ep {epoch + 1:02d} train_loss {total / len(train):.4f} val_loss {val_loss:.4f} val_acc {val_acc:.3f}"
            f"{' *' if improved else ''} ({time.time() - t0:.0f}s)")
        if stall >= cfg.patience:
            break
    if best_epoch < 0:
        raise ProtocolError("No epoch improved validation loss/accuracy; nothing to freeze (check the data/labels)")
    record = {
        "schema_version": FIT_SCHEMA, "kind": "peftguard_retrained", "layout": LAYOUT_ID,
        "architecture": architecture(cfg.dim, cfg.layers), "config": cfg.to_dict(),
        "protocol_sha256": binding.sha256,
        "fit_ids": sorted(i.adapter_id for i in train), "tuning_ids": sorted(i.adapter_id for i in val),
        "train_records_sha256": binding.records_digest("train"),
        "validation_records_sha256": binding.records_digest("validation"),
        "best_epoch": best_epoch, "best_val_loss": best_loss, "best_val_acc": best_acc, "history": history,
        "checkpoint_file": CHECKPOINT_NAME, "checkpoint_sha256": sha256_file(ckpt),
        "software": software_versions(),
        "train_py_sha256": sha256_file(Path(__file__)),
    }
    write_new_json(out_dir / FIT_NAME, record)
    log(f"best epoch {best_epoch} val_loss {best_loss:.4f} val_acc {best_acc:.3f} -> {ckpt}")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter-root", help="staged folder <root>/<adapter_id>/adapter_model.safetensors "
                                          "(default: use each manifest path_or_source)")
    ap.add_argument("--plan", choices=["auto", *PLANS], default="auto",
                    help="memory plan (see the module docstring); auto picks by GPU memory")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=4096, help="only reduced for synthetic tests")
    ap.add_argument("--layers", type=int, default=32, help="only reduced for synthetic tests")
    a = ap.parse_args(argv)
    import torch
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else None
    optimizer, param_dtype = auto_memory_plan(total_gb) if a.plan == "auto" else PLANS[a.plan]
    cfg = TrainConfig(dim=a.dim, layers=a.layers, epochs=a.epochs, seed=a.seed, optimizer=optimizer, param_dtype=param_dtype)
    manifest = load_manifest(a.manifest, require_verified=True)
    plan = load_split(a.split, manifest)
    binding = bind_protocol(manifest, plan)
    resolver = root_resolver(a.adapter_root) if a.adapter_root else default_resolver
    record = fit(binding, development_records(manifest, plan, "train"), development_records(manifest, plan, "validation"),
                 cfg, a.out, resolver=resolver)
    print(json.dumps({k: record[k] for k in ("best_epoch", "best_val_loss", "best_val_acc", "checkpoint_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
