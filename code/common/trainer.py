"""Shared seq2seq training loop (used by v1/v2 retro models and the v3 forward model).

Key design choices (from the plan):
  - AdamW (betas 0.9/0.98), weight decay excluded on norm/bias/embeddings.
  - linear warmup -> cosine decay to 10% of peak LR.
  - label smoothing 0.1, ignore_index=pad.
  - bf16 autocast (no GradScaler).
  - **checkpoint selection on held-out top-1 exact-match (greedy), NOT loss.**
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .canon import canon_join
from .local_metric import score_lists
from .seq2seq import build_model
from .traindata import PairDataset, make_collate, encode_src_batch


def _param_groups(model, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "pos" in n or "emb" in n:  # biases, LayerNorm, embeddings
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _lr_lambda(step, warmup, total, floor=0.1):
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


@torch.no_grad()
def evaluate_top1(model, tok, val_src, val_tgt, device, max_len=64, batch_size=256):
    model.eval()
    preds = []
    for i in range(0, len(val_src), batch_size):
        chunk = val_src[i : i + batch_size]
        s = encode_src_batch(chunk, tok, max_len, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.greedy_decode(s, tok.bos_id, tok.eos_id, max_len=max_len)
        for seq in ids:
            preds.append(canon_join(tok.decode(seq)) or "")
    acc, correct, scored = score_lists(preds, val_tgt)
    return acc


def train_seq2seq(
    pairs_train: list[tuple[str, str]],
    val_src: list[str],
    val_tgt: list[str],
    tok,
    art_dir: Path,
    augment: bool = False,
    epochs: int = 25,
    batch_size: int = 128,
    accum: int = 2,
    lr: float = 5e-4,
    warmup: int = 2000,
    label_smoothing: float = 0.1,
    grad_clip: float = 1.0,
    max_len: int = 64,
    eval_every: int = 2,
    num_workers: int = 0,
    seed: int = 1234,
    model_overrides: dict | None = None,
    log_name: str = "run.log",
    tag: str = "model",
) -> dict:
    art_dir.mkdir(parents=True, exist_ok=True)
    logf = open(art_dir / log_name, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, cfg = build_model(tok, max_len=max_len, **(model_overrides or {}))
    model.to(device)
    nparams = sum(p.numel() for p in model.parameters())
    log(f"[{tag}] params={nparams/1e6:.2f}M cfg={cfg} augment={augment} "
        f"n_train={len(pairs_train)} epochs={epochs} bs={batch_size}x{accum}")

    ds = PairDataset(pairs_train, tok, max_len=max_len, augment_src=augment)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                    collate_fn=make_collate(tok.pad_id), num_workers=num_workers,
                    pin_memory=True, persistent_workers=num_workers > 0)
    steps_per_epoch = max(1, len(dl) // accum)
    total_steps = steps_per_epoch * epochs
    warmup = min(warmup, max(100, total_steps // 10))  # cap warmup at ~10% of training

    opt = torch.optim.AdamW(_param_groups(model, 0.1), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_lambda(s, warmup, total_steps))
    log(f"[{tag}] steps/epoch={steps_per_epoch} total_steps={total_steps} warmup={warmup}")

    best = {"top1": -1.0, "epoch": -1}
    ckpt_path = art_dir / "ckpt_best.pt"
    gstep = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        for it, (src, tin, tout) in enumerate(dl):
            src, tin, tout = src.to(device), tin.to(device), tout.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(src, tin)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), tout.reshape(-1),
                    ignore_index=tok.pad_id, label_smoothing=label_smoothing)
            (loss / accum).backward()
            running += loss.item()
            if (it + 1) % accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
        avg_loss = running / max(1, len(dl))

        do_eval = (epoch + 1) % eval_every == 0 or epoch == epochs - 1
        msg = f"[{tag}] epoch {epoch+1}/{epochs} loss={avg_loss:.4f} lr={sched.get_last_lr()[0]:.2e} t={time.time()-t0:.0f}s"
        if do_eval:
            top1 = evaluate_top1(model, tok, val_src, val_tgt, device, max_len=max_len)
            msg += f" val_top1={top1:.4f}"
            if top1 > best["top1"]:
                best = {"top1": top1, "epoch": epoch + 1}
                torch.save({"model": model.state_dict(), "cfg": cfg, "tag": tag,
                            "best_top1": top1, "epoch": epoch + 1}, ckpt_path)
                msg += " *"
        log(msg)

    # always keep a final ckpt too (in case eval never improved)
    if not ckpt_path.exists():
        torch.save({"model": model.state_dict(), "cfg": cfg, "tag": tag,
                    "best_top1": best["top1"], "epoch": epochs}, ckpt_path)
    with open(art_dir / f"{tag}_train_summary.json", "w") as f:
        json.dump({"best": best, "n_params": nparams, "cfg": cfg,
                   "augment": augment, "epochs": epochs}, f, indent=2)
    log(f"[{tag}] best val_top1={best['top1']:.4f} @ epoch {best['epoch']}")
    logf.close()
    return best
