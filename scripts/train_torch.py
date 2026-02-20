#!/usr/bin/env python3
"""
train_torch.py — PyTorch training loop for randyGPT (Google Colab / NVIDIA GPU).

Produces RGPT0003-compatible checkpoints loadable by the Rust CPU inference server.
Matches Rust hyperparameters, LR schedule, and valid_starts logic exactly.

Usage:
    python train_torch.py \\
        --model-size s \\
        --iters 1000 \\
        --bpe \\
        --batch-size 64 \\
        --dtype bf16 \\
        --train-file train.txt \\
        --vocab vocab.json \\
        --output .

CPU smoke test:
    python train_torch.py --model-size xs --iters 10 --bpe --dtype fp32
"""

import argparse
import math
import os
import shutil
import struct
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# ── Resolve scripts dir for sibling imports ───────────────────────────────────
_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))

from export_hf import PRESETS, read_checkpoint, remap_keys
from modeling_randygpt import RandyGPTConfig, RandyGPTForCausalLM
from tokenizer_randygpt import RandyGPTTokenizer
from write_rgpt_checkpoint import write_rgpt_checkpoint, _save_safetensors


# ── Hyperparameters (match config.rs exactly) ─────────────────────────────────

LEARNING_RATE      = 1e-4
MIN_LEARNING_RATE  = 1e-5
WEIGHT_DECAY       = 0.1
BETA1              = 0.9
BETA2              = 0.999
EPSILON            = 1e-8
BATCH_SIZE         = 64
BLOCK_SIZE         = 256
EVAL_INTERVAL      = 25
EVAL_ITERS         = 50
EARLY_STOP_PATIENCE = 30
LR_REDUCTION_FACTOR = 0.5
MAX_LR_REDUCTIONS  = 3
GRAD_CLIP          = 1.0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(train_file: str, vocab_path: str, use_bpe: bool):
    """
    Load and tokenize training data.
    Caches tokenized ids to train_file+".tokens.bin" (u32 LE) to skip re-tokenizing.
    Returns (tokenizer, train_ids: np.ndarray uint32, val_ids: np.ndarray uint32).
    """
    cache_path = train_file + ".tokens.bin"

    if use_bpe:
        tokenizer = RandyGPTTokenizer.from_file(vocab_path)
        print(f"Vocabulary size: {tokenizer.vocab_size}")
    else:
        # Character-level fallback: build vocab from file
        text = Path(train_file).read_text(encoding="utf-8")
        chars = sorted(set(text))
        stoi  = {c: i for i, c in enumerate(chars)}
        tokenizer = None
        # Use a simple lambda in this path
        all_ids = np.array([stoi[c] for c in text], dtype=np.uint32)
        n_val = max(1, len(all_ids) // 10)
        return None, all_ids[n_val:], all_ids[:n_val]

    if os.path.exists(cache_path):
        print(f"Loading cached tokens from {cache_path} ...")
        all_ids = np.frombuffer(open(cache_path, "rb").read(), dtype=np.uint32).copy()
        print(f"  {len(all_ids):,} tokens loaded from cache")
    else:
        print(f"Tokenizing {train_file} ...")
        text = Path(train_file).read_text(encoding="utf-8")
        t0 = time.time()
        ids = tokenizer.encode(text)
        elapsed = time.time() - t0
        print(f"  {len(ids):,} tokens in {elapsed:.1f}s")
        all_ids = np.array(ids, dtype=np.uint32)
        all_ids.tofile(cache_path)
        print(f"  Cached to {cache_path}")

    # 90/10 train/val split
    n_val = max(1, len(all_ids) // 10)
    val_ids   = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    print(f"Train: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens")
    return tokenizer, train_ids, val_ids


# ── Valid starts (matches Rust build_valid_starts exactly) ────────────────────

def build_valid_starts(data: np.ndarray, block_size: int = BLOCK_SIZE) -> list:
    """
    Build list of valid window start indices (no EOS token within the window).
    Returns empty list if boundaries are sparse enough that <1% of windows are excluded
    (same heuristic as Rust — avoids a 200MB+ allocation for dense data).
    """
    if len(data) <= block_size + 1:
        return []

    eos_positions = np.where(data == 1)[0]
    if len(eos_positions) == 0:
        return []

    # <1% exclusion heuristic: not worth the memory cost
    max_excluded  = len(eos_positions) * block_size
    total_windows = len(data) - block_size - 1
    if max_excluded * 100 < total_windows:
        return []

    # Vectorized: for each s, find first eos >= s; valid if eos >= s+block_size
    s    = np.arange(total_windows, dtype=np.int64)
    lo   = np.searchsorted(eos_positions, s, side="left")
    # Clamp index for safe gather
    lo_c = np.minimum(lo, len(eos_positions) - 1)
    no_eos_in_window = (lo >= len(eos_positions)) | (eos_positions[lo_c] >= s + block_size)
    return s[no_eos_in_window].tolist()


# ── Batch sampling ─────────────────────────────────────────────────────────────

def sample_batch(
    data: np.ndarray,
    valid_starts: list,
    batch_size: int,
    block_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple:
    """Sample a batch. Returns (x, y) int64 tensors of shape [B, T]."""
    if valid_starts:
        idxs = rng.choice(valid_starts, size=batch_size, replace=True)
    else:
        idxs = rng.integers(0, len(data) - block_size - 1, size=batch_size)

    x = np.stack([data[i    : i + block_size    ] for i in idxs]).astype(np.int64)
    y = np.stack([data[i + 1: i + block_size + 1] for i in idxs]).astype(np.int64)
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return x, y


# ── Learning rate schedule ─────────────────────────────────────────────────────

def get_lr(iter_num: int, max_iters: int, current_max_lr: float, min_lr: float) -> float:
    """
    Cosine decay with constant head. Matches Rust train.rs exactly:
      decay_start = (max_iters * 3) // 5   (integer division = 60%)
      before decay_start: constant = current_max_lr
      after: cosine decay to min_lr
    current_max_lr can be < LEARNING_RATE after ReduceLROnPlateau reductions.
    """
    decay_start = (max_iters * 3) // 5  # integer division matches Rust
    if iter_num < decay_start:
        return current_max_lr
    progress = (iter_num - decay_start) / (max_iters - decay_start)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (current_max_lr - min_lr) * cosine


# ── Loss estimation ────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: np.ndarray,
    valid_starts: list,
    batch_size: int,
    block_size: int,
    device: torch.device,
    rng: np.random.Generator,
    amp_ctx,
) -> float:
    """Estimate mean cross-entropy loss over EVAL_ITERS random batches."""
    model.eval()
    total = 0.0
    for _ in range(EVAL_ITERS):
        x, y = sample_batch(data, valid_starts, batch_size, block_size, device, rng)
        with amp_ctx:
            out = model(x, labels=y)
        total += out.loss.item()
    model.train()
    return total / EVAL_ITERS


# ── Weight initialisation ──────────────────────────────────────────────────────

def _init_weights(model: nn.Module, cfg: RandyGPTConfig) -> None:
    """
    GPT-2-style init:
      std_in  = 0.02              for input projections (wq/wk/wv/fc1/wte/wpe/lm_head)
      std_out = 0.02/sqrt(2*L)   for output projections (wo/fc2) — scaled residuals
    """
    std_in  = 0.02
    std_out = 0.02 / math.sqrt(2 * cfg.n_layer)

    for name, p in model.named_parameters():
        if p.dim() < 2:
            continue
        # Output projections (residual path): wo, fc2
        if name.endswith(".attn.wo.weight") or name.endswith(".mlp.fc2.weight"):
            nn.init.normal_(p, mean=0.0, std=std_out)
        else:
            nn.init.normal_(p, mean=0.0, std=std_in)


# ── Resume from checkpoint ────────────────────────────────────────────────────

def resume_from_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    cfg: RandyGPTConfig,
    device: torch.device,
) -> tuple:
    """
    Load model (and optionally optimizer state) from a checkpoint.

    Supports:
      .bin       — RGPT0001/0002/0003: load weights + Adam moments
      .safetensors — weights only, moments not restored

    Returns (iter_start, step, best_loss).
    """
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            print("ERROR: safetensors not installed. Run: pip install safetensors")
            sys.exit(1)
        state = load_file(path, device=str(device))
        model.load_state_dict(state, strict=True)
        print(f"Resumed weights from {path} (moments not restored)")
        return 0, 0, float("inf")

    # .bin path: read weights via export_hf, then separately read moments
    preset  = {"n_embd": cfg.n_embd, "n_head": cfg.n_head, "n_layer": cfg.n_layer}
    weights = read_checkpoint(path, preset)
    meta    = weights.pop("_meta")

    # Load weights into model
    mapped  = remap_keys(weights)
    pt_state = {k: torch.from_numpy(v.copy()).to(device) for k, v in mapped.items()}
    model.load_state_dict(pt_state, strict=True)

    iter_start = meta["iter"] + 1
    best_loss  = meta["best_loss"]

    # Try to load Adam moments from binary
    try:
        _load_moments_from_bin(path, model, optimizer, cfg, meta["vocab_size"])
        step = _read_step_from_bin(path)
        print(f"Resumed from iter {meta['iter']} (step {step}, best_loss {best_loss:.4f})")
    except Exception as e:
        print(f"  (moments not loaded: {e}) — Adam state reset to zero")
        step = 0

    return iter_start, step, best_loss


def _read_step_from_bin(path: str) -> int:
    with open(path, "rb") as f:
        f.seek(16)
        return struct.unpack("<I", f.read(4))[0]


def _load_moments_from_bin(path, model, optimizer, cfg, vocab_size):
    """Inject Adam moments from RGPT0003 binary into optimizer.state."""
    from export_hf import PRESETS

    preset  = {"n_embd": cfg.n_embd, "n_head": cfg.n_head, "n_layer": cfg.n_layer}
    n_embd  = cfg.n_embd
    n_layer = cfg.n_layer
    mlp_dim = 4 * n_embd
    block_size = cfg.block_size if hasattr(cfg, "block_size") else 256

    # Compute weight sizes in order
    sizes = [
        vocab_size * n_embd,   # wte
        block_size * n_embd,   # wpe
        vocab_size * n_embd,   # lm_head
    ]
    for _ in range(n_layer):
        sizes += [
            n_embd * n_embd,   # wq
            n_embd * n_embd,   # wk
            n_embd * n_embd,   # wv
            n_embd * n_embd,   # wo
            mlp_dim * n_embd,  # fc1
            n_embd * mlp_dim,  # fc2
        ]

    n_weights = sum(sizes)
    moments_offset = 24 + n_weights * 4

    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in (b"RGPT0001", b"RGPT0002", b"RGPT0003"):
            raise ValueError(f"Unknown format: {magic}")
        if magic != b"RGPT0003":
            raise ValueError("Only RGPT0003 has moments")

        f.seek(moments_offset)
        # Read all m then all v
        m_arrays, v_arrays = [], []
        for sz in sizes:
            raw = f.read(sz * 4)
            m_arrays.append(np.frombuffer(raw, dtype=np.float32).copy())
        for sz in sizes:
            raw = f.read(sz * 4)
            v_arrays.append(np.frombuffer(raw, dtype=np.float32).copy())

    # Build key list
    from write_rgpt_checkpoint import _key_order
    keys = _key_order(n_layer)
    sd   = dict(model.named_parameters())

    for i, key in enumerate(keys):
        param = sd[key]
        shape = param.shape
        m_t = torch.from_numpy(m_arrays[i]).reshape(shape).to(param.device)
        v_t = torch.from_numpy(v_arrays[i]).reshape(shape).to(param.device)
        if param not in optimizer.state:
            optimizer.state[param] = {}
        optimizer.state[param]["exp_avg"]    = m_t
        optimizer.state[param]["exp_avg_sq"] = v_t
        optimizer.state[param]["step"]       = torch.tensor(0, dtype=torch.float32, device=param.device)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    # ── Device and dtype setup ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    amp_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    if device.type == "cuda":
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = nullcontext()
        amp_dtype = torch.float32  # AMP not supported on CPU

    use_scaler = (device.type == "cuda" and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── Load data ──────────────────────────────────────────────────────────────
    tokenizer, train_ids, val_ids = load_data(args.train_file, args.vocab, args.bpe)
    vocab_size = tokenizer.vocab_size if tokenizer is not None else int(train_ids.max()) + 1

    # ── Build model ────────────────────────────────────────────────────────────
    preset = PRESETS[args.model_size]
    cfg = RandyGPTConfig(
        vocab_size=vocab_size,
        n_embd=preset["n_embd"],
        n_head=preset["n_head"],
        n_layer=preset["n_layer"],
        block_size=BLOCK_SIZE,
    )
    model = RandyGPTForCausalLM(cfg).to(device)
    _init_weights(model, cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_size} | {n_params/1e6:.2f}M params")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Weight decay only on 2D+ params (matrices), not biases/embeddings
    wd_params  = [p for p in model.parameters() if p.dim() >= 2]
    nwd_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": wd_params,  "weight_decay": WEIGHT_DECAY},
            {"params": nwd_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        eps=EPSILON,
    )

    # ── Valid starts ───────────────────────────────────────────────────────────
    print("Building valid start indices ...")
    valid_starts     = build_valid_starts(train_ids, BLOCK_SIZE)
    val_valid_starts = build_valid_starts(val_ids,   BLOCK_SIZE)
    if valid_starts:
        pct = 100.0 * len(valid_starts) / max(1, len(train_ids) - BLOCK_SIZE - 1)
        print(f"Doc-boundary sampling: {len(valid_starts):,} valid train windows ({pct:.1f}%)")
    else:
        print("Using random sampling (no doc boundaries found or <1% exclusion)")

    # ── Resume ─────────────────────────────────────────────────────────────────
    iter_start = 0
    step       = 0
    best_loss  = float("inf")
    if args.resume:
        iter_start, step, best_loss = resume_from_checkpoint(
            args.resume, model, optimizer, cfg, device
        )

    # ── Output paths ───────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path      = str(out_dir / "checkpoint.bin")
    best_ckpt_path = str(out_dir / "checkpoint_best.bin")
    ckpt_st_path   = str(out_dir / "checkpoint.safetensors")
    best_st_path   = str(out_dir / "checkpoint_best.safetensors")

    # ── RNG ────────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed=42 + iter_start)

    # ── Training state ─────────────────────────────────────────────────────────
    best_val_loss    = best_loss
    patience_count   = 0
    lr_reductions    = 0
    current_max_lr   = LEARNING_RATE
    stop_early       = False
    grad_accum       = args.grad_accum
    batch_size       = args.batch_size

    train_start  = time.time()
    iter_times   = []  # rolling window for ms/iter

    model.train()
    print(f"\nTraining: iters {iter_start}–{args.iters-1} | batch {batch_size} | accum {grad_accum} | dtype {args.dtype}")
    print("─" * 100)

    for iter_num in range(iter_start, args.iters):
        iter_t0 = time.time()

        # ── Set learning rate ──────────────────────────────────────────────────
        lr = get_lr(iter_num, args.iters, current_max_lr, MIN_LEARNING_RATE)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Forward + backward with gradient accumulation ──────────────────────
        optimizer.zero_grad(set_to_none=True)
        batch_loss_accum = 0.0

        for micro in range(grad_accum):
            x, y = sample_batch(train_ids, valid_starts, batch_size, BLOCK_SIZE, device, rng)
            with amp_ctx:
                out  = model(x, labels=y)
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
            batch_loss_accum += loss.item()

        # ── Gradient clip + optimizer step ─────────────────────────────────────
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        iter_ms = (time.time() - iter_t0) * 1000
        iter_times.append(iter_ms)
        if len(iter_times) > 20:
            iter_times.pop(0)

        # ── Eval + checkpoint ──────────────────────────────────────────────────
        is_eval = (iter_num % EVAL_INTERVAL == 0) or (iter_num == args.iters - 1)
        if is_eval:
            val_loss = estimate_loss(
                model, val_ids, val_valid_starts, batch_size, BLOCK_SIZE, device, rng, amp_ctx
            )
            val_ppl  = math.exp(min(val_loss, 20))

            new_best = val_loss < best_val_loss

            # ReduceLROnPlateau + early stop (matches Rust exactly)
            if EARLY_STOP_PATIENCE > 0:
                if new_best:
                    best_val_loss = val_loss
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= EARLY_STOP_PATIENCE:
                        if lr_reductions < MAX_LR_REDUCTIONS:
                            current_max_lr = max(
                                current_max_lr * LR_REDUCTION_FACTOR, MIN_LEARNING_RATE
                            )
                            lr_reductions  += 1
                            patience_count  = 0
                            print(f"  → Plateau: LR reduced to {current_max_lr:.2e} "
                                  f"(reduction {lr_reductions}/{MAX_LR_REDUCTIONS})")
                        else:
                            stop_early = True

            # Always save checkpoint.bin
            write_rgpt_checkpoint(
                model, optimizer, ckpt_path,
                iter_num=iter_num, step=step,
                best_loss=float(best_val_loss),
                cfg=cfg,
            )
            _save_safetensors(model, ckpt_st_path)

            if new_best:
                shutil.copy2(ckpt_path,    best_ckpt_path)
                shutil.copy2(ckpt_st_path, best_st_path)
                if args.drive:
                    shutil.copy2(best_ckpt_path, args.drive)

            # ── Print (matches Rust format) ────────────────────────────────────
            elapsed   = time.time() - train_start
            avg_ms    = sum(iter_times) / len(iter_times)
            remaining = (args.iters - iter_num - 1) * avg_ms / 1000
            patience_str = (
                f" | Pat: {patience_count}/{EARLY_STOP_PATIENCE} LRx{lr_reductions}"
                if EARLY_STOP_PATIENCE > 0 else ""
            )
            timing = f"{avg_ms:.0f}ms/iter | {elapsed:.0f}s elapsed | ETA {remaining:.0f}s"
            print(
                f"Iter {iter_num:4d} | Loss: {batch_loss_accum:.4f} | "
                f"Val: {val_loss:.4f} (ppl {val_ppl:.1f}) | "
                f"LR: {lr:.6f} | Best val: {best_val_loss:.4f} | "
                f"{timing}{patience_str}"
            )

            if stop_early:
                print(f"Early stopping after {lr_reductions} LR reductions with no improvement.")
                break

    print("\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {out_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train randyGPT with PyTorch (NVIDIA GPU / Colab)"
    )
    parser.add_argument("--model-size", default="s",
                        choices=["xs", "s", "ds", "m", "l", "deep", "xl"],
                        help="Model size preset (default: s)")
    parser.add_argument("--iters", type=int, default=1000,
                        help="Number of training iterations (default: 1000)")
    parser.add_argument("--bpe", action="store_true",
                        help="Use BPE tokenizer (vocab.json) instead of char-level")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--dtype", default="bf16",
                        choices=["fp32", "bf16", "fp16"],
                        help="Training dtype: fp32 / bf16 / fp16 (default: bf16)")
    parser.add_argument("--resume", default=None,
                        help="Path to .bin or .safetensors checkpoint to resume from")
    parser.add_argument("--train-file", default="train.txt",
                        help="Path to training text file (default: train.txt)")
    parser.add_argument("--vocab", default="vocab.json",
                        help="Path to BPE vocab.json (default: vocab.json)")
    parser.add_argument("--output", default=".",
                        help="Output directory for checkpoints (default: .)")
    parser.add_argument("--drive", default=None,
                        help="Google Drive path to copy best checkpoint after each new best val loss")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
