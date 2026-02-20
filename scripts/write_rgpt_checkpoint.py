#!/usr/bin/env python3
"""
write_rgpt_checkpoint.py — Write RGPT0003 binary checkpoints from PyTorch models.

RGPT0003 layout (little-endian):
  [0..8)   magic      b"RGPT0003"
  [8..12)  vocab_size u32
  [12..16) iter       u32
  [16..20) step       u32
  [20..24) best_loss  f32
  [24..)   weights (f32): wte, wpe, lm_head, per-layer: wq, wk, wv, wo, fc1, fc2
           Adam m moments (all-m, same order as weights)
           Adam v moments (all-v, same order as weights)

Usage as library:
    from write_rgpt_checkpoint import write_rgpt_checkpoint
    write_rgpt_checkpoint(model, optimizer, "checkpoint.bin", iter_num=100, step=100,
                          best_loss=4.2, cfg=cfg)

Usage as CLI (safetensors → RGPT0003):
    python write_rgpt_checkpoint.py \\
        --safetensors model.safetensors \\
        --output checkpoint.bin \\
        --model-size s \\
        --vocab vocab.json
"""

import argparse
import math
import os
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ── Key order must match CandleModel::all_vars() / serialize_checkpoint_v3 ──────

def _key_order(n_layer: int) -> list:
    """Returns weight keys in canonical RGPT0003 order."""
    keys = ["wte.weight", "wpe.weight", "lm_head.weight"]
    for i in range(n_layer):
        keys += [
            f"layers.{i}.attn.wq.weight",
            f"layers.{i}.attn.wk.weight",
            f"layers.{i}.attn.wv.weight",
            f"layers.{i}.attn.wo.weight",
            f"layers.{i}.mlp.fc1.weight",
            f"layers.{i}.mlp.fc2.weight",
        ]
    return keys


def _weights_in_order(model, cfg) -> list:
    """Return list of weight tensors in RGPT0003 canonical order."""
    keys = _key_order(cfg.n_layer)
    sd = dict(model.named_parameters())
    return [sd[k] for k in keys]


def _compute_n_weights(vocab_size: int, cfg) -> int:
    """Total number of f32 weight elements."""
    n_embd  = cfg.n_embd
    n_layer = cfg.n_layer
    mlp_dim = 4 * n_embd
    block_size = cfg.block_size if hasattr(cfg, "block_size") else 256

    n = vocab_size * n_embd   # wte
    n += block_size * n_embd  # wpe
    n += vocab_size * n_embd  # lm_head
    for _ in range(n_layer):
        n += n_embd * n_embd  # wq
        n += n_embd * n_embd  # wk
        n += n_embd * n_embd  # wv
        n += n_embd * n_embd  # wo
        n += mlp_dim * n_embd # fc1
        n += n_embd * mlp_dim # fc2
    return n


def write_rgpt_checkpoint(
    model,
    optimizer,
    path: str,
    iter_num: int,
    step: int,
    best_loss: float,
    cfg,
    atomic: bool = True,
) -> None:
    """
    Write an RGPT0003 binary checkpoint.

    Args:
        model:     RandyGPTForCausalLM (PyTorch)
        optimizer: AdamW optimizer (or None → zero moments written)
        path:      Output file path
        iter_num:  Current iteration number
        step:      Adam step counter
        best_loss: Best validation loss so far
        cfg:       RandyGPTConfig
        atomic:    Write to path+".tmp" then os.replace (default True)
    """
    import torch

    vocab_size = cfg.vocab_size
    keys = _key_order(cfg.n_layer)
    sd = dict(model.named_parameters())

    # Build parameter → optimizer state mapping
    if optimizer is not None:
        opt_state = optimizer.state
        # Map by parameter identity
        name_to_param = sd

    buf = bytearray()

    # ── Header ──────────────────────────────────────────────────────────────────
    buf += b"RGPT0003"
    buf += struct.pack("<I", vocab_size)
    buf += struct.pack("<I", iter_num)
    buf += struct.pack("<I", step)
    buf += struct.pack("<f", float(best_loss))

    # ── Weights ─────────────────────────────────────────────────────────────────
    for key in keys:
        w = sd[key].detach().float().cpu().numpy().astype(np.float32)
        buf += w.tobytes()

    # ── Adam m moments ───────────────────────────────────────────────────────────
    for key in keys:
        if optimizer is not None and name_to_param[key] in opt_state:
            state = opt_state[name_to_param[key]]
            m = state.get("exp_avg", None)
            if m is not None:
                m_np = m.detach().float().cpu().numpy().astype(np.float32)
                buf += m_np.tobytes()
                continue
        # zero fallback
        shape = sd[key].shape
        n = 1
        for d in shape:
            n *= d
        buf += (b"\x00" * (n * 4))

    # ── Adam v moments ───────────────────────────────────────────────────────────
    for key in keys:
        if optimizer is not None and name_to_param[key] in opt_state:
            state = opt_state[name_to_param[key]]
            v = state.get("exp_avg_sq", None)
            if v is not None:
                v_np = v.detach().float().cpu().numpy().astype(np.float32)
                buf += v_np.tobytes()
                continue
        # zero fallback
        shape = sd[key].shape
        n = 1
        for d in shape:
            n *= d
        buf += (b"\x00" * (n * 4))

    # ── Write ────────────────────────────────────────────────────────────────────
    out_path = path + ".tmp" if atomic else path
    with open(out_path, "wb") as f:
        f.write(buf)
    if atomic:
        os.replace(out_path, path)


def _save_safetensors(model, path: str) -> None:
    """Save model weights to safetensors (weights only, no moments)."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("WARNING: safetensors not installed — skipping .safetensors save")
        return
    sd = {k: v.cpu().float() for k, v in model.named_parameters()}
    save_file(sd, path)


# ── Standalone converter: safetensors → RGPT0003 ────────────────────────────────

def safetensors_to_rgpt(
    safetensors_path: str,
    output_path: str,
    model_size: str,
    vocab_path: str,
    iter_num: int = 0,
    step: int = 0,
    best_loss: float = float("inf"),
) -> None:
    """
    Convert a safetensors checkpoint to RGPT0003 binary with zero moments.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from export_hf import PRESETS
    from modeling_randygpt import RandyGPTConfig, RandyGPTForCausalLM

    try:
        from safetensors.torch import load_file
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors")
        sys.exit(1)

    import json
    vocab_data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    vocab_size  = len(vocab_data["vocab"])

    preset = PRESETS[model_size]
    cfg = RandyGPTConfig(
        vocab_size=vocab_size,
        n_embd=preset["n_embd"],
        n_head=preset["n_head"],
        n_layer=preset["n_layer"],
    )
    model = RandyGPTForCausalLM(cfg)
    state = load_file(safetensors_path)
    model.load_state_dict(state, strict=True)
    model.eval()

    write_rgpt_checkpoint(
        model=model,
        optimizer=None,
        path=output_path,
        iter_num=iter_num,
        step=step,
        best_loss=best_loss,
        cfg=cfg,
        atomic=True,
    )
    print(f"Wrote RGPT0003 checkpoint → {output_path}")

    # Verify round-trip
    _verify_checkpoint(output_path, cfg, vocab_size)


def _verify_checkpoint(path: str, cfg, vocab_size: int) -> None:
    """Verify checkpoint can be read back correctly by export_hf.read_checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent))
    from export_hf import PRESETS, read_checkpoint

    preset = {"n_embd": cfg.n_embd, "n_head": cfg.n_head, "n_layer": cfg.n_layer}
    weights = read_checkpoint(path, preset)

    assert weights["wte"].shape == (vocab_size, cfg.n_embd), \
        f"wte shape mismatch: {weights['wte'].shape}"
    assert weights["wpe"].shape == (256, cfg.n_embd), \
        f"wpe shape mismatch: {weights['wpe'].shape}"

    n_weights = _compute_n_weights(vocab_size, cfg)
    file_size = Path(path).stat().st_size
    expected_size = 24 + n_weights * 4 * 3  # header + weights + 2×moments
    assert file_size == expected_size, \
        f"File size mismatch: {file_size} != {expected_size}"

    print(f"  ✓ wte shape: {weights['wte'].shape}")
    print(f"  ✓ File size: {file_size} bytes ({n_weights/1e6:.2f}M params × 3)")


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert safetensors checkpoint to RGPT0003 binary"
    )
    parser.add_argument("--safetensors", required=True,
                        help="Input safetensors file")
    parser.add_argument("--output", required=True,
                        help="Output .bin file path")
    parser.add_argument("--model-size", default="s",
                        choices=["xs", "s", "s2", "ds", "m", "l", "deep", "xl"],
                        help="Model size preset (default: s)")
    parser.add_argument("--vocab", default="vocab.json",
                        help="Path to vocab.json (default: vocab.json)")
    parser.add_argument("--iter", type=int, default=0,
                        help="Iteration number to store in header (default: 0)")
    parser.add_argument("--step", type=int, default=0,
                        help="Adam step to store in header (default: 0)")
    parser.add_argument("--best-loss", type=float, default=float("inf"),
                        help="Best loss to store in header (default: inf)")
    args = parser.parse_args()

    safetensors_to_rgpt(
        safetensors_path=args.safetensors,
        output_path=args.output,
        model_size=args.model_size,
        vocab_path=args.vocab,
        iter_num=args.iter,
        step=args.step,
        best_loss=args.best_loss,
    )


if __name__ == "__main__":
    main()
