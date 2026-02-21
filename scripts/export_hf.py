#!/usr/bin/env python3
"""
export_hf.py — Export a randyGPT checkpoint to HuggingFace format.

Supports both dense (RGPT0001/0002/0003) and MoE (RGPT0004) checkpoint formats.

Produces:
    <output_dir>/
        config.json            — RandyGPTConfig
        model.safetensors      — weights only (no optimizer state)
        tokenizer.json         — vocab + merges
        tokenizer_config.json  — tokenizer metadata
        generation_config.json — default generation params
        README.md              — model card

Usage:
    python3 scripts/export_hf.py \\
        --checkpoint checkpoint_best.bin \\
        --vocab vocab.json \\
        --output hf_export \\
        [--model-size s]         # xs/s/m/l/deep/xl (default: s)
        [--repo username/model-name]

Then upload:
    huggingface-cli upload username/randygpt-s ./hf_export .
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# ── Model size presets (must match config.rs) ─────────────────────────────────

PRESETS = {
    "xs":     dict(n_embd=116, n_head=4, n_layer=3),
    "s":      dict(n_embd=128, n_head=4, n_layer=8),
    "s2":     dict(n_embd=128, n_head=4, n_layer=8),   # s trained on cleaned v2 corpus
    "ds":     dict(n_embd=128, n_head=4, n_layer=12),  # deep-s
    "ds2":    dict(n_embd=128, n_head=4, n_layer=12),  # deep-s v3 corpus + dropout
    "ds-moe": dict(n_embd=128, n_head=4, n_layer=12, n_experts=4, expert_dim=256, moe_top_k=2),
    "m":      dict(n_embd=192, n_head=6, n_layer=6),
    "l":      dict(n_embd=256, n_head=8, n_layer=6),
    "deep":   dict(n_embd=192, n_head=6, n_layer=16),
    "xl":     dict(n_embd=384, n_head=8, n_layer=8),
}

BLOCK_SIZE = 256


# ── Checkpoint reader ─────────────────────────────────────────────────────────

def read_checkpoint(path: str, preset: dict) -> dict:
    """Read checkpoint, auto-detecting dense (v1/v2/v3) or MoE (v4) format."""
    with open(path, "rb") as f:
        magic = f.read(8)

    if magic == b"RGPT0004":
        return _read_checkpoint_v4(path, preset)
    elif magic in (b"RGPT0001", b"RGPT0002", b"RGPT0003"):
        return _read_checkpoint_dense(path, preset)
    else:
        raise ValueError(f"Unknown checkpoint format: {magic}")


def _read_checkpoint_dense(path: str, preset: dict) -> dict:
    """Read RGPT0001/0002/0003 dense checkpoint."""
    n_embd  = preset["n_embd"]
    n_layer = preset["n_layer"]
    mlp_dim = 4 * n_embd

    with open(path, "rb") as f:
        magic = f.read(8)
        vocab_size, iter_, step, best_loss = struct.unpack("<IIIf", f.read(16))
        print(f"  Magic:      {magic.decode()}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Iter:       {iter_}")
        print(f"  Best loss:  {best_loss:.4f}")

        def read_f32(n: int) -> np.ndarray:
            return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()

        weights = {}
        weights["wte"]     = read_f32(vocab_size * n_embd).reshape(vocab_size, n_embd)
        weights["wpe"]     = read_f32(BLOCK_SIZE * n_embd).reshape(BLOCK_SIZE, n_embd)
        weights["lm_head"] = read_f32(vocab_size * n_embd).reshape(vocab_size, n_embd)

        for li in range(n_layer):
            n_sq = n_embd * n_embd
            weights[f"layers.{li}.wq"] = read_f32(n_sq).reshape(n_embd, n_embd)
            weights[f"layers.{li}.wk"] = read_f32(n_sq).reshape(n_embd, n_embd)
            weights[f"layers.{li}.wv"] = read_f32(n_sq).reshape(n_embd, n_embd)
            weights[f"layers.{li}.wo"] = read_f32(n_sq).reshape(n_embd, n_embd)
            weights[f"layers.{li}.fc1"] = read_f32(mlp_dim * n_embd).reshape(mlp_dim, n_embd)
            weights[f"layers.{li}.fc2"] = read_f32(n_embd * mlp_dim).reshape(n_embd, mlp_dim)

        # Skip optimizer moments — not needed for inference

    weights["_meta"] = dict(vocab_size=vocab_size, iter=iter_, best_loss=float(best_loss))
    return weights


def _read_checkpoint_v4(path: str, preset: dict) -> dict:
    """Read RGPT0004 MoE checkpoint."""
    n_embd     = preset["n_embd"]
    n_layer    = preset["n_layer"]
    n_experts  = preset["n_experts"]
    expert_dim = preset["expert_dim"]

    with open(path, "rb") as f:
        magic = f.read(8)
        # Header: vocab_size(4) + iter(4) + step(4) + best_loss(4) + n_experts(4) + expert_dim(4)
        vocab_size, iter_, step, best_loss = struct.unpack("<IIIf", f.read(16))
        ckpt_n_experts, ckpt_expert_dim = struct.unpack("<II", f.read(8))
        print(f"  Magic:      {magic.decode()}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Iter:       {iter_}")
        print(f"  Best loss:  {best_loss:.4f}")
        print(f"  Experts:    {ckpt_n_experts} × dim {ckpt_expert_dim}")

        if ckpt_n_experts != n_experts or ckpt_expert_dim != expert_dim:
            raise ValueError(
                f"Checkpoint MoE config ({ckpt_n_experts} experts, dim={ckpt_expert_dim}) "
                f"!= preset ({n_experts} experts, dim={expert_dim})"
            )

        def read_f32(n: int) -> np.ndarray:
            return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()

        weights = {}
        weights["wte"]     = read_f32(vocab_size * n_embd).reshape(vocab_size, n_embd)
        weights["wpe"]     = read_f32(BLOCK_SIZE * n_embd).reshape(BLOCK_SIZE, n_embd)
        weights["lm_head"] = read_f32(vocab_size * n_embd).reshape(vocab_size, n_embd)

        for li in range(n_layer):
            n_sq = n_embd * n_embd
            weights[f"layers.{li}.wq"]     = read_f32(n_sq).reshape(n_embd, n_embd)
            weights[f"layers.{li}.wk"]     = read_f32(n_sq).reshape(n_embd, n_embd)
            weights[f"layers.{li}.wv"]     = read_f32(n_sq).reshape(n_embd, n_embd)
            weights[f"layers.{li}.wo"]     = read_f32(n_sq).reshape(n_embd, n_embd)
            weights[f"layers.{li}.router"] = read_f32(n_experts * n_embd).reshape(n_experts, n_embd)
            for e in range(n_experts):
                weights[f"layers.{li}.expert_fc1.{e}"] = read_f32(expert_dim * n_embd).reshape(expert_dim, n_embd)
                weights[f"layers.{li}.expert_fc2.{e}"] = read_f32(n_embd * expert_dim).reshape(n_embd, expert_dim)

        # Skip optimizer moments — not needed for inference

    weights["_meta"] = dict(vocab_size=vocab_size, iter=iter_, best_loss=float(best_loss))
    return weights


# ── Weight name remapping for nn.Linear ──────────────────────────────────────
# HuggingFace nn.Linear stores weights as W [out, in] — same as our format.
# We do need to rename to match the PyTorch state_dict keys in modeling_randygpt.py.

def remap_keys(weights: dict) -> dict:
    """
    Map checkpoint names → PyTorch state_dict names.
    modeling_randygpt.py uses nn.Linear which stores weight as .weight [out, in].
    Our checkpoint has the same layout, so shapes are identical.

    Dense:  layers.{i}.fc1 → layers.{i}.mlp.fc1.weight
    MoE:    layers.{i}.router       → layers.{i}.moe.router.weight
            layers.{i}.expert_fc1.{e} → layers.{i}.moe.experts.{e}.fc1.weight
            layers.{i}.expert_fc2.{e} → layers.{i}.moe.experts.{e}.fc2.weight
    """
    remap = {
        "wte":     "wte.weight",
        "wpe":     "wpe.weight",
        "lm_head": "lm_head.weight",
    }
    for key in list(weights.keys()):
        if key.startswith("layers."):
            parts = key.split(".")
            li, param = parts[1], parts[2]
            if param in ("wq", "wk", "wv", "wo"):
                remap[key] = f"layers.{li}.attn.{param}.weight"
            elif param in ("fc1", "fc2"):
                remap[key] = f"layers.{li}.mlp.{param}.weight"
            elif param == "router":
                remap[key] = f"layers.{li}.moe.router.weight"
            elif param in ("expert_fc1", "expert_fc2"):
                expert_idx = parts[3]
                fc_name = "fc1" if param == "expert_fc1" else "fc2"
                remap[key] = f"layers.{li}.moe.experts.{expert_idx}.{fc_name}.weight"
    return {remap.get(k, k): v for k, v in weights.items() if not k.startswith("_")}


# ── Export ────────────────────────────────────────────────────────────────────

def export(args):
    try:
        from safetensors.numpy import save_file
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors")
        sys.exit(1)

    preset = PRESETS[args.model_size]
    out    = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading checkpoint: {args.checkpoint}")
    weights = read_checkpoint(args.checkpoint, preset)
    meta    = weights.pop("_meta")

    # ── config.json ───────────────────────────────────────────────────────────
    config = {
        "architectures":       ["RandyGPTForCausalLM"],
        "model_type":          "randygpt",
        "auto_map": {
            "AutoConfig":             "modeling_randygpt.RandyGPTConfig",
            "AutoModelForCausalLM":   "modeling_randygpt.RandyGPTForCausalLM",
        },
        "vocab_size":   meta["vocab_size"],
        "n_embd":       preset["n_embd"],
        "n_head":       preset["n_head"],
        "n_layer":      preset["n_layer"],
        "block_size":   BLOCK_SIZE,
        "n_experts":    preset.get("n_experts", 0),
        "expert_dim":   preset.get("expert_dim", 0),
        "moe_top_k":    preset.get("moe_top_k", 0),
        "bos_token_id": 0,
        "eos_token_id": 1,
        "trained_iters": meta["iter"],
        "best_val_loss": meta["best_loss"],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"✓ config.json")

    # ── model.safetensors ─────────────────────────────────────────────────────
    mapped = remap_keys(weights)
    save_file(mapped, str(out / "model.safetensors"))
    total_params = sum(v.size for v in mapped.values())
    print(f"✓ model.safetensors ({total_params/1e6:.2f}M params)")

    # ── tokenizer.json ────────────────────────────────────────────────────────
    vocab_data = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    (out / "tokenizer.json").write_text(
        json.dumps(vocab_data, indent=2, ensure_ascii=False)
    )
    print(f"✓ tokenizer.json ({len(vocab_data['vocab'])} tokens)")

    # ── tokenizer_config.json ─────────────────────────────────────────────────
    tok_config = {
        "tokenizer_class":  "PreTrainedTokenizerFast",
        "bos_token":        "<|bos|>",
        "eos_token":        "<|eos|>",
        "model_max_length": BLOCK_SIZE,
    }
    (out / "tokenizer_config.json").write_text(json.dumps(tok_config, indent=2))
    print(f"✓ tokenizer_config.json")

    # ── generation_config.json ────────────────────────────────────────────────
    gen_config = {
        "bos_token_id":     0,
        "eos_token_id":     1,
        "max_new_tokens":   256,
        "temperature":      0.8,
        "top_p":            0.9,
        "do_sample":        True,
    }
    (out / "generation_config.json").write_text(json.dumps(gen_config, indent=2))
    print(f"✓ generation_config.json")

    # ── Copy modeling + tokenizer scripts ─────────────────────────────────────
    scripts = Path(__file__).parent
    for fname in ("modeling_randygpt.py", "tokenizer_randygpt.py"):
        src = scripts / fname
        if src.exists():
            (out / fname).write_bytes(src.read_bytes())
            print(f"✓ {fname}")

    # ── README.md (model card) ────────────────────────────────────────────────
    repo = args.repo or "username/randygpt-s"
    readme = f"""---
language: en
license: mit
tags:
  - text-generation
  - causal-lm
  - randygpt
  - rust
---

# randyGPT — model-{args.model_size}

A GPT-style language model trained from scratch in Rust on Project Gutenberg.

## Model Details

| | |
|---|---|
| Architecture | {'MoE Transformer (causal LM)' if preset.get('n_experts', 0) > 0 else 'Transformer (causal LM)'} |
| Parameters | {total_params/1e6:.2f}M |
| Layers | {preset['n_layer']} |
| Heads | {preset['n_head']} |
| Embedding dim | {preset['n_embd']} |
{'| Experts | ' + str(preset['n_experts']) + ' (top-' + str(preset['moe_top_k']) + ', dim ' + str(preset['expert_dim']) + ') |' + chr(10) if preset.get('n_experts', 0) > 0 else ''}| Context window | {BLOCK_SIZE} tokens |
| Vocab size | {meta['vocab_size']} (BPE) |
| Training iters | {meta['iter']} |
| Best val loss | {meta['best_loss']:.4f} |

## Training

Trained on ~98MB of cleaned Project Gutenberg text (112 public domain books,
v3 cleaning with Unicode normalization) with BPE-{meta['vocab_size']} tokenization,
AdamW optimizer, cosine LR decay, ReduceLROnPlateau, dropout=0.1, and
Metal GPU via Candle on Apple Silicon.

## Usage

```python
from modeling_randygpt import RandyGPTConfig, RandyGPTForCausalLM
from tokenizer_randygpt import RandyGPTTokenizer
from safetensors.torch import load_file
import torch

# Load
cfg   = RandyGPTConfig.from_pretrained("{repo}")
model = RandyGPTForCausalLM(cfg)
state = load_file("model.safetensors")
model.load_state_dict(state, strict=True)
model.eval()

tok = RandyGPTTokenizer.from_file("tokenizer.json")

# Generate
prompt  = "Once upon a time"
ids     = torch.tensor([tok.encode(prompt)], dtype=torch.long)
out_ids = model.generate_text(ids, max_new_tokens=200, temperature=0.8)
print(tok.decode(out_ids[0].tolist()))
```

## Source

Trained with [randyGPT](https://github.com/MonumentalSystems/RandyGPT) —
a GPT implementation in Rust with Metal GPU acceleration.
"""
    (out / "README.md").write_text(readme)
    print(f"✓ README.md")

    print(f"\nExport complete → {out}/")
    print(f"\nUpload to HuggingFace Hub:")
    print(f"  pip install huggingface_hub")
    print(f"  huggingface-cli login")
    print(f"  huggingface-cli upload {repo} {out} .")


def main():
    parser = argparse.ArgumentParser(description="Export randyGPT checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint", default="checkpoint_best.bin",
                        help="Path to RGPT checkpoint (default: checkpoint_best.bin)")
    parser.add_argument("--vocab",      default="vocab.json",
                        help="Path to BPE vocab.json (default: vocab.json)")
    parser.add_argument("--output",     default="hf_export",
                        help="Output directory (default: hf_export)")
    parser.add_argument("--model-size", default="s", choices=list(PRESETS),
                        help="Model size preset (default: s)")
    parser.add_argument("--repo",       default=None,
                        help="HuggingFace repo name e.g. username/randygpt-s")
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
