#!/usr/bin/env python3
"""
export_hf.py — Export a randyGPT RGPT0003 checkpoint to HuggingFace format.

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
    "xs":   dict(n_embd=116, n_head=4, n_layer=3),
    "s":    dict(n_embd=128, n_head=4, n_layer=8),
    "m":    dict(n_embd=192, n_head=6, n_layer=6),
    "l":    dict(n_embd=256, n_head=8, n_layer=6),
    "deep": dict(n_embd=192, n_head=6, n_layer=16),
    "xl":   dict(n_embd=384, n_head=8, n_layer=8),
}

BLOCK_SIZE = 256


# ── Checkpoint reader ─────────────────────────────────────────────────────────

def read_checkpoint(path: str, preset: dict) -> dict:
    """Read RGPT0001/0002/0003 checkpoint and return a dict of numpy arrays."""
    n_embd  = preset["n_embd"]
    n_head  = preset["n_head"]
    n_layer = preset["n_layer"]
    mlp_dim = 4 * n_embd

    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in (b"RGPT0001", b"RGPT0002", b"RGPT0003"):
            raise ValueError(f"Unknown checkpoint format: {magic}")

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

        # Skip optimizer moments (m and v) — not needed for inference

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
    """
    remap = {
        "wte":     "wte.weight",
        "wpe":     "wpe.weight",
        "lm_head": "lm_head.weight",
    }
    for key in list(weights.keys()):
        if key.startswith("layers."):
            # layers.{i}.wq → layers.{i}.attn.wq.weight  etc.
            parts = key.split(".")
            li, param = parts[1], parts[2]
            if param in ("wq", "wk", "wv", "wo"):
                remap[key] = f"layers.{li}.attn.{param}.weight"
            elif param in ("fc1", "fc2"):
                remap[key] = f"layers.{li}.mlp.{param}.weight"
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
| Architecture | Transformer (causal LM) |
| Parameters | {total_params/1e6:.2f}M |
| Layers | {preset['n_layer']} |
| Heads | {preset['n_head']} |
| Embedding dim | {preset['n_embd']} |
| Context window | {BLOCK_SIZE} tokens |
| Vocab size | {meta['vocab_size']} (BPE) |
| Training iters | {meta['iter']} |
| Best val loss | {meta['best_loss']:.4f} |

## Training

Trained on ~103MB of cleaned Project Gutenberg text (114 public domain books)
with BPE-{meta['vocab_size']} tokenization, AdamW optimizer, cosine LR decay,
and ReduceLROnPlateau. Metal GPU via Candle on Apple Silicon.

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
