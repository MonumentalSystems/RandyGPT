# Usage Guide

## Quick Start

```bash
# Build (requires Rust nightly)
rustup default nightly
cargo build --release

# Train on Shakespeare for 1000 iterations
./target/release/randygpt --iters 1000

# Resume if you need to stop and continue later
./target/release/randygpt --iters 2000 --resume
```

## CLI Arguments

```
./randygpt [--iters N] [--resume [path]] [--generate PROMPT] [--bpe [N]] [--serve [addr]] [--api-key KEY]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--iters N` | 1000 | Total iterations to train to (not additional — the target) |
| `--resume` | — | Resume from `checkpoint.bin` (auto path) |
| `--resume <path>` | — | Resume from a specific `.bin` file |
| `--generate PROMPT` | — | CPU-only inference; skips data loading and Metal init |
| `--bpe [N]` | — | Use BPE tokenization with vocab size N (default 2000) |
| `--serve [addr]` | `0.0.0.0:8080` | Start HTTP inference server; requires `--bpe` and a checkpoint |
| `--api-key KEY` | — | Require `Authorization: Bearer KEY` on all `/` requests |
| `--lr LR` | — | Override learning rate at runtime |
| `--min-lr LR` | — | Override minimum learning rate at runtime |

### Examples

```bash
# Fresh 1000-iteration run
./target/release/randygpt --iters 1000

# Continue to 2000 from where you left off
./target/release/randygpt --iters 2000 --resume

# Continue from the best-loss checkpoint
./target/release/randygpt --iters 2000 --resume checkpoint_best.bin

# Custom checkpoint file
./target/release/randygpt --iters 5000 --resume runs/shakespeare_500.bin
```

> **Note**: `--iters` is a *target*, not an *additional count*. If you trained to 1000 and want 500 more, pass `--iters 1500`.

## Serve Mode

`--serve` loads a trained checkpoint and starts a lightweight HTTP server for on-demand generation. It requires BPE mode (`--bpe`) and a checkpoint.

```bash
# Default address (0.0.0.0:8080) — auto-loads checkpoint_best.bin or checkpoint.bin
./target/release/randygpt --bpe --serve

# Custom address
./target/release/randygpt --bpe --serve 127.0.0.1:9000

# With bearer-token auth
./target/release/randygpt --bpe --serve --api-key mysecret

# Explicit checkpoint
./target/release/randygpt --bpe --resume runs/my_run.bin --serve
```

### Request format

`POST /` with a JSON body:

```json
{
  "prompt": "Once upon a time",
  "max_tokens": 200,
  "temperature": 0.8
}
```

`max_tokens` and `temperature` are optional (defaults: 2048 and 0.7).

### Response format

```json
{
  "text": "Once upon a time in a land far away...",
  "model": "randygpt-6L-8H-256D",
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 196
  }
}
```

### Authentication

If `--api-key` is provided, every request must include:
```
Authorization: Bearer <key>
```
Requests without or with a wrong token receive `401 Unauthorized`.

### Example

```bash
curl -s http://localhost:8080/ \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer mysecret' \
  -d '{"prompt":"ROMEO:","max_tokens":100,"temperature":0.9}'
```

Press **Ctrl-C** to stop the server cleanly.

## Checkpoints

State is serialized to memory every iteration and flushed to disk only when training finishes or you press Ctrl-C. There's **zero disk I/O inside the hot training loop**, so CPU utilization stays at full multi-core the whole time.

Two checkpoint files are maintained:

| File | Flushed when |
|------|-------------|
| `checkpoint.bin` | Training completes or Ctrl-C |
| `checkpoint_best.bin` | Training completes or Ctrl-C (if best loss improved) |

**What's saved**: Model weights + full AdamW optimizer state (m/v moments) + iteration/step counters. The LR schedule resumes exactly where it left off.

**File size**: ~55 MB for the current 4.82M-param config (scales with param count). Uses RGPT0003 binary format (weights + full GPU AdamW moments).

**Ctrl-C**: Pressing Ctrl-C mid-training saves both checkpoints before exiting:
```
Interrupted at iteration 89. Saving checkpoint...
✓ Saved checkpoint.bin (iter 89)
✓ Saved checkpoint_best.bin (best loss 3.9917 @89)
```

**Guard**: If you resume with `--iters` already reached, the program exits cleanly with a message instead of running 0 iterations silently.

### Backing up a checkpoint

```bash
cp checkpoint_best.bin runs/shakespeare_1000.bin
```

## Training Data

Place your training text in `train.txt`. The model uses character-level tokenization — any plain text file works.

```bash
# Tiny Shakespeare (classic)
curl -o train.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Your own text
cp my_corpus.txt train.txt

# Multiple files combined
cat docs/*.md > train.txt
```

If `train.txt` is absent the model falls back to a small built-in sample (good for testing the binary, not for actual training).

### Data Size Recommendations

| Model Size | Min | Recommended |
|------------|-----|-------------|
| Tiny (~80K params) | 10 KB | 50 KB |
| Small (~1.2M params) | 100 KB | 500 KB–2 MB |
| **Current (~4.77M params)** | 500 KB | 2–10 MB |
| Large (~40M params) | 5 MB | 50 MB+ |

## Scaling the Model

Edit constants in `src/config.rs` and rebuild. The main levers:

```rust
pub const N_EMBD: usize     = 256;  // Embedding dimension
pub const N_HEAD: usize     = 8;    // Attention heads (must divide N_EMBD evenly)
pub const N_LAYER: usize    = 6;    // Transformer layers
pub const BLOCK_SIZE: usize = 64;   // Context window (tokens)
pub const BATCH_SIZE: usize = 128;  // Batch size
```

### Preset configs

| Preset | N_EMBD | N_HEAD | N_LAYER | Params |
|--------|--------|--------|---------|--------|
| Tiny | 64 | 4 | 2 | ~80K |
| Small | 128 | 8 | 6 | ~1.2M |
| **Current (default)** | **256** | **8** | **6** | **~4.77M** |
| Large | 512 | 16 | 12 | ~40M |

> **Checkpoint compatibility**: Changing any architecture constant invalidates existing checkpoints. Start fresh (delete `.bin` files) after changing model shape.

## Generation Parameters

Generation runs automatically after training. The prompts and parameters live in `main()`:

```rust
let sample = generate(
    &mut model,
    &tokenizer,
    "ROMEO:",   // Prompt string
    100,        // Max new tokens
    0.8,        // Temperature
    0.9,        // Top-p (nucleus sampling)
    &mut rng,
);
```

### Temperature
- **0.1–0.5**: Deterministic, repetitive
- **0.6–0.9**: Balanced (default: 0.8)
- **1.0–2.0**: Creative / random

### Top-P
- **0.5–0.7**: Conservative
- **0.8–0.9**: Balanced (default: 0.9)
- **0.95–1.0**: Maximum diversity

## Metal GPU Acceleration (Apple Silicon)

On M-series Macs, Metal is used automatically for the entire training loop — you'll see:

```
✓ Metal GPU enabled on device: Metal(MetalDevice(DeviceId(1)))
Metal GPU: enabled — training via Candle autograd
```

The forward pass, backward pass (Candle autograd), and AdamW optimizer (moments as Metal `Var` tensors) all run on GPU — zero CPU transfers in the hot loop.

If Metal is unavailable (non-Apple hardware or driver issue), the code falls back to CPU (Rayon + Accelerate BLAS) transparently.

## Performance Tips

1. **Always use `--release`** — debug builds are 20–50× slower
2. **Use a checkpoint interval** — the default 100-iter checkpoint means at most 100 iters lost if interrupted
3. **Prefer `checkpoint_best.bin`** for long resumptions — it reflects the model's best state, not just the most recent
4. **RSS memory**: `real` RSS stays ~400 MB; Activity Monitor shows ~3 GB — that's normal on Apple Silicon. Metal tensors live in unified memory, which shows up in the process virtual address space but not in RSS.
5. **First iteration is slower** — Metal JIT-compiles kernels on the first batch; subsequent iters are at full speed

## Troubleshooting

### "Already at iteration N (target M). Nothing to train."
Your `--iters` target is at or below what's in the checkpoint. Increase it:
```bash
./target/release/randygpt --iters 3000 --resume
```

### "Bad magic bytes in checkpoint"
The `.bin` file is corrupt or not a randyGPT checkpoint. Use a different file or start fresh.

### "Checkpoint vocab_size N != model vocab_size M"
The checkpoint was saved with a different training corpus (different character set → different vocab). Either use the same `train.txt` or delete the checkpoint and start fresh.

### Out of memory
Reduce `N_EMBD`, `N_LAYER`, or `BLOCK_SIZE` in `src/config.rs` and rebuild.

### Very slow training
- Confirm you're using `--release`
- Check that Rayon is using multiple cores (`Cores available: N` in output)
- Consider reducing batch size or model size for faster iteration
