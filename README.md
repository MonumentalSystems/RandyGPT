# randyGPT - Mini GPT in Rust

A GPT-style language model implemented from scratch in Rust. Training runs on Metal GPU via Candle autograd on Apple Silicon, with a CPU fallback (Rayon + Accelerate BLAS) on other systems.

## Features

- **Transformer Architecture**: Multi-head causal attention, feed-forward layers, residual connections, RMSNorm
- **Metal GPU Training**: Candle autograd on Apple M-series — forward + backward + AdamW fully on GPU
- **CPU / Linux Fallback**: Rayon multi-core + Accelerate BLAS when Metal unavailable; builds natively on Linux
- **Full Training Loop**: AdamW, gradient clipping, dropout, ReduceLROnPlateau, cosine decay
- **GPT-2 Style Init**: Scaled output projections for stable deep training
- **Checkpoint Save/Resume**: RGPT0003 format (weights + optimizer moments); resumes LR schedule exactly
- **Character-Level + BPE Tokenization**: char vocab or learned BPE with `--bpe [N]`
- **True KV Cache**: Single-token decode — prefill prompt once, O(1) projections per new token
- **Top-P + Temperature Sampling**: Nucleus sampling for text generation
- **HTTP Inference Server**: `--serve` flag; no output token cap; returns completion only (prompt stripped)

## Model Presets (select at build time)

| Preset | Dim | Heads | Layers | Params | Build flag |
|--------|-----|-------|--------|--------|------------|
| model-xs | 116 | 4 | 3 | ~746K | `--features model-xs` |
| **model-s** | **128** | **4** | **8** | **~1.6M** | `--features model-s` |
| model-m | 192 | 6 | 6 | ~2.7M | `--features model-m` |
| model-l *(default)* | 256 | 8 | 6 | ~4.82M | *(none)* |
| model-deep | 192 | 6 | 16 | ~7.5M | `--features model-deep` |
| model-xl | 384 | 8 | 8 | ~10.8M | `--features model-xl` |

Default hyperparameters (v0.9.5):

| Hyperparameter | Value |
|----------------|-------|
| Block size | 256 tokens |
| Batch size | 64 |
| Learning rate | 1e-4 → 1e-5 (cosine) |
| Weight decay | 0.1 |
| Dropout | 0.1 |
| Early stopping | ReduceLROnPlateau (patience=30, ×3 reductions) |

## Usage

### Build

```bash
cargo build --release
```

### Train from scratch

```bash
# model-l (default), char-level
./target/release/randygpt --iters 5000

# model-s with BPE-500 vocab (recommended for novels dataset)
cargo build --release --features model-s
./target/release/randygpt --iters 25000 --bpe 500
```

### Resume training from a checkpoint

```bash
# Resume from last checkpoint (auto path)
./target/release/randygpt --iters 2000 --resume

# Resume from the best-loss checkpoint
./target/release/randygpt --iters 2000 --resume checkpoint_best.bin

# Resume from a specific file
./target/release/randygpt --iters 5000 --resume my_run.bin
```

### Run inference server

```bash
# Serve on default address (0.0.0.0:8080) — auto-loads checkpoint_best.bin
./target/release/randygpt --bpe --serve

# Custom address
./target/release/randygpt --bpe --serve 127.0.0.1:9000

# With bearer-token authentication
./target/release/randygpt --bpe --serve --api-key mysecret
```

The server accepts `POST /` with a JSON body and returns generated text:

```bash
curl -s http://localhost:8080/ \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":200,"temperature":0.8}'
# → {"text":"…","model":"randygpt-6L-8H-256D","usage":{"prompt_tokens":4,"completion_tokens":196}}
```

| Field | Default | Description |
|-------|---------|-------------|
| `prompt` | required | Text prompt |
| `max_tokens` | 256 (BLOCK_SIZE) | Max new tokens; capped at context window |
| `temperature` | 0.7 | Sampling temperature |

If `--api-key` is set, every request must include `Authorization: Bearer <key>`.

### Checkpoint files

Weights are serialized to memory every `EVAL_INTERVAL` iterations and flushed to disk when training finishes or you press **Ctrl-C** — zero disk I/O in the hot loop.

| File | Contents |
|------|---------|
| `checkpoint.bin` | Latest periodic checkpoint |
| `checkpoint_best.bin` | Best-loss checkpoint |

Both files use the RGPT0002 format and include model weights + full AdamW optimizer state, so the LR schedule resumes correctly from exactly where you left off.

Checkpoint size: ~55 MB for the current 4.77M-param config.

### Training data

Place your training text in `train.txt`:

```bash
# Shakespeare (classic benchmark)
curl -o train.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Or any plain text
cp my_text.txt train.txt
```

If `train.txt` is absent, the model falls back to a tiny built-in sample.

## Architecture Details

### Forward Pass

1. **Token + Position Embedding** — learned lookup tables `[vocab, D]` and `[T, D]`
2. **Transformer Layers** (×6):
   - RMSNorm → Multi-head causal self-attention → Residual
   - RMSNorm → MLP (squared ReLU, 4× expansion) → Residual
   - Dropout after attention projection and MLP output (training only)
3. **LM Head** — linear projection to vocab logits

### Training (Metal GPU path)

- **Forward + Backward**: Candle autograd on Metal — fully batched `[B, T, D]` tensor ops
- **Loss**: `candle_nn::loss::cross_entropy` over all sequence positions
- **Optimizer**: Full GPU AdamW via `GpuAdamState` — moments live as Metal `Var`s, zero CPU transfers in hot loop
- **Gradient clipping**: L∞ norm clipped at 1.0
- **LR schedule**: constant at `max_lr` → cosine decay to `min_lr` over final 40%; ReduceLROnPlateau halves `max_lr` on stagnation (up to 3×)

### Training (CPU fallback)

- **Rayon**: Each batch item processed on a separate thread; gradients summed after
- **BLAS**: Accelerate `cblas_sgemv` for d_x, `cblas_sgemm` for batched d_W (one call per weight matrix instead of T rank-1 updates)

## Performance (Apple M-series, Shakespeare 1.1M tokens)

| Version | ms/iter | 1000 iter | Val ppl @1000 | Notes |
|---------|---------|-----------|----------------|-------|
| v0.7.1 CPU | ~964ms | ~96s | — | 12 cores, batch=128, SGEMM backward |
| **v0.8.0 GPU** | **~488ms** | **~49s** | **10.8** | Metal autograd, batch=128, 60.9% GPU |

RSS memory: ~400 MB real; Activity Monitor shows ~3 GB (Metal unified memory mapped into virtual space, not CPU-resident).

## Parameter Scaling Reference

| Config | Layers | Dim | Heads | Params |
|--------|--------|-----|-------|--------|
| Tiny | 2 | 64 | 4 | ~80K |
| Small | 6 | 128 | 8 | ~1.2M |
| **Current** | **6** | **256** | **8** | **~4.77M** |
| Large | 12 | 512 | 16 | ~40M |

## Completed Improvements

- [x] Backward pass for training (v0.2)
- [x] Adam optimizer (v0.2)
- [x] Multi-core training via Rayon (v0.3)
- [x] Gradient clipping, LR scheduling, dropout (v0.4)
- [x] GPT-2 style initialization (v0.4)
- [x] Model checkpointing (v0.5)
- [x] Metal GPU inference via Candle (v0.5)
- [x] Full attention gradient computation (v0.6)
- [x] Validation split and perplexity metrics (v0.6)
- [x] Accelerate BLAS for CPU matmuls (v0.7.0)
- [x] SGEMM batched backward, 256-dim, batch=128 (v0.7.1)
- [x] Metal GPU training via Candle autograd (v0.8.0)
- [x] BPE tokenization with `--bpe [N]` flag (v0.9.2)
- [x] Fast CPU-only `--generate` mode (v0.9.3)
- [x] HTTP inference server via `--serve` (v0.9.4)
- [x] True KV-cache single-token decode (v0.9.5)
- [x] Generation returns completion only — prompt stripped (v0.9.5)
- [x] ReduceLROnPlateau — halves LR on stagnation, resumes; no premature early stop (v0.9.5)
- [x] LR warmup removed — redundant with gradient clipping + Adam bias correction (v0.9.5)
- [x] Linux / cross-platform builds — Metal conditional on macOS (v0.9.5)
- [x] Systemd deployment to `oss.xenon.fun.local` (v0.9.5)

## Planned

- [ ] **Gnostr model distribution** — publish/fetch checkpoints as nostr events via gnostr CLI

## Credits

Based on mini_gpt.rs by @RandyMcMillan
Enhanced by Claude Sonnet 4.5 / Monumental Systems

## License

MIT
