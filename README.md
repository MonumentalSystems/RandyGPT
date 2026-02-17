# randyGPT - Mini GPT in Rust

A GPT-style language model implemented from scratch in Rust. Training runs on Metal GPU via Candle autograd on Apple Silicon, with a CPU fallback (Rayon + Accelerate BLAS) on other systems.

## Features

- **Transformer Architecture**: Multi-head causal attention, feed-forward layers, residual connections, RMSNorm
- **Metal GPU Training**: Candle autograd on Apple M-series — forward + backward on GPU, AdamW on CPU
- **CPU Fallback**: Rayon multi-core + Accelerate BLAS (sgemm batched backward) when Metal unavailable
- **Full Training Loop**: AdamW optimizer, gradient clipping, dropout, LR warmup + cosine decay
- **GPT-2 Style Init**: Scaled output projections for stable deep training
- **Checkpoint Save/Resume**: RGPT0002 format; save and continue across sessions
- **Character-Level Tokenization**: Up to 512-char vocabulary with BOS/EOS tokens
- **KV Cache**: Efficient autoregressive generation
- **Top-P + Temperature Sampling**: Nucleus sampling for text generation

## Current Model Configuration (v0.8.0)

| Hyperparameter | Value |
|----------------|-------|
| Embedding dim | 256 |
| Attention heads | 8 |
| Layers | 6 |
| Block size | 64 tokens |
| MLP hidden dim | 1024 (4× embd) |
| Parameters | ~4.77M |
| Dropout | 0.1 |
| Optimizer | AdamW (wd=0.01) |
| Batch size | 128 |

## Usage

### Build

```bash
cargo build --release
```

### Train from scratch

```bash
./target/release/randygpt --iters 1000
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

- **Forward + Backward**: Candle autograd on Metal — fully batched `[128, 64, 256]` tensor ops
- **Loss**: `candle_nn::loss::cross_entropy` over all sequence positions
- **Gradients**: `loss.backward()` → `GradStore` → pulled to CPU via `.to_vec1::<f32>()`
- **Optimizer**: AdamW moments stay on CPU; updated weights re-uploaded via `Var::set()`
- **Gradient clipping**: L∞ norm clipped at 1.0

### Training (CPU fallback)

- **Rayon**: Each batch item processed on a separate thread; gradients summed after
- **BLAS**: Accelerate `cblas_sgemv` for d_x, `cblas_sgemm` for batched d_W (one call per weight matrix instead of T rank-1 updates)

## Performance (Apple M-series, Shakespeare 1.1M tokens)

| Version | ms/iter | 1000 iter | Notes |
|---------|---------|-----------|-------|
| v0.7.1 CPU | ~964ms | ~96s | 12 cores, batch=128, SGEMM backward |
| **v0.8.0 GPU** | **~518ms** | **~52s** | Metal autograd, batch=128 |

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

## Credits

Based on mini_gpt.rs by @RandyMcMillan
Enhanced by Claude Sonnet 4.5 / Monumental Systems

## License

MIT
