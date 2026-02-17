# randyGPT - Mini GPT in Rust

A GPT-style language model implemented from scratch in Rust. No external ML frameworks for training — pure Rust with Rayon for parallelism and Candle/Metal for accelerated inference on Apple Silicon.

## Features

- **Transformer Architecture**: Multi-head attention, feed-forward layers, residual connections, RMSNorm
- **Multi-Core Training**: Rayon parallelization across all CPU cores
- **Full Training Loop**: Backward pass, AdamW optimizer, gradient clipping, dropout
- **GPT-2 Style Init**: Scaled output projections for stable deep training
- **LR Schedule**: Linear warmup → constant → cosine decay
- **Metal GPU Inference**: Batched matmuls via Candle on Apple M-series chips (inference/eval only)
- **Checkpoint Save/Resume**: Save and continue training across sessions
- **Character-Level Tokenization**: With BOS/EOS tokens, up to 512-char vocabulary
- **KV Cache**: Efficient autoregressive generation
- **Top-P + Temperature Sampling**: Nucleus sampling for text generation

## Current Model Configuration (v0.5)

| Hyperparameter | Value |
|----------------|-------|
| Embedding dim | 128 |
| Attention heads | 8 |
| Layers | 6 |
| Block size | 64 tokens |
| MLP hidden dim | 512 (4× embd) |
| Parameters | ~1.20M |
| Dropout | 0.1 |
| Optimizer | AdamW (wd=0.01) |

## Usage

### Build

```bash
cargo build --release
```

Requires Rust nightly (for Candle Metal support):
```bash
rustup default nightly
```

### Train from scratch

```bash
./target/release/randygpt --iters 1000
```

### Resume training from a checkpoint

```bash
# Resume from the last periodic checkpoint (auto path)
./target/release/randygpt --iters 2000 --resume

# Resume from the best-loss checkpoint
./target/release/randygpt --iters 2000 --resume checkpoint_best.bin

# Resume from a specific checkpoint file
./target/release/randygpt --iters 5000 --resume my_run.bin
```

### Checkpoint files

State is serialized to memory every iteration and flushed to disk only when training finishes or you press **Ctrl-C** — zero disk I/O in the hot loop, so CPU stays at full multi-core.

| File | Flushed when |
|------|-------------|
| `checkpoint.bin` | Training completes or Ctrl-C |
| `checkpoint_best.bin` | Training completes or Ctrl-C (if loss improved) |

Both files include model weights **and** full AdamW optimizer state, so the LR schedule resumes correctly from exactly where you left off.

Checkpoint size: ~14 MB for the current 1.2M-param config.

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

1. **Token + Position Embedding** — learned lookup tables
2. **Transformer Layers** (×N_LAYER):
   - RMSNorm → Multi-head causal self-attention → Residual
   - RMSNorm → MLP (squared ReLU, 4× expansion) → Residual
   - Dropout after attention projection and MLP output (training only)
3. **LM Head** — linear projection to vocab logits

### Training

- **Loss**: Cross-entropy over all sequence positions (not just last token)
- **Backward**: MLP and embedding gradients; attention handled via simplified path
- **Optimizer**: AdamW — Adam with decoupled weight decay (β₁=0.9, β₂=0.999, wd=0.01)
- **Gradient clipping**: L∞ norm clipped at 1.0
- **Parallelism**: Rayon processes each batch item on a separate thread; gradients summed after

### Metal GPU Acceleration

On Apple Silicon Macs, Metal is used for batched matrix multiplications during inference and loss estimation. Training stays on CPU (the backward pass is CPU-only, and Rayon's parallelism already uses all cores effectively).

The benchmark for the current config:
- CPU QKV [64×128]*[128×128]: ~0.6ms
- Metal QKV: <0.01ms (>100× faster for batched ops)

## Performance

- **Training speed**: ~45s per 100 iterations (Shakespeare, 1.1M tokens, 12 cores)
- **CPU usage**: ~350% (3–4 cores effectively via Rayon at current batch/model size)
- **Memory**: ~420 MB RSS (stable, no growth)

## Parameter Scaling Reference

| Config | Layers | Dim | Heads | Params |
|--------|--------|-----|-------|--------|
| Tiny | 2 | 64 | 4 | ~80K |
| Current | 6 | 128 | 8 | ~1.2M |
| Medium | 6 | 256 | 8 | ~4.8M |
| Large | 12 | 512 | 16 | ~40M |

## TODO / Future Improvements

- [x] ~~Implement backward pass for training~~ ✅ (v0.2)
- [x] ~~Add Adam optimizer~~ ✅ (v0.2)
- [x] ~~Multi-core training~~ ✅ Rayon (v0.3)
- [x] ~~Gradient clipping~~ ✅ (v0.4)
- [x] ~~Learning rate scheduling~~ ✅ (v0.4)
- [x] ~~Dropout~~ ✅ (v0.4)
- [x] ~~GPT-2 style initialization~~ ✅ (v0.4)
- [x] ~~CLI arguments~~ ✅ (v0.4)
- [x] ~~Model checkpointing (save/load weights)~~ ✅ (v0.5)
- [x] ~~Metal/GPU inference acceleration~~ ✅ Candle (v0.5)
- [ ] Full attention gradient computation
- [ ] Validation split and perplexity metrics
- [ ] BPE tokenization
- [ ] Mixed precision training

## Credits

Based on mini_gpt.rs by @RandyMcMillan
Enhanced by Claude Sonnet 4.5 / Monumental Systems

## License

MIT
