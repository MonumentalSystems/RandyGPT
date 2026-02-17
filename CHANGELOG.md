# Changelog

All notable changes to randyGPT are documented here.

## Version Comparison

| Feature | v0.1 | v0.2 | v0.3 | v0.4 | v0.5.1 | v0.6.0 | v0.7.0 | v0.7.1 | v0.8.0 |
|---------|------|------|------|------|--------|--------|--------|--------|--------|
| **Layers** | 1 | 4 | 4 | 6 | 6 | 6 | 6 | 6 | 6 |
| **Embedding Dim** | 32 | 128 | 128 | 128* | 128 | 128 | 128 | 256 | 256 |
| **Parameters** | ~10K | ~800K | ~800K | ~1.2M | ~1.2M | ~1.2M | ~1.2M | ~4.77M | ~4.77M |
| **Training** | ❌ | ✅ Single-core | ✅ Multi-core | ✅ Multi-core | ✅ Multi-core | ✅ Multi-core | ✅ Multi-core | ✅ Multi-core | ✅ **GPU autograd** |
| **Attention grads** | ❌ | Q only | Q only | Q only | Q only | ✅ Q+K+V | ✅ Q+K+V | ✅ Q+K+V | ✅ Candle autograd |
| **Optimizer** | - | Adam | Adam | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW (CPU moments) |
| **LR Schedule** | - | Immediate decay | Immediate decay | Constant→Decay | Constant→Decay | Constant→Decay | Constant→Decay | Warmup→60%→Decay | Warmup→60%→Decay |
| **Initialization** | Random | Standard | Standard | GPT-2 style | GPT-2 style | GPT-2 style | GPT-2 style | GPT-2 style | GPT-2 style |
| **Dropout** | ❌ | ❌ | ❌ | ✅ (0.1) | ✅ (0.1) | ✅ (0.1) | ✅ (0.1) | ✅ (0.1) | ✅ (0.1) |
| **Checkpoints** | ❌ | ❌ | ❌ | ❌ | ✅ memory-buffered | ✅ | ✅ | ✅ | ✅ RGPT0002 |
| **Ctrl-C save** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Val loss / ppl** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Metal GPU** | ❌ | ❌ | ❌ | ✅ | ✅ (stable) | ✅ | ✅ | ✅ | ✅ **training** |
| **BLAS (Accelerate)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ sgemv/sger | ✅ +sgemm | ✅ (CPU fallback) |
| **Batch size** | - | - | - | - | - | - | 32 | 128 | 128 |
| **Timing output** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ ms/iter + ETA | ✅ ms/iter + ETA |
| **Code structure** | 1 file | 1 file | 1 file | 1 file | 1 file | 10 modules | 10 modules | 10 modules | 10 modules |
| **Memory (RSS)** | ~50MB | ~100MB | ~300MB | 43GB⚠ | ~420MB | ~420MB | ~420MB | ~1.6GB | ~400MB real† |
| **Speed (1000 iter)** | N/A | ~600s‡ | ~78s | ~450s | ~450s | ~450s | ~215s | ~96s§ | ~49s¶ |

†RSS ~400MB real; Activity Monitor shows ~3GB (Metal unified memory in virtual space — not CPU-resident)
‡Estimated
§SGEMM batched backward; measured ~964ms/iter CPU with 12 cores; batch=128
¶Candle Metal autograd (~490ms/iter steady-state, ~2× vs v0.7.1 CPU); batch=128; 60.9% GPU utilization
\*v0.4 targeted 256-dim but shipped at 128 due to the Metal memory issue fixed in v0.5

---

## [0.8.0] - 2026-02-16

### Candle Autograd Metal Training (Strategy A Hybrid)

#### GPU Training via Candle Autograd (~3.4× speedup)
- **Forward and backward passes now run on Metal GPU** via Candle autograd
- New `CandleModel` / `CandleLayer` structs: all weights stored as `candle_core::Var` on Metal device
- New `forward_candle_train()`: fully batched `[BATCH_SIZE, BLOCK_SIZE, N_EMBD]` forward pass using Candle tensor ops
  - Embeddings via `index_select` + positional encoding via `narrow`
  - RMSNorm, multi-head causal attention, squared-ReLU MLP — all Candle ops
  - Cross-entropy via `candle_nn::loss::cross_entropy`
- `loss.backward()` → `GradStore` replaces hand-written two-pass SGEMM backward
- **Optimizer stays on CPU (Strategy A)**: AdamW moments remain `Vec<f32>`, `adam_step()` reused unchanged
- Gradients pulled off GPU via `.flatten_all().to_vec1::<f32>()`, clipped, Adam-updated, re-uploaded via `Var::set()`
- **Measured speedup**: ~490ms/iter steady-state vs ~964ms/iter (v0.7.1 CPU) = **~2×**; 60.9% GPU utilization

#### RGPT0002 Checkpoint Format
- New magic bytes `b"RGPT0002"` — incompatible with prior RGPT0001 checkpoints
- Same binary layout (f32 arrays in same order); weights extracted from Vars via `.flatten_all().to_vec1::<f32>()`
- `serialize_checkpoint_v2()` / `load_checkpoint_v2()` in `checkpoint.rs`
- Resume path tries RGPT0002 first, falls back to RGPT0001 (CPU path)

#### CPU Fallback Preserved
- On systems without Metal, falls back to existing `train()` (BLAS SGEMM backward, unchanged)
- All inference paths (`estimate_loss`, `generate`) unchanged

---

## [0.7.1] - 2026-02-16

### SGEMM Batched Backward + Scale to 256-dim + Training Improvements

#### SGEMM Batched Weight Gradient (~2.2× additional speedup)
- **Replaced 64 `cblas_sger` rank-1 updates per weight matrix with one `cblas_sgemm`**
- New `linear_bwd_dw_batched()`: computes `d_W += D^T · X` where D is `[T×nout]` and X is `[T×nin]`
- New `linear_bwd_dx_only()`: just the `cblas_sgemv` for d_x, no d_w allocation
- Two-pass backward: per-position loop (d_x, sequential) → SGEMM pass (all d_w, batched)
- Eliminates throwaway weight-gradient allocations in the d_x loop
- **Measured**: ~964ms/iter (12-core, batch=128) vs ~3879ms/iter (256-dim single-core baseline) = **~4×**

#### Scale to 256-Dimensional Embeddings
- `N_EMBD: 128 → 256` — model grows from ~1.2M to **~4.77M parameters**
- Incompatible with prior checkpoints (delete old `.bin` files before training)

#### Training Improvements
- **LR decay start**: moved from 80% → **60%** of total iterations (cosine decay kicks in earlier, prevents late-training instability)
- **Batch size**: 32 → **128** (4× gradient quality per step; ~1.6 GB RSS)
- **Timing output**: ms/iter and ETA added to every log line:
  ```
  Iter  100 | Loss: 3.4521 | Val: 3.5812 (ppl 35.9) | LR: 0.000030 | Best: 3.4521 @100 | 1774ms/iter | 177s elapsed | ETA 1597s
  ```
- **Training complete summary**: `Total time: 177.4s | Avg: 1774ms/iter (100 iters)`

---

## [0.7.0] - 2026-02-16

### Accelerate BLAS for CPU Matrix Ops (~2× training speedup)

- **`linear_fwd`**: replaced manual dot-product loop with `cblas_sgemv` (matrix-vector multiply)
- **`linear_bwd`**: replaced double loop with `cblas_sgemv` (W^T · d_out) + `cblas_sger` (outer-product update for d_w)
- Both use Apple's Accelerate framework — always available on macOS, no additional dependencies
- `build.rs` added to emit `cargo:rustc-link-lib=framework=Accelerate`
- **Measured speedup**: ~215s/1000 iter (down from ~450s) — **2.1× faster**
- All other compute paths unchanged; Metal inference path unaffected

---

## [0.6.0] - 2026-02-16

### Full Attention Gradients + Validation Tracking + Module Refactor

#### Full Attention Backward Pass
- **K and V gradients now computed** — previously `d_k` and `d_v` were zeroed, meaning key/value projection weights (`wk`, `wv`) never learned
- Gradient for K at current position: `d_k[j] += d_scores[pos] * scale * q[j]`
- Gradient for V at current position: `d_v[j] += attn_weights[pos] * d_attn_out[j]`
- Only the **current position's** K and V are backpropagated (correct for autoregressive models — past positions' KV were computed in prior forward passes and are not revisited)
- All 6 projection matrices (`wq`, `wk`, `wv`, `wo`, `fc1`, `fc2`) now receive gradients every iteration

#### Validation Split + Perplexity
- Training data split **90% train / 10% validation** at load time
- Val loss + perplexity reported at every eval interval:
  ```
  Iter  100 | Loss: 3.4521 | Val: 3.5812 (ppl 35.9) | LR: 0.000030 | Best: 3.4521 @100
  ```
- Perplexity = `exp(val_loss)` — a more interpretable metric: ppl 20 ≈ model is choosing from ~20 equally likely next tokens
- Initial and final val loss/perplexity reported at start and end of training
- Val set is never trained on; divergence between train and val loss indicates overfitting

#### Module Refactor (1869-line main.rs → 10 focused modules)
- `config.rs` — all hyperparameters as `pub const`
- `rng.rs` — xorshift PRNG
- `tokenizer.rs` — character-level tokenizer
- `model.rs` — `GPTModel`, `LayerWeights`, `PosActs`, `GradientBuffer` structs
- `ops.rs` — `linear_fwd/bwd`, `rmsnorm_fwd`, `softmax_fwd/bwd`, `cross_entropy_loss`, `apply_dropout`, `clip_gradients`
- `metal.rs` — `METAL_DEVICE` lazy init, `metal_matmul_batch`
- `forward.rs` — per-token CPU forward (training), batched Metal forward (inference)
- `optimizer.rs` — `adam_step`, `zero_grads`, `get_learning_rate`
- `checkpoint.rs` — binary checkpoint serialize/load
- `train.rs` — training loop, `estimate_loss`, `generate`
- `main.rs` — CLI, data loading, model init, orchestration (~187 lines)

---

## [0.5.1] - 2026-02-16

### 🔧 Zero-IO Training Loop + Ctrl-C Save

- **Memory-buffered checkpoints**: `serialize_checkpoint()` serializes state to a `Vec<u8>` in RAM on every iteration — no disk writes inside the hot loop
- **Flush only on completion or interrupt**: `flush_checkpoint()` does a single atomic write (temp-file + rename) at end-of-training or on Ctrl-C
- **Ctrl-C handler** (`ctrlc` crate): pressing Ctrl-C sets an atomic flag; the training loop detects it after the current iteration, flushes both buffers, and exits cleanly:
  ```
  Interrupted at iteration 89. Saving checkpoint...
  ✓ Saved checkpoint.bin (iter 89)
  ✓ Saved checkpoint_best.bin (best loss 3.9917 @89)
  ```
- **Result**: CPU utilization restored to full multi-core (was dropping to 1-2 cores due to 14 MB synchronous disk writes per iteration)

---

## [0.5.0] - 2026-02-16

### 🚀 Checkpoint Save / Resume

- **`serialize_checkpoint` / `load_checkpoint`** — compact binary format (`RGPT0001` magic)
  - Saves model weights + full AdamW optimizer state (m/v moments) + iter/step counters
  - LR schedule resumes correctly because the step counter is preserved
  - ~14 MB per checkpoint for the current 1.2M-param config
- **Two automatic checkpoint files**:
  - `checkpoint.bin` — written at completion (and on Ctrl-C)
  - `checkpoint_best.bin` — written whenever training loss hits a new best (on Ctrl-C if improved)
- **CLI resume flag**: `--resume [path]`
  - `--resume` → loads `checkpoint.bin`
  - `--resume <path>` → loads any `.bin` file
  - Guard: prints a clear message if `--iters` target is already reached
- **`--iters` flag** added (replaces positional argument; positional still works for compat)

### 🔧 Metal Memory Fix

- Fixed catastrophic memory explosion (43 GB → 430 MB) caused by calling Metal GPU per-vector during training
- Root cause: `linear_fwd` was routing every single matrix-vector op through Metal, allocating ~98K GPU tensors per training iteration that Metal's runtime couldn't free fast enough
- Fix: training always uses CPU (`linear_fwd_cpu` via Rayon); Metal is used only in `forward_metal_logits` for batched inference in `estimate_loss` and `generate`
- Removed the unused `linear_fwd_metal` function

### 📊 Architecture (unchanged from v0.4)

| Feature | v0.5 |
|---------|------|
| Embedding Dim | 128 |
| Layers | 6 |
| Parameters | ~1.2M |
| Metal | ✅ Inference only |
| Checkpointing | ✅ |

---

## [0.4.0] - 2026-02-16

### 🚀 Major Improvements

#### Scaled Model Architecture
- **256-dimensional embeddings** (up from 128)
- **6 transformer layers** (up from 4)
- **~4.77M parameters** (up from ~800K, 6x increase)
- Better model capacity for learning complex patterns

#### Training Optimizations
- **Fixed learning rate schedule**: Constant until 80% of training, then cosine decay
  - Previous: Started decaying at iter 100, causing premature plateau
  - Now: Full learning until 80% complete, smooth decay in final 20%
- **AdamW optimizer**: Added weight decay (0.01) for better generalization
- **GPT-2 style initialization**: Output projections scaled by 1/sqrt(2*N_LAYER)
  - Accounts for residual accumulation across deeper network
- **Dropout regularization**: Fully integrated during training (rate: 0.1)
  - Applied after attention and MLP output projections
  - Randomly zeros 10% of activations (scaled by 1/0.9)
  - Disabled during evaluation and generation
  - Thread-safe implementation with per-thread RNG

#### CLI Improvements
- **Command-line arguments**: Pass iteration count as `./randygpt <iterations>`
- No more recompilation to change training length

### 📊 Architecture Comparison

| Feature | v0.3 | v0.4 |
|---------|------|------|
| **Embedding Dim** | 128 | 256 |
| **Layers** | 4 | 6 |
| **Parameters** | ~800K | ~4.77M |
| **LR Schedule** | Immediate decay | Constant → Cosine |
| **Optimizer** | Adam | AdamW (with weight decay) |
| **Initialization** | Standard | GPT-2 style |
| **Dropout** | ❌ | ✅ (0.1 rate) |
| **CLI Args** | ❌ | ✅ |

### 🔧 Technical Details

**Learning Rate Schedule**:
```
Warmup: 0-100 iterations (linear warmup)
Constant: 100-80% of total (full learning rate)
Decay: Last 20% (cosine decay to min_lr)
```

**Weight Decay**: 0.01 (L2 regularization on all parameters)

**Initialization**:
- Input projections: std = 0.02
- Output projections: std = 0.02 / sqrt(2 * N_LAYER)

**Dropout**:
- Rate: 0.1 (10% of activations zeroed)
- Applied: After attention output projection, after MLP output
- Scaling: 1 / (1 - dropout_rate) = 1.11x during training
- Thread-safe: Unique RNG seed per parallel batch item

### 📝 Documentation
- Updated parameter counts in README
- Added CLI usage instructions

---

## [0.3.0] - 2026-02-16

### 🚀 Major Features

#### Multi-Core Training with Rayon
- **Added Rayon parallelization** for batch processing
- **8x speedup** on multi-core CPUs (825% CPU usage on 12-core system)
- Thread-safe gradient computation with local buffers
- Efficient gradient aggregation after parallel computation

#### Training Performance
- Training time: **78 seconds** for 1000 iterations
- Dataset: Shakespeare (1.1MB, 1.1M tokens)
- Loss improvement: **5.88 → 3.23** (45% reduction)
- Uses all available CPU cores efficiently

### ✨ Improvements
- Increased default iterations from 200 to 1000
- Added `GradientBuffer` struct for thread-local gradients
- Parallel forward + backward passes per batch item
- Real-time loss reporting every 100 iterations

### 📊 Performance Metrics
```
Real time:  78.56 seconds
User time:  643.04 seconds (parallel work)
System time: 5.55 seconds
CPU usage:  825% (8.25 cores average)
Cores used: 12 available, ~8 effectively utilized
```

### 📝 Documentation
- Added `PERFORMANCE.md` with detailed benchmarks
- Updated README with training status
- Added multi-core usage examples

---

## [0.2.0] - 2026-02-16

### 🎓 Training Implementation

#### Working Training Loop
- **Backward pass implemented** for MLP layers
- **Adam optimizer** with momentum and variance tracking
- Gradient computation and weight updates working
- Loss decreases during training (proof of learning!)

#### Architecture
- 4-layer transformer (up from 1 layer)
- 128-dimensional embeddings (up from 32)
- 8 attention heads (up from 4)
- 64-token context window (up from 8)
- ~800K parameters (up from ~10K)

### ✨ Features
- Character-level tokenizer with BOS/EOS
- Data loading from text files
- Loss estimation and tracking
- Top-p sampling for generation
- Temperature control
- KV caching for efficiency

### 📝 Documentation
- `README.md` - Feature overview
- `QUICKSTART.md` - 5-minute setup guide
- `USAGE.md` - Detailed usage instructions
- `IMPROVEMENTS.md` - Changelog from original
- `STATUS.md` - Current status
- `WHY_GIBBERISH.md` - Explains untrained output
- `PARALLELIZATION.md` - Multi-core guide

### 🔧 Infrastructure
- Gradient buffers for all parameters
- Adam optimizer state (m and v vectors)
- Training loop with batch sampling
- Cross-entropy loss computation

### 📊 Initial Results
- Training works but slow (single-core)
- Loss decreases: ~5.5 → ~3.5 in 200 iterations
- Output improving but needs more training
- Model learning basic character patterns

---

## [0.1.0] - Original mini_gpt.rs

### Initial Implementation
- Based on @RandyMcMillan's mini_gpt.rs gist
- Tiny architecture: 1 layer, 32-dim, 4 heads
- ~10K parameters
- No training (random weights only)
- Forward pass and generation only
- Simple character-level vocabulary

### Features
- Minimal transformer architecture
- PRNG for weight initialization
- Top-p sampling
- Example with "hello world"

---

## Future Roadmap

### v0.4.0 - Quality Improvements ✅ Done
- [x] Gradient clipping
- [x] Learning rate scheduling (warmup → constant → cosine decay)
- [x] Dropout regularization
- [x] GPT-2 style initialization
- [ ] Full attention gradient computation (still simplified)
- [ ] Validation split and perplexity metrics

### v0.5.0 - Checkpointing & Metal ✅ Done
- [x] Checkpoint saving/loading with resume
- [x] Metal GPU inference acceleration (Candle, M-series)
- [x] Memory safety (fixed per-vector Metal allocation explosion)
- [x] Zero-IO training loop (memory-buffered checkpoints)
- [x] Ctrl-C graceful save

### v0.6.0 - Training Quality ✅ Done
- [x] Full attention gradient computation (K and V projections now backprop)
- [x] Validation split (90/10) and perplexity tracking

### v0.7.0 - BLAS Performance ✅ Done
- [x] Accelerate BLAS for CPU matmuls (2.1× speedup: ~450s → ~215s / 1000 iter)

### v0.7.1 - Scale + SGEMM ✅ Done
- [x] SGEMM batched backward (2.2× speedup for 256-dim model)
- [x] 256-dim embeddings (4.77M params)
- [x] Batch size 128, LR decay at 60%, ms/iter + ETA output

### v1.0.0 - Production Ready
- [ ] Multiple model size presets via CLI
- [ ] BPE tokenization
- [ ] Model evaluation suite
- [ ] Mixed precision training

---

## Credits

**Original Inspiration:** mini_gpt.rs by @RandyMcMillan
**Enhanced By:** Claude Sonnet 4.5
**Organization:** Monumental Systems

## License

MIT
