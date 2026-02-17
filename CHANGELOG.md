# Changelog

All notable changes to randyGPT are documented here.

## Version Comparison

| Feature | v0.1 | v0.2 | v0.3 | v0.4 | v0.5.1 | v0.6.0 | v0.7.0 | v0.7.1 | v0.8.0 | v0.8.5 | v0.9.1 | v0.9.2 |
|---------|------|------|------|------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Layers** | 1 | 4 | 4 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| **Embedding Dim** | 32 | 128 | 128 | 128* | 128 | 128 | 128 | 256 | 256 | 256 | 256 | 256 |
| **Parameters** | ~10K | ~800K | ~800K | ~1.2M | ~1.2M | ~1.2M | ~1.2M | ~4.77M | ~4.77M | ~4.77M | ~4.82M | ~4.82M |
| **Training** | ❌ | ✅ Single-core | ✅ Multi-core | ✅ Multi-core | ✅ Multi-core | ✅ Multi-core | ✅ Multi-core | ✅ Multi-core | ✅ **GPU autograd** | ✅ **Full GPU** | ✅ **Full GPU** | ✅ **Full GPU** |
| **Attention grads** | ❌ | Q only | Q only | Q only | Q only | ✅ Q+K+V | ✅ Q+K+V | ✅ Q+K+V | ✅ Candle autograd | ✅ Candle autograd | ✅ Candle autograd | ✅ Candle autograd |
| **Optimizer** | - | Adam | Adam | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW (CPU moments) | **AdamW (GPU moments)** | **AdamW (GPU moments)** | **AdamW (GPU moments)** |
| **LR Schedule** | - | Immediate decay | Immediate decay | Constant→Decay | Constant→Decay | Constant→Decay | Constant→Decay | Warmup→60%→Decay | Warmup→60%→Decay | Warmup→60%→Decay | Warmup→60%→Decay + **--lr flag** | same |
| **Tokenizer** | char | char | char | char | char | char | char | char | char | char | char | **char + BPE** |
| **Dropout** | ❌ | ❌ | ❌ | ✅ (0.1) | ✅ (0.1) | ✅ (0.1) | ✅ (0.1) | ✅ (0.1) | ✅ (0.1) | ✅ (0.1) | ✅ (0.1) | ✅ (0.1) |
| **Checkpoints** | ❌ | ❌ | ❌ | ❌ | ✅ memory-buffered | ✅ | ✅ | ✅ | ✅ RGPT0002 | ✅ **RGPT0003** | ✅ RGPT0003 **best=val** | ✅ RGPT0003 |
| **Ctrl-C save** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Val loss / ppl** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Early stopping** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ patience=20 | ✅ **val-based** | ✅ val-based |
| **Metal GPU** | ❌ | ❌ | ❌ | ✅ | ✅ (stable) | ✅ | ✅ | ✅ | ✅ training | ✅ **fwd+bwd+optim** | ✅ fwd+bwd+optim | ✅ fwd+bwd+optim |
| **BLAS (Accelerate)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ sgemv/sger | ✅ +sgemm | ✅ (CPU fallback) | ✅ (CPU fallback) | ✅ (CPU fallback) | ✅ (CPU fallback) |
| **Batch size** | - | - | - | - | - | - | 32 | 128 | 128 | 64 | 64 | 64 |
| **Context window** | 64 | 64 | 64 | 64 | 64 | 64 | 64 | 64 | 64 | **256** | **256** | **256** |
| **Best val ppl** | - | - | - | - | - | - | - | 10.8 | 10.8 | - | **10.6** | TBD (BPE) |
| **Speed (ms/iter)** | - | ~600 | ~78s total | ~450 | ~450 | ~450 | ~215 | ~96§ | ~49¶ | ~1835 | ~1835 | ~1870 |

†RSS ~1GB real at T=256; Activity Monitor shows 4-8GB (Metal unified memory pool — not CPU-resident)
‡Estimated
§SGEMM batched backward; measured ~964ms/iter CPU with 12 cores; batch=128
¶Candle Metal autograd; confirmed 488ms/iter over 500 iters; ~2× vs v0.7.1 CPU; 60.9% GPU; val ppl 10.8 @ iter 1000
\*v0.4 targeted 256-dim but shipped at 128 due to the Metal memory issue fixed in v0.5

---

## [0.9.2] - 2026-02-17

### BPE Tokenization (`--bpe` flag)

#### Implementation
- Added `BpeTokenizer` to `src/tokenizer.rs` alongside existing char-level tokenizer — same public interface (`encode`, `decode`, `vocab_size`), zero changes to training loop, model, or checkpoint format
- BPE training: incremental pair-counting with max-heap (lazy deletion) — O(n + merges × log n) vs naive O(n × merges²). Trains 2000-token vocab on 7.6MB corpus in ~15 seconds
- BPE encoding: one pass per merge priority level, applied to parallel line-chunks via rayon — full 7.6MB corpus encodes in seconds vs minutes for naive approach
- Vocab saved to `vocab.json` (serde_json) on first run, loaded automatically on resume — no retraining needed
- `--bpe [N]` CLI flag: `--bpe` uses default 2000-token vocab, `--bpe 3000` uses custom size
- Char-level checkpoints incompatible with BPE checkpoints — caught cleanly by existing `vocab_size` mismatch check in `checkpoint.rs`
- Added `serde` + `serde_json` dependencies for vocab persistence

#### Usage
```bash
# First run — trains BPE vocab, saves vocab.json, trains model
./randygpt --bpe --iters 3000

# Resume — loads vocab.json automatically
./randygpt --bpe --resume --iters 3000

# Custom vocab size
./randygpt --bpe 3000 --iters 3000
```

#### Early BPE Results (model-L, 2000 tokens, in progress)
- Initial ppl ~2000 (full vocab entropy: log(2000) ≈ 7.6 nats) — expected for fresh init
- iter 100: ppl 663, iter 200: ppl 416, iter 250: ppl 318 — steep descent, patience resetting every eval
- Generation at iter 100 already shows real Shakespeare words ("hath", "father", "speak", "friend") vs char-level noise at same iter count
- Speed: ~1870ms/iter (same as char-level — wte/lm_head are larger but GPU-bound compute unchanged)
- Full results TBD when 3000-iter run completes

---

## [0.9.1] - 2026-02-17

### Training Quality Fixes + `--lr` / `--min-lr` Flags

#### `checkpoint_best.bin` Now Tracks Best Val Loss
- Previously `checkpoint_best.bin` saved the checkpoint at the best *train* loss iteration — a misleading signal since train loss can diverge from val loss late in training
- Fixed: `checkpoint_best.bin` now captures the checkpoint whenever `val_loss < best_val_loss` (patience reset), making it the true best-generalization checkpoint
- `best_val_loss` seeded from checkpoint header on resume — patience counter starts relative to the previous run's best, not `f32::INFINITY`

#### `--lr` and `--min-lr` CLI Flags
- Override `LEARNING_RATE` and `MIN_LEARNING_RATE` at runtime without recompiling:
  ```
  ./randygpt --resume --iters 3000 --lr 1e-5 --min-lr 1e-6
  ```
- LR schedule ceiling and floor are now runtime parameters passed through to both `train()` and `train_candle()`
- Printed at startup when overridden: `LR override: 0.00001 → 0.000001`
- Enables fine-tuning runs at reduced LR from an existing checkpoint

#### Confirmed v0.9.x Results (Shakespeare, ~4.82M params, T=256)
| Metric | Value |
|--------|-------|
| Best val loss | **2.3716** |
| Best val perplexity | **10.6** |
| Iters to best | ~929 (from scratch) |
| Speed | ~1835ms/iter |
| Memory (real RSS) | ~1GB |
| Memory (virtual) | 4–8GB (Metal pool, normal) |
| LR at best | 3e-5 (full, pre-decay) |

- T=256 context window outperforms T=64 peak (ppl 10.8): model sees full Shakespeare speeches per sample
- Val floor ~2.37 is the data ceiling for 5M params on ~1MB Shakespeare corpus
- Further improvement requires more data or a larger model (v1.0 roadmap)

#### Model Size Scaling Experiment (model-S, 7.6MB corpus)
- Built `--features model-s` binary (128-dim, 4-head, 4-layer, ~0.85M params) and trained from scratch on the 7.6MB expanded corpus
- Result: best val loss **2.4918 (ppl 12.1)**, final val **2.5316 (ppl 12.6)** at iter 2000, 592ms/iter
- Comparison: model-L (4.82M) + 1MB = ppl **10.6** vs model-S (0.85M) + 7.6MB = ppl **12.1**
- **Conclusion: model capacity is the binding constraint** — 7× more data does not compensate for 5.7× fewer parameters at this scale. Expanding the corpus pays off only when paired with a model large enough to use it.

#### Model-L + 7.6MB Experiment
- Same model-L architecture (4.82M params) that achieved ppl 10.6 on 1MB, now trained on 7.6MB corpus
- Result: best val loss **2.4474 (ppl 11.6)** at iter ~850, early stopping fired at iter 1050 (patience 20/20)
- Resumed from best checkpoint at lr 1e-5; best improved to **2.4474 (ppl 11.6)** — did not beat 10.6 baseline
- **Finding:** more data alone did not help model-L — the model plateaued at ppl 11.6 with 7.6MB vs 10.6 with 1MB
- Root cause: 7.6MB corpus has more stylistic diversity (sonnets + 8 plays) than the focused 1MB corpus; char-level model capacity is the bottleneck, not data volume at this architecture
- **Conclusion: char-level tokenization is the fundamental ceiling** — model needs to learn to spell before learning style. BPE tokenization is the right next step.

#### Model Size Scaling Experiment (model-M, 7.6MB corpus)
- Built `--features model-m` binary (192-dim, 6-head, 6-layer, ~2.75M params) and trained on the 7.6MB expanded corpus
- Run was fragmented across multiple `--iters` segments (500 → 1000) due to interactive testing; total ~1000 iters trained
- Result at iter 1000: best val loss **2.4993 (ppl 12.5)**, speed ~1370ms/iter
- Descent trajectory: ppl 116.9 @ iter 0 → 30.9 @ iter 100 → 15.2 @ iter 270 → 13.5 @ iter 400 → **12.5 @ iter 1000**
- Model-M is clearly stronger than model-S early (ppl 30.9 vs ~65 at iter 100) but didn't reach ppl 10.6 in 1000 iters
- **Note:** fragmented LR schedule (each `--iters N` restart re-anchors cosine decay) meant the model never got a full 1800-iter flat LR phase; true potential likely higher
- Generation at ppl 12.5 shows more word-like fragments than model-S ("have my insure that", "the meldo cout") but still char-level noise
- Next: model-L (4.82M) + 7.6MB — same architecture as the ppl 10.6 baseline, now with 7× more data

| Metric | model-S + 7.6MB | model-M + 7.6MB | model-L + 1MB (baseline) |
|--------|----------------|----------------|--------------------------|
| Params | ~0.85M | ~2.75M | ~4.82M |
| Best ppl | 12.1 | 12.5* | **10.6** |
| Iters | 2000 | ~1000 (fragmented) | ~929 |
| ms/iter | 592 | 1370 | 1835 |

*fragmented LR schedule — not a fair comparison to a clean full run

---

## [0.9.0] - 2026-02-16

### 4× Context Window (BLOCK_SIZE 64 → 256)

#### Architecture Change
- `BLOCK_SIZE: 64 → 256` — model now sees ~4 full lines of Shakespeare per training sample
- `BATCH_SIZE: 128 → 64` — halved to keep Metal activation tape (`[B, T, D]`) memory-stable at 4× context
- Token throughput per iter unchanged: 128×64 = 64×256 = 8,192 tokens/batch
- Parameters: ~4.77M → **~4.82M** (wpe table grows: 192 × N_EMBD extra floats)

#### Why This Helps
- At T=64 the model saw ~1 line of text per sample — enough for character n-gram statistics, not enough for verse structure
- At T=256 each sample spans a full speech: speaker cues, line breaks, iambic rhythm, multi-sentence flow
- Attention heads can now attend across 256 positions — rhyme schemes, callback words, dialogue turns all fit in one window

#### Measured Results vs T=64 Baseline (500 iters, Shakespeare)
| iter | T=64 val ppl | T=256 val ppl |
|------|-------------|--------------|
| 200 | 12.3 | 13.4 |
| 300 | 11.8 | 12.0 |
| 400 | 11.2 | 11.2 ← tied |
| 500 | 12.4 ← degrading | **11.8 ← still improving** |

- T=64 overfit and reversed after iter 400; T=256 had tight train/val gap and continued improving
- Speed: ~1912ms/iter (vs ~434ms at T=64) — attention O(T²) accounts for most of the increase
- Memory: ~200MB real RSS, ~8GB virtual (Metal unified memory, unchanged pattern)

#### Fix: Initial Loss Estimate on Resume
- Previously `--resume` showed initial loss ~4.20 (freshly-initialized CPU model) instead of resumed checkpoint loss
- Fixed: sync `CandleModel` weights back to CPU `GPTModel` before `estimate_loss` when resuming on Metal

---

## [0.8.5] - 2026-02-16

### Full GPU AdamW via GpuAdamState

#### All Three Training Phases Now on Metal
- **v0.8.0 hybrid**: Forward + backward on GPU; AdamW moments stayed on CPU (`Vec<f32>`)
- **v0.8.5**: Optimizer moments now live as `Var` tensors on Metal — zero CPU transfers in the hot loop

#### GpuAdamState
- New `pub mod gpu_adam` in `optimizer.rs` with `GpuAdamState` struct
- Moment Vars (`m`, `v`) allocated on the same Metal device as weight Vars via `Var::zeros()`
- Full AdamW step as Candle tensor ops:
  - Gradient clipping: `g.clamp(-GRAD_CLIP, GRAD_CLIP)?` (stays on GPU, no download)
  - EMA updates: `m = m * β₁ + g * (1-β₁)`, `v = v * β₂ + g² * (1-β₂)`
  - Bias correction: scalar multiply (no data movement)
  - Weight decay: `θ *= (1 - lr * wd)` in-place via `Var::set()`
  - Parameter update: `θ -= lr * m̂ / (√v̂ + ε)`
  - All ops stay on GPU; `Var::set()` triggers no allocation

#### CandleModel Simplification
- Removed all `m_*/v_*` Vec<f32> moment fields from `CandleModel` and `CandleLayer`
- `CandleLayer` is now 6 weight Vars only: `wq, wk, wv, wo, fc1, fc2`
- New `all_vars()` method returns Vars in canonical order: `wte, wpe, lm_head, [N_LAYER × wq wk wv wo fc1 fc2]`
- `GpuAdamState` indexed by the same order — guaranteed consistency
- `var_to_vec` made `pub` for use in checkpoint serialization

#### train_candle() Simplification
- Replaced per-weight `update_var!` macro (9 calls per iteration) with:
  ```rust
  let vars = model.all_vars();
  opt.step(&grads, &vars, lr).unwrap();
  ```
- One call handles all weights; GpuAdamState iterates over `grads.get(var)` internally

#### RGPT0003 Checkpoint Format
- New magic `b"RGPT0003"` — includes optimizer state as GPU Var data
- Layout: header + weights (all_vars() order) + `opt.m` moments + `opt.v` moments
- `serialize_checkpoint_v3(model, opt, iter, step, best_loss)` — pulls all data off GPU via `.flatten_all().to_vec1()`
- `load_checkpoint_v3(path, model, opt)` — uploads weights and moments as Vars, restores `step_t`
- Resume chain: RGPT0003 → RGPT0002 (weights only, fresh moments) → RGPT0001 (CPU path)
- RGPT0002 `serialize_checkpoint_v2` kept but weights-only (moments now in GpuAdamState)

---

## [0.8.0] - 2026-02-16

### Candle Autograd Metal Training (Strategy A Hybrid)

#### GPU Training via Candle Autograd (~2× speedup)
- **Forward and backward passes now run on Metal GPU** via Candle autograd
- New `CandleModel` / `CandleLayer` structs: all weights stored as `candle_core::Var` on Metal device
- New `forward_candle_train()`: fully batched `[BATCH_SIZE, BLOCK_SIZE, N_EMBD]` forward pass using Candle tensor ops
  - Embeddings via `index_select` + positional encoding via `narrow`
  - RMSNorm, multi-head causal attention, squared-ReLU MLP — all Candle ops
  - Cross-entropy via `candle_nn::loss::cross_entropy`
- `loss.backward()` → `GradStore` replaces hand-written two-pass SGEMM backward
- **Optimizer stays on CPU (Strategy A)**: AdamW moments remain `Vec<f32>`, `adam_step()` reused unchanged
- Gradients pulled off GPU via `.flatten_all().to_vec1::<f32>()`, clipped, Adam-updated, re-uploaded via `Var::set()`
- **Measured speedup**: ~488ms/iter steady-state vs ~964ms/iter (v0.7.1 CPU) = **~2×**; 60.9% GPU utilization

#### Confirmed 1000-Iteration Results (Shakespeare, 4.77M params)
| Metric | Value |
|--------|-------|
| Avg ms/iter | 488ms |
| Total time (500 iters) | 245s |
| Val loss @ iter 1000 | 2.38 |
| Val perplexity @ iter 1000 | 10.8 |
| Best val loss | 2.25 @ iter 962 |
| GPU utilization | 60.9% |
| RSS (real) | ~400 MB |

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

### v0.8.x - Full GPU + Context Window ✅ Done
- [x] Full GPU AdamW via GpuAdamState (moments as Metal Vars)
- [x] RGPT0003 checkpoint format (weights + moments)
- [x] BLOCK_SIZE 64 → 256 (4× context window)
- [x] Early stopping (patience=20, val-based)
- [x] 50-sample val estimates for stability

### v0.9.x - Training Quality ✅ Done
- [x] `checkpoint_best.bin` tracks best val loss (not train loss)
- [x] `best_val_loss` seeded from checkpoint on resume
- [x] `--lr` / `--min-lr` CLI flags for runtime LR override
- [x] Confirmed best: val ppl **10.6** (2.3716 loss) at T=256

### v1.0.0 - BPE Tokenization + Quality Breakthrough
- [x] Multiple model size presets (S/M/L/XL) via `--features` at build time
- [x] Larger training corpus: 7.6MB (8× Shakespeare works from Project Gutenberg)
- [x] **model-M (2.7M) + 7.6MB** — tested (ppl 12.5 at iter 1000, fragmented schedule)
- [x] **model-L (4.82M) + 7.6MB** — char-level ceiling confirmed at ppl 11.6; more data alone insufficient
- [x] **BPE tokenization** — `--bpe [N]` flag, 2000-token vocab, heap-based training ~15s, parallel encoding
- [ ] **BPE 3000-iter run to completion** — target: first coherent multi-word Shakespeare phrases ← **running now**
- [ ] Mixed precision (f16) training on Metal

---

## Credits

**Original Inspiration:** mini_gpt.rs by @RandyMcMillan
**Enhanced By:** Claude Sonnet 4.5
**Organization:** Monumental Systems

## License

MIT
