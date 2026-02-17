/* ------------------------------------------------------------------ */
/* Hyperparameters and global constants                               */
/* ------------------------------------------------------------------ */
//
// Model size presets — select at build time with:
//   cargo build --release --features model-s   (~1.1M params, ~500ms/iter)
//   cargo build --release --features model-m   (~2.7M params, ~1100ms/iter)
//   cargo build --release --features model-l   (~4.82M params, ~1835ms/iter)  ← default
//   cargo build --release --features model-xl  (~10.8M params, ~4000ms/iter)
//
// All presets use BLOCK_SIZE=256, BATCH_SIZE=64, same training constants.
// Checkpoints are NOT cross-compatible between sizes (different weight shapes).

// ── Architecture ──────────────────────────────────────────────────────────

#[cfg(feature = "model-s")]
pub const N_EMBD:  usize = 128;
#[cfg(feature = "model-s")]
pub const N_HEAD:  usize = 4;
#[cfg(feature = "model-s")]
pub const N_LAYER: usize = 4;

#[cfg(feature = "model-m")]
pub const N_EMBD:  usize = 192;
#[cfg(feature = "model-m")]
pub const N_HEAD:  usize = 6;
#[cfg(feature = "model-m")]
pub const N_LAYER: usize = 6;

#[cfg(feature = "model-xl")]
pub const N_EMBD:  usize = 384;
#[cfg(feature = "model-xl")]
pub const N_HEAD:  usize = 8;
#[cfg(feature = "model-xl")]
pub const N_LAYER: usize = 8;

// Default (model-l): 256-dim, 8-head, 6-layer — ~4.82M params
#[cfg(not(any(feature = "model-s", feature = "model-m", feature = "model-xl")))]
pub const N_EMBD:  usize = 256;
#[cfg(not(any(feature = "model-s", feature = "model-m", feature = "model-xl")))]
pub const N_HEAD:  usize = 8;
#[cfg(not(any(feature = "model-s", feature = "model-m", feature = "model-xl")))]
pub const N_LAYER: usize = 6;

pub const BLOCK_SIZE: usize = 256;
pub const HEAD_DIM:   usize = N_EMBD / N_HEAD;
pub const MLP_DIM:    usize = 4 * N_EMBD;
pub const MAX_VOCAB:  usize = 8192;   // raised for BPE (char-level uses ~117)

// ── BPE tokenizer ─────────────────────────────────────────────────────────
pub const BPE_VOCAB_SIZE: usize = 2000; // default target vocab for --bpe mode
pub const BPE_VOCAB_PATH: &str  = "vocab.json";

// ── Training ──────────────────────────────────────────────────────────────

pub const BATCH_SIZE: usize = 64;
// Gradient accumulation: run this many micro-batches before each optimizer step.
// Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS. Set to 1 to disable.
pub const GRAD_ACCUM_STEPS: usize = 1;
pub const LEARNING_RATE: f32 = 3e-5;
pub const MIN_LEARNING_RATE: f32 = 3e-6;
pub const WEIGHT_DECAY: f32 = 0.01;
pub const DROPOUT_RATE: f32 = 0.1;
pub const BETA1: f32 = 0.9;
pub const BETA2: f32 = 0.999;
pub const EPSILON: f32 = 1e-8;
pub const MAX_ITERS: usize = 1000;
pub const EVAL_INTERVAL: usize = 10;
pub const GRAD_CLIP: f32 = 1.0;
// Early stopping: halt if val loss hasn't improved for this many eval intervals.
// Set to 0 to disable. E.g. patience=20 + EVAL_INTERVAL=10 → stops after
// 200 consecutive iters with no val improvement.
pub const EARLY_STOP_PATIENCE: usize = 20;

// ── Metal ─────────────────────────────────────────────────────────────────

pub const USE_METAL: bool = true;
#[allow(dead_code)]
pub const CANDLE_TRAIN: bool = true; // use Candle autograd for training when Metal available
