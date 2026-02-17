/* ------------------------------------------------------------------ */
/* Hyperparameters and global constants                               */
/* ------------------------------------------------------------------ */

// Model architecture
pub const N_EMBD: usize = 256;
pub const N_HEAD: usize = 8;
pub const N_LAYER: usize = 6;
pub const BLOCK_SIZE: usize = 256;
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;
pub const MLP_DIM: usize = 4 * N_EMBD;
pub const MAX_VOCAB: usize = 512;

// Training
pub const BATCH_SIZE: usize = 64;
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
// Set to 0 to disable. E.g. with EVAL_INTERVAL=100 and patience=5 → stops after
// 500 consecutive iters with no val improvement.
pub const EARLY_STOP_PATIENCE: usize = 10;

// Metal
pub const USE_METAL: bool = true;
#[allow(dead_code)]
pub const CANDLE_TRAIN: bool = true; // use Candle autograd for training when Metal available
