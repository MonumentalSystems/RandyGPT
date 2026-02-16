// Small Model Configuration - Balanced (Default)
// ~800K parameters, good quality/speed tradeoff

pub const N_EMBD: usize = 128;
pub const N_HEAD: usize = 8;
pub const N_LAYER: usize = 4;
pub const BLOCK_SIZE: usize = 64;
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;
pub const MLP_DIM: usize = 4 * N_EMBD;
pub const MAX_VOCAB: usize = 512;

// Training parameters
pub const BATCH_SIZE: usize = 32;
pub const LEARNING_RATE: f32 = 3e-4;
pub const MAX_ITERS: usize = 5000;
pub const EVAL_INTERVAL: usize = 100;

// Good for: General purpose, decent quality
// Training time: ~1 hour on CPU for small dataset
// Memory: ~15MB
