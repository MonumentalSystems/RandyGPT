// Tiny Model Configuration - Fast testing
// ~100K parameters, quick iteration

pub const N_EMBD: usize = 64;
pub const N_HEAD: usize = 4;
pub const N_LAYER: usize = 2;
pub const BLOCK_SIZE: usize = 32;
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;
pub const MLP_DIM: usize = 4 * N_EMBD;
pub const MAX_VOCAB: usize = 256;

// Training parameters
pub const BATCH_SIZE: usize = 16;
pub const LEARNING_RATE: f32 = 5e-4;
pub const MAX_ITERS: usize = 2000;
pub const EVAL_INTERVAL: usize = 50;

// Good for: Quick experiments, testing code changes
// Training time: Minutes on CPU
// Memory: ~5MB
