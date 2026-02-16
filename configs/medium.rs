// Medium Model Configuration - Better quality
// ~3M parameters, noticeable improvement

pub const N_EMBD: usize = 256;
pub const N_HEAD: usize = 8;
pub const N_LAYER: usize = 6;
pub const BLOCK_SIZE: usize = 128;
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;
pub const MLP_DIM: usize = 4 * N_EMBD;
pub const MAX_VOCAB: usize = 1024;

// Training parameters
pub const BATCH_SIZE: usize = 32;
pub const LEARNING_RATE: f32 = 2e-4;
pub const MAX_ITERS: usize = 10000;
pub const EVAL_INTERVAL: usize = 200;

// Good for: Higher quality output, larger datasets
// Training time: Several hours on CPU
// Memory: ~40MB
