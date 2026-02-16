// Large Model Configuration - Best quality
// ~20M parameters, production-quality

pub const N_EMBD: usize = 512;
pub const N_HEAD: usize = 16;
pub const N_LAYER: usize = 12;
pub const BLOCK_SIZE: usize = 256;
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;
pub const MLP_DIM: usize = 4 * N_EMBD;
pub const MAX_VOCAB: usize = 2048;

// Training parameters
pub const BATCH_SIZE: usize = 16;  // Smaller due to memory
pub const LEARNING_RATE: f32 = 1e-4;
pub const MAX_ITERS: usize = 20000;
pub const EVAL_INTERVAL: usize = 500;

// Good for: Best possible output, large datasets
// Training time: Many hours on CPU, consider GPU
// Memory: ~200MB
// Note: This size really benefits from GPU acceleration
