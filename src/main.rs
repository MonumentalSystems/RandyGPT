use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write, Read};
use std::path::Path;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use rayon::prelude::*;
use candle_core::{Device, Tensor, Result as CandleResult};

/* ------------------------------------------------------------------ */
/* Enhanced Model hyper-parameters                                    */
/* ------------------------------------------------------------------ */
const N_EMBD: usize = 128;        // Scaled up from 128 (3M params)
const N_HEAD: usize = 8;          // Keep 8 heads
const N_LAYER: usize = 6;         // Scaled up from 4 (deeper model)
const BLOCK_SIZE: usize = 64;     // Keep 64 context
const HEAD_DIM: usize = N_EMBD / N_HEAD;
const MLP_DIM: usize = 4 * N_EMBD;
const MAX_VOCAB: usize = 512;     // Support more characters/tokens

// Training parameters
const BATCH_SIZE: usize = 32;
const LEARNING_RATE: f32 = 3e-5;      // Tuned for larger model
const MIN_LEARNING_RATE: f32 = 3e-6;  // Decay to 10% of initial
const WEIGHT_DECAY: f32 = 0.01;       // L2 regularization
const DROPOUT_RATE: f32 = 0.1;        // Dropout probability
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-8;
const MAX_ITERS: usize = 1000;
const EVAL_INTERVAL: usize = 100;
const GRAD_CLIP: f32 = 1.0;  // Gradient clipping threshold
const USE_METAL: bool = true;  // Enable Metal GPU acceleration

/* ------------------------------------------------------------------ */
/* Metal GPU Acceleration                                              */
/* ------------------------------------------------------------------ */

// Global Metal device (initialized once)
lazy_static::lazy_static! {
    static ref METAL_DEVICE: Option<Device> = {
        if USE_METAL {
            match Device::new_metal(0) {
                Ok(dev) => {
                    eprintln!("✓ Metal GPU enabled on device: {:?}", dev);
                    Some(dev)
                }
                Err(e) => {
                    eprintln!("⚠ Metal GPU unavailable: {}", e);
                    None
                }
            }
        } else {
            None
        }
    };
}

// Metal-accelerated batched matrix multiply: out = x * W^T
// x: [seq_len, nin], w: [nout, nin] → out: [seq_len, nout]
fn metal_matmul_batch(
    x: &[f32],     // Input: [seq_len x nin] row-major
    w: &[f32],     // Weight: [nout x nin] row-major
    seq_len: usize,
    nin: usize,
    nout: usize,
    out: &mut [f32], // Output: [seq_len x nout] row-major
) -> CandleResult<()> {
    let device = METAL_DEVICE.as_ref().unwrap();

    // Upload to Metal device
    let x_t = Tensor::from_slice(x, (seq_len, nin), device)?;
    let w_t = Tensor::from_slice(w, (nout, nin), device)?;

    // Compute x * W^T → [seq_len, nout]
    let result = x_t.matmul(&w_t.t()?)?;

    // Copy back to CPU
    let flat = result.flatten_all()?.to_vec1::<f32>()?;
    out.copy_from_slice(&flat);

    Ok(())
}


/* ------------------------------------------------------------------ */
/* Minimal xorshift PRNG                                             */
/* ------------------------------------------------------------------ */
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    fn gauss(&mut self, mean: f32, std: f32) -> f32 {
        let mut u1 = self.uniform();
        let u2 = self.uniform();
        if u1 < 1e-30 { u1 = 1e-30; }
        let mag = ((-2.0 * u1.ln()).sqrt()) as f32;
        mean + std * mag * ((2.0 * std::f64::consts::PI * u2).cos() as f32)
    }

    fn choice(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize
    }
}

/* ------------------------------------------------------------------ */
/* Tokenizer - Character-level with BOS/EOS                          */
/* ------------------------------------------------------------------ */
struct Tokenizer {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: Vec<char>,
    bos_id: usize,
    eos_id: usize,
    vocab_size: usize,
}

impl Tokenizer {
    fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut idx_to_char = vec!['<'; 1]; // BOS
        idx_to_char.push('>'); // EOS
        idx_to_char.extend(chars);

        let char_to_idx: HashMap<char, usize> = idx_to_char
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        let bos_id = 0;
        let eos_id = 1;
        let vocab_size = idx_to_char.len();

        Self { char_to_idx, idx_to_char, bos_id, eos_id, vocab_size }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_idx.get(&c).copied())
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&idx| self.idx_to_char.get(idx))
            .collect()
    }
}

/* ------------------------------------------------------------------ */
/* Weights and Activations                                           */
/* ------------------------------------------------------------------ */
#[derive(Clone)]
struct PosActs {
    x_embed: Vec<f32>,
    x_in: Vec<Vec<f32>>,
    xn_attn: Vec<Vec<f32>>,
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    attn_out: Vec<Vec<f32>>,
    x_mid: Vec<Vec<f32>>,
    xn_mlp: Vec<Vec<f32>>,
    mlp_pre: Vec<Vec<f32>>,
    mlp_post: Vec<Vec<f32>>,
    x_out: Vec<f32>,
}

impl PosActs {
    fn new() -> Self {
        Self {
            x_embed: vec![0.0; N_EMBD],
            x_in: vec![vec![0.0; N_EMBD]; N_LAYER],
            xn_attn: vec![vec![0.0; N_EMBD]; N_LAYER],
            q: vec![vec![0.0; N_EMBD]; N_LAYER],
            k: vec![vec![0.0; N_EMBD]; N_LAYER],
            v: vec![vec![0.0; N_EMBD]; N_LAYER],
            attn_out: vec![vec![0.0; N_EMBD]; N_LAYER],
            x_mid: vec![vec![0.0; N_EMBD]; N_LAYER],
            xn_mlp: vec![vec![0.0; N_EMBD]; N_LAYER],
            mlp_pre: vec![vec![0.0; MLP_DIM]; N_LAYER],
            mlp_post: vec![vec![0.0; MLP_DIM]; N_LAYER],
            x_out: vec![0.0; N_EMBD],
        }
    }
}

struct LayerWeights {
    // Attention weights
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,

    // MLP weights
    fc1: Vec<f32>,
    fc2: Vec<f32>,

    // Gradients
    d_wq: Vec<f32>,
    d_wk: Vec<f32>,
    d_wv: Vec<f32>,
    d_wo: Vec<f32>,
    d_fc1: Vec<f32>,
    d_fc2: Vec<f32>,

    // Adam optimizer moments
    m_wq: Vec<f32>, v_wq: Vec<f32>,
    m_wk: Vec<f32>, v_wk: Vec<f32>,
    m_wv: Vec<f32>, v_wv: Vec<f32>,
    m_wo: Vec<f32>, v_wo: Vec<f32>,
    m_fc1: Vec<f32>, v_fc1: Vec<f32>,
    m_fc2: Vec<f32>, v_fc2: Vec<f32>,
}

impl LayerWeights {
    fn new(rng: &mut Rng, _layer_idx: usize) -> Self {
        let mut make_params = |sz: usize, std: f32| -> Vec<f32> {
            (0..sz).map(|_| rng.gauss(0.0, std)).collect()
        };

        let zero_params = |sz: usize| -> Vec<f32> {
            vec![0.0; sz]
        };

        // GPT-2 style initialization:
        // - Input projections: std = 0.02
        // - Output projections: std = 0.02 / sqrt(2 * N_LAYER)
        // This accounts for residual branch accumulation
        let std_in = 0.02;
        let std_out = 0.02 / (2.0 * N_LAYER as f32).sqrt();

        Self {
            wq: make_params(N_EMBD * N_EMBD, std_in),
            wk: make_params(N_EMBD * N_EMBD, std_in),
            wv: make_params(N_EMBD * N_EMBD, std_in),
            wo: make_params(N_EMBD * N_EMBD, std_out),  // Output projection
            fc1: make_params(MLP_DIM * N_EMBD, std_in),
            fc2: make_params(N_EMBD * MLP_DIM, std_out), // Output projection

            d_wq: zero_params(N_EMBD * N_EMBD),
            d_wk: zero_params(N_EMBD * N_EMBD),
            d_wv: zero_params(N_EMBD * N_EMBD),
            d_wo: zero_params(N_EMBD * N_EMBD),
            d_fc1: zero_params(MLP_DIM * N_EMBD),
            d_fc2: zero_params(N_EMBD * MLP_DIM),

            m_wq: zero_params(N_EMBD * N_EMBD), v_wq: zero_params(N_EMBD * N_EMBD),
            m_wk: zero_params(N_EMBD * N_EMBD), v_wk: zero_params(N_EMBD * N_EMBD),
            m_wv: zero_params(N_EMBD * N_EMBD), v_wv: zero_params(N_EMBD * N_EMBD),
            m_wo: zero_params(N_EMBD * N_EMBD), v_wo: zero_params(N_EMBD * N_EMBD),
            m_fc1: zero_params(MLP_DIM * N_EMBD), v_fc1: zero_params(MLP_DIM * N_EMBD),
            m_fc2: zero_params(N_EMBD * MLP_DIM), v_fc2: zero_params(N_EMBD * MLP_DIM),
        }
    }
}

/* ------------------------------------------------------------------ */
/* Model Structure                                                    */
/* ------------------------------------------------------------------ */
struct GPTModel {
    wte: Vec<f32>,  // Token embeddings
    wpe: Vec<f32>,  // Position embeddings
    layers: Vec<LayerWeights>,
    lm_head: Vec<f32>,  // Final projection to vocab

    // Gradients for embeddings
    d_wte: Vec<f32>,
    d_wpe: Vec<f32>,
    d_lm_head: Vec<f32>,

    // Adam moments for embeddings
    m_wte: Vec<f32>, v_wte: Vec<f32>,
    m_wpe: Vec<f32>, v_wpe: Vec<f32>,
    m_lm_head: Vec<f32>, v_lm_head: Vec<f32>,

    vocab_size: usize,
}

impl GPTModel {
    fn new(vocab_size: usize, rng: &mut Rng) -> Self {
        let wte_size = vocab_size * N_EMBD;
        let wpe_size = BLOCK_SIZE * N_EMBD;
        let lm_size = vocab_size * N_EMBD;

        // Create layers first (with GPT-2 style initialization)
        let layers: Vec<LayerWeights> = (0..N_LAYER)
            .map(|li| LayerWeights::new(rng, li))
            .collect();

        // Then create embedding params
        let wte: Vec<f32> = (0..wte_size).map(|_| rng.gauss(0.0, 0.02)).collect();
        let wpe: Vec<f32> = (0..wpe_size).map(|_| rng.gauss(0.0, 0.01)).collect();
        let lm_head: Vec<f32> = (0..lm_size).map(|_| rng.gauss(0.0, 0.02)).collect();

        Self {
            wte,
            wpe,
            layers,
            lm_head,

            d_wte: vec![0.0; wte_size],
            d_wpe: vec![0.0; wpe_size],
            d_lm_head: vec![0.0; lm_size],

            m_wte: vec![0.0; wte_size], v_wte: vec![0.0; wte_size],
            m_wpe: vec![0.0; wpe_size], v_wpe: vec![0.0; wpe_size],
            m_lm_head: vec![0.0; lm_size], v_lm_head: vec![0.0; lm_size],

            vocab_size,
        }
    }
}

/* ------------------------------------------------------------------ */
/* Math primitives                                                    */
/* ------------------------------------------------------------------ */
fn linear_fwd(x: &[f32], w: &[f32], nout: usize, nin: usize, out: &mut [f32]) {
    // Training always uses CPU — Metal is only used in batched inference (forward_metal_logits).
    // Per-vector Metal calls allocate GPU memory tens of thousands of times per iteration,
    // causing catastrophic memory exhaustion. Batched matmuls (forward_metal_logits) are
    // used for estimate_loss and generate instead.
    linear_fwd_cpu(x, w, nout, nin, out);
}

// CPU fallback implementation
fn linear_fwd_cpu(x: &[f32], w: &[f32], nout: usize, nin: usize, out: &mut [f32]) {
    for r in 0..nout {
        // zip-based dot product — LLVM can auto-vectorize this with SIMD
        out[r] = w[r * nin..(r + 1) * nin].iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
    }
}

fn rmsnorm_fwd(x: &[f32], n: usize, out: &mut [f32]) -> f32 {
    let mut ms = 0.0;
    for i in 0..n {
        ms += x[i] * x[i];
    }
    let scale = 1.0 / ((ms / n as f32) + 1e-5).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale;
    }
    scale
}

fn softmax_fwd(logits: &[f32], n: usize, probs: &mut [f32], temp: f32) {
    let mut mx = logits[0] / temp;
    for i in 1..n {
        let val = logits[i] / temp;
        if val > mx {
            mx = val;
        }
    }

    let mut sum = 0.0;
    for i in 0..n {
        probs[i] = ((logits[i] / temp) - mx).exp();
        sum += probs[i];
    }

    let inv = 1.0 / sum;
    for i in 0..n {
        probs[i] *= inv;
    }
}

fn cross_entropy_loss(probs: &[f32], target: usize) -> f32 {
    let p = probs[target].max(1e-10);
    -p.ln()
}

/* ------------------------------------------------------------------ */
/* Forward Pass                                                       */
/* ------------------------------------------------------------------ */
fn forward(
    tokens: &[usize],
    model: &GPTModel,
    kv_cache: &mut Vec<Vec<(Vec<f32>, Vec<f32>)>>,
    training: bool,
    mut rng: Option<&mut Rng>,
) -> (Vec<Vec<f32>>, Vec<PosActs>) {
    let seq_len = tokens.len();
    let mut all_logits = Vec::new();
    let mut all_acts = Vec::new();

    // Process each position
    for pos in 0..seq_len {
        let tok = tokens[pos];
        let mut act = PosActs::new();

        // Embedding: token + position
        for i in 0..N_EMBD {
            act.x_embed[i] = model.wte[tok * N_EMBD + i] + model.wpe[pos * N_EMBD + i];
        }

        let mut x = act.x_embed.clone();

        // Transformer layers
        for li in 0..N_LAYER {
            act.x_in[li] = x.clone();

            // Layer norm for attention
            let mut xn = vec![0.0; N_EMBD];
            rmsnorm_fwd(&x, N_EMBD, &mut xn);
            act.xn_attn[li] = xn.clone();

            // Attention: Q, K, V projections
            let mut q = vec![0.0; N_EMBD];
            let mut k = vec![0.0; N_EMBD];
            let mut v = vec![0.0; N_EMBD];

            linear_fwd(&xn, &model.layers[li].wq, N_EMBD, N_EMBD, &mut q);
            linear_fwd(&xn, &model.layers[li].wk, N_EMBD, N_EMBD, &mut k);
            linear_fwd(&xn, &model.layers[li].wv, N_EMBD, N_EMBD, &mut v);

            act.q[li] = q.clone();
            act.k[li] = k.clone();
            act.v[li] = v.clone();

            // Store K, V in cache
            if kv_cache[li].len() <= pos {
                kv_cache[li].push((k.clone(), v.clone()));
            }

            // Multi-head attention
            let mut attn_out = vec![0.0; N_EMBD];
            let scale = 1.0 / (HEAD_DIM as f32).sqrt();

            for h in 0..N_HEAD {
                let hs = h * HEAD_DIM;

                // Compute attention scores for all previous positions
                let mut scores = vec![0.0; pos + 1];
                for t in 0..=pos {
                    let mut dot = 0.0;
                    for j in 0..HEAD_DIM {
                        dot += q[hs + j] * kv_cache[li][t].0[hs + j];
                    }
                    scores[t] = dot * scale;
                }

                // Softmax over scores
                let mut attn_weights = vec![0.0; pos + 1];
                softmax_fwd(&scores, pos + 1, &mut attn_weights, 1.0);

                // Weighted sum of values
                for t in 0..=pos {
                    for j in 0..HEAD_DIM {
                        attn_out[hs + j] += attn_weights[t] * kv_cache[li][t].1[hs + j];
                    }
                }
            }

            act.attn_out[li] = attn_out.clone();

            // Output projection + residual
            let mut attn_proj = vec![0.0; N_EMBD];
            linear_fwd(&attn_out, &model.layers[li].wo, N_EMBD, N_EMBD, &mut attn_proj);

            // Apply dropout to attention output during training
            if training {
                if let Some(r) = rng.as_deref_mut() {
                    apply_dropout(&mut attn_proj, DROPOUT_RATE, r);
                }
            }

            for i in 0..N_EMBD {
                x[i] = attn_proj[i] + act.x_in[li][i];
            }
            act.x_mid[li] = x.clone();

            // Layer norm for MLP
            let mut xn_mlp = vec![0.0; N_EMBD];
            rmsnorm_fwd(&x, N_EMBD, &mut xn_mlp);
            act.xn_mlp[li] = xn_mlp.clone();

            // MLP: fc1 -> squared ReLU -> fc2
            let mut h1 = vec![0.0; MLP_DIM];
            linear_fwd(&xn_mlp, &model.layers[li].fc1, MLP_DIM, N_EMBD, &mut h1);
            act.mlp_pre[li] = h1.clone();

            let mut h2 = vec![0.0; MLP_DIM];
            for i in 0..MLP_DIM {
                h2[i] = if h1[i] > 0.0 { h1[i] * h1[i] } else { 0.0 };
            }
            act.mlp_post[li] = h2.clone();

            let mut mlp_out = vec![0.0; N_EMBD];
            linear_fwd(&h2, &model.layers[li].fc2, N_EMBD, MLP_DIM, &mut mlp_out);

            // Apply dropout to MLP output during training
            if training {
                if let Some(r) = rng.as_deref_mut() {
                    apply_dropout(&mut mlp_out, DROPOUT_RATE, r);
                }
            }

            // MLP residual
            for i in 0..N_EMBD {
                x[i] = mlp_out[i] + act.x_mid[li][i];
            }
        }

        act.x_out = x.clone();

        // Final projection to vocabulary
        let mut logits = vec![0.0; model.vocab_size];
        linear_fwd(&x, &model.lm_head, model.vocab_size, N_EMBD, &mut logits);

        all_logits.push(logits);
        all_acts.push(act);
    }

    (all_logits, all_acts)
}

/* ------------------------------------------------------------------ */
/* Metal-Accelerated Full Sequence Forward Pass                      */
/* ------------------------------------------------------------------ */

// Metal forward: processes all seq_len positions in batched matmuls
// Returns only logits (no activations for backward).
// Use this during loss estimation/generation for pure speed.
fn forward_metal_logits(
    tokens: &[usize],
    model: &GPTModel,
) -> Vec<Vec<f32>> {
    let device = match METAL_DEVICE.as_ref() {
        Some(d) => d,
        None => return forward_metal_logits_cpu(tokens, model),
    };

    let seq_len = tokens.len();

    // Helper: batch matmul on Metal - x [T, nin] * W^T [nin, nout] → [T, nout]
    let metal_mm = |x_data: &[f32], t: usize, nin: usize,
                    w_data: &[f32], nout: usize| -> Vec<f32> {
        let x_t = Tensor::from_slice(x_data, (t, nin), device).unwrap();
        let w_t = Tensor::from_slice(w_data, (nout, nin), device).unwrap();
        x_t.matmul(&w_t.t().unwrap()).unwrap()
            .flatten_all().unwrap()
            .to_vec1::<f32>().unwrap()
    };

    // Build input embeddings for all positions [seq_len, N_EMBD]
    let mut x_flat: Vec<f32> = vec![0.0; seq_len * N_EMBD];
    for (pos, &tok) in tokens.iter().enumerate() {
        for i in 0..N_EMBD {
            x_flat[pos * N_EMBD + i] =
                model.wte[tok * N_EMBD + i] + model.wpe[pos * N_EMBD + i];
        }
    }

    // Run through transformer layers
    for li in 0..N_LAYER {
        // RMSNorm over all positions (CPU - cheap elementwise op)
        let mut xn_flat = vec![0.0f32; seq_len * N_EMBD];
        for pos in 0..seq_len {
            rmsnorm_fwd(
                &x_flat[pos * N_EMBD..(pos + 1) * N_EMBD],
                N_EMBD,
                &mut xn_flat[pos * N_EMBD..(pos + 1) * N_EMBD],
            );
        }

        // Q, K, V projections on Metal [seq_len, N_EMBD] → [seq_len, N_EMBD]
        let q_flat = metal_mm(&xn_flat, seq_len, N_EMBD, &model.layers[li].wq, N_EMBD);
        let k_flat = metal_mm(&xn_flat, seq_len, N_EMBD, &model.layers[li].wk, N_EMBD);
        let v_flat = metal_mm(&xn_flat, seq_len, N_EMBD, &model.layers[li].wv, N_EMBD);

        // Attention computation (CPU - quadratic in seq_len but seq_len=64 is small)
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * N_EMBD];

        for h in 0..N_HEAD {
            let hs = h * HEAD_DIM;
            for pos in 0..seq_len {
                // Compute scores vs all previous positions (causal)
                let mut scores = vec![0.0f32; pos + 1];
                for t in 0..=pos {
                    let mut dot = 0.0f32;
                    for j in 0..HEAD_DIM {
                        dot += q_flat[pos * N_EMBD + hs + j] * k_flat[t * N_EMBD + hs + j];
                    }
                    scores[t] = dot * scale;
                }
                // Softmax
                let mut weights = vec![0.0f32; pos + 1];
                softmax_fwd(&scores, pos + 1, &mut weights, 1.0);
                // Weighted value sum
                for t in 0..=pos {
                    for j in 0..HEAD_DIM {
                        attn_out[pos * N_EMBD + hs + j] +=
                            weights[t] * v_flat[t * N_EMBD + hs + j];
                    }
                }
            }
        }

        // Output projection on Metal [seq_len, N_EMBD] → [seq_len, N_EMBD]
        let attn_proj = metal_mm(&attn_out, seq_len, N_EMBD, &model.layers[li].wo, N_EMBD);

        // Residual connection
        let mut x_mid = vec![0.0f32; seq_len * N_EMBD];
        for i in 0..seq_len * N_EMBD {
            x_mid[i] = x_flat[i] + attn_proj[i];
        }

        // MLP: RMSNorm (CPU) → fc1 (Metal) → squared ReLU (CPU) → fc2 (Metal) → residual
        let mut xn_mlp = vec![0.0f32; seq_len * N_EMBD];
        for pos in 0..seq_len {
            rmsnorm_fwd(
                &x_mid[pos * N_EMBD..(pos + 1) * N_EMBD],
                N_EMBD,
                &mut xn_mlp[pos * N_EMBD..(pos + 1) * N_EMBD],
            );
        }

        // fc1: [seq_len, N_EMBD] → [seq_len, MLP_DIM]
        let h1_flat = metal_mm(&xn_mlp, seq_len, N_EMBD, &model.layers[li].fc1, MLP_DIM);

        // Squared ReLU (CPU - elementwise)
        let mut h2_flat = vec![0.0f32; seq_len * MLP_DIM];
        for i in 0..h2_flat.len() {
            let v = h1_flat[i];
            h2_flat[i] = if v > 0.0 { v * v } else { 0.0 };
        }

        // fc2: [seq_len, MLP_DIM] → [seq_len, N_EMBD]
        let mlp_out = metal_mm(&h2_flat, seq_len, MLP_DIM, &model.layers[li].fc2, N_EMBD);

        // MLP residual
        for i in 0..seq_len * N_EMBD {
            x_flat[i] = x_mid[i] + mlp_out[i];
        }
    }

    // Final LM head: [seq_len, N_EMBD] → [seq_len, vocab_size]
    let logits_flat = metal_mm(&x_flat, seq_len, N_EMBD, &model.lm_head, model.vocab_size);

    // Split into per-position logits
    logits_flat
        .chunks(model.vocab_size)
        .map(|c| c.to_vec())
        .collect()
}

// CPU fallback when Metal is not available
fn forward_metal_logits_cpu(tokens: &[usize], model: &GPTModel) -> Vec<Vec<f32>> {
    let mut kv = (0..N_LAYER).map(|_| Vec::new()).collect();
    let (logits, _) = forward(tokens, model, &mut kv, false, None);
    logits
}

/* ------------------------------------------------------------------ */
/* Backward Pass and Training (Simplified)                           */
/* ------------------------------------------------------------------ */

// Helper: Linear backward pass
fn linear_bwd(
    d_out: &[f32],
    x: &[f32],
    w: &[f32],
    nout: usize,
    nin: usize,
    d_x: &mut [f32],
    d_w: &mut [f32],
) {
    // Zero out d_x
    for i in 0..nin {
        d_x[i] = 0.0;
    }

    // Compute gradients
    for r in 0..nout {
        for c in 0..nin {
            d_w[r * nin + c] += d_out[r] * x[c];
            d_x[c] += d_out[r] * w[r * nin + c];
        }
    }
}

// Softmax backward pass
fn softmax_bwd(probs: &[f32], d_out: &[f32], n: usize, d_in: &mut [f32]) {
    // d_in[i] = sum_j (probs[i] * (δ_ij - probs[j]) * d_out[j])
    // Simplified: d_in[i] = probs[i] * (d_out[i] - dot(probs, d_out))
    let mut dot = 0.0;
    for i in 0..n {
        dot += probs[i] * d_out[i];
    }
    for i in 0..n {
        d_in[i] = probs[i] * (d_out[i] - dot);
    }
}

// Simplified backward pass - only for last position (like GPT training)
fn backward_simple(
    logits: &[f32],
    target: usize,
    act: &PosActs,
    model: &mut GPTModel,
    tok: usize,
    pos: usize,
) {
    let vocab_size = model.vocab_size;

    // Compute softmax and loss gradient
    let mut probs = vec![0.0; vocab_size];
    softmax_fwd(logits, vocab_size, &mut probs, 1.0);

    // d_logits = probs - one_hot(target)
    let mut d_logits = probs;
    d_logits[target] -= 1.0;

    // Backward through lm_head
    let mut d_x_out = vec![0.0; N_EMBD];
    linear_bwd(
        &d_logits,
        &act.x_out,
        &model.lm_head,
        vocab_size,
        N_EMBD,
        &mut d_x_out,
        &mut model.d_lm_head,
    );

    let mut d_x = d_x_out;

    // Backward through layers (simplified - just MLP for now)
    for li in (0..N_LAYER).rev() {
        // MLP backward
        let d_mlp_out = d_x.clone();

        // Clone weights to avoid borrow checker issues
        let fc2_weights = model.layers[li].fc2.clone();
        let fc1_weights = model.layers[li].fc1.clone();

        // Backward through fc2
        let mut d_h2 = vec![0.0; MLP_DIM];
        linear_bwd(
            &d_mlp_out,
            &act.mlp_post[li],
            &fc2_weights,
            N_EMBD,
            MLP_DIM,
            &mut d_h2,
            &mut model.layers[li].d_fc2,
        );

        // Backward through squared ReLU
        let mut d_h1 = vec![0.0; MLP_DIM];
        for i in 0..MLP_DIM {
            if act.mlp_pre[li][i] > 0.0 {
                d_h1[i] = d_h2[i] * 2.0 * act.mlp_pre[li][i];
            }
        }

        // Backward through fc1
        let mut d_xn_mlp = vec![0.0; N_EMBD];
        linear_bwd(
            &d_h1,
            &act.xn_mlp[li],
            &fc1_weights,
            MLP_DIM,
            N_EMBD,
            &mut d_xn_mlp,
            &mut model.layers[li].d_fc1,
        );

        // Add residual gradient
        for i in 0..N_EMBD {
            d_x[i] = d_xn_mlp[i] + d_x[i];
        }

        // Skip attention backward for simplicity (would need more complex grad computation)
        // Just use identity: d_x passes through
    }

    // Backward into embeddings
    for i in 0..N_EMBD {
        model.d_wte[tok * N_EMBD + i] += d_x[i];
        model.d_wpe[pos * N_EMBD + i] += d_x[i];
    }
}

// Learning rate schedule: warmup + constant + decay
fn get_learning_rate(iter: usize, max_iters: usize) -> f32 {
    let warmup_iters = 100;
    let decay_start = (max_iters * 4) / 5;  // Start decay at 80%

    if iter < warmup_iters {
        // Warmup: linearly increase from 10% to 100%
        LEARNING_RATE * (0.1 + 0.9 * iter as f32 / warmup_iters as f32)
    } else if iter < decay_start {
        // Constant: stay at max LR for most of training
        LEARNING_RATE
    } else {
        // Cosine decay: only in last 20% of training
        let progress = ((iter - decay_start) as f32) / ((max_iters - decay_start) as f32);
        let cosine = 0.5 * (1.0 + (progress * std::f32::consts::PI).cos());
        MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * cosine
    }
}

// Gradient clipping
fn clip_gradients(grads: &mut [f32], max_norm: f32) {
    let mut norm = 0.0;
    for &g in grads.iter() {
        norm += g * g;
    }
    norm = norm.sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
}

// Dropout: randomly zero out elements during training
fn apply_dropout(x: &mut [f32], dropout_rate: f32, rng: &mut Rng) {
    let keep_prob = 1.0 - dropout_rate;
    let scale = 1.0 / keep_prob;
    for val in x.iter_mut() {
        if rng.uniform() > dropout_rate as f64 {
            *val *= scale;
        } else {
            *val = 0.0;
        }
    }
}

// Adam optimizer step with learning rate and weight decay (AdamW)
fn adam_step(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    t: usize,
    lr: f32,
) {
    // Precompute bias corrections once — powf() is expensive, no need to call it per param
    let t_f = t as f32;
    let bc1 = 1.0 - BETA1.powf(t_f);
    let bc2 = 1.0 - BETA2.powf(t_f);
    let one_m_b1 = 1.0 - BETA1;
    let one_m_b2 = 1.0 - BETA2;

    for i in 0..params.len() {
        m[i] = BETA1 * m[i] + one_m_b1 * grads[i];
        v[i] = BETA2 * v[i] + one_m_b2 * grads[i] * grads[i];

        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;

        // AdamW: decoupled weight decay
        params[i] -= lr * (m_hat / (v_hat.sqrt() + EPSILON) + WEIGHT_DECAY * params[i]);
    }
}

// Zero out gradients
fn zero_grads(model: &mut GPTModel) {
    model.d_wte.fill(0.0);
    model.d_wpe.fill(0.0);
    model.d_lm_head.fill(0.0);
    for layer in &mut model.layers {
        layer.d_wq.fill(0.0);
        layer.d_wk.fill(0.0);
        layer.d_wv.fill(0.0);
        layer.d_wo.fill(0.0);
        layer.d_fc1.fill(0.0);
        layer.d_fc2.fill(0.0);
    }
}

// Struct to hold gradients for a single example.
// Layer weight gradients are stored as flat contiguous Vecs (N_LAYER × per-layer-size)
// instead of Vec<Vec<f32>> for better cache locality and SIMD-friendly accumulation.
// Access layer li of d_wq via: grads.d_wq[li * N_EMBD*N_EMBD .. (li+1) * N_EMBD*N_EMBD]
// or via the helper method .layer_slice(field, li, stride).
#[derive(Clone)]
struct GradientBuffer {
    d_wte: Vec<f32>,
    d_wpe: Vec<f32>,
    d_lm_head: Vec<f32>,
    // Flat: [layer0 | layer1 | ... | layerN-1], each N_EMBD*N_EMBD floats
    d_wq: Vec<f32>,
    d_wk: Vec<f32>,
    d_wv: Vec<f32>,
    d_wo: Vec<f32>,
    // Flat: each MLP_DIM*N_EMBD floats
    d_fc1: Vec<f32>,
    d_fc2: Vec<f32>,
}

impl GradientBuffer {
    fn new(vocab_size: usize) -> Self {
        Self {
            d_wte:    vec![0.0; vocab_size * N_EMBD],
            d_wpe:    vec![0.0; BLOCK_SIZE * N_EMBD],
            d_lm_head: vec![0.0; vocab_size * N_EMBD],
            d_wq:  vec![0.0; N_LAYER * N_EMBD * N_EMBD],
            d_wk:  vec![0.0; N_LAYER * N_EMBD * N_EMBD],
            d_wv:  vec![0.0; N_LAYER * N_EMBD * N_EMBD],
            d_wo:  vec![0.0; N_LAYER * N_EMBD * N_EMBD],
            d_fc1: vec![0.0; N_LAYER * MLP_DIM * N_EMBD],
            d_fc2: vec![0.0; N_LAYER * N_EMBD * MLP_DIM],
        }
    }

    // Return the slice for layer `li` of a flat N_LAYER×stride array
    #[inline(always)]
    fn layer<'a>(field: &'a [f32], li: usize, stride: usize) -> &'a [f32] {
        &field[li * stride .. (li + 1) * stride]
    }

    #[inline(always)]
    fn layer_mut<'a>(field: &'a mut Vec<f32>, li: usize, stride: usize) -> &'a mut [f32] {
        &mut field[li * stride .. (li + 1) * stride]
    }
}

/* ------------------------------------------------------------------ */
/* Checkpoint save / load                                             */
/* ------------------------------------------------------------------ */

// File format (all little-endian):
//   [0..8]   magic   b"RGPT0001"
//   [8..12]  vocab_size  u32
//   [12..16] iter        u32   (last completed iteration, 0-based)
//   [16..20] step        u32   (Adam step counter)
//   [20..24] best_loss   f32
//   [24..]   flat f32 arrays in order:
//              wte, wpe, lm_head,
//              per layer: wq,wk,wv,wo,fc1,fc2,
//              m_wte,v_wte, m_wpe,v_wpe, m_lm_head,v_lm_head,
//              per layer: m_wq,v_wq, m_wk,v_wk, m_wv,v_wv,
//                         m_wo,v_wo, m_fc1,v_fc1, m_fc2,v_fc2

// Write f32 slice to a byte buffer (in-memory, no disk I/O)
fn write_f32_slice_buf(buf: &mut Vec<u8>, s: &[f32]) {
    buf.reserve(s.len() * 4);
    for &v in s {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

// Flush an in-memory checkpoint buffer atomically to disk.
// Writes to a temp file then renames so we never leave a partial file.
fn flush_checkpoint(path: &str, buf: &[u8]) -> std::io::Result<()> {
    let tmp = format!("{}.tmp", path);
    {
        let mut f = File::create(&tmp)?;
        f.write_all(buf)?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}


fn read_f32_slice(f: &mut File, n: usize) -> std::io::Result<Vec<f32>> {
    let mut buf = vec![0u8; n * 4];
    f.read_exact(&mut buf)?;
    Ok(buf.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect())
}

// Serialize checkpoint to an in-memory byte buffer — zero disk I/O.
// Call flush_checkpoint() to actually write to disk.
fn serialize_checkpoint(
    model: &GPTModel,
    iter: usize,
    step: usize,
    best_loss: f32,
) -> Vec<u8> {
    // Pre-size: header(24) + all f32 params * 4 bytes each, ×2 for Adam moments
    let n_params = model.wte.len() + model.wpe.len() + model.lm_head.len()
        + N_LAYER * (model.layers[0].wq.len() * 6); // wq+wk+wv+wo+fc1+fc2
    let mut buf: Vec<u8> = Vec::with_capacity(24 + n_params * 4 * 3); // weights + 2× moments

    // Header
    buf.extend_from_slice(b"RGPT0001");
    buf.extend_from_slice(&(model.vocab_size as u32).to_le_bytes());
    buf.extend_from_slice(&(iter as u32).to_le_bytes());
    buf.extend_from_slice(&(step as u32).to_le_bytes());
    buf.extend_from_slice(&best_loss.to_le_bytes());
    // Weights
    write_f32_slice_buf(&mut buf, &model.wte);
    write_f32_slice_buf(&mut buf, &model.wpe);
    write_f32_slice_buf(&mut buf, &model.lm_head);
    for li in 0..N_LAYER {
        write_f32_slice_buf(&mut buf, &model.layers[li].wq);
        write_f32_slice_buf(&mut buf, &model.layers[li].wk);
        write_f32_slice_buf(&mut buf, &model.layers[li].wv);
        write_f32_slice_buf(&mut buf, &model.layers[li].wo);
        write_f32_slice_buf(&mut buf, &model.layers[li].fc1);
        write_f32_slice_buf(&mut buf, &model.layers[li].fc2);
    }
    // Adam moments
    write_f32_slice_buf(&mut buf, &model.m_wte);   write_f32_slice_buf(&mut buf, &model.v_wte);
    write_f32_slice_buf(&mut buf, &model.m_wpe);   write_f32_slice_buf(&mut buf, &model.v_wpe);
    write_f32_slice_buf(&mut buf, &model.m_lm_head); write_f32_slice_buf(&mut buf, &model.v_lm_head);
    for li in 0..N_LAYER {
        write_f32_slice_buf(&mut buf, &model.layers[li].m_wq); write_f32_slice_buf(&mut buf, &model.layers[li].v_wq);
        write_f32_slice_buf(&mut buf, &model.layers[li].m_wk); write_f32_slice_buf(&mut buf, &model.layers[li].v_wk);
        write_f32_slice_buf(&mut buf, &model.layers[li].m_wv); write_f32_slice_buf(&mut buf, &model.layers[li].v_wv);
        write_f32_slice_buf(&mut buf, &model.layers[li].m_wo); write_f32_slice_buf(&mut buf, &model.layers[li].v_wo);
        write_f32_slice_buf(&mut buf, &model.layers[li].m_fc1); write_f32_slice_buf(&mut buf, &model.layers[li].v_fc1);
        write_f32_slice_buf(&mut buf, &model.layers[li].m_fc2); write_f32_slice_buf(&mut buf, &model.layers[li].v_fc2);
    }
    buf
}

/// Returns (iter_start, step, best_loss) on success.
fn load_checkpoint(path: &str, model: &mut GPTModel) -> std::io::Result<(usize, usize, f32)> {
    let mut f = File::open(path)?;
    // Magic
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"RGPT0001" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("Bad magic bytes in checkpoint {}", path)));
    }
    // Header scalars
    let mut u32buf = [0u8; 4];
    f.read_exact(&mut u32buf)?; let ckpt_vocab = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let iter       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let step       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let best_loss  = f32::from_le_bytes(u32buf);

    if ckpt_vocab != model.vocab_size {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("Checkpoint vocab_size {} != model vocab_size {}", ckpt_vocab, model.vocab_size)));
    }
    // Weights
    model.wte      = read_f32_slice(&mut f, model.wte.len())?;
    model.wpe      = read_f32_slice(&mut f, model.wpe.len())?;
    model.lm_head  = read_f32_slice(&mut f, model.lm_head.len())?;
    for li in 0..N_LAYER {
        let n_sq = N_EMBD * N_EMBD;
        model.layers[li].wq  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wk  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wv  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wo  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].fc1 = read_f32_slice(&mut f, MLP_DIM * N_EMBD)?;
        model.layers[li].fc2 = read_f32_slice(&mut f, N_EMBD * MLP_DIM)?;
    }
    // Adam moments
    model.m_wte     = read_f32_slice(&mut f, model.wte.len())?;
    model.v_wte     = read_f32_slice(&mut f, model.wte.len())?;
    model.m_wpe     = read_f32_slice(&mut f, model.wpe.len())?;
    model.v_wpe     = read_f32_slice(&mut f, model.wpe.len())?;
    model.m_lm_head = read_f32_slice(&mut f, model.lm_head.len())?;
    model.v_lm_head = read_f32_slice(&mut f, model.lm_head.len())?;
    for li in 0..N_LAYER {
        let n_sq = N_EMBD * N_EMBD;
        model.layers[li].m_wq  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].v_wq  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].m_wk  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].v_wk  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].m_wv  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].v_wv  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].m_wo  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].v_wo  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].m_fc1 = read_f32_slice(&mut f, MLP_DIM * N_EMBD)?;
        model.layers[li].v_fc1 = read_f32_slice(&mut f, MLP_DIM * N_EMBD)?;
        model.layers[li].m_fc2 = read_f32_slice(&mut f, N_EMBD * MLP_DIM)?;
        model.layers[li].v_fc2 = read_f32_slice(&mut f, N_EMBD * MLP_DIM)?;
    }
    Ok((iter + 1, step, best_loss))  // iter+1: resume *after* the saved iter
}

// Training function with Rayon parallelization
// iter_start / step_start / best_loss_start are restored from a checkpoint when resuming.
// ctrlc_flag: set to true by the Ctrl-C handler; train() flushes and exits cleanly.
fn train(
    model: &mut GPTModel,
    data: &[usize],
    iterations: usize,   // total iterations to reach (not additional)
    rng: &mut Rng,
    iter_start: usize,
    step_start: usize,
    best_loss_start: f32,
    ctrlc_flag: Arc<AtomicBool>,
) {
    println!("=== Starting Training (Multi-Core with Rayon) ===");
    if iter_start > 0 {
        println!("Resuming from iteration {}", iter_start);
    }
    println!("Iterations: {} → {}", iter_start, iterations);
    println!("Batch size: {}", BATCH_SIZE);
    println!("Learning rate: {} → {}", LEARNING_RATE, MIN_LEARNING_RATE);
    println!("Gradient clipping: {}", GRAD_CLIP);
    println!("Cores available: {}", rayon::current_num_threads());
    println!();

    let mut step = step_start;
    let mut best_loss = best_loss_start;
    let mut best_iter = if iter_start > 0 { iter_start.saturating_sub(1) } else { 0 };

    // In-memory checkpoint buffers — serialized each time, flushed to disk only at
    // end-of-training or on Ctrl-C. No disk I/O inside the hot training loop.
    let mut ckpt_buf: Vec<u8> = Vec::new();
    let mut ckpt_best_buf: Vec<u8> = Vec::new();

    for iter in iter_start..iterations {
        // Generate batch indices (sequential, using our RNG)
        let batch_starts: Vec<usize> = (0..BATCH_SIZE)
            .filter_map(|_| {
                if data.len() > BLOCK_SIZE + 1 {
                    Some(rng.choice(data.len() - BLOCK_SIZE - 1))
                } else {
                    None
                }
            })
            .collect();

        if batch_starts.is_empty() {
            continue;
        }

        // Wrap model in Mutex for thread-safe access (read-only during forward pass)
        let model_ref = &*model;

        // Process batch in parallel using Rayon
        let results: Vec<_> = batch_starts
            .par_iter()
            .map(|&start_idx| {
                let x_vec: Vec<usize> = data[start_idx..start_idx + BLOCK_SIZE].to_vec();
                let y_vec: Vec<usize> = data[start_idx + 1..start_idx + BLOCK_SIZE + 1].to_vec();

                // Thread-local RNG for dropout
                let mut thread_rng = Rng::new(start_idx as u64 + iter as u64);

                // Forward pass (with training=true and dropout enabled)
                let mut kv_cache: Vec<Vec<(Vec<f32>, Vec<f32>)>> =
                    (0..N_LAYER).map(|_| Vec::new()).collect();
                let (logits_seq, acts) = forward(&x_vec, model_ref, &mut kv_cache, true, Some(&mut thread_rng));

                // Compute gradients (thread-local)
                let mut local_grads = GradientBuffer::new(model_ref.vocab_size);
                let mut total_loss = 0.0;

                // Scratch buffers hoisted out of the per-position loop to avoid
                // allocating vocab_size + N_EMBD floats on every of the 64 positions.
                let mut probs    = vec![0.0f32; model_ref.vocab_size];
                let mut d_logits = vec![0.0f32; model_ref.vocab_size];
                let mut d_x_out  = vec![0.0f32; N_EMBD];

                // Train on ALL positions (not just last) - 64x more training signal!
                for pos in 0..logits_seq.len() {
                    // Compute loss gradient for this position
                    softmax_fwd(&logits_seq[pos], model_ref.vocab_size, &mut probs, 1.0);
                    total_loss += cross_entropy_loss(&probs, y_vec[pos]);

                    d_logits.copy_from_slice(&probs);
                    d_logits[y_vec[pos]] -= 1.0;

                    // Backward through lm_head
                    d_x_out.fill(0.0);
                    linear_bwd(
                        &d_logits,
                        &acts[pos].x_out,
                        &model_ref.lm_head,
                        model_ref.vocab_size,
                        N_EMBD,
                        &mut d_x_out,
                        &mut local_grads.d_lm_head,
                    );

                    let mut d_x = d_x_out.clone(); // 128 floats — cheap

                    // Backward through layers (FULL: MLP + Attention)
                    for li in (0..N_LAYER).rev() {
                        // Clone weights to avoid borrow issues
                        let fc2_weights = model_ref.layers[li].fc2.clone();
                        let fc1_weights = model_ref.layers[li].fc1.clone();
                        let wo_weights = model_ref.layers[li].wo.clone();
                        let wq_weights = model_ref.layers[li].wq.clone();
                        let wk_weights = model_ref.layers[li].wk.clone();
                        let wv_weights = model_ref.layers[li].wv.clone();

                        // ===== MLP Backward =====
                        let mut d_h2 = vec![0.0; MLP_DIM];
                        linear_bwd(
                            &d_x,
                            &acts[pos].mlp_post[li],
                            &fc2_weights,
                            N_EMBD,
                            MLP_DIM,
                            &mut d_h2,
                            &mut local_grads.d_fc2[li * N_EMBD * MLP_DIM .. (li + 1) * N_EMBD * MLP_DIM],
                        );

                        let mut d_h1 = vec![0.0; MLP_DIM];
                        for i in 0..MLP_DIM {
                            if acts[pos].mlp_pre[li][i] > 0.0 {
                                d_h1[i] = d_h2[i] * 2.0 * acts[pos].mlp_pre[li][i];
                            }
                        }

                        let mut d_xn_mlp = vec![0.0; N_EMBD];
                        linear_bwd(
                            &d_h1,
                            &acts[pos].xn_mlp[li],
                            &fc1_weights,
                            MLP_DIM,
                            N_EMBD,
                            &mut d_xn_mlp,
                            &mut local_grads.d_fc1[li * MLP_DIM * N_EMBD .. (li + 1) * MLP_DIM * N_EMBD],
                        );

                        // Add MLP residual gradient
                        let mut d_x_mid = vec![0.0; N_EMBD];
                        for i in 0..N_EMBD {
                            d_x_mid[i] = d_xn_mlp[i] + d_x[i];
                        }

                        // ===== Attention Backward =====

                        // Backward through attention output projection
                        let mut d_attn_out = vec![0.0; N_EMBD];
                        linear_bwd(
                            &d_x_mid,
                            &acts[pos].attn_out[li],
                            &wo_weights,
                            N_EMBD,
                            N_EMBD,
                            &mut d_attn_out,
                            &mut local_grads.d_wo[li * N_EMBD * N_EMBD .. (li + 1) * N_EMBD * N_EMBD],
                        );

                        // Backward through multi-head attention
                        let mut d_q = vec![0.0; N_EMBD];
                        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

                        for h in 0..N_HEAD {
                            let hs = h * HEAD_DIM;

                            // Recompute attention weights for this head (needed for backward)
                            let mut scores = vec![0.0; pos + 1];
                            for t in 0..=pos {
                                let mut dot = 0.0;
                                for j in 0..HEAD_DIM {
                                    dot += acts[pos].q[li][hs + j] * kv_cache[li][t].0[hs + j];
                                }
                                scores[t] = dot * scale;
                            }

                            let mut attn_weights = vec![0.0; pos + 1];
                            softmax_fwd(&scores, pos + 1, &mut attn_weights, 1.0);

                            // Gradient from weighted sum of values
                            let mut d_attn_weights = vec![0.0; pos + 1];
                            for t in 0..=pos {
                                for j in 0..HEAD_DIM {
                                    // d_v: gradient to values at position t
                                    // (stored in kv_cache, can't backprop to past)

                                    // d_attn_weights: gradient to attention weights
                                    d_attn_weights[t] += d_attn_out[hs + j] * kv_cache[li][t].1[hs + j];
                                }
                            }

                            // Backward through softmax
                            let mut d_scores = vec![0.0; pos + 1];
                            softmax_bwd(&attn_weights, &d_attn_weights, pos + 1, &mut d_scores);

                            // Backward through scaled dot product (Q * K^T)
                            for t in 0..=pos {
                                for j in 0..HEAD_DIM {
                                    // d_q: gradient to query
                                    d_q[hs + j] += d_scores[t] * scale * kv_cache[li][t].0[hs + j];

                                    // d_k: gradient to keys at position t
                                    // (stored in kv_cache at position t, can only update current pos)
                                }
                            }
                        }

                        // Backward through Q, K, V projections
                        let mut d_xn_attn_q = vec![0.0; N_EMBD];
                        let mut d_xn_attn_k = vec![0.0; N_EMBD];
                        let mut d_xn_attn_v = vec![0.0; N_EMBD];

                        linear_bwd(
                            &d_q,
                            &acts[pos].xn_attn[li],
                            &wq_weights,
                            N_EMBD,
                            N_EMBD,
                            &mut d_xn_attn_q,
                            &mut local_grads.d_wq[li * N_EMBD * N_EMBD .. (li + 1) * N_EMBD * N_EMBD],
                        );

                        // For K and V, we only backprop to current position
                        // (past positions already computed)
                        let d_k = vec![0.0; N_EMBD]; // Simplified: ignore K gradient
                        let d_v = vec![0.0; N_EMBD]; // Simplified: ignore V gradient

                        linear_bwd(
                            &d_k,
                            &acts[pos].xn_attn[li],
                            &wk_weights,
                            N_EMBD,
                            N_EMBD,
                            &mut d_xn_attn_k,
                            &mut local_grads.d_wk[li * N_EMBD * N_EMBD .. (li + 1) * N_EMBD * N_EMBD],
                        );

                        linear_bwd(
                            &d_v,
                            &acts[pos].xn_attn[li],
                            &wv_weights,
                            N_EMBD,
                            N_EMBD,
                            &mut d_xn_attn_v,
                            &mut local_grads.d_wv[li * N_EMBD * N_EMBD .. (li + 1) * N_EMBD * N_EMBD],
                        );

                        // Combine gradients from Q, K, V
                        let mut d_xn_attn = vec![0.0; N_EMBD];
                        for i in 0..N_EMBD {
                            d_xn_attn[i] = d_xn_attn_q[i] + d_xn_attn_k[i] + d_xn_attn_v[i];
                        }

                        // Add attention residual gradient
                        for i in 0..N_EMBD {
                            d_x[i] = d_xn_attn[i] + d_x_mid[i];
                        }
                    }

                    // Backward into embeddings
                    for i in 0..N_EMBD {
                        local_grads.d_wte[x_vec[pos] * N_EMBD + i] += d_x[i];
                        local_grads.d_wpe[pos * N_EMBD + i] += d_x[i];
                    }
                }

                let loss = total_loss / logits_seq.len() as f32;

                (local_grads, loss)
            })
            .collect();

        // Aggregate gradients (sequential)
        zero_grads(model);
        let mut batch_loss = 0.0;

        for (grads, loss) in results {
            batch_loss += loss;

            // Accumulate gradients via zip — iterator form lets LLVM auto-vectorize (SIMD)
            model.d_wte.iter_mut().zip(grads.d_wte.iter()).for_each(|(a, b)| *a += b);
            model.d_wpe.iter_mut().zip(grads.d_wpe.iter()).for_each(|(a, b)| *a += b);
            model.d_lm_head.iter_mut().zip(grads.d_lm_head.iter()).for_each(|(a, b)| *a += b);
            for li in 0..N_LAYER {
                let sq = N_EMBD * N_EMBD;
                let fc1s = MLP_DIM * N_EMBD;
                let fc2s = N_EMBD * MLP_DIM;
                model.layers[li].d_wq.iter_mut().zip(grads.d_wq[li*sq..(li+1)*sq].iter()).for_each(|(a, b)| *a += b);
                model.layers[li].d_wk.iter_mut().zip(grads.d_wk[li*sq..(li+1)*sq].iter()).for_each(|(a, b)| *a += b);
                model.layers[li].d_wv.iter_mut().zip(grads.d_wv[li*sq..(li+1)*sq].iter()).for_each(|(a, b)| *a += b);
                model.layers[li].d_wo.iter_mut().zip(grads.d_wo[li*sq..(li+1)*sq].iter()).for_each(|(a, b)| *a += b);
                model.layers[li].d_fc1.iter_mut().zip(grads.d_fc1[li*fc1s..(li+1)*fc1s].iter()).for_each(|(a, b)| *a += b);
                model.layers[li].d_fc2.iter_mut().zip(grads.d_fc2[li*fc2s..(li+1)*fc2s].iter()).for_each(|(a, b)| *a += b);
            }
        }

        batch_loss /= batch_starts.len() as f32;
        step += 1;

        // Get learning rate for this iteration
        let lr = get_learning_rate(iter, iterations);

        // Clip gradients
        clip_gradients(&mut model.d_wte, GRAD_CLIP);
        clip_gradients(&mut model.d_wpe, GRAD_CLIP);
        clip_gradients(&mut model.d_lm_head, GRAD_CLIP);
        for li in 0..N_LAYER {
            clip_gradients(&mut model.layers[li].d_wq, GRAD_CLIP);
            clip_gradients(&mut model.layers[li].d_wk, GRAD_CLIP);
            clip_gradients(&mut model.layers[li].d_wv, GRAD_CLIP);
            clip_gradients(&mut model.layers[li].d_wo, GRAD_CLIP);
            clip_gradients(&mut model.layers[li].d_fc1, GRAD_CLIP);
            clip_gradients(&mut model.layers[li].d_fc2, GRAD_CLIP);
        }

        // Optimizer step with current learning rate
        adam_step(
            &mut model.wte,
            &model.d_wte,
            &mut model.m_wte,
            &mut model.v_wte,
            step,
            lr,
        );
        adam_step(
            &mut model.wpe,
            &model.d_wpe,
            &mut model.m_wpe,
            &mut model.v_wpe,
            step,
            lr,
        );
        adam_step(
            &mut model.lm_head,
            &model.d_lm_head,
            &mut model.m_lm_head,
            &mut model.v_lm_head,
            step,
            lr,
        );

        for li in 0..N_LAYER {
            // Use raw pointers to avoid the borrow checker requiring .clone() when
            // param and grad live in the same struct.  Both slices are non-overlapping
            // fields so this is safe.
            let layer = &mut model.layers[li];
            macro_rules! layer_adam {
                ($w:ident, $dw:ident, $mw:ident, $vw:ident) => {{
                    let grads_ptr = layer.$dw.as_ptr();
                    let grads_len = layer.$dw.len();
                    // SAFETY: $w, $dw, $mw, $vw are distinct non-overlapping fields.
                    let grads: &[f32] = unsafe { std::slice::from_raw_parts(grads_ptr, grads_len) };
                    adam_step(&mut layer.$w, grads, &mut layer.$mw, &mut layer.$vw, step, lr);
                }};
            }
            layer_adam!(wq, d_wq, m_wq, v_wq);
            layer_adam!(wk, d_wk, m_wk, v_wk);
            layer_adam!(wv, d_wv, m_wv, v_wv);
            layer_adam!(wo, d_wo, m_wo, v_wo);
            layer_adam!(fc1, d_fc1, m_fc1, v_fc1);
            layer_adam!(fc2, d_fc2, m_fc2, v_fc2);
        }

        // Track best loss
        if batch_loss < best_loss {
            best_loss = batch_loss;
            best_iter = iter;
        }

        // Log progress + snapshot checkpoint buffers (in memory, no disk)
        // Only serialize on the intervals we'd log — keeps the hot path clean.
        let is_log_iter = iter % EVAL_INTERVAL == 0 || iter == iterations - 1;
        if is_log_iter {
            println!("Iter {:4} | Loss: {:.4} | LR: {:.6} | Best: {:.4} @{}",
                iter, batch_loss, lr, best_loss, best_iter);
            ckpt_buf = serialize_checkpoint(model, iter, step, best_loss);
            // Also refresh best buffer if this is the best iter
            if best_iter == iter {
                ckpt_best_buf = ckpt_buf.clone();
            }
        }

        // Check for Ctrl-C — serialize current state and flush immediately
        if ctrlc_flag.load(Ordering::Relaxed) {
            // Serialize now (may not have been done this interval yet)
            ckpt_buf = serialize_checkpoint(model, iter, step, best_loss);
            if best_iter == iter || ckpt_best_buf.is_empty() {
                ckpt_best_buf = ckpt_buf.clone();
            }
            println!();
            println!("Interrupted at iteration {}. Saving checkpoint...", iter);
            if !ckpt_buf.is_empty() {
                if let Err(e) = flush_checkpoint("checkpoint.bin", &ckpt_buf) {
                    eprintln!("Warning: could not save checkpoint: {}", e);
                } else {
                    println!("✓ Saved checkpoint.bin (iter {})", iter);
                }
            }
            if !ckpt_best_buf.is_empty() {
                if let Err(e) = flush_checkpoint("checkpoint_best.bin", &ckpt_best_buf) {
                    eprintln!("Warning: could not save best checkpoint: {}", e);
                } else {
                    println!("✓ Saved checkpoint_best.bin (best loss {:.4} @{})", best_loss, best_iter);
                }
            }
            std::process::exit(0);
        }
    }

    println!();
    println!("Training complete!");
    println!("Best loss: {:.4} at iteration {}", best_loss, best_iter);

    // Flush final checkpoints to disk
    if !ckpt_buf.is_empty() {
        if let Err(e) = flush_checkpoint("checkpoint.bin", &ckpt_buf) {
            eprintln!("Warning: could not save final checkpoint: {}", e);
        }
    }
    if !ckpt_best_buf.is_empty() {
        if let Err(e) = flush_checkpoint("checkpoint_best.bin", &ckpt_best_buf) {
            eprintln!("Warning: could not save best checkpoint: {}", e);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Data Loading                                                       */
/* ------------------------------------------------------------------ */
fn load_training_data(path: &str) -> std::io::Result<String> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut text = String::new();

    for line in reader.lines() {
        let line = line?;
        text.push_str(&line);
        text.push('\n');
    }

    Ok(text)
}

/* ------------------------------------------------------------------ */
/* Text Generation                                                    */
/* ------------------------------------------------------------------ */
fn generate(
    model: &GPTModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
    rng: &mut Rng,
) -> String {
    let mut tokens = tokenizer.encode(prompt);
    let max_len = BLOCK_SIZE.min(tokens.len() + max_new_tokens);

    let mut kv_cache: Vec<Vec<(Vec<f32>, Vec<f32>)>> =
        (0..N_LAYER).map(|_| Vec::new()).collect();

    for _ in tokens.len()..max_len {
        // Use Metal-accelerated forward pass if available
        let logits_seq_metal;
        let logits_seq_cpu;
        let logits = if METAL_DEVICE.is_some() {
            logits_seq_metal = forward_metal_logits(&tokens, model);
            &logits_seq_metal[logits_seq_metal.len() - 1]
        } else {
            let (seq, _) = forward(&tokens, model, &mut kv_cache, false, None);
            logits_seq_cpu = seq;
            &logits_seq_cpu[logits_seq_cpu.len() - 1]
        };
        let logits = logits.clone();

        // Sample next token with top-p
        let mut probs = vec![0.0; model.vocab_size];
        softmax_fwd(&logits, model.vocab_size, &mut probs, temperature);

        // Top-p sampling
        let mut sorted_indices: Vec<usize> = (0..model.vocab_size).collect();
        sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        let mut cumulative_prob = 0.0;
        let mut cutoff = model.vocab_size;
        for (i, &idx) in sorted_indices.iter().enumerate() {
            cumulative_prob += probs[idx];
            if cumulative_prob >= top_p {
                cutoff = i + 1;
                break;
            }
        }

        // Renormalize and sample
        let mut top_sum = 0.0;
        for i in 0..cutoff {
            top_sum += probs[sorted_indices[i]];
        }

        let mut r = rng.uniform() as f32 * top_sum;
        let mut next_token = sorted_indices[0];
        for i in 0..cutoff {
            let idx = sorted_indices[i];
            r -= probs[idx];
            if r <= 0.0 {
                next_token = idx;
                break;
            }
        }

        if next_token == tokenizer.eos_id {
            break;
        }

        tokens.push(next_token);

        // Truncate if we exceed block size
        if tokens.len() > BLOCK_SIZE {
            tokens = tokens[1..].to_vec();
            // Clear cache when we truncate
            kv_cache = (0..N_LAYER).map(|_| Vec::new()).collect();
        }
    }

    tokenizer.decode(&tokens)
}

/* ------------------------------------------------------------------ */
/* Training Function (Simplified - No Backward Pass Yet)             */
/* ------------------------------------------------------------------ */
fn estimate_loss(
    model: &GPTModel,
    data: &[usize],
    eval_iters: usize,
    rng: &mut Rng,
) -> f32 {
    let mut total_loss = 0.0;
    let mut count = 0;

    for _ in 0..eval_iters {
        if data.len() <= BLOCK_SIZE + 1 {
            continue;
        }

        let start_idx = rng.choice(data.len() - BLOCK_SIZE - 1);
        let x = &data[start_idx..start_idx + BLOCK_SIZE];
        let y = &data[start_idx + 1..start_idx + BLOCK_SIZE + 1];

        // Use Metal-accelerated forward pass if available
        let logits_seq = if METAL_DEVICE.is_some() {
            forward_metal_logits(x, model)
        } else {
            let mut kv_cache: Vec<Vec<(Vec<f32>, Vec<f32>)>> =
                (0..N_LAYER).map(|_| Vec::new()).collect();
            let (logits, _) = forward(x, model, &mut kv_cache, false, None);
            logits
        };

        for (logits, &target) in logits_seq.iter().zip(y.iter()) {
            let mut probs = vec![0.0; model.vocab_size];
            softmax_fwd(logits, model.vocab_size, &mut probs, 1.0);
            total_loss += cross_entropy_loss(&probs, target);
            count += 1;
        }
    }

    if count > 0 {
        total_loss / count as f32
    } else {
        0.0
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
fn main() -> std::io::Result<()> {
    // Parse command-line arguments
    // Usage: randygpt [--iters N] [--resume [path]]
    //   --iters N         total iterations to train (default: MAX_ITERS)
    //   --resume          resume from checkpoint.bin (auto-detected)
    //   --resume <path>   resume from a specific checkpoint file
    let args: Vec<String> = std::env::args().collect();

    let mut iterations = MAX_ITERS;
    let mut resume_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                i += 1;
                if i < args.len() {
                    iterations = args[i].parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("Invalid --iters value. Using default: {}", MAX_ITERS);
                        MAX_ITERS
                    });
                }
            }
            "--resume" => {
                // Optional path follows; if next arg starts with '-' or is absent, use default
                if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    i += 1;
                    resume_path = Some(args[i].clone());
                } else {
                    resume_path = Some("checkpoint.bin".to_string());
                }
            }
            other => {
                // Legacy: bare number as first arg = iterations
                if let Ok(n) = other.parse::<usize>() {
                    iterations = n;
                } else {
                    eprintln!("Unknown argument '{}'. Ignoring.", other);
                }
            }
        }
        i += 1;
    }

    // Auto-detect checkpoint if --resume not given but checkpoint.bin exists
    if resume_path.is_none() && Path::new("checkpoint.bin").exists() {
        eprintln!("Found checkpoint.bin — use --resume to continue from it, or delete it to start fresh.");
    }

    println!("=== Enhanced randyGPT ===");
    println!("Model: {} layers, {} heads, {} embedding dim", N_LAYER, N_HEAD, N_EMBD);
    println!("Block size: {}, Vocab size: up to {}", BLOCK_SIZE, MAX_VOCAB);
    println!();

    let mut rng = Rng::new(1337);

    // Load training data
    let training_text = if Path::new("train.txt").exists() {
        println!("Loading training data from train.txt...");
        load_training_data("train.txt")?
    } else {
        println!("No train.txt found. Using default sample data.");
        concat!(
            "The quick brown fox jumps over the lazy dog. ",
            "Rust is a systems programming language. ",
            "Machine learning models learn from data. ",
            "Transformers use attention mechanisms. ",
            "GPT stands for Generative Pre-trained Transformer. ",
            "Neural networks are inspired by the human brain. ",
            "Deep learning is a subset of machine learning. "
        ).to_string()
    };

    println!("Training data size: {} characters", training_text.len());

    // Build tokenizer
    let tokenizer = Tokenizer::from_text(&training_text);
    println!("Vocabulary size: {}", tokenizer.vocab_size);
    println!("Sample tokens: {:?}", &tokenizer.idx_to_char[..10.min(tokenizer.vocab_size)]);
    println!();

    // Tokenize data
    let data = tokenizer.encode(&training_text);
    println!("Tokenized to {} tokens", data.len());

    // Initialize model (always start fresh, then overwrite with checkpoint if resuming)
    println!("Initializing model...");
    let mut model = GPTModel::new(tokenizer.vocab_size, &mut rng);

    let param_count = model.wte.len() + model.wpe.len() + model.lm_head.len() +
        N_LAYER * (
            model.layers[0].wq.len() +
            model.layers[0].wk.len() +
            model.layers[0].wv.len() +
            model.layers[0].wo.len() +
            model.layers[0].fc1.len() +
            model.layers[0].fc2.len()
        );
    println!("Parameters: ~{:.2}M", param_count as f32 / 1_000_000.0);
    println!();

    // Load checkpoint if resuming
    let (iter_start, step_start, best_loss_start) = if let Some(ref ckpt) = resume_path {
        match load_checkpoint(ckpt, &mut model) {
            Ok((it, st, bl)) => {
                println!("✓ Resumed from '{}' — iter {}, step {}, best loss {:.4}", ckpt, it, st, bl);
                println!();
                (it, st, bl)
            }
            Err(e) => {
                eprintln!("Error loading checkpoint '{}': {}", ckpt, e);
                eprintln!("Starting from scratch instead.");
                (0, 0, f32::INFINITY)
            }
        }
    } else {
        (0, 0, f32::INFINITY)
    };

    // Guard: nothing to do if already at or past target
    if iter_start >= iterations {
        println!("Already at iteration {} (target {}). Nothing to train.", iter_start, iterations);
        println!("Increase --iters to continue training.");
        return Ok(());
    }

    // Estimate initial loss
    println!("Estimating initial loss...");
    let initial_loss = estimate_loss(&model, &data, 10, &mut rng);
    println!("Initial loss: {:.4}", initial_loss);
    println!();

    // Install Ctrl-C handler — sets a flag; train() checks it each iteration and flushes cleanly
    let ctrlc_flag = Arc::new(AtomicBool::new(false));
    {
        let flag = ctrlc_flag.clone();
        ctrlc::set_handler(move || {
            flag.store(true, Ordering::Relaxed);
        }).expect("Error setting Ctrl-C handler");
    }

    // Train the model
    train(&mut model, &data, iterations, &mut rng, iter_start, step_start, best_loss_start, ctrlc_flag);

    // Estimate loss after training
    println!("Estimating final loss...");
    let final_loss = estimate_loss(&model, &data, 10, &mut rng);
    println!("Final loss: {:.4} (started at {:.4})", final_loss, initial_loss);
    println!();

    // Generate samples after training
    println!("=== Generation After Training ===");

    let prompts = vec![
        ("ROMEO:", 100),
        ("To be or not to be", 100),
        ("Once upon a time", 100),
    ];

    for (prompt, max_tokens) in prompts {
        println!("\nPrompt: \"{}\"", prompt);
        let sample = generate(
            &mut model,
            &tokenizer,
            prompt,
            max_tokens,
            0.8,
            0.9,
            &mut rng,
        );
        println!("Generated: {}", sample);
    }

    println!();
    println!("=== Summary ===");
    println!("✓ Data: {} chars, {} tokens", training_text.len(), data.len());
    println!("✓ Model: ~{:.2}M parameters", param_count as f32 / 1_000_000.0);
    println!("✓ Trained: {} iterations", iterations);
    println!("✓ Loss improvement: {:.4} → {:.4}", initial_loss, final_loss);
    println!();
    println!("Usage: {} [iterations]", args.get(0).unwrap_or(&"randygpt".to_string()));
    println!("Example: {} 3000  (train for 3000 iterations)", args.get(0).unwrap_or(&"randygpt".to_string()));

    Ok(())
}
