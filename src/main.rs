use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use rayon::prelude::*;

/* ------------------------------------------------------------------ */
/* Enhanced Model hyper-parameters                                    */
/* ------------------------------------------------------------------ */
const N_EMBD: usize = 256;        // Scaled up from 128 (3M params)
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
    for r in 0..nout {
        let mut s = 0.0;
        let wr = &w[r * nin..(r + 1) * nin];
        for c in 0..nin {
            s += wr[c] * x[c];
        }
        out[r] = s;
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
    let t_f = t as f32;
    for i in 0..params.len() {
        m[i] = BETA1 * m[i] + (1.0 - BETA1) * grads[i];
        v[i] = BETA2 * v[i] + (1.0 - BETA2) * grads[i] * grads[i];

        let m_hat = m[i] / (1.0 - BETA1.powf(t_f));
        let v_hat = v[i] / (1.0 - BETA2.powf(t_f));

        // AdamW: decoupled weight decay
        params[i] -= lr * (m_hat / (v_hat.sqrt() + EPSILON) + WEIGHT_DECAY * params[i]);
    }
}

// Zero out gradients
fn zero_grads(model: &mut GPTModel) {
    for i in 0..model.d_wte.len() {
        model.d_wte[i] = 0.0;
    }
    for i in 0..model.d_wpe.len() {
        model.d_wpe[i] = 0.0;
    }
    for i in 0..model.d_lm_head.len() {
        model.d_lm_head[i] = 0.0;
    }

    for layer in &mut model.layers {
        for i in 0..layer.d_wq.len() {
            layer.d_wq[i] = 0.0;
        }
        for i in 0..layer.d_wk.len() {
            layer.d_wk[i] = 0.0;
        }
        for i in 0..layer.d_wv.len() {
            layer.d_wv[i] = 0.0;
        }
        for i in 0..layer.d_wo.len() {
            layer.d_wo[i] = 0.0;
        }
        for i in 0..layer.d_fc1.len() {
            layer.d_fc1[i] = 0.0;
        }
        for i in 0..layer.d_fc2.len() {
            layer.d_fc2[i] = 0.0;
        }
    }
}

// Struct to hold gradients for a single example
#[derive(Clone)]
struct GradientBuffer {
    d_wte: Vec<f32>,
    d_wpe: Vec<f32>,
    d_lm_head: Vec<f32>,
    d_wq: Vec<Vec<f32>>,
    d_wk: Vec<Vec<f32>>,
    d_wv: Vec<Vec<f32>>,
    d_wo: Vec<Vec<f32>>,
    d_fc1: Vec<Vec<f32>>,
    d_fc2: Vec<Vec<f32>>,
}

impl GradientBuffer {
    fn new(vocab_size: usize) -> Self {
        Self {
            d_wte: vec![0.0; vocab_size * N_EMBD],
            d_wpe: vec![0.0; BLOCK_SIZE * N_EMBD],
            d_lm_head: vec![0.0; vocab_size * N_EMBD],
            d_wq: vec![vec![0.0; N_EMBD * N_EMBD]; N_LAYER],
            d_wk: vec![vec![0.0; N_EMBD * N_EMBD]; N_LAYER],
            d_wv: vec![vec![0.0; N_EMBD * N_EMBD]; N_LAYER],
            d_wo: vec![vec![0.0; N_EMBD * N_EMBD]; N_LAYER],
            d_fc1: vec![vec![0.0; MLP_DIM * N_EMBD]; N_LAYER],
            d_fc2: vec![vec![0.0; N_EMBD * MLP_DIM]; N_LAYER],
        }
    }
}

// Training function with Rayon parallelization
fn train(
    model: &mut GPTModel,
    data: &[usize],
    iterations: usize,
    rng: &mut Rng,
) {
    println!("=== Starting Training (Multi-Core with Rayon) ===");
    println!("Iterations: {}", iterations);
    println!("Batch size: {}", BATCH_SIZE);
    println!("Learning rate: {} → {}", LEARNING_RATE, MIN_LEARNING_RATE);
    println!("Gradient clipping: {}", GRAD_CLIP);
    println!("Cores available: {}", rayon::current_num_threads());
    println!();

    let mut step = 0;
    let mut best_loss = f32::INFINITY;
    let mut best_iter = 0;

    for iter in 0..iterations {
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

                // Train on ALL positions (not just last) - 64x more training signal!
                for pos in 0..logits_seq.len() {
                    // Compute loss gradient for this position
                    let mut probs = vec![0.0; model_ref.vocab_size];
                    softmax_fwd(&logits_seq[pos], model_ref.vocab_size, &mut probs, 1.0);
                    total_loss += cross_entropy_loss(&probs, y_vec[pos]);

                    let mut d_logits = probs;
                    d_logits[y_vec[pos]] -= 1.0;

                    // Backward through lm_head
                    let mut d_x_out = vec![0.0; N_EMBD];
                    linear_bwd(
                        &d_logits,
                        &acts[pos].x_out,
                        &model_ref.lm_head,
                        model_ref.vocab_size,
                        N_EMBD,
                        &mut d_x_out,
                        &mut local_grads.d_lm_head,
                    );

                    let mut d_x = d_x_out;

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
                            &mut local_grads.d_fc2[li],
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
                            &mut local_grads.d_fc1[li],
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
                            &mut local_grads.d_wo[li],
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
                            &mut local_grads.d_wq[li],
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
                            &mut local_grads.d_wk[li],
                        );

                        linear_bwd(
                            &d_v,
                            &acts[pos].xn_attn[li],
                            &wv_weights,
                            N_EMBD,
                            N_EMBD,
                            &mut d_xn_attn_v,
                            &mut local_grads.d_wv[li],
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

            // Accumulate gradients
            for i in 0..model.d_wte.len() {
                model.d_wte[i] += grads.d_wte[i];
            }
            for i in 0..model.d_wpe.len() {
                model.d_wpe[i] += grads.d_wpe[i];
            }
            for i in 0..model.d_lm_head.len() {
                model.d_lm_head[i] += grads.d_lm_head[i];
            }
            for li in 0..N_LAYER {
                for i in 0..model.layers[li].d_wq.len() {
                    model.layers[li].d_wq[i] += grads.d_wq[li][i];
                }
                for i in 0..model.layers[li].d_wk.len() {
                    model.layers[li].d_wk[i] += grads.d_wk[li][i];
                }
                for i in 0..model.layers[li].d_wv.len() {
                    model.layers[li].d_wv[i] += grads.d_wv[li][i];
                }
                for i in 0..model.layers[li].d_wo.len() {
                    model.layers[li].d_wo[i] += grads.d_wo[li][i];
                }
                for i in 0..model.layers[li].d_fc1.len() {
                    model.layers[li].d_fc1[i] += grads.d_fc1[li][i];
                }
                for i in 0..model.layers[li].d_fc2.len() {
                    model.layers[li].d_fc2[i] += grads.d_fc2[li][i];
                }
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
            let layer = &mut model.layers[li];

            // Update attention weights
            adam_step(
                &mut layer.wq,
                &layer.d_wq.clone(),
                &mut layer.m_wq,
                &mut layer.v_wq,
                step,
                lr,
            );
            adam_step(
                &mut layer.wk,
                &layer.d_wk.clone(),
                &mut layer.m_wk,
                &mut layer.v_wk,
                step,
                lr,
            );
            adam_step(
                &mut layer.wv,
                &layer.d_wv.clone(),
                &mut layer.m_wv,
                &mut layer.v_wv,
                step,
                lr,
            );
            adam_step(
                &mut layer.wo,
                &layer.d_wo.clone(),
                &mut layer.m_wo,
                &mut layer.v_wo,
                step,
                lr,
            );

            // Update MLP weights
            adam_step(
                &mut layer.fc1,
                &layer.d_fc1.clone(),
                &mut layer.m_fc1,
                &mut layer.v_fc1,
                step,
                lr,
            );
            adam_step(
                &mut layer.fc2,
                &layer.d_fc2.clone(),
                &mut layer.m_fc2,
                &mut layer.v_fc2,
                step,
                lr,
            );
        }

        // Track best loss
        if batch_loss < best_loss {
            best_loss = batch_loss;
            best_iter = iter;
        }

        // Log progress
        if iter % EVAL_INTERVAL == 0 || iter == iterations - 1 {
            println!("Iter {:4} | Loss: {:.4} | LR: {:.6} | Best: {:.4} @{}",
                iter, batch_loss, lr, best_loss, best_iter);
        }
    }

    println!();
    println!("Training complete!");
    println!("Best loss: {:.4} at iteration {}", best_loss, best_iter);
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
        // Forward pass (training=false, no dropout)
        let (logits_seq, _) = forward(&tokens, model, &mut kv_cache, false, None);
        let logits = &logits_seq[logits_seq.len() - 1];

        // Sample next token with top-p
        let mut probs = vec![0.0; model.vocab_size];
        softmax_fwd(logits, model.vocab_size, &mut probs, temperature);

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

        let mut kv_cache: Vec<Vec<(Vec<f32>, Vec<f32>)>> =
            (0..N_LAYER).map(|_| Vec::new()).collect();

        // Evaluation mode (training=false, no dropout)
        let (logits_seq, _) = forward(x, model, &mut kv_cache, false, None);

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
    let args: Vec<String> = std::env::args().collect();

    let iterations = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or_else(|_| {
            eprintln!("Invalid iterations argument. Using default: {}", MAX_ITERS);
            MAX_ITERS
        })
    } else {
        MAX_ITERS
    };

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

    // Initialize model
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

    // Estimate initial loss
    println!("Estimating initial loss...");
    let initial_loss = estimate_loss(&model, &data, 10, &mut rng);
    println!("Initial loss: {:.4}", initial_loss);
    println!();

    // Train the model
    train(&mut model, &data, iterations, &mut rng);

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
