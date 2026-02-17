/* ------------------------------------------------------------------ */
/* Math primitives: linear layers, norms, activations, loss          */
/* ------------------------------------------------------------------ */

use crate::rng::Rng;

// Linear forward: out[nout] = W[nout×nin] · x[nin]
// Always CPU — Metal is used only in batched inference (forward_metal_logits).
// Per-vector Metal calls allocate GPU memory tens of thousands of times per
// iteration, causing memory exhaustion. Batched matmuls are used instead.
pub fn linear_fwd(x: &[f32], w: &[f32], nout: usize, nin: usize, out: &mut [f32]) {
    for r in 0..nout {
        // zip-based dot product — LLVM can auto-vectorize with SIMD
        out[r] = w[r * nin..(r + 1) * nin].iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
    }
}

// Linear backward:
//   d_w[r,c] += d_out[r] * x[c]
//   d_x[c]   += d_out[r] * w[r,c]
pub fn linear_bwd(
    d_out: &[f32],
    x: &[f32],
    w: &[f32],
    nout: usize,
    nin: usize,
    d_x: &mut [f32],
    d_w: &mut [f32],
) {
    d_x[..nin].fill(0.0);
    for r in 0..nout {
        for c in 0..nin {
            d_w[r * nin + c] += d_out[r] * x[c];
            d_x[c]           += d_out[r] * w[r * nin + c];
        }
    }
}

// RMSNorm: out[i] = x[i] / rms(x); returns the scale factor
pub fn rmsnorm_fwd(x: &[f32], n: usize, out: &mut [f32]) -> f32 {
    let ms: f32 = x[..n].iter().map(|v| v * v).sum::<f32>() / n as f32;
    let scale = 1.0 / (ms + 1e-5).sqrt();
    for i in 0..n { out[i] = x[i] * scale; }
    scale
}

// Softmax with temperature
pub fn softmax_fwd(logits: &[f32], n: usize, probs: &mut [f32], temp: f32) {
    let mx = logits[..n].iter().map(|&v| v / temp).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for i in 0..n {
        probs[i] = ((logits[i] / temp) - mx).exp();
        sum += probs[i];
    }
    let inv = 1.0 / sum;
    for i in 0..n { probs[i] *= inv; }
}

// Softmax backward: d_in[i] = probs[i] * (d_out[i] - dot(probs, d_out))
pub fn softmax_bwd(probs: &[f32], d_out: &[f32], n: usize, d_in: &mut [f32]) {
    let dot: f32 = probs[..n].iter().zip(d_out[..n].iter()).map(|(p, d)| p * d).sum();
    for i in 0..n { d_in[i] = probs[i] * (d_out[i] - dot); }
}

pub fn cross_entropy_loss(probs: &[f32], target: usize) -> f32 {
    -probs[target].max(1e-10).ln()
}

// Dropout: randomly zero elements and scale remaining by 1/(1-rate)
pub fn apply_dropout(x: &mut [f32], dropout_rate: f32, rng: &mut Rng) {
    let scale = 1.0 / (1.0 - dropout_rate);
    for val in x.iter_mut() {
        if rng.uniform() > dropout_rate as f64 {
            *val *= scale;
        } else {
            *val = 0.0;
        }
    }
}

// Gradient clipping: L2 norm clip
pub fn clip_gradients(grads: &mut [f32], max_norm: f32) {
    let norm = grads.iter().map(|&g| g * g).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() { *g *= scale; }
    }
}
