/* ------------------------------------------------------------------ */
/* Math primitives: linear layers, norms, activations, loss          */
/* ------------------------------------------------------------------ */

use crate::rng::Rng;

// ── BLAS FFI (Accelerate framework on macOS) ──────────────────────
// cblas_sgemv: matrix-vector multiply
// cblas_sger:  rank-1 outer-product update
#[allow(non_camel_case_types)]
mod blas {
    type c_int = i32;
    type c_float = f32;

    // CBLAS_ORDER / CBLAS_TRANSPOSE values
    pub const ROW_MAJOR: c_int = 101;
    pub const NO_TRANS:  c_int = 111;
    pub const TRANS:     c_int = 112;

    extern "C" {
        // y = alpha * A * x + beta * y   (or A^T * x when trans=TRANS)
        pub fn cblas_sgemv(
            order: c_int, trans: c_int,
            m: c_int, n: c_int,
            alpha: c_float,
            a: *const c_float, lda: c_int,
            x: *const c_float, incx: c_int,
            beta: c_float,
            y: *mut c_float, incy: c_int,
        );

        // A += alpha * x * y^T   (outer-product update)
        pub fn cblas_sger(
            order: c_int,
            m: c_int, n: c_int,
            alpha: c_float,
            x: *const c_float, incx: c_int,
            y: *const c_float, incy: c_int,
            a: *mut c_float, lda: c_int,
        );
    }
}

// Linear forward: out[nout] = W[nout×nin] · x[nin]
// Uses cblas_sgemv for the matrix-vector product.
pub fn linear_fwd(x: &[f32], w: &[f32], nout: usize, nin: usize, out: &mut [f32]) {
    unsafe {
        blas::cblas_sgemv(
            blas::ROW_MAJOR, blas::NO_TRANS,
            nout as i32, nin as i32,
            1.0,
            w.as_ptr(), nin as i32,
            x.as_ptr(), 1,
            0.0,
            out.as_mut_ptr(), 1,
        );
    }
}

// Linear backward:
//   d_x[nin]       = W^T · d_out           (cblas_sgemv with TRANS)
//   d_w[nout×nin] += d_out ⊗ x             (cblas_sger outer product)
pub fn linear_bwd(
    d_out: &[f32],
    x: &[f32],
    w: &[f32],
    nout: usize,
    nin: usize,
    d_x: &mut [f32],
    d_w: &mut [f32],
) {
    unsafe {
        // d_x = W^T · d_out
        blas::cblas_sgemv(
            blas::ROW_MAJOR, blas::TRANS,
            nout as i32, nin as i32,
            1.0,
            w.as_ptr(), nin as i32,
            d_out.as_ptr(), 1,
            0.0,
            d_x.as_mut_ptr(), 1,
        );

        // d_w += d_out ⊗ x  (accumulate — alpha=1.0)
        blas::cblas_sger(
            blas::ROW_MAJOR,
            nout as i32, nin as i32,
            1.0,
            d_out.as_ptr(), 1,
            x.as_ptr(), 1,
            d_w.as_mut_ptr(), nin as i32,
        );
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
