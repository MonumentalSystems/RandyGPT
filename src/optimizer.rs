/* ------------------------------------------------------------------ */
/* AdamW optimizer and gradient utilities                            */
/* ------------------------------------------------------------------ */

use crate::config::*;
use crate::model::GPTModel;

// AdamW step: Adam with decoupled weight decay.
// Bias corrections (powf) are precomputed once per call, not per-parameter.
pub fn adam_step(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    t: usize,
    lr: f32,
) {
    let t_f    = t as f32;
    let bc1    = 1.0 - BETA1.powf(t_f);
    let bc2    = 1.0 - BETA2.powf(t_f);
    let one_m_b1 = 1.0 - BETA1;
    let one_m_b2 = 1.0 - BETA2;

    for i in 0..params.len() {
        m[i] = BETA1 * m[i] + one_m_b1 * grads[i];
        v[i] = BETA2 * v[i] + one_m_b2 * grads[i] * grads[i];
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        params[i] -= lr * (m_hat / (v_hat.sqrt() + EPSILON) + WEIGHT_DECAY * params[i]);
    }
}

// Zero all gradient buffers in the model
pub fn zero_grads(model: &mut GPTModel) {
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

// Learning rate schedule: linear warmup → constant → cosine decay
pub fn get_learning_rate(iter: usize, max_iters: usize) -> f32 {
    let warmup_iters = 100;
    let decay_start  = (max_iters * 3) / 5; // start decay at 60%

    if iter < warmup_iters {
        LEARNING_RATE * (0.1 + 0.9 * iter as f32 / warmup_iters as f32)
    } else if iter < decay_start {
        LEARNING_RATE
    } else {
        let progress = (iter - decay_start) as f32 / (max_iters - decay_start) as f32;
        let cosine   = 0.5 * (1.0 + (progress * std::f32::consts::PI).cos());
        MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * cosine
    }
}
