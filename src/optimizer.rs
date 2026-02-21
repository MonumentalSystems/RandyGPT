/* ------------------------------------------------------------------ */
/* AdamW optimizer and gradient utilities                            */
/* ------------------------------------------------------------------ */

use crate::config::*;
use crate::model::GPTModel;

// GPU AdamW state: moment Vars live on the Metal device.
// Used by train_candle() — no CPU transfers in the hot loop.
pub use gpu_adam::GpuAdamState;

pub mod gpu_adam {
    use candle_core::{backprop::GradStore, Result as CResult, Tensor, Var};
    use crate::config::*;

    pub struct GpuAdamState {
        pub m: Vec<Var>,  // first moments, same order as CandleModel::all_vars()
        pub v: Vec<Var>,  // second moments
        pub step_t: usize,
    }

    impl GpuAdamState {
        /// Allocate zero moment Vars on the same device as each weight Var.
        pub fn new(vars: &[Var]) -> CResult<Self> {
            let mut m = Vec::with_capacity(vars.len());
            let mut v = Vec::with_capacity(vars.len());
            for var in vars {
                m.push(Var::zeros(var.shape(), candle_core::DType::F32, var.device())?);
                v.push(Var::zeros(var.shape(), candle_core::DType::F32, var.device())?);
            }
            Ok(Self { m, v, step_t: 0 })
        }

        /// Full GPU AdamW step with per-element gradient clipping.
        /// `vars` must be the same slice (same order) used in `new()`.
        /// `grad_scale` divides gradients before clipping (1.0 = no scaling).
        pub fn step(
            &mut self,
            grads: &GradStore,
            vars: &[Var],
            lr: f32,
            grad_scale: f64,
        ) -> CResult<()> {
            self.step_t += 1;
            let t = self.step_t as i32;
            let bc1 = 1.0f64 - (BETA1 as f64).powi(t);
            let bc2 = 1.0f64 - (BETA2 as f64).powi(t);
            let scale_m = 1.0 / bc1;
            let scale_v = 1.0 / bc2;
            let lr_wd   = lr as f64 * WEIGHT_DECAY as f64;
            let inv_scale = 1.0 / grad_scale;

            for (i, var) in vars.iter().enumerate() {
                if let Some(g) = grads.get(var) {
                    // Unscale (fp16 loss scaling) + L∞ gradient clip — stays on GPU
                    let g = (g * inv_scale)?.clamp(-GRAD_CLIP as f64, GRAD_CLIP as f64)?;

                    let m_prev = self.m[i].as_tensor();
                    let v_prev = self.v[i].as_tensor();

                    let next_m = ((m_prev * BETA1 as f64)? + (g.clone() * (1.0 - BETA1 as f64))?)?;
                    let next_v = ((v_prev * BETA2 as f64)? + (g.sqr()? * (1.0 - BETA2 as f64))?)?;

                    let m_hat = (&next_m * scale_m)?;
                    let v_hat = (&next_v * scale_v)?;

                    // weight decay: theta *= (1 - lr * wd)
                    let theta_wd = (var.as_tensor() * (1.0 - lr_wd))?;
                    let delta    = (m_hat / (v_hat.sqrt()? + EPSILON as f64)?)? * lr as f64;
                    let next_theta = (theta_wd - delta)?;

                    var.set(&next_theta)?;
                    self.m[i].set(&next_m)?;
                    self.v[i].set(&next_v)?;
                }
            }
            Ok(())
        }

        /// Pull all moment Vars to CPU Vec<f32> for checkpointing.
        #[allow(dead_code)]
        pub fn moments_to_vecs(&self) -> CResult<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
            let pull = |v: &Var| -> CResult<Vec<f32>> {
                v.as_tensor().flatten_all()?.to_vec1::<f32>()
            };
            let m_vecs: CResult<Vec<_>> = self.m.iter().map(pull).collect();
            let v_vecs: CResult<Vec<_>> = self.v.iter().map(pull).collect();
            Ok((m_vecs?, v_vecs?))
        }

        /// Restore moment Vars from saved Vec<f32> slices (one per Var, in all_vars() order).
        #[allow(dead_code)]
        pub fn from_vecs(
            m_vecs: Vec<Vec<f32>>,
            v_vecs: Vec<Vec<f32>>,
            vars: &[Var],
        ) -> CResult<Self> {
            let mut m = Vec::with_capacity(vars.len());
            let mut v = Vec::with_capacity(vars.len());
            for (i, var) in vars.iter().enumerate() {
                let t_m = Tensor::from_slice(&m_vecs[i], var.shape(), var.device())?;
                let t_v = Tensor::from_slice(&v_vecs[i], var.shape(), var.device())?;
                m.push(Var::from_tensor(&t_m)?);
                v.push(Var::from_tensor(&t_v)?);
            }
            Ok(Self { m, v, step_t: 0 })
        }
    }
}

// ── Dynamic loss scaler for fp16 mixed-precision training ──────────
#[cfg(feature = "fp16")]
pub struct DynamicLossScaler {
    pub scale: f64,
    good_steps: usize,
    overflow_count: usize,
}

#[cfg(feature = "fp16")]
impl DynamicLossScaler {
    pub fn new() -> Self {
        Self { scale: 32768.0, good_steps: 0, overflow_count: 0 }
    }

    /// Check if any gradient contains inf/nan. Queues GPU reductions,
    /// then pulls scalars — first to_scalar flushes the command buffer.
    pub fn has_overflow(grads: &candle_core::backprop::GradStore, vars: &[candle_core::Var]) -> bool {
        for var in vars {
            if let Some(g) = grads.get(var) {
                match g.abs().and_then(|t| t.sum_all()).and_then(|t| t.to_scalar::<f32>()) {
                    Ok(v) if v.is_finite() => continue,
                    _ => return true,
                }
            }
        }
        false
    }

    /// Update scale after a step. Returns true if this step was good (no overflow).
    pub fn update(&mut self, overflow: bool) -> bool {
        if overflow {
            self.scale = (self.scale * 0.5).max(1.0);
            self.good_steps = 0;
            self.overflow_count += 1;
            false
        } else {
            self.good_steps += 1;
            if self.good_steps >= 2000 {
                self.scale *= 2.0;
                self.good_steps = 0;
            }
            true
        }
    }

    pub fn overflow_count(&self) -> usize { self.overflow_count }
}

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

// Learning rate schedule: constant → cosine decay (no warmup)
pub fn get_learning_rate(iter: usize, max_iters: usize) -> f32 {
    let decay_start = (max_iters * 3) / 5; // start decay at 60%

    if iter < decay_start {
        LEARNING_RATE
    } else {
        let progress = (iter - decay_start) as f32 / (max_iters - decay_start) as f32;
        let cosine   = 0.5 * (1.0 + (progress * std::f32::consts::PI).cos());
        MIN_LEARNING_RATE + (LEARNING_RATE - MIN_LEARNING_RATE) * cosine
    }
}
