/* ------------------------------------------------------------------ */
/* Forward pass: per-token CPU and batched Metal paths               */
/* ------------------------------------------------------------------ */

use candle_core::Tensor;
use crate::config::*;
use crate::metal::METAL_DEVICE;
use crate::model::{GPTModel, PosActs};
use crate::ops::{apply_dropout, linear_fwd, rmsnorm_fwd, softmax_fwd};
use crate::rng::Rng;

/// Per-token autoregressive forward pass with KV cache.
/// Returns (logits_per_position, activations_per_position).
/// Used during training so activations are available for backward.
pub fn forward(
    tokens: &[usize],
    model: &GPTModel,
    kv_cache: &mut Vec<Vec<(Vec<f32>, Vec<f32>)>>,
    training: bool,
    mut rng: Option<&mut Rng>,
) -> (Vec<Vec<f32>>, Vec<PosActs>) {
    let seq_len = tokens.len();
    let mut all_logits = Vec::with_capacity(seq_len);
    let mut all_acts   = Vec::with_capacity(seq_len);

    for pos in 0..seq_len {
        let tok = tokens[pos];
        let mut act = PosActs::new();

        // Token + position embedding
        for i in 0..N_EMBD {
            act.x_embed[i] = model.wte[tok * N_EMBD + i] + model.wpe[pos * N_EMBD + i];
        }

        let mut x = act.x_embed.clone();

        for li in 0..N_LAYER {
            act.x_in[li] = x.clone();

            // Attention pre-norm
            let mut xn = vec![0.0; N_EMBD];
            rmsnorm_fwd(&x, N_EMBD, &mut xn);
            act.xn_attn[li] = xn.clone();

            // Q, K, V projections
            let mut q = vec![0.0; N_EMBD];
            let mut k = vec![0.0; N_EMBD];
            let mut v = vec![0.0; N_EMBD];
            linear_fwd(&xn, &model.layers[li].wq, N_EMBD, N_EMBD, &mut q);
            linear_fwd(&xn, &model.layers[li].wk, N_EMBD, N_EMBD, &mut k);
            linear_fwd(&xn, &model.layers[li].wv, N_EMBD, N_EMBD, &mut v);

            act.q[li] = q.clone();
            act.k[li] = k.clone();
            act.v[li] = v.clone();

            // Append K,V to cache for this position
            if kv_cache[li].len() <= pos {
                kv_cache[li].push((k.clone(), v.clone()));
            }

            // Causal multi-head attention
            let mut attn_out = vec![0.0; N_EMBD];
            let scale = 1.0 / (HEAD_DIM as f32).sqrt();

            for h in 0..N_HEAD {
                let hs = h * HEAD_DIM;
                let mut scores = vec![0.0; pos + 1];
                for t in 0..=pos {
                    let dot: f32 = (0..HEAD_DIM)
                        .map(|j| q[hs + j] * kv_cache[li][t].0[hs + j])
                        .sum();
                    scores[t] = dot * scale;
                }
                let mut weights = vec![0.0; pos + 1];
                softmax_fwd(&scores, pos + 1, &mut weights, 1.0);
                for t in 0..=pos {
                    for j in 0..HEAD_DIM {
                        attn_out[hs + j] += weights[t] * kv_cache[li][t].1[hs + j];
                    }
                }
            }

            act.attn_out[li] = attn_out.clone();

            // Output projection + optional dropout + residual
            let mut attn_proj = vec![0.0; N_EMBD];
            linear_fwd(&attn_out, &model.layers[li].wo, N_EMBD, N_EMBD, &mut attn_proj);
            if training {
                if let Some(r) = rng.as_deref_mut() {
                    apply_dropout(&mut attn_proj, DROPOUT_RATE, r);
                }
            }
            for i in 0..N_EMBD { x[i] = attn_proj[i] + act.x_in[li][i]; }
            act.x_mid[li] = x.clone();

            // MLP pre-norm
            let mut xn_mlp = vec![0.0; N_EMBD];
            rmsnorm_fwd(&x, N_EMBD, &mut xn_mlp);
            act.xn_mlp[li] = xn_mlp.clone();

            // fc1 → squared ReLU → fc2 → optional dropout → residual
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
            if training {
                if let Some(r) = rng.as_deref_mut() {
                    apply_dropout(&mut mlp_out, DROPOUT_RATE, r);
                }
            }
            for i in 0..N_EMBD { x[i] = mlp_out[i] + act.x_mid[li][i]; }
        }

        act.x_out = x.clone();

        let mut logits = vec![0.0; model.vocab_size];
        linear_fwd(&x, &model.lm_head, model.vocab_size, N_EMBD, &mut logits);

        all_logits.push(logits);
        all_acts.push(act);
    }

    (all_logits, all_acts)
}

/// Metal-accelerated full-sequence forward pass (inference only, no activations).
/// Uses batched matmuls on the GPU for QKV / output / MLP projections.
/// Falls back to CPU if Metal is unavailable.
pub fn forward_metal_logits(tokens: &[usize], model: &GPTModel) -> Vec<Vec<f32>> {
    let device = match METAL_DEVICE.as_ref() {
        Some(d) => d,
        None    => return forward_metal_logits_cpu(tokens, model),
    };

    let seq_len = tokens.len();

    // Helper closure: [T, nin] * W^T [nout, nin] → flat [T*nout] on Metal
    let metal_mm = |x_data: &[f32], t: usize, nin: usize,
                    w_data: &[f32], nout: usize| -> Vec<f32> {
        let x_t = Tensor::from_slice(x_data, (t, nin), device).unwrap();
        let w_t = Tensor::from_slice(w_data, (nout, nin), device).unwrap();
        x_t.matmul(&w_t.t().unwrap()).unwrap()
            .flatten_all().unwrap()
            .to_vec1::<f32>().unwrap()
    };

    // Build input embeddings [seq_len, N_EMBD]
    let mut x_flat = vec![0.0f32; seq_len * N_EMBD];
    for (pos, &tok) in tokens.iter().enumerate() {
        for i in 0..N_EMBD {
            x_flat[pos * N_EMBD + i] =
                model.wte[tok * N_EMBD + i] + model.wpe[pos * N_EMBD + i];
        }
    }

    for li in 0..N_LAYER {
        // Attention pre-norm (CPU — cheap elementwise)
        let mut xn_flat = vec![0.0f32; seq_len * N_EMBD];
        for pos in 0..seq_len {
            rmsnorm_fwd(
                &x_flat[pos * N_EMBD..(pos + 1) * N_EMBD],
                N_EMBD,
                &mut xn_flat[pos * N_EMBD..(pos + 1) * N_EMBD],
            );
        }

        // Q, K, V on Metal
        let q_flat = metal_mm(&xn_flat, seq_len, N_EMBD, &model.layers[li].wq, N_EMBD);
        let k_flat = metal_mm(&xn_flat, seq_len, N_EMBD, &model.layers[li].wk, N_EMBD);
        let v_flat = metal_mm(&xn_flat, seq_len, N_EMBD, &model.layers[li].wv, N_EMBD);

        // Causal attention (CPU — O(T²·d) but T=64 is small)
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * N_EMBD];

        for h in 0..N_HEAD {
            let hs = h * HEAD_DIM;
            for pos in 0..seq_len {
                let mut scores = vec![0.0f32; pos + 1];
                for t in 0..=pos {
                    let dot: f32 = (0..HEAD_DIM)
                        .map(|j| q_flat[pos * N_EMBD + hs + j] * k_flat[t * N_EMBD + hs + j])
                        .sum();
                    scores[t] = dot * scale;
                }
                let mut weights = vec![0.0f32; pos + 1];
                softmax_fwd(&scores, pos + 1, &mut weights, 1.0);
                for t in 0..=pos {
                    for j in 0..HEAD_DIM {
                        attn_out[pos * N_EMBD + hs + j] +=
                            weights[t] * v_flat[t * N_EMBD + hs + j];
                    }
                }
            }
        }

        // Output projection on Metal
        let attn_proj = metal_mm(&attn_out, seq_len, N_EMBD, &model.layers[li].wo, N_EMBD);

        // Residual
        let mut x_mid = vec![0.0f32; seq_len * N_EMBD];
        for i in 0..seq_len * N_EMBD { x_mid[i] = x_flat[i] + attn_proj[i]; }

        // MLP pre-norm (CPU)
        let mut xn_mlp = vec![0.0f32; seq_len * N_EMBD];
        for pos in 0..seq_len {
            rmsnorm_fwd(
                &x_mid[pos * N_EMBD..(pos + 1) * N_EMBD],
                N_EMBD,
                &mut xn_mlp[pos * N_EMBD..(pos + 1) * N_EMBD],
            );
        }

        // fc1 on Metal
        let h1_flat = metal_mm(&xn_mlp, seq_len, N_EMBD, &model.layers[li].fc1, MLP_DIM);

        // Squared ReLU (CPU — elementwise)
        let mut h2_flat = vec![0.0f32; seq_len * MLP_DIM];
        for i in 0..h2_flat.len() {
            let v = h1_flat[i];
            h2_flat[i] = if v > 0.0 { v * v } else { 0.0 };
        }

        // fc2 on Metal
        let mlp_out = metal_mm(&h2_flat, seq_len, MLP_DIM, &model.layers[li].fc2, N_EMBD);

        // MLP residual
        for i in 0..seq_len * N_EMBD { x_flat[i] = x_mid[i] + mlp_out[i]; }
    }

    // LM head on Metal
    let logits_flat = metal_mm(&x_flat, seq_len, N_EMBD, &model.lm_head, model.vocab_size);

    logits_flat.chunks(model.vocab_size).map(|c| c.to_vec()).collect()
}

fn forward_metal_logits_cpu(tokens: &[usize], model: &GPTModel) -> Vec<Vec<f32>> {
    let mut kv = (0..N_LAYER).map(|_| Vec::new()).collect();
    let (logits, _) = forward(tokens, model, &mut kv, false, None);
    logits
}
