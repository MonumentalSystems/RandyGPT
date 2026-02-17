/* ------------------------------------------------------------------ */
/* Training loop, loss estimation, and text generation               */
/* ------------------------------------------------------------------ */

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use rayon::prelude::*;

use crate::checkpoint::{flush_checkpoint, serialize_checkpoint};
use crate::config::*;
use crate::forward::{forward, forward_metal_logits};
use crate::model::{GPTModel, GradientBuffer};
use crate::ops::{
    clip_gradients, cross_entropy_loss, linear_bwd,
    softmax_bwd, softmax_fwd,
};
use crate::optimizer::{adam_step, get_learning_rate, zero_grads};
use crate::rng::Rng;
use crate::tokenizer::Tokenizer;
use crate::metal::METAL_DEVICE;

/* ------------------------------------------------------------------ */
/* Estimate loss on a dataset (Metal-accelerated where available)    */
/* ------------------------------------------------------------------ */
pub fn estimate_loss(
    model: &GPTModel,
    data: &[usize],
    eval_iters: usize,
    rng: &mut Rng,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    for _ in 0..eval_iters {
        if data.len() <= BLOCK_SIZE + 1 { continue; }

        let start = rng.choice(data.len() - BLOCK_SIZE - 1);
        let x = &data[start..start + BLOCK_SIZE];
        let y = &data[start + 1..start + BLOCK_SIZE + 1];

        let logits_seq = if METAL_DEVICE.is_some() {
            forward_metal_logits(x, model)
        } else {
            let mut kv = (0..N_LAYER).map(|_| Vec::new()).collect();
            let (logits, _) = forward(x, model, &mut kv, false, None);
            logits
        };

        let mut probs = vec![0.0f32; model.vocab_size];
        for (logits, &target) in logits_seq.iter().zip(y.iter()) {
            softmax_fwd(logits, model.vocab_size, &mut probs, 1.0);
            total_loss += cross_entropy_loss(&probs, target);
            count += 1;
        }
    }

    if count > 0 { total_loss / count as f32 } else { 0.0 }
}

/* ------------------------------------------------------------------ */
/* Text generation with top-p sampling                               */
/* ------------------------------------------------------------------ */
pub fn generate(
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

    while tokens.len() < max_len {
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

        let mut probs = vec![0.0f32; model.vocab_size];
        softmax_fwd(&logits, model.vocab_size, &mut probs, temperature);

        // Top-p (nucleus) sampling
        let mut sorted: Vec<usize> = (0..model.vocab_size).collect();
        sorted.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        let mut cumulative = 0.0f32;
        let mut cutoff = model.vocab_size;
        for (i, &idx) in sorted.iter().enumerate() {
            cumulative += probs[idx];
            if cumulative >= top_p { cutoff = i + 1; break; }
        }

        let top_sum: f32 = sorted[..cutoff].iter().map(|&i| probs[i]).sum();
        let mut r = rng.uniform() as f32 * top_sum;
        let mut next_token = sorted[0];
        for i in 0..cutoff {
            let idx = sorted[i];
            r -= probs[idx];
            if r <= 0.0 { next_token = idx; break; }
        }

        if next_token == tokenizer.eos_id { break; }
        tokens.push(next_token);

        if tokens.len() > BLOCK_SIZE {
            tokens = tokens[1..].to_vec();
            kv_cache = (0..N_LAYER).map(|_| Vec::new()).collect();
        }
    }

    tokenizer.decode(&tokens)
}

/* ------------------------------------------------------------------ */
/* Main training loop                                                 */
/* ------------------------------------------------------------------ */
pub fn train(
    model: &mut GPTModel,
    data: &[usize],
    val_data: &[usize],
    iterations: usize,
    rng: &mut Rng,
    iter_start: usize,
    step_start: usize,
    best_loss_start: f32,
    ctrlc_flag: Arc<AtomicBool>,
) {
    println!("=== Starting Training (Multi-Core with Rayon) ===");
    if iter_start > 0 { println!("Resuming from iteration {}", iter_start); }
    println!("Iterations: {} → {}", iter_start, iterations);
    println!("Batch size: {}", BATCH_SIZE);
    println!("Learning rate: {} → {}", LEARNING_RATE, MIN_LEARNING_RATE);
    println!("Gradient clipping: {}", GRAD_CLIP);
    println!("Cores available: {}", rayon::current_num_threads());
    println!();

    let mut step      = step_start;
    let mut best_loss = best_loss_start;
    let mut best_iter = if iter_start > 0 { iter_start.saturating_sub(1) } else { 0 };

    // In-memory checkpoint buffers — flushed only at end or Ctrl-C
    let mut ckpt_buf:      Vec<u8> = Vec::new();
    let mut ckpt_best_buf: Vec<u8> = Vec::new();

    for iter in iter_start..iterations {
        // Sample batch indices
        let batch_starts: Vec<usize> = (0..BATCH_SIZE)
            .filter_map(|_| {
                if data.len() > BLOCK_SIZE + 1 {
                    Some(rng.choice(data.len() - BLOCK_SIZE - 1))
                } else {
                    None
                }
            })
            .collect();

        if batch_starts.is_empty() { continue; }

        let model_ref = &*model;

        // Parallel forward + backward over batch items
        let results: Vec<(GradientBuffer, f32)> = batch_starts
            .par_iter()
            .map(|&start_idx| {
                let x_vec: Vec<usize> = data[start_idx..start_idx + BLOCK_SIZE].to_vec();
                let y_vec: Vec<usize> = data[start_idx + 1..start_idx + BLOCK_SIZE + 1].to_vec();

                let mut thread_rng = Rng::new(start_idx as u64 + iter as u64);

                let mut kv_cache: Vec<Vec<(Vec<f32>, Vec<f32>)>> =
                    (0..N_LAYER).map(|_| Vec::new()).collect();
                let (logits_seq, acts) =
                    forward(&x_vec, model_ref, &mut kv_cache, true, Some(&mut thread_rng));

                let mut local_grads = GradientBuffer::new(model_ref.vocab_size);
                let mut total_loss  = 0.0f32;

                // Scratch buffers hoisted out of per-position loop
                let mut probs    = vec![0.0f32; model_ref.vocab_size];
                let mut d_logits = vec![0.0f32; model_ref.vocab_size];
                let mut d_x_out  = vec![0.0f32; N_EMBD];

                for pos in 0..logits_seq.len() {
                    softmax_fwd(&logits_seq[pos], model_ref.vocab_size, &mut probs, 1.0);
                    total_loss += cross_entropy_loss(&probs, y_vec[pos]);

                    d_logits.copy_from_slice(&probs);
                    d_logits[y_vec[pos]] -= 1.0;

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

                    let mut d_x = d_x_out.clone();

                    for li in (0..N_LAYER).rev() {
                        let fc2_w = model_ref.layers[li].fc2.clone();
                        let fc1_w = model_ref.layers[li].fc1.clone();
                        let wo_w  = model_ref.layers[li].wo.clone();
                        let wq_w  = model_ref.layers[li].wq.clone();
                        let wk_w  = model_ref.layers[li].wk.clone();
                        let wv_w  = model_ref.layers[li].wv.clone();

                        // ----- MLP backward -----
                        let mut d_h2 = vec![0.0f32; MLP_DIM];
                        linear_bwd(
                            &d_x,
                            &acts[pos].mlp_post[li],
                            &fc2_w,
                            N_EMBD,
                            MLP_DIM,
                            &mut d_h2,
                            &mut local_grads.d_fc2[li * N_EMBD * MLP_DIM .. (li + 1) * N_EMBD * MLP_DIM],
                        );

                        let mut d_h1 = vec![0.0f32; MLP_DIM];
                        for i in 0..MLP_DIM {
                            if acts[pos].mlp_pre[li][i] > 0.0 {
                                d_h1[i] = d_h2[i] * 2.0 * acts[pos].mlp_pre[li][i];
                            }
                        }

                        let mut d_xn_mlp = vec![0.0f32; N_EMBD];
                        linear_bwd(
                            &d_h1,
                            &acts[pos].xn_mlp[li],
                            &fc1_w,
                            MLP_DIM,
                            N_EMBD,
                            &mut d_xn_mlp,
                            &mut local_grads.d_fc1[li * MLP_DIM * N_EMBD .. (li + 1) * MLP_DIM * N_EMBD],
                        );

                        let mut d_x_mid = vec![0.0f32; N_EMBD];
                        for i in 0..N_EMBD { d_x_mid[i] = d_xn_mlp[i] + d_x[i]; }

                        // ----- Attention backward -----
                        let mut d_attn_out = vec![0.0f32; N_EMBD];
                        linear_bwd(
                            &d_x_mid,
                            &acts[pos].attn_out[li],
                            &wo_w,
                            N_EMBD,
                            N_EMBD,
                            &mut d_attn_out,
                            &mut local_grads.d_wo[li * N_EMBD * N_EMBD .. (li + 1) * N_EMBD * N_EMBD],
                        );

                        let mut d_q = vec![0.0f32; N_EMBD];
                        let mut d_k = vec![0.0f32; N_EMBD];
                        let mut d_v = vec![0.0f32; N_EMBD];
                        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

                        for h in 0..N_HEAD {
                            let hs = h * HEAD_DIM;

                            // Recompute attention weights
                            let mut scores = vec![0.0f32; pos + 1];
                            for t in 0..=pos {
                                let dot: f32 = (0..HEAD_DIM)
                                    .map(|j| acts[pos].q[li][hs + j] * kv_cache[li][t].0[hs + j])
                                    .sum();
                                scores[t] = dot * scale;
                            }
                            let mut attn_weights = vec![0.0f32; pos + 1];
                            softmax_fwd(&scores, pos + 1, &mut attn_weights, 1.0);

                            // d_attn_weights and d_v for current position
                            let mut d_attn_weights = vec![0.0f32; pos + 1];
                            for t in 0..=pos {
                                for j in 0..HEAD_DIM {
                                    d_attn_weights[t] +=
                                        d_attn_out[hs + j] * kv_cache[li][t].1[hs + j];
                                    if t == pos {
                                        d_v[hs + j] += attn_weights[t] * d_attn_out[hs + j];
                                    }
                                }
                            }

                            // Softmax backward → d_scores
                            let mut d_scores = vec![0.0f32; pos + 1];
                            softmax_bwd(&attn_weights, &d_attn_weights, pos + 1, &mut d_scores);

                            // d_q and d_k for current position
                            for t in 0..=pos {
                                for j in 0..HEAD_DIM {
                                    d_q[hs + j] +=
                                        d_scores[t] * scale * kv_cache[li][t].0[hs + j];
                                    if t == pos {
                                        d_k[hs + j] +=
                                            d_scores[t] * scale * acts[pos].q[li][hs + j];
                                    }
                                }
                            }
                        }

                        // Backward through Q, K, V projections
                        let mut d_xn_q = vec![0.0f32; N_EMBD];
                        let mut d_xn_k = vec![0.0f32; N_EMBD];
                        let mut d_xn_v = vec![0.0f32; N_EMBD];
                        let sq = N_EMBD * N_EMBD;

                        linear_bwd(&d_q, &acts[pos].xn_attn[li], &wq_w,
                            N_EMBD, N_EMBD, &mut d_xn_q,
                            &mut local_grads.d_wq[li * sq .. (li + 1) * sq]);
                        linear_bwd(&d_k, &acts[pos].xn_attn[li], &wk_w,
                            N_EMBD, N_EMBD, &mut d_xn_k,
                            &mut local_grads.d_wk[li * sq .. (li + 1) * sq]);
                        linear_bwd(&d_v, &acts[pos].xn_attn[li], &wv_w,
                            N_EMBD, N_EMBD, &mut d_xn_v,
                            &mut local_grads.d_wv[li * sq .. (li + 1) * sq]);

                        // Combined attention input gradient + residual
                        for i in 0..N_EMBD {
                            d_x[i] = d_xn_q[i] + d_xn_k[i] + d_xn_v[i] + d_x_mid[i];
                        }
                    }

                    // Embedding gradients
                    for i in 0..N_EMBD {
                        local_grads.d_wte[x_vec[pos] * N_EMBD + i] += d_x[i];
                        local_grads.d_wpe[pos * N_EMBD + i]          += d_x[i];
                    }
                }

                (local_grads, total_loss / logits_seq.len() as f32)
            })
            .collect();

        // Aggregate gradients sequentially
        zero_grads(model);
        let mut batch_loss = 0.0f32;

        for (grads, loss) in results {
            batch_loss += loss;
            model.d_wte.iter_mut().zip(grads.d_wte.iter()).for_each(|(a, b)| *a += b);
            model.d_wpe.iter_mut().zip(grads.d_wpe.iter()).for_each(|(a, b)| *a += b);
            model.d_lm_head.iter_mut().zip(grads.d_lm_head.iter()).for_each(|(a, b)| *a += b);
            for li in 0..N_LAYER {
                let sq   = N_EMBD * N_EMBD;
                let fc1s = MLP_DIM * N_EMBD;
                let fc2s = N_EMBD * MLP_DIM;
                model.layers[li].d_wq.iter_mut().zip(grads.d_wq[li*sq..(li+1)*sq].iter()).for_each(|(a,b)| *a+=b);
                model.layers[li].d_wk.iter_mut().zip(grads.d_wk[li*sq..(li+1)*sq].iter()).for_each(|(a,b)| *a+=b);
                model.layers[li].d_wv.iter_mut().zip(grads.d_wv[li*sq..(li+1)*sq].iter()).for_each(|(a,b)| *a+=b);
                model.layers[li].d_wo.iter_mut().zip(grads.d_wo[li*sq..(li+1)*sq].iter()).for_each(|(a,b)| *a+=b);
                model.layers[li].d_fc1.iter_mut().zip(grads.d_fc1[li*fc1s..(li+1)*fc1s].iter()).for_each(|(a,b)| *a+=b);
                model.layers[li].d_fc2.iter_mut().zip(grads.d_fc2[li*fc2s..(li+1)*fc2s].iter()).for_each(|(a,b)| *a+=b);
            }
        }

        batch_loss /= batch_starts.len() as f32;
        step += 1;

        let lr = get_learning_rate(iter, iterations);

        // Gradient clipping
        clip_gradients(&mut model.d_wte, GRAD_CLIP);
        clip_gradients(&mut model.d_wpe, GRAD_CLIP);
        clip_gradients(&mut model.d_lm_head, GRAD_CLIP);
        for li in 0..N_LAYER {
            clip_gradients(&mut model.layers[li].d_wq,  GRAD_CLIP);
            clip_gradients(&mut model.layers[li].d_wk,  GRAD_CLIP);
            clip_gradients(&mut model.layers[li].d_wv,  GRAD_CLIP);
            clip_gradients(&mut model.layers[li].d_wo,  GRAD_CLIP);
            clip_gradients(&mut model.layers[li].d_fc1, GRAD_CLIP);
            clip_gradients(&mut model.layers[li].d_fc2, GRAD_CLIP);
        }

        // Adam optimizer update
        adam_step(&mut model.wte, &model.d_wte, &mut model.m_wte, &mut model.v_wte, step, lr);
        adam_step(&mut model.wpe, &model.d_wpe, &mut model.m_wpe, &mut model.v_wpe, step, lr);
        adam_step(&mut model.lm_head, &model.d_lm_head, &mut model.m_lm_head, &mut model.v_lm_head, step, lr);

        for li in 0..N_LAYER {
            // SAFETY: $w, $dw, $mw, $vw are distinct non-overlapping fields.
            let layer = &mut model.layers[li];
            macro_rules! layer_adam {
                ($w:ident, $dw:ident, $mw:ident, $vw:ident) => {{
                    let grads_ptr = layer.$dw.as_ptr();
                    let grads_len = layer.$dw.len();
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

        if batch_loss < best_loss {
            best_loss = batch_loss;
            best_iter = iter;
        }

        // Log + snapshot checkpoint buffers
        let is_log_iter = iter % EVAL_INTERVAL == 0 || iter == iterations - 1;
        if is_log_iter {
            if !val_data.is_empty() {
                let val_loss = estimate_loss(model, val_data, 10, rng);
                let val_ppl  = val_loss.exp();
                println!(
                    "Iter {:4} | Loss: {:.4} | Val: {:.4} (ppl {:.1}) | LR: {:.6} | Best: {:.4} @{}",
                    iter, batch_loss, val_loss, val_ppl, lr, best_loss, best_iter
                );
            } else {
                println!(
                    "Iter {:4} | Loss: {:.4} | LR: {:.6} | Best: {:.4} @{}",
                    iter, batch_loss, lr, best_loss, best_iter
                );
            }
            ckpt_buf = serialize_checkpoint(model, iter, step, best_loss);
            if best_iter == iter { ckpt_best_buf = ckpt_buf.clone(); }
        }

        // Ctrl-C: flush and exit
        if ctrlc_flag.load(Ordering::Relaxed) {
            ckpt_buf = serialize_checkpoint(model, iter, step, best_loss);
            if best_iter == iter || ckpt_best_buf.is_empty() {
                ckpt_best_buf = ckpt_buf.clone();
            }
            println!();
            println!("Interrupted at iteration {}. Saving checkpoint...", iter);
            flush_checkpoint("checkpoint.bin", &ckpt_buf)
                .map(|_| println!("✓ Saved checkpoint.bin (iter {})", iter))
                .unwrap_or_else(|e| eprintln!("Warning: {}", e));
            flush_checkpoint("checkpoint_best.bin", &ckpt_best_buf)
                .map(|_| println!("✓ Saved checkpoint_best.bin (best loss {:.4} @{})", best_loss, best_iter))
                .unwrap_or_else(|e| eprintln!("Warning: {}", e));
            std::process::exit(0);
        }
    }

    println!();
    println!("Training complete!");
    println!("Best loss: {:.4} at iteration {}", best_loss, best_iter);

    if !ckpt_buf.is_empty() {
        flush_checkpoint("checkpoint.bin", &ckpt_buf)
            .unwrap_or_else(|e| eprintln!("Warning: could not save checkpoint: {}", e));
    }
    if !ckpt_best_buf.is_empty() {
        flush_checkpoint("checkpoint_best.bin", &ckpt_best_buf)
            .unwrap_or_else(|e| eprintln!("Warning: could not save best checkpoint: {}", e));
    }
}
