/* ------------------------------------------------------------------ */
/* Checkpoint save / load                                            */
/* ------------------------------------------------------------------ */
//
// File format (little-endian):
//   [0..8]   magic      b"RGPT0001"
//   [8..12]  vocab_size u32
//   [12..16] iter       u32   (last completed iteration, 0-based)
//   [16..20] step       u32   (Adam step counter)
//   [20..24] best_loss  f32
//   [24..]   flat f32 arrays:
//              wte, wpe, lm_head,
//              per layer: wq, wk, wv, wo, fc1, fc2
//              m_wte, v_wte, m_wpe, v_wpe, m_lm_head, v_lm_head,
//              per layer: m_wq, v_wq, m_wk, v_wk, m_wv, v_wv,
//                         m_wo, v_wo, m_fc1, v_fc1, m_fc2, v_fc2

use std::fs::File;
use std::io::{Read, Write};
use crate::config::*;
use crate::model::GPTModel;

// ── In-memory helpers ──────────────────────────────────────────────

fn write_f32s(buf: &mut Vec<u8>, s: &[f32]) {
    buf.reserve(s.len() * 4);
    for &v in s { buf.extend_from_slice(&v.to_le_bytes()); }
}

fn read_f32_slice(f: &mut File, n: usize) -> std::io::Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 4];
    f.read_exact(&mut raw)?;
    Ok(raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

// ── Public API ─────────────────────────────────────────────────────

/// Serialize model + optimizer state to an in-memory byte buffer.
/// No disk I/O — call flush_checkpoint() to write to disk.
pub fn serialize_checkpoint(
    model: &GPTModel,
    iter: usize,
    step: usize,
    best_loss: f32,
) -> Vec<u8> {
    let n_params = model.wte.len() + model.wpe.len() + model.lm_head.len()
        + N_LAYER * model.layers[0].wq.len() * 6;
    let mut buf: Vec<u8> = Vec::with_capacity(24 + n_params * 4 * 3);

    // Header
    buf.extend_from_slice(b"RGPT0001");
    buf.extend_from_slice(&(model.vocab_size as u32).to_le_bytes());
    buf.extend_from_slice(&(iter as u32).to_le_bytes());
    buf.extend_from_slice(&(step as u32).to_le_bytes());
    buf.extend_from_slice(&best_loss.to_le_bytes());

    // Weights
    write_f32s(&mut buf, &model.wte);
    write_f32s(&mut buf, &model.wpe);
    write_f32s(&mut buf, &model.lm_head);
    for li in 0..N_LAYER {
        let l = &model.layers[li];
        write_f32s(&mut buf, &l.wq);
        write_f32s(&mut buf, &l.wk);
        write_f32s(&mut buf, &l.wv);
        write_f32s(&mut buf, &l.wo);
        write_f32s(&mut buf, &l.fc1);
        write_f32s(&mut buf, &l.fc2);
    }

    // Adam moments
    write_f32s(&mut buf, &model.m_wte);    write_f32s(&mut buf, &model.v_wte);
    write_f32s(&mut buf, &model.m_wpe);    write_f32s(&mut buf, &model.v_wpe);
    write_f32s(&mut buf, &model.m_lm_head); write_f32s(&mut buf, &model.v_lm_head);
    for li in 0..N_LAYER {
        let l = &model.layers[li];
        write_f32s(&mut buf, &l.m_wq); write_f32s(&mut buf, &l.v_wq);
        write_f32s(&mut buf, &l.m_wk); write_f32s(&mut buf, &l.v_wk);
        write_f32s(&mut buf, &l.m_wv); write_f32s(&mut buf, &l.v_wv);
        write_f32s(&mut buf, &l.m_wo); write_f32s(&mut buf, &l.v_wo);
        write_f32s(&mut buf, &l.m_fc1); write_f32s(&mut buf, &l.v_fc1);
        write_f32s(&mut buf, &l.m_fc2); write_f32s(&mut buf, &l.v_fc2);
    }

    buf
}

/// Atomically flush a checkpoint buffer to disk (write to .tmp then rename).
pub fn flush_checkpoint(path: &str, buf: &[u8]) -> std::io::Result<()> {
    let tmp = format!("{}.tmp", path);
    {
        let mut f = File::create(&tmp)?;
        f.write_all(buf)?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Load a checkpoint from disk into `model`.
/// Returns (iter_start, step, best_loss) — iter_start is the saved iter + 1
/// so training resumes *after* the last completed iteration.
pub fn load_checkpoint(
    path: &str,
    model: &mut GPTModel,
) -> std::io::Result<(usize, usize, f32)> {
    let mut f = File::open(path)?;

    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"RGPT0001" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Bad magic bytes in checkpoint {}", path),
        ));
    }

    let mut u32buf = [0u8; 4];
    f.read_exact(&mut u32buf)?; let ckpt_vocab = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let iter       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let step       = u32::from_le_bytes(u32buf) as usize;
    f.read_exact(&mut u32buf)?; let best_loss  = f32::from_le_bytes(u32buf);

    if ckpt_vocab != model.vocab_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Checkpoint vocab_size {} != model vocab_size {}", ckpt_vocab, model.vocab_size),
        ));
    }

    model.wte     = read_f32_slice(&mut f, model.wte.len())?;
    model.wpe     = read_f32_slice(&mut f, model.wpe.len())?;
    model.lm_head = read_f32_slice(&mut f, model.lm_head.len())?;
    for li in 0..N_LAYER {
        let n_sq = N_EMBD * N_EMBD;
        model.layers[li].wq  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wk  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wv  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].wo  = read_f32_slice(&mut f, n_sq)?;
        model.layers[li].fc1 = read_f32_slice(&mut f, MLP_DIM * N_EMBD)?;
        model.layers[li].fc2 = read_f32_slice(&mut f, N_EMBD * MLP_DIM)?;
    }

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

    Ok((iter + 1, step, best_loss))
}
