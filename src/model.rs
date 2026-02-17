/* ------------------------------------------------------------------ */
/* Model structs: weights, activations, gradient buffer              */
/* ------------------------------------------------------------------ */

use candle_core::{Device, Result as CResult, Tensor, Var};
use crate::config::*;
use crate::rng::Rng;

/* Per-position activations stored during forward pass for backward */
#[derive(Clone)]
pub struct PosActs {
    pub x_embed: Vec<f32>,
    pub x_in: Vec<Vec<f32>>,
    pub xn_attn: Vec<Vec<f32>>,
    pub q: Vec<Vec<f32>>,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub attn_out: Vec<Vec<f32>>,
    pub x_mid: Vec<Vec<f32>>,
    pub xn_mlp: Vec<Vec<f32>>,
    pub mlp_pre: Vec<Vec<f32>>,
    pub mlp_post: Vec<Vec<f32>>,
    pub x_out: Vec<f32>,
}

impl PosActs {
    pub fn new() -> Self {
        Self {
            x_embed:  vec![0.0; N_EMBD],
            x_in:     vec![vec![0.0; N_EMBD]; N_LAYER],
            xn_attn:  vec![vec![0.0; N_EMBD]; N_LAYER],
            q:        vec![vec![0.0; N_EMBD]; N_LAYER],
            k:        vec![vec![0.0; N_EMBD]; N_LAYER],
            v:        vec![vec![0.0; N_EMBD]; N_LAYER],
            attn_out: vec![vec![0.0; N_EMBD]; N_LAYER],
            x_mid:    vec![vec![0.0; N_EMBD]; N_LAYER],
            xn_mlp:   vec![vec![0.0; N_EMBD]; N_LAYER],
            mlp_pre:  vec![vec![0.0; MLP_DIM]; N_LAYER],
            mlp_post: vec![vec![0.0; MLP_DIM]; N_LAYER],
            x_out:    vec![0.0; N_EMBD],
        }
    }
}

/* Weights + gradients + Adam moments for one transformer layer */
pub struct LayerWeights {
    // Attention projections
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,

    // MLP projections
    pub fc1: Vec<f32>,
    pub fc2: Vec<f32>,

    // Gradients
    pub d_wq: Vec<f32>,
    pub d_wk: Vec<f32>,
    pub d_wv: Vec<f32>,
    pub d_wo: Vec<f32>,
    pub d_fc1: Vec<f32>,
    pub d_fc2: Vec<f32>,

    // Adam moments
    pub m_wq: Vec<f32>, pub v_wq: Vec<f32>,
    pub m_wk: Vec<f32>, pub v_wk: Vec<f32>,
    pub m_wv: Vec<f32>, pub v_wv: Vec<f32>,
    pub m_wo: Vec<f32>, pub v_wo: Vec<f32>,
    pub m_fc1: Vec<f32>, pub v_fc1: Vec<f32>,
    pub m_fc2: Vec<f32>, pub v_fc2: Vec<f32>,
}

impl LayerWeights {
    pub fn new(rng: &mut Rng, _layer_idx: usize) -> Self {
        let mut make_params = |sz: usize, std: f32| -> Vec<f32> {
            (0..sz).map(|_| rng.gauss(0.0, std)).collect()
        };
        let zeros = |sz: usize| -> Vec<f32> { vec![0.0; sz] };

        // GPT-2 style init: output projections scaled down by 1/sqrt(2*N_LAYER)
        let std_in  = 0.02;
        let std_out = 0.02 / (2.0 * N_LAYER as f32).sqrt();

        Self {
            wq: make_params(N_EMBD * N_EMBD, std_in),
            wk: make_params(N_EMBD * N_EMBD, std_in),
            wv: make_params(N_EMBD * N_EMBD, std_in),
            wo: make_params(N_EMBD * N_EMBD, std_out),
            fc1: make_params(MLP_DIM * N_EMBD, std_in),
            fc2: make_params(N_EMBD * MLP_DIM, std_out),

            d_wq:  zeros(N_EMBD * N_EMBD),
            d_wk:  zeros(N_EMBD * N_EMBD),
            d_wv:  zeros(N_EMBD * N_EMBD),
            d_wo:  zeros(N_EMBD * N_EMBD),
            d_fc1: zeros(MLP_DIM * N_EMBD),
            d_fc2: zeros(N_EMBD * MLP_DIM),

            m_wq: zeros(N_EMBD * N_EMBD), v_wq: zeros(N_EMBD * N_EMBD),
            m_wk: zeros(N_EMBD * N_EMBD), v_wk: zeros(N_EMBD * N_EMBD),
            m_wv: zeros(N_EMBD * N_EMBD), v_wv: zeros(N_EMBD * N_EMBD),
            m_wo: zeros(N_EMBD * N_EMBD), v_wo: zeros(N_EMBD * N_EMBD),
            m_fc1: zeros(MLP_DIM * N_EMBD), v_fc1: zeros(MLP_DIM * N_EMBD),
            m_fc2: zeros(N_EMBD * MLP_DIM), v_fc2: zeros(N_EMBD * MLP_DIM),
        }
    }
}

/* Full GPT model: embeddings + N_LAYER transformer layers + LM head */
pub struct GPTModel {
    pub wte: Vec<f32>,      // Token embeddings  [vocab × N_EMBD]
    pub wpe: Vec<f32>,      // Position embeddings [BLOCK_SIZE × N_EMBD]
    pub layers: Vec<LayerWeights>,
    pub lm_head: Vec<f32>,  // Final projection [vocab × N_EMBD]

    pub d_wte: Vec<f32>,
    pub d_wpe: Vec<f32>,
    pub d_lm_head: Vec<f32>,

    pub m_wte: Vec<f32>, pub v_wte: Vec<f32>,
    pub m_wpe: Vec<f32>, pub v_wpe: Vec<f32>,
    pub m_lm_head: Vec<f32>, pub v_lm_head: Vec<f32>,

    pub vocab_size: usize,
}

impl GPTModel {
    pub fn new(vocab_size: usize, rng: &mut Rng) -> Self {
        let wte_sz  = vocab_size * N_EMBD;
        let wpe_sz  = BLOCK_SIZE * N_EMBD;
        let head_sz = vocab_size * N_EMBD;

        let layers: Vec<LayerWeights> = (0..N_LAYER)
            .map(|li| LayerWeights::new(rng, li))
            .collect();

        let wte:     Vec<f32> = (0..wte_sz).map(|_| rng.gauss(0.0, 0.02)).collect();
        let wpe:     Vec<f32> = (0..wpe_sz).map(|_| rng.gauss(0.0, 0.01)).collect();
        let lm_head: Vec<f32> = (0..head_sz).map(|_| rng.gauss(0.0, 0.02)).collect();

        Self {
            wte, wpe, layers, lm_head,
            d_wte:    vec![0.0; wte_sz],
            d_wpe:    vec![0.0; wpe_sz],
            d_lm_head: vec![0.0; head_sz],
            m_wte: vec![0.0; wte_sz], v_wte: vec![0.0; wte_sz],
            m_wpe: vec![0.0; wpe_sz], v_wpe: vec![0.0; wpe_sz],
            m_lm_head: vec![0.0; head_sz], v_lm_head: vec![0.0; head_sz],
            vocab_size,
        }
    }
}

/* Per-example gradient accumulation buffer (flat layout for SIMD) */
#[derive(Clone)]
pub struct GradientBuffer {
    pub d_wte:     Vec<f32>,
    pub d_wpe:     Vec<f32>,
    pub d_lm_head: Vec<f32>,
    // Flat: [layer0 | layer1 | ... | layerN-1]
    pub d_wq:  Vec<f32>,   // N_LAYER * N_EMBD * N_EMBD
    pub d_wk:  Vec<f32>,
    pub d_wv:  Vec<f32>,
    pub d_wo:  Vec<f32>,
    pub d_fc1: Vec<f32>,   // N_LAYER * MLP_DIM * N_EMBD
    pub d_fc2: Vec<f32>,   // N_LAYER * N_EMBD * MLP_DIM
}

impl GradientBuffer {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            d_wte:     vec![0.0; vocab_size * N_EMBD],
            d_wpe:     vec![0.0; BLOCK_SIZE * N_EMBD],
            d_lm_head: vec![0.0; vocab_size * N_EMBD],
            d_wq:  vec![0.0; N_LAYER * N_EMBD * N_EMBD],
            d_wk:  vec![0.0; N_LAYER * N_EMBD * N_EMBD],
            d_wv:  vec![0.0; N_LAYER * N_EMBD * N_EMBD],
            d_wo:  vec![0.0; N_LAYER * N_EMBD * N_EMBD],
            d_fc1: vec![0.0; N_LAYER * MLP_DIM * N_EMBD],
            d_fc2: vec![0.0; N_LAYER * N_EMBD * MLP_DIM],
        }
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn layer<'a>(field: &'a [f32], li: usize, stride: usize) -> &'a [f32] {
        &field[li * stride .. (li + 1) * stride]
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn layer_mut<'a>(field: &'a mut Vec<f32>, li: usize, stride: usize) -> &'a mut [f32] {
        &mut field[li * stride .. (li + 1) * stride]
    }
}

/* ------------------------------------------------------------------ */
/* Candle (Metal GPU) model: weights as Var, moments as Vec<f32>     */
/* ------------------------------------------------------------------ */

fn make_var(data: &[f32], shape: (usize, usize), device: &Device) -> CResult<Var> {
    Var::from_tensor(&Tensor::from_slice(data, shape, device)?)
}

fn var_to_vec(v: &Var) -> CResult<Vec<f32>> {
    v.as_tensor().flatten_all()?.to_vec1::<f32>()
}

pub struct CandleLayer {
    pub wq: Var, pub wk: Var, pub wv: Var, pub wo: Var,
    pub fc1: Var, pub fc2: Var,
    // AdamW moments stay on CPU
    pub m_wq: Vec<f32>, pub v_wq: Vec<f32>,
    pub m_wk: Vec<f32>, pub v_wk: Vec<f32>,
    pub m_wv: Vec<f32>, pub v_wv: Vec<f32>,
    pub m_wo: Vec<f32>, pub v_wo: Vec<f32>,
    pub m_fc1: Vec<f32>, pub v_fc1: Vec<f32>,
    pub m_fc2: Vec<f32>, pub v_fc2: Vec<f32>,
}

pub struct CandleModel {
    pub wte:     Var,  // [vocab_size, N_EMBD]
    pub wpe:     Var,  // [BLOCK_SIZE, N_EMBD]
    pub lm_head: Var,  // [vocab_size, N_EMBD]
    pub layers:  Vec<CandleLayer>,
    pub m_wte: Vec<f32>, pub v_wte: Vec<f32>,
    pub m_wpe: Vec<f32>, pub v_wpe: Vec<f32>,
    pub m_lm_head: Vec<f32>, pub v_lm_head: Vec<f32>,
    pub vocab_size: usize,
    pub device: Device,
}

impl CandleModel {
    /// Upload CPU GPTModel weights to Metal Vars. Moments start at zero.
    pub fn from_gpt(m: &GPTModel, device: &Device) -> CResult<Self> {
        let layers = (0..N_LAYER).map(|li| {
            let l = &m.layers[li];
            Ok(CandleLayer {
                wq:  make_var(&l.wq,  (N_EMBD, N_EMBD), device)?,
                wk:  make_var(&l.wk,  (N_EMBD, N_EMBD), device)?,
                wv:  make_var(&l.wv,  (N_EMBD, N_EMBD), device)?,
                wo:  make_var(&l.wo,  (N_EMBD, N_EMBD), device)?,
                fc1: make_var(&l.fc1, (MLP_DIM, N_EMBD), device)?,
                fc2: make_var(&l.fc2, (N_EMBD, MLP_DIM), device)?,
                m_wq: l.m_wq.clone(), v_wq: l.v_wq.clone(),
                m_wk: l.m_wk.clone(), v_wk: l.v_wk.clone(),
                m_wv: l.m_wv.clone(), v_wv: l.v_wv.clone(),
                m_wo: l.m_wo.clone(), v_wo: l.v_wo.clone(),
                m_fc1: l.m_fc1.clone(), v_fc1: l.v_fc1.clone(),
                m_fc2: l.m_fc2.clone(), v_fc2: l.v_fc2.clone(),
            })
        }).collect::<CResult<Vec<_>>>()?;

        Ok(Self {
            wte:     make_var(&m.wte,     (m.vocab_size, N_EMBD), device)?,
            wpe:     make_var(&m.wpe,     (BLOCK_SIZE, N_EMBD),   device)?,
            lm_head: make_var(&m.lm_head, (m.vocab_size, N_EMBD), device)?,
            layers,
            m_wte: m.m_wte.clone(), v_wte: m.v_wte.clone(),
            m_wpe: m.m_wpe.clone(), v_wpe: m.v_wpe.clone(),
            m_lm_head: m.m_lm_head.clone(), v_lm_head: m.v_lm_head.clone(),
            vocab_size: m.vocab_size,
            device: device.clone(),
        })
    }

    /// Download Var weights back to a CPU GPTModel (for inference / checkpointing).
    pub fn to_gpt(&self) -> CResult<GPTModel> {
        let vocab_size = self.vocab_size;

        let mut rng = crate::rng::Rng::new(0); // dummy — weights come from Vars
        let mut gpt = GPTModel::new(vocab_size, &mut rng);

        gpt.wte     = var_to_vec(&self.wte)?;
        gpt.wpe     = var_to_vec(&self.wpe)?;
        gpt.lm_head = var_to_vec(&self.lm_head)?;
        gpt.m_wte     = self.m_wte.clone(); gpt.v_wte     = self.v_wte.clone();
        gpt.m_wpe     = self.m_wpe.clone(); gpt.v_wpe     = self.v_wpe.clone();
        gpt.m_lm_head = self.m_lm_head.clone(); gpt.v_lm_head = self.v_lm_head.clone();

        for li in 0..N_LAYER {
            let cl = &self.layers[li];
            gpt.layers[li].wq  = var_to_vec(&cl.wq)?;
            gpt.layers[li].wk  = var_to_vec(&cl.wk)?;
            gpt.layers[li].wv  = var_to_vec(&cl.wv)?;
            gpt.layers[li].wo  = var_to_vec(&cl.wo)?;
            gpt.layers[li].fc1 = var_to_vec(&cl.fc1)?;
            gpt.layers[li].fc2 = var_to_vec(&cl.fc2)?;
            gpt.layers[li].m_wq  = cl.m_wq.clone(); gpt.layers[li].v_wq  = cl.v_wq.clone();
            gpt.layers[li].m_wk  = cl.m_wk.clone(); gpt.layers[li].v_wk  = cl.v_wk.clone();
            gpt.layers[li].m_wv  = cl.m_wv.clone(); gpt.layers[li].v_wv  = cl.v_wv.clone();
            gpt.layers[li].m_wo  = cl.m_wo.clone(); gpt.layers[li].v_wo  = cl.v_wo.clone();
            gpt.layers[li].m_fc1 = cl.m_fc1.clone(); gpt.layers[li].v_fc1 = cl.v_fc1.clone();
            gpt.layers[li].m_fc2 = cl.m_fc2.clone(); gpt.layers[li].v_fc2 = cl.v_fc2.clone();
        }
        Ok(gpt)
    }

    /// Replace a Var with a new tensor built from updated CPU weights.
    #[allow(dead_code)]
    pub fn set_var(v: &mut Var, data: &[f32], shape: (usize, usize), device: &Device) -> CResult<()> {
        *v = make_var(data, shape, device)?;
        Ok(())
    }

    /// Build a new Var from updated CPU weight data (used in train_candle update_var! macro).
    #[allow(dead_code)]
    pub fn set_var_ret(data: Vec<f32>, shape: (usize, usize), device: &Device) -> CResult<Var> {
        make_var(&data, shape, device)
    }
}
