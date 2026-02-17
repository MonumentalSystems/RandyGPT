/* ------------------------------------------------------------------ */
/* Metal GPU device and batch matmul helper                          */
/* ------------------------------------------------------------------ */

use candle_core::{Device, Result as CandleResult, Tensor};
use crate::config::USE_METAL;

lazy_static::lazy_static! {
    pub static ref METAL_DEVICE: Option<Device> = {
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

/// Batched matmul on Metal: x [T, nin] * W^T [nout, nin] → [T, nout]
#[allow(dead_code)]
pub fn metal_matmul_batch(
    x: &[f32],
    w: &[f32],
    seq_len: usize,
    nin: usize,
    nout: usize,
    out: &mut [f32],
) -> CandleResult<()> {
    let device = METAL_DEVICE.as_ref().unwrap();
    let x_t = Tensor::from_slice(x, (seq_len, nin), device)?;
    let w_t = Tensor::from_slice(w, (nout, nin), device)?;
    let result = x_t.matmul(&w_t.t()?)?;
    let flat = result.flatten_all()?.to_vec1::<f32>()?;
    out.copy_from_slice(&flat);
    Ok(())
}
