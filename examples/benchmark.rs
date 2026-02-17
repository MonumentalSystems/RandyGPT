use candle_core::{Device, Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_embd = 256usize;
    let mlp_dim = 1024usize;
    let batch = 32usize;
    let seq_len = 64usize;

    // --- CPU benchmark ---
    let cpu = Device::Cpu;
    let iters = 100;

    // Q projection: [N_EMBD x N_EMBD] matmul [seq*batch x N_EMBD]
    let x_cpu = Tensor::zeros((batch * seq_len, n_embd), candle_core::DType::F32, &cpu)?;
    let w_cpu = Tensor::zeros((n_embd, n_embd), candle_core::DType::F32, &cpu)?;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = x_cpu.matmul(&w_cpu.t()?)?;
    }
    let cpu_time = start.elapsed().as_millis();
    println!("CPU  QKV-like matmul [{batch}x{seq_len}x{n_embd}] * [{n_embd}x{n_embd}]: {}ms for {iters} iters ({:.1}ms avg)",
             cpu_time, cpu_time as f64 / iters as f64);

    // FC1: [MLP_DIM x N_EMBD] matmul
    let x2_cpu = Tensor::zeros((batch * seq_len, n_embd), candle_core::DType::F32, &cpu)?;
    let w2_cpu = Tensor::zeros((mlp_dim, n_embd), candle_core::DType::F32, &cpu)?;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = x2_cpu.matmul(&w2_cpu.t()?)?;
    }
    let cpu_fc1 = start.elapsed().as_millis();
    println!("CPU  FC1 matmul [{batch}x{seq_len}x{n_embd}] * [{mlp_dim}x{n_embd}]:     {}ms for {iters} iters ({:.1}ms avg)",
             cpu_fc1, cpu_fc1 as f64 / iters as f64);

    // --- Metal benchmark ---
    let metal = Device::new_metal(0)?;

    let x_gpu = Tensor::zeros((batch * seq_len, n_embd), candle_core::DType::F32, &metal)?;
    let w_gpu = Tensor::zeros((n_embd, n_embd), candle_core::DType::F32, &metal)?;
    // Warmup
    for _ in 0..5 {
        let _ = x_gpu.matmul(&w_gpu.t()?)?;
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = x_gpu.matmul(&w_gpu.t()?)?;
    }
    let gpu_time = start.elapsed().as_millis();
    println!("Metal QKV-like matmul [{batch}x{seq_len}x{n_embd}] * [{n_embd}x{n_embd}]: {}ms for {iters} iters ({:.1}ms avg)",
             gpu_time, gpu_time as f64 / iters as f64);

    let x2_gpu = Tensor::zeros((batch * seq_len, n_embd), candle_core::DType::F32, &metal)?;
    let w2_gpu = Tensor::zeros((mlp_dim, n_embd), candle_core::DType::F32, &metal)?;
    // Warmup
    for _ in 0..5 {
        let _ = x2_gpu.matmul(&w2_gpu.t()?)?;
    }
    let start = Instant::now();
    for _ in 0..iters {
        let _ = x2_gpu.matmul(&w2_gpu.t()?)?;
    }
    let gpu_fc1 = start.elapsed().as_millis();
    println!("Metal FC1 matmul [{batch}x{seq_len}x{n_embd}] * [{mlp_dim}x{n_embd}]:     {}ms for {iters} iters ({:.1}ms avg)",
             gpu_fc1, gpu_fc1 as f64 / iters as f64);

    println!("\n--- Results ---");
    println!("QKV speedup: {:.2}x", cpu_time as f64 / gpu_time as f64);
    println!("FC1 speedup: {:.2}x", cpu_fc1 as f64 / gpu_fc1 as f64);

    Ok(())
}
