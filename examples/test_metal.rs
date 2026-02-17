use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Metal Device ===");
    
    // Test Metal device creation
    let device = Device::new_metal(0)?;
    println!("✓ Metal device created: {:?}", device);
    
    // Test simple tensor operations
    let a = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
    let b = Tensor::new(&[4.0f32, 5.0, 6.0], &device)?;
    let c = (a + b)?;
    println!("✓ Tensor addition: {:?}", c.to_vec1::<f32>()?);
    
    // Test matrix multiplication (GEMM - the key operation)
    let m1 = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
    let m2 = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]], &device)?;
    let result = m1.matmul(&m2)?;
    println!("✓ Matrix mult: {:?}", result.to_vec2::<f32>()?);
    
    println!("\n=== Metal GPU is working! ===");
    Ok(())
}
