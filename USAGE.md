# Usage Guide

## Quick Start

```bash
# Build and run
cargo run --release

# The model will:
# 1. Load train.txt if it exists
# 2. Build a vocabulary from the training data
# 3. Initialize model parameters
# 4. Estimate initial loss
# 5. Generate sample text (untrained)
```

## Scaling the Model

Edit the hyperparameters in `src/main.rs`:

### Tiny Model (Fast, Low Memory)
```rust
const N_EMBD: usize = 64;
const N_HEAD: usize = 4;
const N_LAYER: usize = 2;
const BLOCK_SIZE: usize = 32;
```
**~200K parameters, good for testing**

### Small Model (Default)
```rust
const N_EMBD: usize = 128;
const N_HEAD: usize = 8;
const N_LAYER: usize = 4;
const BLOCK_SIZE: usize = 64;
```
**~800K parameters, balanced**

### Medium Model (Slower, Better Quality)
```rust
const N_EMBD: usize = 256;
const N_HEAD: usize = 8;
const N_LAYER: usize = 6;
const BLOCK_SIZE: usize = 128;
```
**~3M parameters, requires more memory**

### Large Model (Very Slow, Best Quality)
```rust
const N_EMBD: usize = 512;
const N_HEAD: usize = 16;
const N_LAYER: usize = 12;
const BLOCK_SIZE: usize = 256;
```
**~20M parameters, CPU-intensive**

## Training Data

### Format
Plain text files work best. The model learns character-level patterns.

### Examples

**Programming Documentation:**
```bash
cat docs/*.md > train.txt
```

**Code Examples:**
```bash
find src/ -name "*.rs" -exec cat {} \; > train.txt
```

**Shakespeare:**
```bash
cp examples/shakespeare.txt train.txt
```

**Custom Text:**
```bash
echo "Your custom text here. The model will learn from this." > train.txt
```

### Data Size Recommendations

| Model Size | Min Data | Recommended | Max Useful |
|------------|----------|-------------|------------|
| Tiny | 10KB | 50KB | 500KB |
| Small | 50KB | 200KB | 2MB |
| Medium | 200KB | 1MB | 10MB |
| Large | 1MB | 5MB | 50MB |

More data generally helps, but diminishing returns after the "Recommended" amount for untrained models.

## Generation Parameters

Edit the `generate()` call in `main()`:

```rust
let sample = generate(
    &model,
    &tokenizer,
    "The quick brown",  // Prompt
    100,                // Max new tokens
    0.8,                // Temperature (0.1-2.0)
    0.9,                // Top-p (0.0-1.0)
    &mut rng,
);
```

### Temperature
- **Low (0.1-0.5)**: More deterministic, repetitive
- **Medium (0.6-0.9)**: Balanced creativity
- **High (1.0-2.0)**: More random, creative

### Top-P (Nucleus Sampling)
- **Low (0.5-0.7)**: Conservative, focused
- **Medium (0.8-0.9)**: Balanced variety
- **High (0.95-1.0)**: Maximum diversity

## Performance Tips

1. **Always use `--release` builds**
   ```bash
   cargo build --release
   ```

2. **For faster iteration**, use a smaller model during development

3. **Profile your code** if generation is too slow:
   ```bash
   cargo install flamegraph
   cargo flamegraph
   ```

4. **Memory usage** scales roughly with:
   - `N_EMBD²` for attention layers
   - `vocab_size × N_EMBD` for embeddings
   - `BLOCK_SIZE × N_LAYER × N_EMBD` for KV cache

## Adding Real Training

To implement training, you'll need to add:

1. **Backward Pass**: Compute gradients for all parameters
   - Add `backward()` function that computes ∂loss/∂params
   - Use chain rule through all operations

2. **Optimizer Step**: Update weights using Adam
   ```rust
   fn adam_update(param: &mut [f32], grad: &[f32], m: &mut [f32], v: &mut [f32], t: usize) {
       for i in 0..param.len() {
           m[i] = BETA1 * m[i] + (1.0 - BETA1) * grad[i];
           v[i] = BETA2 * v[i] + (1.0 - BETA2) * grad[i] * grad[i];

           let m_hat = m[i] / (1.0 - BETA1.powi(t as i32));
           let v_hat = v[i] / (1.0 - BETA2.powi(t as i32));

           param[i] -= LEARNING_RATE * m_hat / (v_hat.sqrt() + EPSILON);
       }
   }
   ```

3. **Training Loop**:
   ```rust
   for iter in 0..MAX_ITERS {
       // Sample batch
       let batch = sample_batch(&data, BATCH_SIZE, BLOCK_SIZE);

       // Forward pass
       let (logits, acts) = forward(&batch.x, &model, &mut kv_cache);

       // Compute loss
       let loss = compute_loss(&logits, &batch.y);

       // Backward pass
       backward(&acts, &batch.y, &mut model);

       // Update parameters
       optimizer_step(&mut model, iter + 1);

       if iter % EVAL_INTERVAL == 0 {
           println!("Iter {}: loss = {:.4}", iter, loss);
       }
   }
   ```

4. **Checkpointing**: Save/load model weights
   ```rust
   use std::fs::File;
   use std::io::{Write, Read};

   fn save_model(model: &GPTModel, path: &str) -> std::io::Result<()> {
       let mut file = File::create(path)?;
       // Serialize weights (use serde or custom binary format)
       Ok(())
   }
   ```

## Troubleshooting

### Out of Memory
- Reduce `N_EMBD`, `N_LAYER`, or `BLOCK_SIZE`
- Reduce `MAX_VOCAB` if you have a large character set
- Use smaller training data

### Slow Generation
- Make sure you're using `--release` build
- Reduce model size
- Reduce `max_new_tokens` in generation

### Poor Quality Output
- Model is untrained (random weights)
- Implement training (backward pass)
- Use more training data
- Increase model size

### Compilation Errors
- Make sure you're using Rust 1.70+
- Check for typos in constant definitions
- Ensure all imports are correct

## Next Steps

1. **Implement Training**: Add backward pass and optimizer
2. **Add Validation**: Track train/val loss
3. **Better Tokenization**: Implement BPE or WordPiece
4. **Model Checkpointing**: Save/load weights
5. **CLI Arguments**: Make hyperparameters configurable
6. **Batch Training**: Process multiple sequences at once
7. **GPU Support**: Use compute shaders or CUDA
