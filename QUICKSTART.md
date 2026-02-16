# Quick Start Guide

## 5-Minute Setup

```bash
# 1. Build the project
cargo build --release

# 2. Run with default data
cargo run --release

# That's it! The model will generate sample text (untrained)
```

## Add Your Own Training Data

```bash
# Option 1: Write your own
echo "Your training text here. Add as much as you want." > train.txt
cargo run --release

# Option 2: Use Shakespeare example
cp examples/shakespeare.txt train.txt
cargo run --release

# Option 3: Download datasets
./download_data.sh
# Follow the prompts, then:
cp data/tiny_shakespeare.txt train.txt
cargo run --release
```

## Customize the Model

Edit `src/main.rs` and change these constants at the top:

```rust
// Make it SMALLER (faster, testing)
const N_EMBD: usize = 64;
const N_LAYER: usize = 2;
const BLOCK_SIZE: usize = 32;

// Make it BIGGER (better quality)
const N_EMBD: usize = 256;
const N_LAYER: usize = 6;
const BLOCK_SIZE: usize = 128;
```

See `configs/` directory for pre-made configurations.

## Customize Generation

Find the `generate()` call in `main()` and modify:

```rust
let sample = generate(
    &model,
    &tokenizer,
    "Once upon a time",  // ← Your prompt here
    100,                 // ← How many new tokens to generate
    0.8,                 // ← Temperature (0.1=focused, 2.0=creative)
    0.9,                 // ← Top-p (0.5=safe, 0.99=diverse)
    &mut rng,
);
```

## What You Get

After running, you'll see:
1. Model statistics (parameters, vocab size)
2. Initial loss estimate
3. Generated sample text (will be random/nonsensical since untrained)

## Next Steps

### To Train the Model
Currently, the model generates random text because training isn't implemented yet. To add training:

1. See `USAGE.md` - section "Adding Real Training"
2. Implement backward pass
3. Add optimizer step
4. Create training loop

The infrastructure is 90% ready - just need backpropagation!

### To Improve Quality (Without Training)
Since training isn't implemented, you can still:
1. **Increase model size** - More parameters = more capacity
2. **Use better data** - Clean, structured text helps
3. **Adjust generation params** - Try different temperature/top-p

### To Learn More
- `README.md` - Full feature overview
- `USAGE.md` - Detailed usage instructions
- `IMPROVEMENTS.md` - What's new vs original
- `configs/` - Pre-made model configurations

## Common Issues

**"Out of memory"**
→ Reduce N_EMBD, N_LAYER, or BLOCK_SIZE

**"Generation is too slow"**
→ Make sure you used `--release` build
→ Try smaller model size

**"Output is gibberish"**
→ This is expected! Model is untrained (random weights)
→ Implement training to get real results

**"Vocabulary size is huge"**
→ Clean your training data (remove emoji, special chars)
→ Or increase MAX_VOCAB constant

## Example Session

```bash
$ cargo run --release
=== Enhanced randyGPT ===
Model: 4 layers, 8 heads, 128 embedding dim
Block size: 64, Vocab size: up to 512

Loading training data from train.txt...
Training data size: 2153 characters
Vocabulary size: 50
Sample tokens: ['<', '>', '\n', ' ', '+', ',', '-', '.', ':', 'B']

Tokenized to 2153 tokens
Initializing model...
Parameters: ~0.81M

Estimating initial loss...
Initial loss: 5.0875

=== Generation before training ===
The :itjn:
OiC
[... random output ...]

Training not yet implemented (requires backward pass).
Model is randomly initialized but functional for generation.
```

## File Structure

```
randyGPT/
├── src/
│   └── main.rs              # Main implementation
├── configs/                  # Pre-made configurations
│   ├── tiny.rs              #   Fast testing
│   ├── small.rs             #   Balanced (default)
│   ├── medium.rs            #   Better quality
│   └── large.rs             #   Best quality
├── examples/
│   └── shakespeare.txt      # Sample training data
├── data/                     # Downloaded datasets (created by script)
├── train.txt                 # Your training data
├── Cargo.toml               # Rust project config
├── README.md                # Full documentation
├── QUICKSTART.md            # This file
├── USAGE.md                 # Detailed usage
├── IMPROVEMENTS.md          # Changelog from original
└── download_data.sh         # Dataset downloader

```

## Quick Tips

1. **Always build with `--release`** - 10-50x faster than debug builds
2. **Start small** - Use tiny config for testing, then scale up
3. **Good training data matters** - Clean, consistent text works best
4. **Be patient** - Larger models are slow on CPU (but worth it!)
5. **Save your work** - Copy train.txt and note your hyperparameters

## Getting Help

- Check `USAGE.md` for detailed instructions
- Look at `configs/` for model sizing examples
- See `IMPROVEMENTS.md` for what changed from original
- File issues on GitHub if you find bugs

---

**Ready to go?** Just run `cargo run --release` and you're off!
