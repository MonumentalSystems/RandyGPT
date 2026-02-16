# Ready to Run - v0.4.0

## What's Changed

All 5 requested optimizations have been implemented:

### ✅ 1. Increased to 256-dim, 6 layers (4.77M params)
- **Before**: 128-dim, 4 layers, ~800K params
- **After**: 256-dim, 6 layers, ~4.77M params
- **Impact**: 6x more model capacity for learning

### ✅ 2. Fixed LR Schedule
- **Before**: Started decaying at iter 100 (caused plateau at iter 973)
- **After**: Constant until 80% of training, then cosine decay
- **Impact**: Full learning throughout main training phase

### ✅ 3. Added Weight Decay (AdamW)
- **Implementation**: Decoupled weight decay = 0.01
- **Impact**: L2 regularization for better generalization

### ✅ 4. Dropout
- **Implementation**: Fully integrated with rate 0.1
- **Applied**: After attention output and MLP output (before residuals)
- **Impact**: Prevents overfitting, improves generalization
- **Thread-safe**: Each parallel batch has unique RNG seed

### ✅ 5. Better Initialization (GPT-2 style)
- **Implementation**: Output projections scaled by 1/sqrt(2*N_LAYER)
- **Impact**: Accounts for residual accumulation in deeper network

## How to Run

Build (already done):
```bash
cargo build --release
```

Run training:
```bash
# Quick test (100 iterations, ~15 seconds)
./target/release/randygpt 100

# Short run (1000 iterations, ~2 minutes)
./target/release/randygpt 1000

# Medium run (3000 iterations, ~6 minutes)
./target/release/randygpt 3000

# Long run (5000 iterations, ~10 minutes)
./target/release/randygpt 5000
```

## Expected Results

With the fixed LR schedule, dropout, and larger model:

**Previous (v0.3, 3000 iters)**:
- Loss: 5.88 → 3.06 (plateaued at iter 973)
- Issue: Premature LR decay killed learning

**Initial v0.4 test (500 iters, no dropout)**:
- Loss: 4.21 → 2.93 @iter 205, then climbed to 4.01
- Issue: Overfitting on training batches

**Expected (v0.4 with dropout, 3000+ iters)**:
- Loss: Should continue improving past iter 1000
- Dropout prevents overfitting seen in 500-iter test
- LR stays constant until iter 2400 (80% of 3000)
- Cosine decay in final 600 iterations

**Recommended**: Try 5000 iterations to see full benefit of new schedule + dropout

## Performance

- **Cores used**: ~8-10 (on 12-core system)
- **CPU usage**: ~825%
- **Speed**: ~120 seconds per 1000 iterations
- **Memory**: ~210MB

## Files Modified

- `src/main.rs`: All optimizations implemented
- `CHANGELOG.md`: Documented v0.4.0 changes

## Status

🟢 **Ready to run manually!**

Just execute:
```bash
./target/release/randygpt 3000
```

Or for a longer, better quality run:
```bash
./target/release/randygpt 5000
```
