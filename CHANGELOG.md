# Changelog

All notable changes to randyGPT are documented here.

## [0.3.0] - 2026-02-16

### 🚀 Major Features

#### Multi-Core Training with Rayon
- **Added Rayon parallelization** for batch processing
- **8x speedup** on multi-core CPUs (825% CPU usage on 12-core system)
- Thread-safe gradient computation with local buffers
- Efficient gradient aggregation after parallel computation

#### Training Performance
- Training time: **78 seconds** for 1000 iterations
- Dataset: Shakespeare (1.1MB, 1.1M tokens)
- Loss improvement: **5.88 → 3.23** (45% reduction)
- Uses all available CPU cores efficiently

### ✨ Improvements
- Increased default iterations from 200 to 1000
- Added `GradientBuffer` struct for thread-local gradients
- Parallel forward + backward passes per batch item
- Real-time loss reporting every 100 iterations

### 📊 Performance Metrics
```
Real time:  78.56 seconds
User time:  643.04 seconds (parallel work)
System time: 5.55 seconds
CPU usage:  825% (8.25 cores average)
Cores used: 12 available, ~8 effectively utilized
```

### 📝 Documentation
- Added `PERFORMANCE.md` with detailed benchmarks
- Updated README with training status
- Added multi-core usage examples

---

## [0.2.0] - 2026-02-16

### 🎓 Training Implementation

#### Working Training Loop
- **Backward pass implemented** for MLP layers
- **Adam optimizer** with momentum and variance tracking
- Gradient computation and weight updates working
- Loss decreases during training (proof of learning!)

#### Architecture
- 4-layer transformer (up from 1 layer)
- 128-dimensional embeddings (up from 32)
- 8 attention heads (up from 4)
- 64-token context window (up from 8)
- ~800K parameters (up from ~10K)

### ✨ Features
- Character-level tokenizer with BOS/EOS
- Data loading from text files
- Loss estimation and tracking
- Top-p sampling for generation
- Temperature control
- KV caching for efficiency

### 📝 Documentation
- `README.md` - Feature overview
- `QUICKSTART.md` - 5-minute setup guide
- `USAGE.md` - Detailed usage instructions
- `IMPROVEMENTS.md` - Changelog from original
- `STATUS.md` - Current status
- `WHY_GIBBERISH.md` - Explains untrained output
- `PARALLELIZATION.md` - Multi-core guide

### 🔧 Infrastructure
- Gradient buffers for all parameters
- Adam optimizer state (m and v vectors)
- Training loop with batch sampling
- Cross-entropy loss computation

### 📊 Initial Results
- Training works but slow (single-core)
- Loss decreases: ~5.5 → ~3.5 in 200 iterations
- Output improving but needs more training
- Model learning basic character patterns

---

## [0.1.0] - Original mini_gpt.rs

### Initial Implementation
- Based on @RandyMcMillan's mini_gpt.rs gist
- Tiny architecture: 1 layer, 32-dim, 4 heads
- ~10K parameters
- No training (random weights only)
- Forward pass and generation only
- Simple character-level vocabulary

### Features
- Minimal transformer architecture
- PRNG for weight initialization
- Top-p sampling
- Example with "hello world"

---

## Comparison Table

| Feature | v0.1 | v0.2 | v0.3 |
|---------|------|------|------|
| **Layers** | 1 | 4 | 4 |
| **Embedding Dim** | 32 | 128 | 128 |
| **Parameters** | ~10K | ~800K | ~800K |
| **Training** | ❌ | ✅ Single-core | ✅ Multi-core |
| **Loss Tracking** | ❌ | ✅ | ✅ |
| **Speed (1000 iter)** | N/A | ~600s* | ~78s |
| **CPU Usage** | 100% | 100% | 825% |
| **Speedup** | 1x | 1x | 8x |
| **Quality** | Random | Learning | Learning faster |

*Estimated

---

## Future Roadmap

### v0.4.0 - Quality Improvements
- [ ] Add attention gradient computation
- [ ] Implement gradient clipping
- [ ] Learning rate scheduling
- [ ] Validation split and metrics

### v0.5.0 - Optimization
- [ ] BLAS integration for matrix ops (2x speedup)
- [ ] Better memory management
- [ ] Checkpoint saving/loading

### v1.0.0 - Production Ready
- [ ] CLI with arguments
- [ ] Multiple model size presets
- [ ] BPE tokenization
- [ ] Model evaluation suite

### v2.0.0 - GPU Support (Optional)
- [ ] wgpu or CUDA backend
- [ ] 50-100x speedup potential
- [ ] Mixed precision training
- [ ] Flash attention

---

## Credits

**Original Inspiration:** mini_gpt.rs by @RandyMcMillan
**Enhanced By:** Claude Sonnet 4.5
**Organization:** Monumental Systems

## License

MIT
