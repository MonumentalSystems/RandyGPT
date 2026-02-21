# Candle Metal Backend Issues

Edge cases and bugs encountered during MoE development on Apple Metal (M-series).
Reference for filing upstream issues at https://github.com/huggingface/candle.

## 1. `scatter_add` U32 index dtype lost on Metal

**Symptom**: `Metal error scatter-add ids should be u8/u32/i64, expected: U32, got: F32`

**Repro**: Create a 2D U32 index tensor and pass to `scatter_add` on a Metal device:
```rust
let idx_data: Vec<u32> = vec![0u32; n_tok * N_EMBD]; // definitely u32
let idx_2d = Tensor::from_vec(idx_data, (n_tok, N_EMBD), &metal_device)?;
result.scatter_add(&src, &idx_2d, 0)?; // panics: "got: F32"
```

**Attempted fixes (all failed)**:
- Constructing index via `broadcast_as` + `contiguous()` on a 1D U32 tensor — F32
- Constructing index directly from `Vec<u32>` via `Tensor::from_vec` — still F32
- Both 1D→broadcast and direct 2D construction produce F32 on Metal

**Note**: 1D U32 tensors via `Tensor::from_vec` work correctly with `index_select` on Metal. The issue appears specific to larger/2D U32 tensors or the `scatter_add` Metal kernel path.

**Workaround**: Avoid `scatter_add` entirely. Use a gather-only pattern:
compute per-expert outputs sparsely, `cat` them, then use `index_select` (1D, U32)
to gather each token's K expert results. See `forward_candle_train` MoE block.

**Candle version**: 0.9.2

---

## 2. `broadcast_as` + `contiguous()` may convert U32 → F32 on Metal

**Symptom**: A U32 tensor that goes through `unsqueeze` → `broadcast_as` → `contiguous()`
arrives as F32 dtype on Metal.

```rust
let idx_1d = Tensor::from_vec(vec![0u32; n], (n,), &metal_device)?; // U32 ✓
let idx_2d = idx_1d.unsqueeze(1)?
    .broadcast_as((n, m))?
    .contiguous()?; // dtype is now F32 on Metal
```

**Note**: This may be the root cause of issue #1 — if `contiguous()` on Metal always
materializes broadcast views as F32 regardless of source dtype.

**Workaround**: Construct the full tensor from a `Vec<u32>` on CPU and upload, or
avoid broadcast+contiguous on integer tensors entirely.

---

## 3. Per-matmul fp16 cast overhead negates throughput gain for small models

**Symptom**: Mixed-precision via per-matmul `to_dtype(F16)` + matmul + `to_dtype(F32)`
is **slower** than pure fp32 for small embedding dims (N_EMBD=128).

**Observed**: 3110ms/iter (fp16) vs 3020ms/iter (fp32) on model-ds (128-dim, 12-layer).

**Root cause**: Each `to_dtype` is a separate GPU kernel dispatch. With ~10 matmuls per
layer × 12 layers = ~120 matmuls, the 3 extra dtype casts per matmul add 360 GPU kernel
dispatches. For memory-bandwidth-bound matmuls (small matrices), the cast overhead exceeds
the fp16 compute throughput gain.

**When fp16 helps**: Larger models (256+ dim) where matmuls are compute-bound and the
2× fp16 throughput outweighs casting overhead. Also benefits from persistent fp16
(cast once, stay in fp16 through chains of matmuls, cast back only for norms/softmax).

**Workaround**: For small models, skip fp16. For larger models, use persistent fp16
(cast at layer boundaries, not per-matmul) to minimize cast overhead.

---

## 4. GPU↔CPU sync for MoE router top-K causes pipeline stalls

**Symptom**: Periodic GPU utilization dips to 25-50% during MoE training, visible in
`nvtop`/GPU monitor. ~5-15ms stall per layer × 12 layers = 60-180ms per iteration.

**Root cause**: `router_probs.flatten_all()?.to_vec1::<f32>()?` forces the Metal GPU
to flush its command buffer and block until all prior work completes, then copies data
to CPU for top-K sorting. This stalls the GPU pipeline 12× per forward pass.

**Impact**: Measured ~170ms improvement (3000→2830ms/iter) from eliminating the sync,
despite switching from sparse dispatch (50% expert FLOPs) to dense dispatch (100% FLOPs).
At 128-dim, the sync overhead dominates over the FLOP savings.

**Workaround**: Compute top-K mask entirely on GPU using `arg_sort_last_dim` + `gather`
+ `ge` threshold comparison. Use dense expert dispatch (all experts on all tokens,
gating weights zero out non-selected). Eliminates all CPU↔GPU syncs in the forward pass.

**Tradeoff**: Dense dispatch uses 2× expert FLOPs but eliminates sync stalls, all
`index_select`/`cat`/reverse-mapping overhead, and reduces peak memory usage.
Net positive for small models; sparse dispatch may win for larger models where the
FLOP cost dominates.

---

*Last updated: 2026-02-21*
