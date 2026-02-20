# model-ds2 — Gutenberg v3 (complete)

## Run Summary

| | |
|---|---|
| Model preset | model-ds (deep-s) |
| Architecture | 12 layers, 4 heads, 128-dim, ~2.90M params |
| Vocab | BPE-2000 (built from v3 corpus) |
| Training data | Gutenberg v3 cleaned, 112 books, ~97.9MB, ~33M tokens |
| Block size | 256 tokens |
| Batch size | 64 |
| Optimizer | AdamW, LR 1e-4 → cosine → ReduceLROnPlateau |
| Total iters | 15000 |
| Total time | ~19,408s (~5.4 hours) |
| Avg iter time | ~1993ms/iter (Metal GPU) |
| Best val loss | **3.8242** (iter 14375) |
| Final val loss | 3.8674 (ppl 47.8) |
| Final train loss | 3.7930 |
| LR reductions | 1× (5e-5 at iter ~13425) |
| Build flag | `--features model-ds` |

## Improvements over ds v1

- Dropout 0.1 in Metal/Candle training path (attn proj + MLP output)
- v3 corpus cleaning: image refs, HTML tags, any-length footnote refs stripped
- Unicode → ASCII normalization (smart quotes, em dashes, ellipsis, etc.)
- Fresh BPE-2000 vocab built from normalized corpus

## Val Loss Trajectory

| Iter | Val Loss |
|------|----------|
| 0 | 7.60 |
| 750 | 5.39 |
| 4300 | 4.42 |
| 5300 | 4.32 |
| 9000 | 4.05 |
| 11000 | 3.94 |
| 13000 | 3.87 |
| 13425 | LR reduced → 5e-5 |
| 13675 | 3.8327 |
| 13850 | 3.8318 |
| 14375 | **3.8242** (best) |
| 14999 | 3.8635 |

## Files

| File | Description |
|------|-------------|
| `checkpoint_ds2_best.bin` | Best val loss checkpoint (RGPT0003, iter 14375, val 3.8242) |
| `vocab_v3.json` | BPE-2000 tokenizer vocab + merges |

## Generation Samples (end of training)

```
Prompt: "Once upon a time"
. With whom his article and his life was the very titude with such a
broad as a great deal in every body he would hold me and keep them
into Cosette on a coat, by his simpless he would not be allowing from
him in his body and had season, he would make no longer overset in her.

Prompt: "To be or not to be"
respect to communications, and what I have numbered with the second in
proceeding; it was to be my brother. The point was unable to begin
to advance of it. At these idea that he had not temperoration;
"Have I know that my journey may be too good?"
```

## Publish Command

```bash
./scripts/publish_hf.sh ds2 checkpoint_ds2_best.bin
```

## Notes

- LR reduced once at iter ~13425 (patience=30, factor=0.5): 1e-4 → 5e-5
- LR reached minimum (~1e-5) by iter ~14600; no further improvement
- Final patience: 25/30 (stopped at iter 14999)
- Trained ~4.5× Chinchilla optimal (2.9M params × 20 = 58M optimal; ~262M tokens seen)
