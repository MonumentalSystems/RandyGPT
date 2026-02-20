# randyGPT Colab Training Runbook

## Overview

Training flow:
1. **Local** — generate vocab + token cache, upload to Drive via rclone
2. **Colab** — mount Drive, run training (tokens already cached → starts instantly)
3. **Auto-backup** — `checkpoint_best.bin` copied to Drive on every new best val loss
4. **Resume** — set `RESUME` in Cell 4, re-run

---

## One-Time Setup

### Install rclone (macOS)
```bash
brew install rclone
rclone config
```
In the config wizard:
- `n` → new remote
- name: `gdrive`
- storage: `drive` (Google Drive)
- Follow OAuth flow in browser
- Scope: `drive` (full access)
- Leave everything else default

Verify:
```bash
rclone lsd richardj:
```

### Create Drive folder
```bash
rclone mkdir richardj:randyGPT
```

---

## Pre-Session Prep (run locally before each Colab session)

### Step 1 — Generate token cache
Run this from the project root. Takes ~2 min on M4, saves 15+ min on Colab.
```bash
cd /rjs/AI/randyGPT
python3 -c "
import sys, numpy as np, time
sys.path.insert(0, 'scripts')
from tokenizer_randygpt import RandyGPTTokenizer
tok = RandyGPTTokenizer.from_file('vocab.json')
print('Tokenizing...')
text = open('train.txt', encoding='utf-8').read()
t0 = time.time()
ids = tok.encode(text)
print(f'  {len(ids):,} tokens in {time.time()-t0:.1f}s')
np.array(ids, dtype=np.uint32).tofile('train.txt.tokens.bin')
print('  Saved train.txt.tokens.bin')
"
```

### Step 2 — Upload everything to Drive
```bash
cd /rjs/AI/randyGPT

# Training data (only when train.txt or vocab changes)
rclone copy train.txt          richardj:randyGPT/ --progress
rclone copy vocab.json         richardj:randyGPT/ --progress
rclone copy train.txt.tokens.bin richardj:randyGPT/ --progress

# Scripts (only when scripts change)
rclone copy scripts/export_hf.py            richardj:randyGPT/scripts/ --progress
rclone copy scripts/modeling_randygpt.py    richardj:randyGPT/scripts/ --progress
rclone copy scripts/tokenizer_randygpt.py   richardj:randyGPT/scripts/ --progress
rclone copy scripts/train_torch.py          richardj:randyGPT/scripts/ --progress
rclone copy scripts/write_rgpt_checkpoint.py richardj:randyGPT/scripts/ --progress

# Notebook (only when notebook changes)
rclone copy colab/randygpt_train.ipynb richardj:randyGPT/ --progress
```

Or all at once (re-uploads only changed files):
```bash
rclone sync scripts/ richardj:randyGPT/scripts/ \
  --include "*.py" \
  --exclude "__pycache__/**" \
  --progress
```

### What triggers a re-upload
| File | Re-upload when |
|------|---------------|
| `train.txt` | Training data changes |
| `vocab.json` | BPE vocab regenerated |
| `train.txt.tokens.bin` | Either of the above change |
| `scripts/*.py` | Any script edited |
| `colab/randygpt_train.ipynb` | Notebook edited |

---

## Colab Session

### Cell 1 — GPU check
- Verify: `GPU: Tesla T4` and `CUDA available: True`
- If no GPU: Runtime → Change runtime type → T4 GPU

### Cell 2 — Mount Drive
- Click the auth link, sign in
- Verify `DRIVE_DIR = /content/drive/MyDrive/randyGPT` exists

### Cell 3 — Copy files from Drive
Add/replace the upload block with:
```python
import os, sys

# Copy data
!cp "$DRIVE_DIR/train.txt"              /content/train.txt
!cp "$DRIVE_DIR/vocab.json"             /content/vocab.json
!cp "$DRIVE_DIR/train.txt.tokens.bin"   /content/train.txt.tokens.bin  # pre-built cache

# Copy scripts
os.makedirs('/content/randyGPT/scripts', exist_ok=True)
!cp "$DRIVE_DIR"/scripts/*.py /content/randyGPT/scripts/

sys.path.insert(0, '/content/randyGPT/scripts')

# Verify
for f in ['/content/train.txt', '/content/vocab.json',
          '/content/train.txt.tokens.bin']:
    size = os.path.getsize(f) / 1e6
    print(f'  ✓ {f} ({size:.1f} MB)')
```

### Cell 4 — Config
```python
MODEL_SIZE  = 'xs'      # xs / s / ds / m / l / deep / xl
ITERS       = 2000      # xs@T4 ~250ms/iter → 2000 iters ≈ 8 min
DTYPE       = 'fp16'    # T4: fp16  |  A100: bf16
BATCH_SIZE  = 64
GRAD_ACCUM  = 1
RESUME      = ''        # empty = fresh start (see Resume section below)
```

### Cell 5 — Train
- Token cache found instantly → training starts in seconds
- Output every 25 iters:
  ```
  Iter    0 | Loss: 7.31 | Val: 7.29 (ppl 1469) | LR: 0.000100 | Best val: 7.29 | 250ms/iter | ...
  Iter   25 | Loss: 5.82 | Val: 5.79 (ppl 327)  | LR: 0.000100 | ...
  ```
- `checkpoint_best.bin` auto-copied to Drive on each new best val loss

### Cells 6 & 7 — Export + generation test
Run after training completes. Cell 7 generates from 3 prompts to verify the model works.

---

## Resume After Disconnect

```python
# Cell 4 — change RESUME before running
RESUME = f'{DRIVE_DIR}/checkpoint_best.bin'
```

Then re-run Cells 1–5. The checkpoint header stores `iter` so training continues from where it left off.

---

## Expected Results by Model

| Model | Params | T4 fp16 | 2000 iters | Expected val loss |
|-------|--------|---------|------------|-------------------|
| xs    | 746K   | ~250ms  | ~8 min     | ~4.5–5.0          |
| s     | 1.6M   | ~500ms  | ~17 min    | ~4.2–4.6          |
| ds    | 2.8M   | ~800ms  | ~27 min    | ~4.0–4.4          |
| l     | 4.8M   | ~1600ms | ~53 min    | ~3.8–4.2          |

---

## Download Checkpoint Locally

After training, pull the best checkpoint from Drive:
```bash
rclone copy richardj:randyGPT/checkpoint_best.bin /rjs/AI/randyGPT/ --progress
```

Load it with the Rust server:
```bash
./target/release/randygpt --bpe --generate "Once upon a time"
```

Or export to HuggingFace:
```bash
python3 scripts/export_hf.py \
  --checkpoint checkpoint_best.bin \
  --vocab vocab.json \
  --model-size xs \
  --output hf_export
```

---

## Quick Reference

```bash
# Local prep (run before each Colab session if data/scripts changed)
cd /rjs/AI/randyGPT
python3 -c "import sys,numpy as np,time; sys.path.insert(0,'scripts'); from tokenizer_randygpt import RandyGPTTokenizer; tok=RandyGPTTokenizer.from_file('vocab.json'); t=open('train.txt',encoding='utf-8').read(); t0=time.time(); ids=tok.encode(t); print(f'{len(ids):,} tokens in {time.time()-t0:.1f}s'); np.array(ids,dtype=np.uint32).tofile('train.txt.tokens.bin')"
rclone sync . richardj:randyGPT/ --include "*.bin" --include "vocab.json" --include "train.txt" --progress
rclone sync scripts/ richardj:randyGPT/scripts/ --include "*.py" --exclude "__pycache__/**" --progress

# Download best checkpoint after training
rclone copy richardj:randyGPT/checkpoint_best.bin /rjs/AI/randyGPT/ --progress
```
