"""
app.py — randyGPT HuggingFace Space
Loads model weights from the Hub; HF hosts the compute.

Repo: MonumentalSystems/randygpt-s
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ── Inline model definition (no external import needed in the Space) ──────────

class RandyGPTConfig:
    def __init__(self, **kw):
        self.vocab_size = kw.get("vocab_size", 1500)
        self.n_embd     = kw.get("n_embd",     128)
        self.n_head     = kw.get("n_head",     4)
        self.n_layer    = kw.get("n_layer",    8)
        self.block_size = kw.get("block_size", 256)
        self.head_dim   = self.n_embd // self.n_head
        self.mlp_dim    = 4 * self.n_embd

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(**d)


def rmsnorm(x, eps=1e-5):
    return x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(cfg.head_dim)
        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) * self.scale
        mask = torch.full((T, T), float('-inf'), device=x.device).triu(1)
        attn = F.softmax(scores + mask, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, cfg.mlp_dim, bias=False)
        self.fc2 = nn.Linear(cfg.mlp_dim, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)).pow(2))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


class RandyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg     = cfg
        self.wte     = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe     = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.layers  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.wte(ids) + self.wpe(pos)
        for block in self.layers:
            x = block(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=200, temperature=0.8, top_p=0.9):
        self.eval()
        for _ in range(max_new_tokens):
            ctx    = ids[:, -self.cfg.block_size:]
            logits = self(ctx)[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cum    = sp.cumsum(-1)
            sp[cum - sp > top_p] = 0.0
            sp /= sp.sum()
            nxt = si[0, torch.multinomial(sp[0], 1)]
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
            if nxt.item() == 1:   # <|eos|>
                break
        return ids


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class Tokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.t2i   = {s: i for i, s in enumerate(vocab)}
        self.bos   = self.t2i.get("<|bos|>", 0)
        self.eos   = self.t2i.get("<|eos|>", 1)
        self.merge_map = {}
        for left, right in merges:
            l, r, m = self.t2i.get(left), self.t2i.get(right), self.t2i.get(left + right)
            if l is not None and r is not None and m is not None:
                self.merge_map.setdefault((l, r), m)

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])

    def _encode_chunk(self, text):
        tokens = [self.t2i[c] for c in text if c in self.t2i]
        if len(tokens) < 2:
            return tokens
        while True:
            best = None
            for i in range(len(tokens) - 1):
                m = self.merge_map.get((tokens[i], tokens[i+1]))
                if m is not None and (best is None or m < best):
                    best = m
            if best is None:
                break
            out, i = [], 0
            while i < len(tokens):
                if i+1 < len(tokens) and self.merge_map.get((tokens[i], tokens[i+1])) == best:
                    out.append(best); i += 2
                else:
                    out.append(tokens[i]); i += 1
            tokens = out
        return tokens

    def encode(self, text):
        nl = self.t2i.get("\n")
        lines, result = text.split("\n"), []
        for i, line in enumerate(lines):
            result.extend(self._encode_chunk(line))
            if i < len(lines) - 1 and nl is not None:
                result.append(nl)
        return result

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids
                       if i not in (self.bos, self.eos) and 0 <= i < len(self.vocab))


# ── Load model once at startup ────────────────────────────────────────────────

import os
REPO = os.environ.get("MODEL_REPO", "MonumentalSystems/randygpt-s")
DEVICE = "cpu"   # HF free-tier Spaces use CPU

print(f"Loading model from {REPO} …")
cfg_path = hf_hub_download(repo_id=REPO, filename="config.json")
st_path  = hf_hub_download(repo_id=REPO, filename="model.safetensors")
tok_path = hf_hub_download(repo_id=REPO, filename="tokenizer.json")

cfg   = RandyGPTConfig.from_json(cfg_path)
tok   = Tokenizer.from_json(tok_path)
model = RandyGPT(cfg)
model.load_state_dict(load_file(st_path, device=DEVICE))
model.eval()
print(f"Model ready — vocab {cfg.vocab_size}, {cfg.n_layer}L×{cfg.n_embd}D")


# ── Inference ─────────────────────────────────────────────────────────────────

def generate(prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    prompt = prompt.strip()
    if not prompt:
        return "(enter a prompt)"
    ids = tok.encode(prompt)
    if not ids:
        return "(could not tokenize prompt)"
    tensor = torch.tensor([ids], dtype=torch.long)
    out    = model.generate(tensor, max_new_tokens=max_tokens,
                            temperature=temperature, top_p=top_p)
    full   = tok.decode(out[0].tolist())
    return full


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="randyGPT") as demo:
    gr.Markdown(
        "# randyGPT\n"
        "A GPT-style language model trained from scratch in Rust on 114 Project Gutenberg books.\n\n"
        f"**Model:** `{REPO}` · {cfg.n_layer} layers · {cfg.n_embd}-dim · {cfg.vocab_size}-token BPE vocab"
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder="Once upon a time",
                lines=3,
            )
            output_box = gr.Textbox(label="Generated text", lines=10, interactive=False)
            run_btn    = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            max_tok  = gr.Slider(20, 200, value=150, step=10,  label="Max new tokens")
            temp     = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            topp     = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p")

    run_btn.click(
        fn=generate,
        inputs=[prompt_box, max_tok, temp, topp],
        outputs=output_box,
    )
    prompt_box.submit(
        fn=generate,
        inputs=[prompt_box, max_tok, temp, topp],
        outputs=output_box,
    )

    gr.Examples(
        examples=[
            ["Once upon a time in a land far away"],
            ["It was the best of times, it was the worst of times"],
            ["The old man sat by the fire and"],
            ["She looked out across the sea and wondered"],
        ],
        inputs=prompt_box,
    )

if __name__ == "__main__":
    demo.launch()
