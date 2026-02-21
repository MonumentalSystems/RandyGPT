"""
server.py — randyGPT OpenAI-compatible inference server
Serves POST /v1/chat/completions and GET /v1/models on port 7860.

Loads model weights from HuggingFace Hub at startup.
Compatible with OpenAI SDK, OpenRouter, LangChain, etc.
"""

import json
import math
import time
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import List, Optional
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ── Inline model ──────────────────────────────────────────────────────────────

class Cfg:
    def __init__(self, **kw):
        self.vocab_size  = kw.get("vocab_size", 1500)
        self.n_embd      = kw.get("n_embd",     128)
        self.n_head      = kw.get("n_head",     4)
        self.n_layer     = kw.get("n_layer",    8)
        self.block_size  = kw.get("block_size", 256)
        self.head_dim    = self.n_embd // self.n_head
        self.mlp_dim     = 4 * self.n_embd
        self.n_experts   = kw.get("n_experts",  0)   # 0 = dense
        self.expert_dim  = kw.get("expert_dim", 0)
        self.moe_top_k   = kw.get("moe_top_k",  0)

def rmsnorm(x, eps=1e-5):
    return x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()

class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.hd = c.n_head, c.head_dim
        self.sc = 1.0 / math.sqrt(c.head_dim)
        self.wq = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.wk = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.wv = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.wo = nn.Linear(c.n_embd, c.n_embd, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        s = q @ k.transpose(-2, -1) * self.sc
        s = s + torch.full((T, T), float('-inf'), device=x.device).triu(1)
        return self.wo((F.softmax(s, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc1 = nn.Linear(c.n_embd, c.mlp_dim, bias=False)
        self.fc2 = nn.Linear(c.mlp_dim, c.n_embd, bias=False)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)).pow(2))

class MoEExpert(nn.Module):
    def __init__(self, n_embd, expert_dim):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, expert_dim, bias=False)
        self.fc2 = nn.Linear(expert_dim, n_embd, bias=False)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)).pow(2))

class MoELayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_experts = c.n_experts
        self.top_k     = c.moe_top_k
        self.router    = nn.Linear(c.n_embd, c.n_experts, bias=False)
        self.experts   = nn.ModuleList([
            MoEExpert(c.n_embd, c.expert_dim) for _ in range(c.n_experts)
        ])
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        logits = self.router(x_flat)
        probs  = F.softmax(logits, dim=-1)
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = (top_idx[:, k] == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    out[mask] += top_vals[mask, k:k+1] * expert_out
        return out.view(B, T, D)

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn = Attn(c)
        if c.n_experts > 0:
            self.moe = MoELayer(c)
        else:
            self.mlp = MLP(c)
    def forward(self, x):
        x = x + self.attn(rmsnorm(x))
        ffn = self.moe if hasattr(self, 'moe') else self.mlp
        return x + ffn(rmsnorm(x))

class RandyGPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c       = c
        self.wte     = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe     = nn.Embedding(c.block_size, c.n_embd)
        self.layers  = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        x = self.wte(ids) + self.wpe(torch.arange(T, device=ids.device).unsqueeze(0))
        for b in self.layers:
            x = b(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=200, temperature=0.8, top_p=0.9):
        self.eval()
        for _ in range(max_new_tokens):
            ctx    = ids[:, -self.c.block_size:]
            logits = self(ctx)[:, -1, :] / max(temperature, 1e-6)
            probs  = F.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cum    = sp.cumsum(-1)
            sp[cum - sp > top_p] = 0.0
            sp /= sp.sum()
            nxt = si[0, torch.multinomial(sp[0], 1)]
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
            if nxt.item() == 1:
                break
        return ids

    @torch.no_grad()
    def generate_stream(self, ids, max_new_tokens=200, temperature=0.8, top_p=0.9):
        """Yields (token_id, is_last) one token at a time."""
        self.eval()
        for i in range(max_new_tokens):
            ctx    = ids[:, -self.c.block_size:]
            logits = self(ctx)[:, -1, :] / max(temperature, 1e-6)
            probs  = F.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cum    = sp.cumsum(-1)
            sp[cum - sp > top_p] = 0.0
            sp /= sp.sum()
            nxt = si[0, torch.multinomial(sp[0], 1)]
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
            token_id = nxt.item()
            is_last  = (token_id == 1) or (i == max_new_tokens - 1)
            yield token_id, is_last
            if token_id == 1:
                break


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class Tokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.t2i   = {s: i for i, s in enumerate(vocab)}
        self.bos   = self.t2i.get("<|bos|>", 0)
        self.eos   = self.t2i.get("<|eos|>", 1)
        self.mmap  = {}
        for l, r in merges:
            li, ri, mi = self.t2i.get(l), self.t2i.get(r), self.t2i.get(l + r)
            if li is not None and ri is not None and mi is not None:
                self.mmap.setdefault((li, ri), mi)

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])

    def _chunk(self, text):
        tokens = [self.t2i[c] for c in text if c in self.t2i]
        if len(tokens) < 2:
            return tokens
        while True:
            best = None
            for i in range(len(tokens) - 1):
                m = self.mmap.get((tokens[i], tokens[i+1]))
                if m is not None and (best is None or m < best):
                    best = m
            if best is None:
                break
            out, i = [], 0
            while i < len(tokens):
                if i+1 < len(tokens) and self.mmap.get((tokens[i], tokens[i+1])) == best:
                    out.append(best); i += 2
                else:
                    out.append(tokens[i]); i += 1
            tokens = out
        return tokens

    def encode(self, text):
        nl = self.t2i.get("\n")
        lines, result = text.split("\n"), []
        for i, line in enumerate(lines):
            result.extend(self._chunk(line))
            if i < len(lines) - 1 and nl is not None:
                result.append(nl)
        return result

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids
                       if i not in (self.bos, self.eos) and 0 <= i < len(self.vocab))


# ── Load model at startup ──────────────────────────────────────────────────────

import os
import threading

REPO     = os.environ.get("MODEL_REPO", "MonumentalSystems/randygpt-s")
MODEL_ID = REPO.split("/")[-1]

_model_lock    = threading.Lock()
_reload_lock   = threading.Lock()   # only one reload at a time
_is_reloading  = False              # debounce flag

def _get_remote_sha() -> str:
    """Fetch the current commit SHA of model.safetensors from Hub metadata."""
    from huggingface_hub import get_paths_info
    infos = list(get_paths_info(REPO, ["model.safetensors"], repo_type="model"))
    return infos[0].lfs.sha256 if infos and infos[0].lfs else ""

def load_model(force_weights=False):
    print(f"Loading {REPO} …")
    cfg_path = hf_hub_download(repo_id=REPO, filename="config.json",        force_download=False)
    st_path  = hf_hub_download(repo_id=REPO, filename="model.safetensors",  force_download=force_weights)
    tok_path = hf_hub_download(repo_id=REPO, filename="tokenizer.json",     force_download=False)
    _cfg  = Cfg(**json.loads(Path(cfg_path).read_text()))
    _tok  = Tokenizer.from_json(tok_path)
    _mdl  = RandyGPT(_cfg)
    _mdl.load_state_dict(load_file(st_path, device="cpu"))
    _mdl.eval()
    print("Model ready.")
    return _cfg, _tok, _mdl

cfg, tok, model = load_model()
_current_sha = _get_remote_sha()


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="randyGPT", version="0.9.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _openai_error(status: int, message: str, err_type: str = "invalid_request_error", code: str = None):
    body = {"error": {"message": message, "type": err_type}}
    if code:
        body["error"]["code"] = code
    return JSONResponse(status_code=status, content=body)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return _openai_error(exc.status_code, str(exc.detail))

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    msg = "; ".join(f"{e['loc'][-1]}: {e['msg']}" for e in exc.errors())
    return _openai_error(422, msg, code="invalid_request_error")


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id":       MODEL_ID,
            "object":   "model",
            "created":  1700000000,
            "owned_by": "MonumentalSystems",
        }]
    }


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model:       Optional[str]   = MODEL_ID
    messages:    List[Message]
    max_tokens:  Optional[int]   = 200
    temperature: Optional[float] = 0.8
    top_p:       Optional[float] = 0.9
    n:           Optional[int]   = 1
    stream:      Optional[bool]  = False


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _stream_completion(ids, max_tokens, temperature, top_p, completion_id, _model, _tok):
    """Generator that yields SSE chunks one token at a time.
    Takes model/tok as arguments (snapshotted at request time) so reloads
    mid-stream don't affect this request."""
    tensor      = torch.tensor([ids], dtype=torch.long)
    token_count = 0

    for token_id, is_last in _model.generate_stream(
        tensor, max_new_tokens=max_tokens,
        temperature=temperature, top_p=top_p
    ):
        token_text    = _tok.decode([token_id])
        token_count  += 1
        finish_reason = ("length" if token_count >= max_tokens else "stop") if is_last else None

        chunk = {
            "id":      completion_id,
            "object":  "chat.completion.chunk",
            "created": int(time.time()),
            "model":   MODEL_ID,
            "choices": [{
                "index": 0,
                "delta": {"content": token_text},
                "finish_reason": finish_reason,
            }],
        }
        yield _sse(chunk)

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    # Snapshot globals at request start — concurrent requests and reloads
    # are both safe because each request holds its own references.
    _m, _t, _c = model, tok, cfg

    prompt = req.messages[-1].content.strip() if req.messages else ""
    if not prompt:
        raise HTTPException(status_code=400, detail="No content in messages")

    ids = _t.encode(prompt)
    if not ids:
        raise HTTPException(status_code=400, detail="Prompt tokenized to empty sequence")

    max_tokens    = max(1, min(req.max_tokens or 200, _c.block_size))
    temperature   = max(0.01, min(req.temperature or 0.8, 2.0))
    top_p         = req.top_p or 0.9
    n             = max(1, min(req.n or 1, 4))
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # ── Streaming ─────────────────────────────────────────────────────────────
    if req.stream:
        return StreamingResponse(
            _stream_completion(ids, max_tokens, temperature, top_p, completion_id, _m, _t),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    # ── Non-streaming ─────────────────────────────────────────────────────────
    choices = []
    total_completion_tokens = 0

    for i in range(n):
        tensor      = torch.tensor([ids], dtype=torch.long)
        out         = _m.generate(tensor, max_new_tokens=max_tokens,
                                  temperature=temperature, top_p=top_p)
        full        = _t.decode(out[0].tolist())
        completion  = full[len(prompt):].lstrip() if full.startswith(prompt) else full
        comp_tokens = len(_t.encode(completion))
        total_completion_tokens += comp_tokens
        choices.append({
            "index":         i,
            "message":       {"role": "assistant", "content": completion},
            "finish_reason": "length" if comp_tokens >= max_tokens else "stop",
        })

    return {
        "id":                 completion_id,
        "object":             "chat.completion",
        "created":            int(time.time()),
        "model":              MODEL_ID,
        "system_fingerprint": f"{MODEL_ID}-v0.9.6",
        "choices":            choices,
        "usage": {
            "prompt_tokens":     len(ids),
            "completion_tokens": total_completion_tokens,
            "total_tokens":      len(ids) + total_completion_tokens,
        },
    }


@app.post("/reload")
def reload_weights():
    """Hot-reload model weights from Hub. Debounced — returns 200 immediately if already reloading.
    Only swaps weights if Hub has a newer version of model.safetensors."""
    global cfg, tok, model, _current_sha, _is_reloading

    # Debounce: if already reloading, return immediately
    if _is_reloading:
        return {"status": "ok", "model": MODEL_ID, "reloaded": False, "reason": "already reloading"}

    with _reload_lock:
        if _is_reloading:
            return {"status": "ok", "model": MODEL_ID, "reloaded": False, "reason": "already reloading"}
        _is_reloading = True

    try:
        new_sha = _get_remote_sha()
        if new_sha == _current_sha:
            return {"status": "ok", "model": MODEL_ID, "reloaded": False, "reason": "weights unchanged"}

        print(f"New weights detected ({_current_sha[:8]} → {new_sha[:8]}), reloading…")
        new_cfg, new_tok, new_model = load_model(force_weights=True)

        with _model_lock:
            cfg, tok, model = new_cfg, new_tok, new_model
            _current_sha = new_sha

        return {"status": "ok", "model": MODEL_ID, "reloaded": True, "sha": new_sha[:16]}
    finally:
        _is_reloading = False


@app.get("/")
def root():
    return {"model": MODEL_ID, "endpoints": ["/v1/models", "/v1/chat/completions", "/reload"]}
