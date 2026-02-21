"""
modeling_randygpt.py — PyTorch implementation of the randyGPT architecture.

Matches the Rust forward pass exactly:
  - RMSNorm with no learnable parameters (x / rms(x))
  - Multi-head causal self-attention (no bias, scaled dot-product)
  - MLP with squared-ReLU activation and 4× expansion (dense mode)
  - MoE layer with top-k expert routing and squared-ReLU experts (MoE mode)
  - No final layer norm before lm_head
  - Learned token + position embeddings

Compatible with HuggingFace transformers via PreTrainedModel/PretrainedConfig.

Usage (standalone):
    from modeling_randygpt import RandyGPTConfig, RandyGPTForCausalLM
    from safetensors.torch import load_file

    cfg   = RandyGPTConfig.from_pretrained("path/to/model")
    model = RandyGPTForCausalLM(cfg)
    model.load_state_dict(load_file("path/to/model.safetensors"), strict=True)
    model.eval()

Usage (HuggingFace Hub):
    model = RandyGPTForCausalLM.from_pretrained("username/randygpt-s")
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


# ── Config ────────────────────────────────────────────────────────────────────

class RandyGPTConfig(PretrainedConfig):
    model_type = "randygpt"

    def __init__(
        self,
        vocab_size: int = 1500,
        n_embd:     int = 128,
        n_head:     int = 4,
        n_layer:    int = 8,
        block_size: int = 256,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size  = vocab_size
        self.n_embd      = n_embd
        self.n_head      = n_head
        self.n_layer     = n_layer
        self.block_size  = block_size
        self.head_dim    = n_embd // n_head
        self.mlp_dim     = 4 * n_embd
        self.n_experts   = kwargs.get("n_experts", 0)    # 0 = dense
        self.expert_dim  = kwargs.get("expert_dim", 0)
        self.moe_top_k   = kwargs.get("moe_top_k", 0)


# ── Modules ───────────────────────────────────────────────────────────────────

def rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm with no learnable scale — matches Rust rmsnorm_fwd exactly."""
    return x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: RandyGPTConfig):
        super().__init__()
        self.n_head   = cfg.n_head
        self.head_dim = cfg.head_dim
        self.n_embd   = cfg.n_embd
        self.scale    = 1.0 / math.sqrt(cfg.head_dim)

        # No bias — matches Rust linear_fwd (pure matmul)
        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,Dh]
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal attention scores
        scores = q @ k.transpose(-2, -1) * self.scale  # [B,H,T,T]
        mask   = torch.full((T, T), float('-inf'), device=x.device).triu(1)
        scores = scores + mask

        attn   = F.softmax(scores, dim=-1)
        out    = attn @ v                                         # [B,H,T,Dh]
        out    = out.transpose(1, 2).contiguous().view(B, T, C)  # [B,T,D]
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, cfg: RandyGPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd,   cfg.mlp_dim, bias=False)
        self.fc2 = nn.Linear(cfg.mlp_dim,  cfg.n_embd,  bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = F.relu(h).pow(2)   # squared ReLU — matches Rust exactly
        return self.fc2(h)


class MoEExpert(nn.Module):
    def __init__(self, n_embd: int, expert_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, expert_dim, bias=False)
        self.fc2 = nn.Linear(expert_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)).pow(2))


class MoELayer(nn.Module):
    def __init__(self, cfg: RandyGPTConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k     = cfg.moe_top_k
        self.router    = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts   = nn.ModuleList([
            MoEExpert(cfg.n_embd, cfg.expert_dim) for _ in range(cfg.n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        logits = self.router(x_flat)                         # [B*T, n_experts]
        probs  = F.softmax(logits, dim=-1)                   # [B*T, n_experts]
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)   # [B*T, top_k]
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)  # renormalize

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = (top_idx[:, k] == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    out[mask] += top_vals[mask, k:k+1] * expert_out
        return out.view(B, T, D)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: RandyGPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        if cfg.n_experts > 0:
            self.moe = MoELayer(cfg)
        else:
            self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rmsnorm(x))   # pre-norm attention + residual
        ffn = self.moe if hasattr(self, 'moe') else self.mlp
        x = x + ffn(rmsnorm(x))         # pre-norm FFN + residual
        return x


# ── Full model ────────────────────────────────────────────────────────────────

class RandyGPTForCausalLM(PreTrainedModel):
    config_class = RandyGPTConfig

    def __init__(self, cfg: RandyGPTConfig):
        super().__init__(cfg)
        self.wte     = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe     = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.layers  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # No final layer norm — matches Rust forward pass

    def forward(
        self,
        input_ids:     torch.Tensor,          # [B, T]
        labels:        torch.Tensor = None,   # [B, T] for loss
        **kwargs,
    ) -> CausalLMOutput:
        B, T = input_ids.shape
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # [1, T]
        x   = self.wte(input_ids) + self.wpe(pos)                    # [B, T, D]

        for block in self.layers:
            x = block(x)

        logits = self.lm_head(x)  # [B, T, V] — no final norm, matches Rust

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_text(
        self,
        prompt_ids:  torch.Tensor,   # [1, T] — already tokenized
        max_new_tokens: int = 200,
        temperature:    float = 0.8,
        top_p:          float = 0.9,
    ) -> torch.Tensor:
        """Simple top-p nucleus sampling. Returns full sequence including prompt."""
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            ctx   = ids[:, -self.config.block_size:]
            out   = self(ctx)
            logits = out.logits[:, -1, :] / temperature  # [1, V]

            # Top-p (nucleus) sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumprobs = sorted_probs.cumsum(dim=-1)
            mask     = cumprobs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum()
            next_id = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]
            ids     = torch.cat([ids, next_id.view(1, 1)], dim=1)

        return ids
