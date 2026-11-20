"""Transformer décodeur de type Llama (RMSNorm, RoPE, SwiGLU, embeddings liés)."""
from dataclasses import dataclass, asdict
import torch, torch.nn as nn, torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    hidden: int = 464      # SwiGLU ; 688 dépasserait 10 Mo en fp16 (voir README)
    block: int = 256
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(T, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(T).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x: B,H,T,hd
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[2]
    c, s = cos[:T], sin[:T]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.hd = c.n_heads, c.dim // c.n_heads
        self.qkv = nn.Linear(c.dim, 3 * c.dim, bias=False)
        self.proj = nn.Linear(c.dim, c.dim, bias=False)
        self.dropout = c.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.dim, c.hidden, bias=False)
        self.w3 = nn.Linear(c.dim, c.hidden, bias=False)
        self.w2 = nn.Linear(c.hidden, c.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.dim), RMSNorm(c.dim)
        self.attn, self.mlp = Attention(c), MLP(c)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.n1(x), cos, sin))
        return x + self.drop(self.mlp(self.n2(x)))


class MiniLLM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.cfg = c
        self.tok = nn.Embedding(c.vocab_size, c.dim)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.dim)
        self.head = nn.Linear(c.dim, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # poids liés
        cos, sin = rope_cache(c.block, c.dim // c.n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():  # init résiduelle réduite
            if n.endswith("proj.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / (2 * c.n_layers) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # poids liés comptés une fois

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=1.0):
        was = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block:])
            logits = logits[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                if top_p < 1.0:
                    sl, si = torch.sort(logits, descending=True)
                    cp = F.softmax(sl, -1).cumsum(-1)
                    rm = cp - F.softmax(sl, -1) > top_p
                    sl[rm] = -float("inf")
                    logits = torch.full_like(logits, -float("inf")).scatter(1, si, sl)
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
        self.train(was)
        return idx


if __name__ == "__main__":
    m = MiniLLM(Config())
    n = m.num_params()
    print(f"params: {n:,} ({n/1e6:.2f}M)  fp16 ~ {2*n/1e6:.2f} MB  int8 ~ {n/1e6:.2f} MB")
