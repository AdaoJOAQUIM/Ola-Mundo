"""Systemes A et D : OLMoE-1B-7B via le moteur 4 bits en streaming disque de echo/monde/.

Une seule passe par question : invite + les 4 options sont rangees dans une seule sequence ; un masque
d'attention empeche les options de se voir entre elles (chacune ne voit que l'invite et elle-meme) et
les positions des options repartent toutes de la fin de l'invite. Resultat identique a 4 passes separees,
pour un quart des lectures d'experts.
"""
import os, sys, time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, "/home/user/Ola-Mundo/echo/monde")
import moteur as mt

WORLD = "/home/user/monde_data/monde"
SRC = "/home/user/monde_data/olmoe"


class OLMoEScorer:
    def __init__(self, ctx_max=2048):
        self.m = mt.Moteur(WORLD, mode="q4", cache_frac=0.0, prefetch=False, max_ctx=8,
                           direct=False, drop=False)
        self.tok = Tokenizer.from_file(f"{SRC}/tokenizer.json")
        m = self.m
        inv = 1.0 / (m.cfg["rope_theta"] ** (torch.arange(0, m.hd, 2, dtype=torch.float) / m.hd))
        fr = torch.arange(ctx_max, dtype=torch.float)[:, None] * inv[None]
        emb = torch.cat((fr, fr), -1)
        self.cos, self.sin = emb.cos(), emb.sin()
        self.eos = self.tok.token_to_id("<|endoftext|>")

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def attention(self, L, h):
        m, p = self.m, self.m.lay[L]
        T = h.shape[0]
        x = mt.rms(h, p["ln1"], m.eps)
        q = mt.rms(x @ p["q"].T, p["qn"], m.eps)
        k = mt.rms(x @ p["k"].T, p["kn"], m.eps)
        v = x @ p["v"].T
        sh = lambda t: t.reshape(T, m.H, m.hd).transpose(0, 1)
        q, k, v = sh(q), sh(k), sh(v)
        q = q * self._cos + mt.rot_half(q) * self._sin
        k = k * self._cos + mt.rot_half(k) * self._sin
        o = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=self._mask)[0]
        return h + o.transpose(0, 1).reshape(T, m.d) @ p["o"].T

    @torch.no_grad()
    def score(self, prompt, opts):
        """-> liste de (somme des log-probas des jetons de l'option, nb de jetons)."""
        m = self.m
        pid = [self.eos] + self.enc(prompt)
        oids = [self.enc(o) for o in opts]
        P = len(pid)
        ids = pid + [t for o in oids for t in o]
        T = len(ids)
        pos = list(range(P))
        seg = [-1] * P
        for i, o in enumerate(oids):
            pos += list(range(P, P + len(o)))
            seg += [i] * len(o)
        pos_t = torch.tensor(pos)
        seg_t = torch.tensor(seg)
        idx = torch.arange(T)
        # visible si : s dans l'invite et s<=t ; ou meme option et s<=t
        causal = idx[None, :] <= idx[:, None]
        same = (seg_t[None, :] == seg_t[:, None]) & (seg_t[:, None] >= 0)
        in_prompt = (seg_t[None, :] == -1)
        mask = causal & (in_prompt | same)
        self._mask, self._cos, self._sin = mask, self.cos[pos_t], self.sin[pos_t]
        h = m.embed[torch.as_tensor(ids)].float()
        for L in range(m.NL):
            h = self.attention(L, h)
            h = m.moe(L, h)
        h = mt.rms(h, m.fnorm, m.eps)
        res, start = [], P
        for i, o in enumerate(oids):
            src = [P - 1] + list(range(start, start + len(o) - 1))
            lg = h[src] @ m.lm_head.T
            lp = torch.log_softmax(lg, -1)
            tg = torch.tensor(o)
            res.append((float(lp[torch.arange(len(o)), tg].sum()), len(o)))
            start += len(o)
        return res

    def close(self):
        self.m.close()
