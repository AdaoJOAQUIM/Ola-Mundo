"""Mini-TRM (Tiny Recursive Model) pour le Sudoku 9x9, variante MLP-mixer sans attention.

Un seul petit réseau `net` (blocs : mélange de jetons MLP sur les 81 positions +
SwiGLU sur les canaux, RMSNorm, résiduel) est ré-appliqué récursivement :
    z <- net(x + y + z)   (n fois)      puis      y <- net(y + z)
Ce « processus de récursion » est répété T fois par pas de supervision (T-1 sans
gradient, le dernier avec gradient). (y, z) sont transmis détachés d'un pas de
supervision au suivant.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CASES = 81


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.a = nn.Linear(d, h, bias=False)
        self.b = nn.Linear(d, h, bias=False)
        self.c = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.c(F.silu(self.a(x)) * self.b(x))


class MelangeJetons(nn.Module):
    """MLP appliquée le long de l'axe des 81 positions."""
    def __init__(self, h):
        super().__init__()
        self.f1 = nn.Linear(N_CASES, h)
        self.f2 = nn.Linear(h, N_CASES)

    def forward(self, x):  # x : [B, 81, D]
        t = x.transpose(1, 2)
        return self.f2(F.silu(self.f1(t))).transpose(1, 2)


class Bloc(nn.Module):
    def __init__(self, d, h_tok, h_can):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.tok = MelangeJetons(h_tok)
        self.can = SwiGLU(d, h_can)

    def forward(self, x):
        x = x + self.tok(self.n1(x))
        return x + self.can(self.n2(x))


class Reseau(nn.Module):
    def __init__(self, d, n_blocs, h_tok, h_can):
        super().__init__()
        self.blocs = nn.ModuleList(Bloc(d, h_tok, h_can) for _ in range(n_blocs))
        self.norme = RMSNorm(d)

    def forward(self, x):
        for b in self.blocs:
            x = b(x)
        return self.norme(x)


class TRM(nn.Module):
    def __init__(self, d=128, n_blocs=2, n=3, T=2, h_tok=162, h_can=344):
        super().__init__()
        self.d, self.n, self.T = d, n, T
        self.emb = nn.Embedding(11, d)           # 0 = vide, 1..9 = chiffres, 10 = réservé
        self.pos = nn.Parameter(torch.randn(N_CASES, d) * 0.02)
        self.net = Reseau(d, n_blocs, h_tok, h_can)
        self.tete = nn.Linear(d, 9)              # logits des 9 chiffres par case
        self.arret = nn.Linear(d, 1)             # tête d'arrêt (halting) sur y poolé
        self.y0 = nn.Parameter(torch.randn(d) * 0.1)
        self.z0 = nn.Parameter(torch.randn(d) * 0.1)
        nn.init.normal_(self.emb.weight, std=0.5)

    def init_etat(self, B):
        y = self.y0.expand(B, N_CASES, self.d).contiguous()
        z = self.z0.expand(B, N_CASES, self.d).contiguous()
        return y, z

    def processus(self, xe, y, z):
        for _ in range(self.n):
            z = self.net(xe + y + z)
            y = self.net(y + z)
        return y, z

    def etape(self, x, y, z):
        """Un pas de supervision. x : [B,81] long. Renvoie (y, z) détachés,
        logits [B,81,9], logit d'arrêt [B]."""
        xe = self.emb(x) + self.pos
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.processus(xe, y, z)
        y, z = self.processus(xe, y, z)
        logits = self.tete(y)
        h = self.arret(y.mean(1)).squeeze(-1)
        return y.detach(), z.detach(), logits, h


def nb_params(m):
    return sum(p.numel() for p in m.parameters())


def taille_export(m):
    """Octets d'export : int8 = 1 octet/paramètre + 4 octets d'échelle par tenseur ;
    fp16 = 2 octets/paramètre."""
    n = nb_params(m)
    nt = sum(1 for _ in m.parameters())
    return {"params": n, "tenseurs": nt, "int8_octets": n + 4 * nt, "fp16_octets": 2 * n}


def quantifier_int8(m):
    """Copie du modèle dont les poids sont quantifiés puis déquantifiés en int8
    (symétrique, par tenseur) ; renvoie (copie, octets exportés réels)."""
    q = copy.deepcopy(m)
    octets = 0
    with torch.no_grad():
        for p in q.parameters():
            s = p.abs().max().clamp_min(1e-12) / 127.0
            qi = torch.round(p / s).clamp(-127, 127).to(torch.int8)
            p.copy_(qi.float() * s)
            octets += qi.numel() + 4
    return q, octets
