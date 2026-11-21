"""PRANC : W ~ sum_j alpha_j R_j, R_j gaussiennes engendrees par graine, alpha par moindres carres.

Faisabilite : une base de n_alpha matrices de la taille complete d'une matrice MLP (884k x 20k
valeurs) ne tient ni en memoire ni en temps. On decoupe donc chaque matrice en blocs de L poids ;
une base unique de m = round(budget*L/16) vecteurs gaussiens (graine fixe) est partagee par tous
les blocs, et les alpha de chaque bloc sont les moindres carres exacts. Nombre total d'alpha
= budget * numel / 16, comme dans le PRANC complet. (Verification sur sous-matrice : voir test.)
"""
import torch, numpy as np

SEED_BASE = 1234


def base(m, L):
    g = torch.Generator().manual_seed(SEED_BASE)
    return torch.randn(m, L, generator=g)


def pranc(W, bpw, L=4096):
    m = max(1, int(round(bpw * L / 16)))
    R = base(m, L)
    flat = W.reshape(-1)
    n = flat.numel()
    pad = (-n) % L
    if pad: flat = torch.cat([flat, flat.new_zeros(pad)])
    Wb = flat.reshape(-1, L)                                   # (B,L)
    A = (R @ R.T).double()
    rhs = (Wb @ R.T).double()                                  # (B,m)
    al = torch.linalg.solve(A, rhs.T).T.float()
    al = al.to(torch.float16).float()                          # alpha stockes en fp16
    What = (al @ R).reshape(-1)[:n].reshape(W.shape)
    bits = Wb.shape[0] * m * 16 + 32                           # + graine de la base
    return What, bits


def pranc_complet(Wmat, m):
    """PRANC 'complet' sur une (petite) matrice : base de m matrices de la taille de Wmat."""
    n = Wmat.numel()
    g = torch.Generator().manual_seed(SEED_BASE)
    R = torch.randn(m, n, generator=g)
    al = torch.linalg.solve((R @ R.T).double(), (R @ Wmat.reshape(-1)).double()).float()
    return (al @ R).reshape(Wmat.shape), m * 16 + 32
