"""HyperNetwork : petit MLP qui engendre les poids MLP du modele a partir de coordonnees.

Entree : plongements appris (couche, type gate/up/down, ligne, bloc de colonnes) concatenes.
Sortie : un bloc de BS=32 poids contigus d'une ligne. Cibles normalisees par l'ecart-type de
chaque matrice (90 ou 72 scalaires fp16, comptes). Tous les parametres sont stockes en fp16.
Le nombre de parametres est fixe par le budget en bits/poids (budget*N/16 parametres).
"""
import time, math, torch, torch.nn as nn, numpy as np

BS = 32
TYPES = {'gate_proj': 0, 'up_proj': 1, 'down_proj': 2}


class Hyper(nn.Module):
    def __init__(self, nl, rmax, cmax, H, dl=16, dt=8, dr=32, dc=32):
        super().__init__()
        self.El, self.Et = nn.Embedding(nl, dl), nn.Embedding(3, dt)
        self.Er, self.Ec = nn.Embedding(rmax, dr), nn.Embedding(cmax, dc)
        d = dl + dt + dr + dc
        self.net = nn.Sequential(nn.Linear(d, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, BS))

    def forward(self, c):
        z = torch.cat([self.El(c[:, 0]), self.Et(c[:, 1]), self.Er(c[:, 2]), self.Ec(c[:, 3])], 1)
        return self.net(z)


def nb_params(nl, rmax, cmax, H, **kw):
    return sum(p.numel() for p in Hyper(nl, rmax, cmax, H, **kw).parameters())


def choisir_H(nl, rmax, cmax, budget_params):
    lo, hi = 8, 8192
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if nb_params(nl, rmax, cmax, mid) <= budget_params: lo = mid
        else: hi = mid - 1
    return lo


def construire_donnees(mats):
    """mats : liste (couche, type, W[out,in]) -> coords (n,4) long, cibles (n,BS), echelles."""
    C, T, scales = [], [], []
    for (l, t, W) in mats:
        s = float(np.float16(W.std().item())); scales.append(s)
        o, i = W.shape
        assert i % BS == 0
        nb = i // BS
        tgt = (W / s).reshape(o * nb, BS)
        r = torch.arange(o).repeat_interleave(nb); cb = torch.arange(nb).repeat(o)
        c = torch.stack([torch.full_like(r, l), torch.full_like(r, t), r, cb], 1)
        C.append(c); T.append(tgt)
    return torch.cat(C), torch.cat(T), scales


def entrainer(coords, cibles, nl, rmax, cmax, bpw, minutes, nmats, bs=4096, lr=2e-3, log=print):
    N = cibles.numel()
    budget = int(bpw * N / 16) - nmats          # parametres fp16 ; 1 echelle fp16 par matrice
    H = choisir_H(nl, rmax, cmax, budget)
    torch.manual_seed(0)
    net = Hyper(nl, rmax, cmax, H)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.time(); step = 0; n = len(cibles)
    tmax = minutes * 60
    while True:
        el = time.time() - t0
        if el > tmax: break
        frac = el / tmax
        for g in opt.param_groups: g['lr'] = lr * (0.5 * (1 + math.cos(math.pi * frac)) * 0.97 + 0.03)
        idx = torch.randint(0, n, (bs,))
        loss = ((net(coords[idx]) - cibles[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % 200 == 0: log(f'  hyper pas {step} t={el:.0f}s mse={loss.item():.4f}')
    return net, H, npar, step, time.time() - t0


@torch.no_grad()
def generer(net, coords, chunk=65536):
    net = net.half().float()               # parametres arrondis fp16, calcul fp32
    return torch.cat([net(coords[i:i + chunk]) for i in range(0, len(coords), chunk)])
