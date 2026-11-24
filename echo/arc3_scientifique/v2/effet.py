"""Modèle d'effet d'action (idée StochasticGoose, réimplémentée) : petit CNN entraîné en ligne.

Entrée : image 64x64 (16 couleurs, one-hot). Sorties : P(l'image change) pour chaque action simple (tête globale,
7 logits) et pour chaque pixel de clic (tête 1x1, 64x64 logits). Entraîné sur (image, action|clic) -> changé ?
avec dédoublonnage par (hash d'état, action). Échantillonnage d'actions proportionnel à la probabilité prédite.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H = W = 64


class Reseau(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.c1 = nn.Conv2d(16, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.c3 = nn.Conv2d(c, c, 3, padding=2, dilation=2)
        self.clic = nn.Conv2d(c, 1, 1)
        self.glob = nn.Linear(2 * c, 7)

    def forward(self, x):
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        g = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return self.glob(g), self.clic(h)[:, 0]


def onehot(frames: np.ndarray) -> torch.Tensor:
    f = torch.from_numpy(np.asarray(frames, dtype=np.int64))
    if f.dim() == 2:
        f = f[None]
    return F.one_hot(f.clamp(0, 15), 16).permute(0, 3, 1, 2).float()


class ModeleEffet:
    def __init__(self, graine=0, c=16, lr=2e-3, threads=None):
        if threads is not None:
            torch.set_num_threads(int(threads))
        torch.manual_seed(graine)
        self.rng = np.random.default_rng(graine)
        self.net = Reseau(c)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.ex = []            # (indice_image, action, y, x, label)
        self.images = []        # images uniques
        self.index = {}         # hash -> indice image
        self.vus = set()        # (hash, action, y, x)
        self.n_pas = 0
        self.version = 0
        self.nouveaux = 0
        self.pos_rate = 0.5
        self.n_ex = 0

    # ---- données
    def ajouter(self, cle, frame, action, label, y=-1, x=-1):
        """action : 1..7 (simple) ou 6 avec (y,x) pour un clic. Renvoie True si l'exemple est nouveau."""
        k = (cle, action, y, x)
        if k in self.vus:
            return False
        self.vus.add(k)
        if cle not in self.index:
            self.index[cle] = len(self.images)
            self.images.append(np.asarray(frame, dtype=np.int8))
        self.ex.append((self.index[cle], action, y, x, float(label)))
        self.nouveaux += 1
        self.n_ex += 1
        n = len(self.ex)
        self.pos_rate += (float(label) - self.pos_rate) / min(n + 1, 200)
        return True

    # ---- entraînement
    def entrainer(self, pas=2, lot=32, force=False):
        if len(self.ex) < 8 or (self.nouveaux < 8 and not force):
            return None
        self.nouveaux = 0
        pertes = []
        n = len(self.ex)
        # rééquilibrage : mélange 50% d'exemples récents, 50% uniformes
        for _ in range(pas):
            idx = list(self.rng.integers(0, n, lot // 2)) + list(self.rng.integers(max(0, n - 64), n, lot - lot // 2))
            L = [self.ex[i] for i in idx]
            imgs = np.stack([self.images[e[0]] for e in L])
            xin = onehot(imgs)
            lg, lc = self.net(xin)
            loss = 0.0
            tot = 0
            for b, (_, a, y, x, lab) in enumerate(L):
                z = lc[b, y, x] if (a == 6 and y >= 0) else lg[b, a - 1]
                loss = loss + F.binary_cross_entropy_with_logits(z, torch.tensor(lab))
                tot += 1
            loss = loss / tot
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            pertes.append(float(loss.detach()))
        self.n_pas += pas
        self.version += 1
        return float(np.mean(pertes))

    # ---- prédiction
    @torch.no_grad()
    def predire(self, frame):
        """Renvoie (p_simples[7] pour ACTION1..7, carte p_clic[64,64])."""
        lg, lc = self.net(onehot(frame))
        return torch.sigmoid(lg[0]).numpy(), torch.sigmoid(lc[0]).numpy()

    def echantillonner(self, probas, plancher=0.02):
        p = np.asarray(probas, dtype=np.float64) + plancher
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p))

    # ---- (dé)sérialisation (points de contrôle petits)
    def etat(self):
        return {k: v.cpu() for k, v in self.net.state_dict().items()}

    def charger(self, sd):
        self.net.load_state_dict(sd)
