"""Hypothèses de but génériques (aucun jeu précis) et petit modèle de valeur transférable d'un niveau au suivant.

Chaque hypothèse est un SIGNAL DE PROGRÈS mesuré par rapport à l'image de départ du niveau :
  - clear      : la part de pixels hors fond diminue (« tout effacer ») ;
  - collect    : le nombre de petits objets diminue (« tout collecter ») ;
  - match      : deux zones de même taille, presque identiques, se rapprochent (« reproduire le motif de référence ») ;
  - transform  : des objets changent de forme/couleur (« clé / transformation ») puis des couleurs disparaissent (« porte »).
Le modèle de valeur = régression logistique sur ces progrès + variation d'histogramme de couleurs ; poids initiaux = a priori
faible ; appris à chaque niveau gagné (positifs = derniers états avant la victoire, négatifs = autres états visités).
"""
from __future__ import annotations

from collections import deque

import numpy as np

import percep

H = W = 64
NOMS = ["clear", "collect", "match", "match_fini", "transform", "couleurs_perdues"] + [f"hist{c}" for c in range(16)]
NF = len(NOMS)
W0 = np.zeros(NF)
W0[:6] = [1.0, 1.0, 2.0, 2.0, 0.3, 0.6]
B0 = -2.0
PETIT = 40


def _boites_multicolores(frame, bg, masque):
    """Composantes 8-connexes du non-fond (multicolores) : liste de (y0,x0,y1,x1) inclusifs."""
    nb = frame != bg
    if masque is not None:
        nb = nb & ~masque
    vu = np.zeros_like(nb)
    res = []
    ys, xs = np.nonzero(nb)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if vu[y0, x0]:
            continue
        vu[y0, x0] = True
        q = deque([(y0, x0)])
        mny = mxy = y0
        mnx = mxx = x0
        n = 0
        while q:
            y, x = q.popleft()
            n += 1
            mny, mxy, mnx, mxx = min(mny, y), max(mxy, y), min(mnx, x), max(mxx, x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and nb[ny, nx] and not vu[ny, nx]:
                        vu[ny, nx] = True
                        q.append((ny, nx))
        res.append((mny, mnx, mxy, mxx, n))
    return res


def trouver_paires(frame, bg, masque):
    """Paires de zones (boîtes de même taille, 3..32) presque identiques mais pas identiques : motif de référence."""
    bx = [b for b in _boites_multicolores(frame, bg, masque) if 3 <= b[2] - b[0] + 1 <= 32 and 3 <= b[3] - b[1] + 1 <= 32][:30]
    paires = []
    for i in range(len(bx)):
        for j in range(i + 1, len(bx)):
            a, b = bx[i], bx[j]
            if (a[2] - a[0], a[3] - a[1]) != (b[2] - b[0], b[3] - b[1]):
                continue
            h, w = a[2] - a[0] + 1, a[3] - a[1] + 1
            if h * w < 9:
                continue
            A = frame[a[0]:a[2] + 1, a[1]:a[3] + 1]
            B = frame[b[0]:b[2] + 1, b[1]:b[3] + 1]
            eq = float((A == B).mean())
            if 0.35 <= eq < 0.999:
                paires.append((eq, a[:4], b[:4]))
    paires.sort(key=lambda t: -t[0])
    return paires[:6]


def analyser(frame, masque):
    """Objets, fond, histogramme (hors HUD) d'une image."""
    objets, bg = percep.composantes(frame, masque)
    f = frame[~masque] if masque is not None and masque.any() else frame.ravel()
    hist = np.bincount(f.ravel().astype(np.int64), minlength=16)[:16]
    return {"objets": objets, "bg": bg, "hist": hist,
            "nb": int(sum(o["taille"] for o in objets)),
            "petits": sum(1 for o in objets if o["taille"] <= PETIT),
            "formes": sorted((o["couleur"], o["taille"]) for o in objets)}


class Racine:
    """Description de l'image de départ d'un niveau, pour mesurer le progrès."""

    def __init__(self, frame, masque):
        self.frame = frame.copy()
        self.an = analyser(frame, masque)
        self.paires = trouver_paires(frame, self.an["bg"], masque)
        self.couleurs = set(np.nonzero(self.an["hist"])[0].tolist())


def caracteristiques(frame, an, racine: Racine, masque):
    nb0 = max(racine.an["nb"], 1)
    f = np.zeros(NF)
    f[0] = (racine.an["nb"] - an["nb"]) / nb0
    f[1] = (racine.an["petits"] - an["petits"]) / max(racine.an["petits"], 1)
    eq_max, fini = 0.0, 0.0
    for eq0, a, b in racine.paires:
        A = frame[a[0]:a[2] + 1, a[1]:a[3] + 1]
        B = frame[b[0]:b[2] + 1, b[1]:b[3] + 1]
        eq = float((A == B).mean())
        eq_max = max(eq_max, eq - eq0)
        if eq >= 0.999:
            fini = 1.0
    f[2], f[3] = eq_max, fini
    ch = (frame != racine.frame)
    if masque is not None and masque.any():
        ch &= ~masque
    f[4] = min(ch.mean() * 10, 1.0)
    cols = set(np.nonzero(an["hist"])[0].tolist())
    f[5] = len(racine.couleurs - cols) / 4.0
    f[6:] = np.clip((an["hist"] - racine.an["hist"]) / 4096.0 * 8, -1, 1)
    return f


class ModeleValeur:
    """Régression logistique ; poids conservés d'un niveau à l'autre."""

    def __init__(self, actif=True):
        self.w = W0.copy()
        self.b = B0
        self.actif = actif
        self.n_maj = 0

    def valeur(self, f):
        if not self.actif:
            return 0.0
        z = float(self.w @ f + self.b)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def apprendre(self, positifs, negatifs, pas=200, lr=0.3, l2=0.02):
        if not self.actif or not positifs:
            return
        X = np.array(positifs + negatifs)
        y = np.array([1.0] * len(positifs) + [0.0] * len(negatifs))
        sw = np.where(y > 0, max(1.0, len(negatifs) / max(len(positifs), 1)) * 0.5, 1.0)
        w, b = self.w.copy(), self.b
        for _ in range(pas):
            p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -20, 20)))
            g = (p - y) * sw
            w -= lr * ((X.T @ g) / len(y) + l2 * (w - W0))
            b -= lr * g.mean()
        self.w, self.b = w, b
        self.n_maj += 1
