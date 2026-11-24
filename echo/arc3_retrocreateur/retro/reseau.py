"""Perception de la grille de cases : réseau (période c, phases), signature des cases invariante à l'échelle,
table des types. Réutilise la détection de composantes de la v1."""
from __future__ import annotations

import numpy as np

from .v1 import percep_v1

HAUT_HUD = None


def comps_bbox(frame, conn8=True):
    objs, bg = percep_v1.composantes(frame, connexite8=conn8)
    return [o["bbox"] for o in objs], bg


def _score_axe(boites, ax, c, p):
    """Part de boîtes cohérentes avec un réseau de période c et phase p sur l'axe ax (0 = x, 1 = y)."""
    ok = 0
    n = 0
    z0 = 0
    for b in boites:
        a, e = (b[0], b[2]) if ax == 0 else (b[1], b[3])
        if min(b[2] - b[0], b[3] - b[1]) + 1 < c - 2:
            continue
        n += 1
        ra = (a - p) % c
        re = (e + 1 - p) % c
        if ra in (0, 1) and re in (0, c - 1):
            ok += 1
            z0 += (ra == 0) + (re == 0)
    return ok, n, z0


def estimer_reseau(frames, cmax=14, cmin=4, seuil=0.85):
    """Cherche la plus grande période c (puis phases px, py) cohérente avec les composantes de `frames`.
    Renvoie (c, px, py, fond) ou None."""
    boites = []
    bg = 0
    for f in frames:
        b, bg = comps_bbox(f)
        boites += b
    for c in range(cmax, cmin - 1, -1):
        meilleur = None
        for ax in (0, 1):
            cand = None
            for p in range(c):
                ok, n, z0 = _score_axe(boites, ax, c, p)
                if n >= 3 and ok / n >= seuil:
                    k = (ok, z0)
                    if cand is None or k > cand[0]:
                        cand = (k, p, ok / n, n)
            if cand is None:
                meilleur = None
                break
            meilleur = (meilleur or []) + [cand]
        if meilleur:
            return c, meilleur[0][1], meilleur[1][1], bg
    return None


# ----------------------------------------------------------------- signature des cases
def _classe(m, c):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    ext = max(w, h) / c
    e = 0 if m.all() else (1 if ext >= 0.55 else 2)
    cm = c // 2
    cen = int(m[cm - 1:cm + 1, cm - 1:cm + 1].any()) if c % 2 == 0 else int(m[cm, cm])
    cor = int(m[ys.min(), xs.min()])
    # composantes 4-connexes (au plus 3)
    vu = np.zeros_like(m)
    nc = 0
    for y, x in zip(ys.tolist(), xs.tolist()):
        if vu[y, x]:
            continue
        nc += 1
        if nc >= 3:
            break
        pile = [(y, x)]
        vu[y, x] = True
        while pile:
            a, b = pile.pop()
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p, q = a + da, b + db
                if 0 <= p < c and 0 <= q < c and m[p, q] and not vu[p, q]:
                    vu[p, q] = True
                    pile.append((p, q))
    # fraction des pixels sur la croix médiane : distingue croix et diamant
    mid = m[cm, :].sum() + m[:, cm].sum() - int(m[cm, cm])
    ax = int(mid >= 0.8 * m.sum())
    return (e, cen, cor, nc, ax)


def signature_case(bloc, bg):
    """Signature d'un bloc c x c : tuple trié de (couleur, classe de forme) ; () = case vide."""
    out = []
    for col in np.unique(bloc).tolist():
        if col == bg:
            continue
        cl = _classe(bloc == col, bloc.shape[0])
        out.append((int(col), cl))
    return tuple(sorted(out))


class Types:
    """Table signature -> identifiant de type (0 = case vide). Persistante d'un niveau à l'autre."""

    def __init__(self):
        self.sig = {(): 0}
        self.inv = [()]

    def id(self, sg):
        i = self.sig.get(sg)
        if i is None:
            i = len(self.inv)
            self.sig[sg] = i
            self.inv.append(sg)
        return i

    def couleurs(self, t):
        return tuple(c for c, _ in self.inv[t])


# ----------------------------------------------------------------- scène
class Reseau:
    """Réseau de cases : période c, phases (px, py), étendue (gx0, gy0, C, R) en indices de cases."""

    def __init__(self, c, px, py, bg, gx0, gy0, C, R):
        self.c, self.px, self.py, self.bg = c, px, py, bg
        self.gx0, self.gy0, self.C, self.R = gx0, gy0, C, R

    def cle(self):
        return (self.c, self.px, self.py, self.gx0, self.gy0, self.C, self.R)

    def blocs(self, frame):
        """Liste (R*C) de blocs c x c."""
        c = self.c
        out = []
        for r in range(self.R):
            y0 = self.py + (self.gy0 + r) * c
            for q in range(self.C):
                x0 = self.px + (self.gx0 + q) * c
                out.append(frame[y0:y0 + c, x0:x0 + c])
        return out


def construire_reseau(frames, rec=None):
    """Estime le réseau et l'étendue du plateau à partir des composantes cohérentes avec le réseau."""
    est = estimer_reseau(frames)
    if est is None:
        return None
    c, px, py, bg = est
    xs0, xs1, ys0, ys1 = [], [], [], []
    for f in frames:
        for b in comps_bbox(f)[0]:
            w, h = b[2] - b[0] + 1, b[3] - b[1] + 1
            if min(w, h) < c - 2:
                continue
            ra, re = (b[0] - px) % c, (b[2] + 1 - px) % c
            sa, se = (b[1] - py) % c, (b[3] + 1 - py) % c
            if ra in (0, 1) and re in (0, c - 1) and sa in (0, 1) and se in (0, c - 1):
                xs0.append((b[0] - px) // c)
                xs1.append((b[2] - px) // c)
                ys0.append((b[1] - py) // c)
                ys1.append((b[3] - py) // c)
    if not xs0:
        return None
    gx0, gx1, gy0, gy1 = min(xs0), max(xs1), min(ys0), max(ys1)
    # objets plus petits qu'une case (points, croix fines) : ils comptent s'ils sont centrés dans une case
    # voisine de l'étendue (le bandeau d'information, lui, n'est en général pas centré sur le réseau)
    ext = [gx0, gx1, gy0, gy1]
    for f in frames:
        for b in comps_bbox(f)[0]:
            w, h = b[2] - b[0] + 1, b[3] - b[1] + 1
            if w > c or h > c:
                continue
            q0, q1 = (b[0] - px) // c, (b[2] - px) // c
            r0, r1 = (b[1] - py) // c, (b[3] - py) // c
            if q0 != q1 or r0 != r1:
                continue
            ccx, ccy = px + q0 * c + (c - 1) / 2, py + r0 * c + (c - 1) / 2
            if abs((b[0] + b[2]) / 2 - ccx) > 1.0 or abs((b[1] + b[3]) / 2 - ccy) > 1.0:
                continue
            if True:
                ext[0], ext[1] = min(ext[0], q0), max(ext[1], q0)
                ext[2], ext[3] = min(ext[2], r0), max(ext[3], r0)
    gx0, gx1, gy0, gy1 = ext
    return Reseau(c, px, py, bg, gx0, gy0, gx1 - gx0 + 1, gy1 - gy0 + 1)
