"""Perception sans grille : entités = composantes connexes de couleur, attributs, appariement entre images.

Aucun a priori de réseau régulier. Attributs : couleur, position (haut-gauche), taille, masque, forme canonique D4.
"""
from __future__ import annotations

import numpy as np


def _d4(m):
    out = []
    for k in range(4):
        r = np.rot90(m, k)
        out.append(r)
        out.append(r[:, ::-1])
    return out


def cle_forme(masque):
    """Clé exacte (même orientation) et clé canonique D4 d'un masque booléen."""
    ex = (masque.shape, masque.tobytes())
    best = None
    for v in _d4(masque):
        k = (v.shape, np.ascontiguousarray(v).tobytes())
        if best is None or k < best:
            best = k
    return hash(ex), hash(best)


class Ent:
    __slots__ = ("col", "r", "c", "h", "w", "masque", "aire", "kex", "kd4", "id", "corps")

    def __init__(self, col, r, c, masque):
        self.col = int(col)
        self.r, self.c = int(r), int(c)
        self.h, self.w = masque.shape
        self.masque = masque
        self.aire = int(masque.sum())
        self.kex, self.kd4 = cle_forme(masque)
        self.id = -1
        self.corps = -1

    @property
    def centre(self):
        return (self.r + (self.h - 1) / 2.0, self.c + (self.w - 1) / 2.0)

    def bbox(self):
        return (self.r, self.c, self.r + self.h, self.c + self.w)

    def __repr__(self):
        return f"E{self.id}(col={self.col} @({self.r},{self.c}) {self.h}x{self.w} a={self.aire})"


def composantes(frame, bg, ignorer=None, connexite=4):
    """Composantes connexes monochromes hors couleur de fond `bg` ; `ignorer` : masque booléen de pixels exclus."""
    H, W = frame.shape
    vu = np.zeros((H, W), bool)
    if ignorer is not None:
        vu |= ignorer
    vu |= (frame == bg)
    res = []
    voisins = ((1, 0), (-1, 0), (0, 1), (0, -1)) if connexite == 4 else \
        ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    fr = frame.tolist()
    vul = vu.tolist()
    for r0 in range(H):
        for c0 in range(W):
            if vul[r0][c0]:
                continue
            col = fr[r0][c0]
            pile = [(r0, c0)]
            vul[r0][c0] = True
            pts = []
            while pile:
                r, c = pile.pop()
                pts.append((r, c))
                for dr, dc in voisins:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and not vul[rr][cc] and fr[rr][cc] == col:
                        vul[rr][cc] = True
                        pile.append((rr, cc))
            rs = [p[0] for p in pts]; cs = [p[1] for p in pts]
            ra, ca = min(rs), min(cs)
            m = np.zeros((max(rs) - ra + 1, max(cs) - ca + 1), bool)
            for r, c in pts:
                m[r - ra, c - ca] = True
            e = Ent(col, ra, ca, m)
            res.append(e)
    for i, e in enumerate(res):
        e.id = i
    return res


def couleur_fond(frame):
    v, n = np.unique(frame, return_counts=True)
    return int(v[np.argmax(n)])


def deplacement_commun(avant, apres, tol_aire=0):
    """Apparie les entités de deux images : même couleur et même forme exacte, plus proche (distance L1 du coin).
    Renvoie (paires[(i,j)], non_appariees_avant, non_appariees_apres)."""
    par_cle = {}
    for j, e in enumerate(apres):
        par_cle.setdefault((e.col, e.kex), []).append(j)
    paires, pris = [], set()
    # d'abord les immobiles (distance 0), puis les plus proches
    cand = []
    for i, e in enumerate(avant):
        for j in par_cle.get((e.col, e.kex), []):
            f = apres[j]
            cand.append((abs(e.r - f.r) + abs(e.c - f.c), i, j))
    cand.sort()
    ua, ub = set(), set()
    for d, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j)
        paires.append((i, j))
    na = [i for i in range(len(avant)) if i not in ua]
    nb = [j for j in range(len(apres)) if j not in ub]
    return paires, na, nb


def composites(frame, bg, ignorer=None, aire_max=1500):
    """Objets composites : composantes 8-connexes multicolores hors fond (un corps articulé = plusieurs couleurs collées).
    Renvoie des dicts {pts (n,2), cols (n,)}."""
    H, W = frame.shape
    vu = np.zeros((H, W), bool)
    if ignorer is not None:
        vu |= ignorer
    vu |= (frame == bg)
    fr = frame
    res = []
    for r0 in range(H):
        for c0 in range(W):
            if vu[r0, c0]:
                continue
            pile = [(r0, c0)]
            vu[r0, c0] = True
            pts = []
            while pile:
                r, c = pile.pop()
                pts.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W and not vu[rr, cc]:
                            vu[rr, cc] = True
                            pile.append((rr, cc))
            if len(pts) <= aire_max:
                p = np.array(pts)
                res.append({"pts": p, "cols": fr[p[:, 0], p[:, 1]].copy()})
    return res
