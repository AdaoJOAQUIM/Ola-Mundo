"""Explication d'une transition (image avant -> image après) par des événements locaux :
translation rigide d'un corps, transformation D4 d'un corps (avec translation), apparition / disparition / recoloration.
"""
from __future__ import annotations

import numpy as np

# D4 comme matrices entières agissant sur (r, c) ; indice 0 = identité.
_R = np.array([[0, -1], [1, 0]])
_F = np.array([[1, 0], [0, -1]])


def _gen():
    mats = []
    for f in (0, 1):
        for k in range(4):
            m = np.linalg.matrix_power(_R, k)
            if f:
                m = m @ _F
            mats.append(m)
    return mats


D4 = _gen()
D4_CLE = {tuple(m.flatten()): i for i, m in enumerate(D4)}


def d4_mul(i, j):
    return D4_CLE[tuple((D4[i] @ D4[j]).flatten())]


def d4_inv(i):
    for j in range(8):
        if d4_mul(i, j) == 0:
            return j


def pixels(ents):
    """Tableau (n, 3) [r, c, col] de tous les pixels des entités."""
    out = []
    for e in ents:
        rr, cc = np.nonzero(e.masque)
        out.append(np.stack([rr + e.r, cc + e.c, np.full(len(rr), e.col)], 1))
    return np.concatenate(out) if out else np.zeros((0, 3), int)


def _ens(p):
    return set(map(tuple, p.tolist()))


def trouver_d4(pa, pb):
    """Cherche (i, t) avec pb == D4[i] @ pa + t (couleurs comprises). Renvoie les solutions (i, (tr, tc))."""
    if len(pa) != len(pb) or len(pa) == 0:
        return []
    sb = _ens(pb)
    res = []
    for i in range(8):
        q = pa[:, :2] @ D4[i].T
        # translation : alignement des coins minimaux
        t = pb[:, :2].min(0) - q.min(0)
        cand = np.concatenate([q + t, pa[:, 2:3]], 1)
        if _ens(cand) == sb:
            res.append((i, (int(t[0]), int(t[1]))))
    return res


def expliquer(Ea, Eb, fa=None, fb=None, hud=None):
    """Renvoie un dict : {'type': 'none'|'T'|'D4'|'autre', ...}.
    'T'  : d (dr, dc), idx_a (indices des entités avant qui bougent)
    'D4' : i (élément), t, idx_a, idx_b
    'autre' : listes disparues / apparues / recolorées (i_a, j_b)."""
    # 1. retirer les entités identiques (même couleur, forme, position)
    clb = {}
    for j, e in enumerate(Eb):
        clb.setdefault((e.col, e.kex, e.r, e.c), []).append(j)
    RA = []
    pris = set()
    for i, e in enumerate(Ea):
        l = clb.get((e.col, e.kex, e.r, e.c))
        if l:
            pris.add(l.pop())
        else:
            RA.append(i)
    RB = [j for j in range(len(Eb)) if j not in pris]
    if not RA and not RB:
        return {"type": "none"}
    # 2. vote de translation parmi les paires de même forme
    votes = {}
    for i in RA:
        for j in RB:
            a, b = Ea[i], Eb[j]
            if a.col == b.col and a.kex == b.kex:
                d = (b.r - a.r, b.c - a.c)
                votes[d] = votes.get(d, 0) + a.aire
    best_d = None
    if votes:
        best_d = max(votes.items(), key=lambda kv: kv[1])[0]
    moved_a, moved_b = [], []
    if best_d is not None:
        libre = set(RB)
        for i in RA:
            a = Ea[i]
            for j in list(libre):
                b = Eb[j]
                if b.col == a.col and b.kex == a.kex and (b.r - a.r, b.c - a.c) == best_d:
                    moved_a.append(i); moved_b.append(j); libre.discard(j)
                    break
    rest_a = [i for i in RA if i not in moved_a]
    rest_b = [j for j in RB if j not in moved_b]
    if moved_a and not rest_a and not rest_b:
        return {"type": "T", "d": best_d, "idx_a": moved_a, "idx_b": moved_b}
    # 3. transformation D4 de l'ensemble des entités changées
    if RA and RB:
        pa, pb = pixels([Ea[i] for i in RA]), pixels([Eb[j] for j in RB])
        sol = trouver_d4(pa, pb)
        sol = [s for s in sol if s[0] != 0 or True]
        if sol:
            # préférence : rotations avant symétries, quart de tour avant demi-tour
            ordre = {1: 0, 3: 1, 2: 2, 0: 5, 4: 3, 5: 4, 6: 3, 7: 3}
            sol.sort(key=lambda s: ordre.get(s[0], 9))
            i, t = sol[0]
            return {"type": "D4", "i": i, "t": t, "idx_a": RA, "idx_b": RB, "pa": pa}
    # 4. translation + résidus
    out = {"type": "autre", "d": best_d if moved_a else None, "idx_a": moved_a, "idx_b": moved_b,
           "gone": [], "new": [], "recol": []}
    ra, rb = list(rest_a), list(rest_b)
    for i in list(ra):
        for j in list(rb):
            a, b = Ea[i], Eb[j]
            if (a.r, a.c) == (b.r, b.c) and a.kex == b.kex and a.col != b.col:
                out["recol"].append((i, j)); ra.remove(i); rb.remove(j)
                break
    out["gone"], out["new"] = ra, rb
    if moved_a and fa is not None and fb is not None:
        # explication par pixels : tout ce qui change est-il dans les empreintes (avant + après) du groupe qui se translate ?
        D = (fa != fb)
        if hud is not None:
            D = D & ~hud
        foot = np.zeros(D.shape, bool)
        for i in moved_a:
            e = Ea[i]
            foot[e.r:e.r + e.h, e.c:e.c + e.w] |= e.masque
            r2, c2 = e.r + best_d[0], e.c + best_d[1]
            if r2 >= 0 and c2 >= 0:
                sub = foot[r2:r2 + e.h, c2:c2 + e.w]
                sub |= e.masque[:sub.shape[0], :sub.shape[1]]
        if (D & ~foot).sum() <= max(2, 0.05 * D.sum()):
            out["type"] = "T"
            return out
    if moved_a:
        w_m = sum(Ea[i].aire for i in moved_a)
        w_r = sum(Ea[i].aire for i in ra) + sum(Eb[j].aire for j in rb) + sum(Ea[i].aire for i, _ in out["recol"])
        if w_m >= w_r:                    # la translation domine : les résidus sont d'autres effets (compteur, consommation)
            out["type"] = "T"
    return out
