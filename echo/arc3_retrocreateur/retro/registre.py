"""Registre : variable cachée affichée par un motif (glyphe), et but « motif égal à la cible ».

Primitive générale (voir ARCHITECTURE.md, section J3) : un glyphe est un petit cadre de couleur unie contenant un motif
n x n (à l'échelle s). Deux glyphes de même palette dont l'un est l'image de l'autre par une isométrie du carré
(groupe D4 : rotations et miroirs) forment une paire « registre / cible » : le registre change quand l'avatar entre sur
certains déclencheurs (la transformation appliquée est un élément de D4, apprise à la première observation), la cible
reste fixe. Le but est de mettre le registre dans l'état de la cible.

Aucune donnée de jeu : seuls des cadres, des couleurs et des symétries du carré.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ groupe D4 sur les carrés n x n
def _elements():
    """8 éléments : (miroir, quarts de tour). Image = rot90(fliplr(a) si miroir sinon a, k)."""
    return [(m, k) for m in (0, 1) for k in range(4)]


ELEMS = _elements()


def appliquer(e, a):
    m, k = ELEMS[e]
    b = np.fliplr(a) if m else a
    return np.rot90(b, k)


def _table():
    t = np.arange(9).reshape(3, 3)
    imgs = [appliquer(e, t).tobytes() for e in range(8)]
    # compose[g][h] = élément x tel que appliquer(x, a) == appliquer(g, appliquer(h, a))
    comp = [[0] * 8 for _ in range(8)]
    for g in range(8):
        for h in range(8):
            r = appliquer(g, appliquer(h, t)).tobytes()
            comp[g][h] = imgs.index(r)
    return comp


COMP = _table()


def composer(g, h):
    """g après h."""
    return COMP[g][h]


def elements_vers(a, b):
    """Éléments e de D4 tels que appliquer(e, a) == b (liste, éventuellement vide ; plusieurs si motif symétrique)."""
    if a.shape != b.shape or a.shape[0] != a.shape[1]:
        return []
    return [e for e in range(8) if np.array_equal(appliquer(e, a), b)]


# ------------------------------------------------------------------ extraction de glyphes
def _composantes(masque):
    """Composantes 4-connexes d'un masque booléen : liste de (ys, xs)."""
    H, W = masque.shape
    vu = np.zeros_like(masque)
    out = []
    for y0, x0 in zip(*np.nonzero(masque)):
        if vu[y0, x0]:
            continue
        pile = [(y0, x0)]
        vu[y0, x0] = True
        ys, xs = [], []
        while pile:
            y, x = pile.pop()
            ys.append(y)
            xs.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p, q = y + dy, x + dx
                if 0 <= p < H and 0 <= q < W and masque[p, q] and not vu[p, q]:
                    vu[p, q] = True
                    pile.append((p, q))
        out.append((np.array(ys), np.array(xs)))
    return out


def _reduire(a):
    """Plus grande échelle s telle que chaque bloc s x s soit uniforme."""
    h, w = a.shape
    for s in range(min(h, w), 1, -1):
        if h % s or w % s:
            continue
        b = a[::s, ::s]
        if np.array_equal(np.kron(b, np.ones((s, s), dtype=a.dtype)), a):
            return b, s
    return a, 1


def glyphes(frame, cote_min=6, cote_max=16):
    """Cadres carrés (couleur unie, bord >= 1 pixel) contenant un motif carré n x n (2 <= n <= 6, au moins deux valeurs).
    Renvoie une liste de dict(bbox=(y0, x0, y1, x1), arr=ndarray n x n)."""
    res = []
    for col in np.unique(frame).tolist():
        m = frame == col
        n = int(m.sum())
        if n < 12:
            continue
        for ys, xs in _composantes(m):
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            h, w = y1 - y0 + 1, x1 - x0 + 1
            if not (cote_min <= h <= cote_max and cote_min <= w <= cote_max and abs(h - w) <= 0):
                continue
            blk = frame[y0:y1 + 1, x0:x1 + 1]
            mb = m[y0:y1 + 1, x0:x1 + 1]
            # cadre : les quatre bords du rectangle sont presque entièrement de la couleur du cadre
            bords = [mb[0, :], mb[-1, :], mb[:, 0], mb[:, -1]]
            if min(b.mean() for b in bords) < 0.9:
                continue
            # épaisseur du bord : nombre de lignes pleines en tête, au minimum des quatre côtés
            def epais(v):
                k = 0
                while k < len(v) and v[k].all():
                    k += 1
                return k
            e = min(epais(mb), epais(mb[::-1]), epais(mb.T), epais(mb.T[::-1]))
            if e < 1 or h - 2 * e < 2:
                continue
            inner = blk[e:h - e, e:w - e]
            if len(np.unique(inner)) < 2:
                continue
            arr, s = _reduire(inner)
            if arr.shape[0] != arr.shape[1] or not (2 <= arr.shape[0] <= 6):
                continue
            res.append({"bbox": (int(y0), int(x0), int(y1), int(x1)), "arr": arr.copy(), "echelle": s, "bord": int(e)})
    # un cadre à l'intérieur d'un autre cadre de même cote n'est pas doublé
    uniq = []
    for g in res:
        if not any(g is not h and h["bbox"] == g["bbox"] for h in uniq):
            uniq.append(g)
    return uniq


def lire_glyphe(frame, bbox, bord, echelle):
    """Relit le motif d'un glyphe à emplacement connu."""
    y0, x0, y1, x1 = bbox
    inner = frame[y0 + bord:y1 + 1 - bord, x0 + bord:x1 + 1 - bord]
    return inner[::echelle, ::echelle].copy()


def paires(gl):
    """Paires (i, j) de glyphes de même palette dont l'un est l'image de l'autre par un élément de D4 (i < j)."""
    res = []
    for i in range(len(gl)):
        for j in range(i + 1, len(gl)):
            a, b = gl[i]["arr"], gl[j]["arr"]
            if a.shape == b.shape and sorted(np.unique(a).tolist()) == sorted(np.unique(b).tolist()) \
                    and elements_vers(a, b):
                res.append((i, j))
    return res


class Registre:
    """Suivi d'un niveau : glyphes appariés, registre (celui qui change), mode courant (élément de D4), cibles.

    mode : élément de D4 tel que contenu courant du registre = appliquer(mode, contenu initial).
    cibles : ensemble de modes pour lesquels le registre est égal à la cible.
    """

    def __init__(self, frame, prefere=None):
        self.gl = glyphes(frame)
        self.pr = paires(self.gl)
        self.actif = bool(self.pr)
        self.reg = None                  # indice du glyphe registre (une fois qu'un changement a été vu)
        self.init = {i: self.gl[i]["arr"].copy() for i in range(len(self.gl))}
        self.mode = 0
        self.dernier = {i: self.gl[i]["arr"].copy() for i in range(len(self.gl))}
        if prefere is not None:
            # le registre garde sa place à l'écran d'un niveau à l'autre (les cibles, elles, changent avec la mise en page)
            for i in self.indices():
                if self.gl[i]["bbox"] == prefere:
                    self.reg = i

    def indices(self):
        s = set()
        for i, j in self.pr:
            s.update((i, j))
        return s

    def cibles(self):
        """Modes qui amènent le registre sur la cible ; None si le registre n'est pas encore identifié."""
        if self.reg is None:
            return None
        autres = [j if i == self.reg else i for i, j in self.pr if self.reg in (i, j)]
        res = set()
        for t in autres:
            res.update(elements_vers(self.init[self.reg], self.dernier[t]))
        return res or None

    def maj(self, frame):
        """Met à jour d'après une nouvelle image. Renvoie le changement de mode (élément g tel que nouveau = g o ancien)
        s'il y en a un, sinon None."""
        if not self.actif:
            return None
        cur = {}
        for i in self.indices():
            g = self.gl[i]
            cur[i] = lire_glyphe(frame, g["bbox"], g["bord"], g["echelle"])
        chg = [i for i in cur if not np.array_equal(cur[i], self.dernier[i])]
        res = None
        if len(chg) == 1 and (self.reg is None or chg[0] == self.reg):
            i = chg[0]
            if self.reg is None:
                self.reg = i
            old_img = self.dernier[i]
            cands = elements_vers(old_img, cur[i])
            if cands:
                # plus petit élément non trivial d'abord (rotation avant miroir), pour une description courte
                g = min(cands, key=lambda e: (e == 0, ELEMS[e][0], ELEMS[e][1]))
                self.mode = composer(g, self.mode)
                res = g
        for i in cur:
            self.dernier[i] = cur[i]
        return res
