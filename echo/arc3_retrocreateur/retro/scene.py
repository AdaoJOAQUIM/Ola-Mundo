"""Scène : du cadre 64x64 à la grille de types (réseau de cases, signatures, avatar)."""
from __future__ import annotations

import numpy as np

from .reseau import Types, construire_reseau, signature_case

PAD = 2


class Geo:
    """Géométrie d'un niveau : grille R x C (étendue + marge), voisins, cases ignorées (bandeau), clic -> pixel."""

    def __init__(self, net, R, C, gx0, gy0, ignore, outer):
        self.net, self.R, self.C = net, R, C
        self.gx0, self.gy0 = gx0, gy0            # indices de case (réseau) de la case (0,0) de la grille
        self.N = R * C
        self.ignore = ignore                      # frozenset d'indices : jamais franchissables, jamais lus
        self.outer = outer                        # frozenset : marge vide hors de l'étendue observée
        nb = []
        for p in range(self.N):
            r, c = divmod(p, C)
            row = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                row.append(rr * C + cc if 0 <= rr < R and 0 <= cc < C else -1)
            nb.append(tuple(row))
        self.nb = nb
        self.dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
        self.cle = (net.cle(), R, C, gx0, gy0, self.ignore, self.outer)

    def voisin(self, p, d):
        return self.nb[p][d]

    def pixel(self, p):
        r, c = divmod(p, self.C)
        n = self.net
        return (n.px + (self.gx0 + c) * n.c + n.c // 2, n.py + (self.gy0 + r) * n.c + n.c // 2)


class Scene:
    def __init__(self, types: Types):
        self.types = types
        self.av_class = None
        self.av_colors = []        # couleurs connues de l'avatar (la 1re = couleur initiale)
        self.geo = None
        self._cache = {}
        self.sig_av = {}           # id brut -> (couleur, classe) de la partie avatar, ou None
        self.av_parts = set()      # (couleur, classe) reconnues comme avatar (variantes : fusion avec une case de même couleur)

    # ------------------------------------------------------------ niveau
    def nouveau_niveau(self, frame):
        net = construire_reseau([frame])
        if net is None:
            self.geo = None
            return None
        c = net.c
        gx0, gy0 = net.gx0 - PAD, net.gy0 - PAD
        gx1, gy1 = net.gx0 + net.C - 1 + PAD, net.gy0 + net.R - 1 + PAD
        # reste dans l'image
        while net.px + gx0 * c < 0:
            gx0 += 1
        while net.py + gy0 * c < 0:
            gy0 += 1
        while net.px + (gx1 + 1) * c > 64:
            gx1 -= 1
        while net.py + (gy1 + 1) * c > 64:
            gy1 -= 1
        C, R = gx1 - gx0 + 1, gy1 - gy0 + 1
        cx0, cy0, cx1, cy1 = net.gx0, net.gy0, net.gx0 + net.C - 1, net.gy0 + net.R - 1
        ignore, outer = set(), set()
        bg = net.bg
        for r in range(R):
            for q in range(C):
                gx, gy = gx0 + q, gy0 + r
                if cx0 <= gx <= cx1 and cy0 <= gy <= cy1:
                    continue
                y0, x0 = net.py + gy * c, net.px + gx * c
                if (frame[y0:y0 + c, x0:x0 + c] != bg).any():
                    ignore.add(r * C + q)
                else:
                    outer.add(r * C + q)
        self._cache = {}
        self.geo = Geo(net, R, C, gx0, gy0, frozenset(ignore), frozenset(outer))
        return self.geo

    # ------------------------------------------------------------ lecture
    def lire(self, frame):
        """Liste des identifiants bruts (signature de chaque case) pour la grille courante."""
        g = self.geo
        net = g.net
        c, bg = net.c, net.bg
        out = []
        for r in range(g.R):
            y0 = net.py + (g.gy0 + r) * c
            for q in range(g.C):
                x0 = net.px + (g.gx0 + q) * c
                blk = frame[y0:y0 + c, x0:x0 + c]
                k = (blk.tobytes(), bg)
                i = self._cache.get(k)
                if i is None:
                    i = self.types.id(signature_case(blk, bg))
                    self._cache[k] = i
                out.append(i)
        return out

    def partie_avatar(self, rid):
        """(couleur, classe) de la partie avatar d'un identifiant brut s'il en contient une, sinon None."""
        if self.av_class is None:
            return None
        v = self.sig_av.get(rid, 0)
        if v == 0:
            v = None
            for col, cl in self.types.inv[rid]:
                if (cl == self.av_class and col in self.av_colors) or (col, cl) in self.av_parts:
                    v = (col, cl)
                    break
            self.sig_av[rid] = v
        return v

    def reset_av(self):
        self.sig_av = {}

    def cellules_avatar(self, raw):
        res = []
        for p, rid in enumerate(raw):
            if p in self.geo.ignore:
                continue
            if self.partie_avatar(rid) is not None:
                res.append(p)
        return res

    def reste(self, rid, partie):
        """Signature restante d'une case une fois la partie avatar retirée -> identifiant de type."""
        sg = tuple(x for x in self.types.inv[rid] if x != partie)
        return self.types.id(sg)

    def ajouter_part(self, part):
        if part not in self.av_parts:
            self.av_parts.add(part)
            self.reset_av()

    def ajouter_couleur_av(self, col):
        if col not in self.av_colors:
            self.av_colors.append(col)
            self.reset_av()

    # ------------------------------------------------------------ amorçage de l'avatar
    def trouver_mobile(self, raw0, raw1):
        """Parties qui ont bougé d'une case à une case voisine : liste (partie, p, q). L'avatar est la tête de chaîne."""
        g = self.geo
        diff = [p for p in range(g.N) if raw0[p] != raw1[p] and p not in g.ignore]
        if not diff or len(diff) > 8:
            return []
        inv = self.types.inv
        res = []
        for p in diff:
            lost = set(inv[raw0[p]]) - set(inv[raw1[p]])
            for d in range(4):
                q = g.nb[p][d]
                if q < 0 or q not in diff:
                    continue
                gained = set(inv[raw1[q]]) - set(inv[raw0[q]])
                for e in lost & gained:
                    res.append((e, p, q))
        if not res:
            return []
        dest = {q for _, _, q in res}
        tete = [x for x in res if x[1] not in dest]
        return tete or res
