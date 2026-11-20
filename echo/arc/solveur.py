#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Solveur ARC-AGI-1 sans paramètre appris, par recherche de programmes.

Bibliothèque standard Python uniquement : pas de numpy, pas de réseau de
neurones, aucun poids appris, aucun appel à un LLM.

Principe
--------
Une grille est un tuple de tuples d'entiers (0..9).  Un « programme » est une
composition de primitives grille -> grille (DSL ci-dessous), éventuellement
terminée par un « finisseur » inféré sur les exemples (table de couleurs).
Un programme n'est accepté que s'il reproduit EXACTEMENT toutes les sorties
d'entraînement.  Jusqu'à 2 programmes distincts (donnant des sorties de test
différentes) sont conservés : ce sont les 2 tentatives (règle ARC-AGI).

Familles de solveurs (essayées dans cet ordre, de la plus simple à la plus
coûteuse, avec un budget CPU par tâche) :
  1. sortie constante ;
  2. recherche en largeur de compositions de primitives (profondeur <= 3),
     avec dédoublonnage des états intermédiaires et vérification paresseuse ;
  3. règle locale cellule-à-cellule apprise (table de voisinage) ;
  4. recoloriage d'objets selon une propriété apprise (taille, trous, ...) ;
  5. complétion de symétrie (cellules « masquées ») ;
  6. méta-opérateurs par objet (transformation appliquée à chaque objet).

Usage CLI :
    python3 solveur.py training | evaluation [--budget 10] [--procs 2]
    python3 solveur.py --tache <id> [--split training]
"""
import json
import os
import sys
import time
import signal
import itertools
from collections import Counter, defaultdict
from functools import lru_cache

M = 10  # couleur « marqueur » : remplacée ensuite par la table de couleurs inférée
DOSSIER = os.path.dirname(os.path.abspath(__file__))


class Delai(Exception):
    """Levée quand le budget CPU est épuisé."""


# --------------------------------------------------------------------------
# 1. Utilitaires de grilles
# --------------------------------------------------------------------------

def H(g):
    return len(g)


def W(g):
    return len(g[0])


def en_grille(l):
    return tuple(tuple(r) for r in l)


@lru_cache(maxsize=200000)
def fond(g):
    """Couleur de fond = couleur la plus fréquente (à égalité, 0 d'abord)."""
    cnt = Counter(v for r in g for v in r)
    return max(cnt.items(), key=lambda kv: (kv[1], kv[0] == 0))[0]


def comptes(g, sans_fond=True):
    cnt = Counter(v for r in g for v in r)
    if sans_fond:
        cnt.pop(fond(g), None)
    return cnt


def rot90(g):
    return tuple(zip(*g[::-1]))


def rot180(g):
    return tuple(r[::-1] for r in g[::-1])


def rot270(g):
    return tuple(zip(*g))[::-1]


def fliph(g):
    return tuple(r[::-1] for r in g)


def flipv(g):
    return g[::-1]


def transp(g):
    return tuple(zip(*g))


def antitransp(g):
    return rot180(transp(g))


def hcat(*gs):
    if len({len(x) for x in gs}) != 1:
        return None
    return tuple(sum((x[i] for x in gs), ()) for i in range(len(gs[0])))


def vcat(*gs):
    if len({len(x[0]) for x in gs}) != 1:
        return None
    return tuple(r for x in gs for r in x)


def sous_grille(g, r0, c0, r1, c1):
    return tuple(row[c0:c1 + 1] for row in g[r0:r1 + 1])


class Obj:
    """Objet = composante connexe. cells: liste de (r,c)."""
    __slots__ = ('cells', 'color', 'r0', 'c0', 'r1', 'c1', 'size', 'ens')

    def __init__(self, cells, color):
        self.cells = cells
        self.color = color
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        self.r0, self.r1, self.c0, self.c1 = min(rs), max(rs), min(cs), max(cs)
        self.size = len(cells)
        self.ens = None

    @property
    def h(self):
        return self.r1 - self.r0 + 1

    @property
    def w(self):
        return self.c1 - self.c0 + 1

    def forme(self):
        return frozenset((r - self.r0, c - self.c0) for r, c in self.cells)


V4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
V8 = V4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


@lru_cache(maxsize=100000)
def objets(g, kind):
    """kind: 'c4','c8' (par couleur, hors fond) ; 'm4','m8' (multicolore hors fond)."""
    h, w = H(g), W(g)
    b = fond(g)
    voisins = V4 if kind[1] == '4' else V8
    multi = kind[0] == 'm'
    vu = [[False] * w for _ in range(h)]
    res = []
    for r in range(h):
        for c in range(w):
            if vu[r][c] or g[r][c] == b:
                continue
            col = g[r][c]
            pile = [(r, c)]
            vu[r][c] = True
            cells = []
            while pile:
                y, x = pile.pop()
                cells.append((y, x))
                for dy, dx in voisins:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not vu[yy][xx]:
                        v = g[yy][xx]
                        if v != b and (multi or v == col):
                            vu[yy][xx] = True
                            pile.append((yy, xx))
            cells.sort()
            res.append(Obj(cells, None if multi else col))
    return tuple(res)


@lru_cache(maxsize=100000)
def trous_bg(g):
    """Ensemble des cellules de fond non reliées (4-conn.) au bord par du fond."""
    h, w = H(g), W(g)
    b = fond(g)
    vu = [[False] * w for _ in range(h)]
    pile = []
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and g[r][c] == b:
                vu[r][c] = True
                pile.append((r, c))
    while pile:
        y, x = pile.pop()
        for dy, dx in V4:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not vu[yy][xx] and g[yy][xx] == b:
                vu[yy][xx] = True
                pile.append((yy, xx))
    return frozenset((r, c) for r in range(h) for c in range(w)
                     if g[r][c] == b and not vu[r][c])


def nb_trous(g, o):
    """Nombre de régions de fond fermées à l'intérieur de la bbox d'un objet."""
    sub = sous_grille(g, o.r0, o.c0, o.r1, o.c1)
    cells = set(o.cells)
    h, w = o.h, o.w
    mur = [[(o.r0 + r, o.c0 + c) in cells for c in range(w)] for r in range(h)]
    vu = [[False] * w for _ in range(h)]
    pile = []
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not mur[r][c]:
                vu[r][c] = True
                pile.append((r, c))
    while pile:
        y, x = pile.pop()
        for dy, dx in V4:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and not vu[yy][xx] and not mur[yy][xx]:
                vu[yy][xx] = True
                pile.append((yy, xx))
    n = 0
    for r in range(h):
        for c in range(w):
            if not mur[r][c] and not vu[r][c]:
                n += 1
                vu[r][c] = True
                pile = [(r, c)]
                while pile:
                    y, x = pile.pop()
                    for dy, dx in V4:
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and not vu[yy][xx] and not mur[yy][xx]:
                            vu[yy][xx] = True
                            pile.append((yy, xx))
    return n


def touche_bord(g, o):
    return o.r0 == 0 or o.c0 == 0 or o.r1 == H(g) - 1 or o.c1 == W(g) - 1


def rogner(g, o, seul):
    """Découpe la bbox d'un objet ; seul=True : n'y garde que les cellules de l'objet."""
    if not seul:
        return sous_grille(g, o.r0, o.c0, o.r1, o.c1)
    b = fond(g)
    cells = set(o.cells)
    return tuple(tuple(g[r][c] if (r, c) in cells else b
                       for c in range(o.c0, o.c1 + 1)) for r in range(o.r0, o.r1 + 1))


def peindre(g, cells, couleur):
    out = [list(r) for r in g]
    for r, c in cells:
        out[r][c] = couleur
    return en_grille(out)


# --------------------------------------------------------------------------
# 2. Sélecteurs d'objets (retournent un objet unique ou None si ambigu)
# --------------------------------------------------------------------------

def _extreme(objs, cle, maxi=True):
    if not objs:
        return None
    vals = [cle(o) for o in objs]
    t = max(vals) if maxi else min(vals)
    c = [o for o, v in zip(objs, vals) if v == t]
    return c[0] if len(c) == 1 else None


def _unique_par(objs, cle):
    """Objet dont la valeur de cle est unique alors que les autres la partagent."""
    if len(objs) < 3:
        return None
    cnt = Counter(cle(o) for o in objs)
    uniq = [o for o in objs if cnt[cle(o)] == 1]
    return uniq[0] if len(uniq) == 1 else None


def _sym(o, g):
    s = rogner(g, o, True)
    return s == fliph(s) or s == flipv(s)


SELECTEURS = {
    'plus_grand': lambda os, g: _extreme(os, lambda o: o.size, True),
    'plus_petit': lambda os, g: _extreme(os, lambda o: o.size, False),
    'bbox_max': lambda os, g: _extreme(os, lambda o: o.h * o.w, True),
    'bbox_min': lambda os, g: _extreme(os, lambda o: o.h * o.w, False),
    'haut': lambda os, g: _extreme(os, lambda o: o.r0, False),
    'bas': lambda os, g: _extreme(os, lambda o: o.r1, True),
    'gauche': lambda os, g: _extreme(os, lambda o: o.c0, False),
    'droite': lambda os, g: _extreme(os, lambda o: o.c1, True),
    'plus_haut_h': lambda os, g: _extreme(os, lambda o: o.h, True),
    'plus_large': lambda os, g: _extreme(os, lambda o: o.w, True),
    'plus_de_trous': lambda os, g: _extreme(os, lambda o: nb_trous(g, o), True),
    'moins_de_trous': lambda os, g: _extreme(os, lambda o: nb_trous(g, o), False),
    'plus_de_couleurs': lambda os, g: _extreme(os, lambda o: len({g[r][c] for r, c in o.cells}), True),
    'couleur_freq': lambda os, g: _extreme(os, lambda o: sum(1 for p in os if p.color == o.color), True),
    'couleur_rare': lambda os, g: _extreme(os, lambda o: sum(1 for p in os if p.color == o.color), False),
    'couleur_unique': lambda os, g: _unique_par(os, lambda o: o.color),
    'forme_unique': lambda os, g: _unique_par(os, lambda o: o.forme()),
    'symetrique': lambda os, g: (lambda l: l[0] if len(l) == 1 else None)([o for o in os if _sym(o, g)]),
    'asymetrique': lambda os, g: (lambda l: l[0] if len(l) == 1 else None)([o for o in os if not _sym(o, g)]),
    'plus_dense': lambda os, g: _extreme(os, lambda o: o.size * 1000 // (o.h * o.w), True),
    'moins_dense': lambda os, g: _extreme(os, lambda o: o.size * 1000 // (o.h * o.w), False),
}


def prim_crop_objet(kind, sel, seul):
    f = SELECTEURS[sel]

    def p(g):
        o = f(list(objets(g, kind)), g)
        return None if o is None else rogner(g, o, seul)
    return p


# --------------------------------------------------------------------------
# 3. Primitives (DSL) : chacune est une fonction pure grille -> grille | None
# --------------------------------------------------------------------------

def crop_bbox(g):
    b = fond(g)
    pts = [(r, c) for r in range(H(g)) for c in range(W(g)) if g[r][c] != b]
    if not pts:
        return None
    r0 = min(p[0] for p in pts); r1 = max(p[0] for p in pts)
    c0 = min(p[1] for p in pts); c1 = max(p[1] for p in pts)
    return sous_grille(g, r0, c0, r1, c1)


def upscale(k):
    def p(g):
        if H(g) * k > 45 or W(g) * k > 45:
            return None
        return tuple(tuple(v for v in r for _ in range(k)) for r in g for _ in range(k))
    return p


def downscale(k):
    def p(g):
        h, w = H(g), W(g)
        if h % k or w % k:
            return None
        o = tuple(tuple(g[r][c] for c in range(0, w, k)) for r in range(0, h, k))
        return o if upscale(k)(o) == g else None
    return p


def downscale_pick(k):
    """Sous-échantillonnage brut (une cellule sur k)."""
    def p(g):
        h, w = H(g), W(g)
        if h % k or w % k or h // k < 1:
            return None
        return tuple(tuple(g[r][c] for c in range(0, w, k)) for r in range(0, h, k))
    return p


def tile(a, b):
    def p(g):
        if H(g) * a > 45 or W(g) * b > 45:
            return None
        return tuple(r * b for _ in range(a) for r in g)
    return p


def miroir_h(g):
    return hcat(g, fliph(g))


def miroir_h2(g):
    return hcat(fliph(g), g)


def miroir_v(g):
    return vcat(g, flipv(g))


def miroir_v2(g):
    return vcat(flipv(g), g)


def miroir4(g):
    t = hcat(g, fliph(g))
    return vcat(t, flipv(t))


def miroir4b(g):
    t = hcat(fliph(g), g)
    return vcat(flipv(t), t)


def rot4(g):
    if H(g) != W(g):
        return None
    return vcat(hcat(g, rot90(g)), hcat(rot270(g), rot180(g)))


def rot4b(g):
    if H(g) != W(g):
        return None
    return vcat(hcat(g, rot270(g)), hcat(rot90(g), rot180(g)))


def fractal(inv):
    def p(g):
        h, w = H(g), W(g)
        if h * h > 45 or w * w > 45:
            return None
        b = 0 if any(0 in r for r in g) else fond(g)
        vide = tuple(tuple(b for _ in range(w)) for _ in range(h))
        lig = []
        for r in range(h):
            blocs = [g if ((g[r][c] != b) != inv) else vide for c in range(w)]
            lig.extend(hcat(*blocs))
        return tuple(lig)
    return p


def moitie(axe, cote):
    def p(g):
        h, w = H(g), W(g)
        if axe == 'h':
            if w < 2:
                return None
            m = w // 2
            return tuple(r[:m] for r in g) if cote == 0 else tuple(r[w - m:] for r in g)
        if h < 2:
            return None
        m = h // 2
        return g[:m] if cote == 0 else g[h - m:]
    return p


def sym_union(modes):
    """Union (priorité aux cellules non fond) de g et de ses images par symétrie."""
    def p(g):
        b = fond(g)
        imgs = [g]
        if 'h' in modes:
            imgs = imgs + [fliph(x) for x in imgs]
        if 'v' in modes:
            imgs = imgs + [flipv(x) for x in imgs]
        if 'r' in modes:
            if H(g) != W(g):
                return None
            imgs = [g, rot90(g), rot180(g), rot270(g)]
        if 't' in modes:
            if H(g) != W(g):
                return None
            imgs = imgs + [transp(x) for x in imgs]
        h, w = H(g), W(g)
        out = [list(r) for r in g]
        for im in imgs:
            for r in range(h):
                for c in range(w):
                    if out[r][c] == b and im[r][c] != b:
                        out[r][c] = im[r][c]
        return en_grille(out)
    return p


def supprimer_fond_lc(g):
    """Supprime lignes et colonnes entièrement de fond."""
    b = fond(g)
    lig = [r for r in range(H(g)) if any(v != b for v in g[r])]
    col = [c for c in range(W(g)) if any(g[r][c] != b for r in range(H(g)))]
    if not lig or not col:
        return None
    return tuple(tuple(g[r][c] for c in col) for r in lig)


def dedoublonner(g):
    """Fusionne lignes puis colonnes consécutives identiques."""
    l = [g[0]]
    for r in g[1:]:
        if r != l[-1]:
            l.append(r)
    t = transp(tuple(l))
    c = [t[0]]
    for r in t[1:]:
        if r != c[-1]:
            c.append(r)
    return transp(tuple(c))


def dedoublonner_l(g):
    l = [g[0]]
    for r in g[1:]:
        if r != l[-1]:
            l.append(r)
    return tuple(l)


def dedoublonner_c(g):
    return transp(dedoublonner_l(transp(g)))


# ---- couleurs ----

def garder_dominante(g):
    cnt = comptes(g)
    if not cnt:
        return None
    c = cnt.most_common(1)[0][0]
    b = fond(g)
    return tuple(tuple(v if v == c else b for v in r) for r in g)


def retirer_rare(g):
    cnt = comptes(g)
    if len(cnt) < 2:
        return None
    c = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
    b = fond(g)
    return tuple(tuple(b if v == c else v for v in r) for r in g)


def retirer_dominante(g):
    cnt = comptes(g)
    if len(cnt) < 2:
        return None
    c = cnt.most_common(1)[0][0]
    b = fond(g)
    return tuple(tuple(b if v == c else v for v in r) for r in g)


def echanger_2(g):
    cs = sorted({v for r in g for v in r})
    if len(cs) != 2:
        return None
    a, b = cs
    return tuple(tuple(b if v == a else a for v in r) for r in g)


def echanger_freq(g):
    cnt = Counter(v for r in g for v in r).most_common()
    if len(cnt) < 2:
        return None
    a, b = cnt[0][0], cnt[-1][0]
    return tuple(tuple(b if v == a else a if v == b else v for v in r) for r in g)


def non_fond_en_M(g):
    b = fond(g)
    return tuple(tuple(M if v != b else b for v in r) for r in g)


def fond_en_M(g):
    b = fond(g)
    return tuple(tuple(M if v == b else v for v in r) for r in g)


# ---- remplissage ----

def remplir_trous(g):
    t = trous_bg(g)
    if not t:
        return None
    return peindre(g, t, M)


def remplir_trous_objets(g):
    """Trous remplis avec la couleur de l'objet qui les entoure (voisin non fond)."""
    t = trous_bg(g)
    if not t:
        return None
    out = [list(r) for r in g]
    for r, c in t:
        for dy, dx in V4:
            y, x = r + dy, c + dx
            if 0 <= y < H(g) and 0 <= x < W(g) and g[y][x] != fond(g):
                out[r][c] = g[y][x]
                break
    return en_grille(out)


def remplir_bbox(kind):
    def p(g):
        b = fond(g)
        out = [list(r) for r in g]
        n = 0
        for o in objets(g, kind):
            for r in range(o.r0, o.r1 + 1):
                for c in range(o.c0, o.c1 + 1):
                    if g[r][c] == b:
                        out[r][c] = M
                        n += 1
        return en_grille(out) if n else None
    return p


def contour_objets(diag):
    def p(g):
        b = fond(g)
        h, w = H(g), W(g)
        vs = V8 if diag else V4
        out = [list(r) for r in g]
        n = 0
        for r in range(h):
            for c in range(w):
                if g[r][c] == b:
                    for dy, dx in vs:
                        y, x = r + dy, c + dx
                        if 0 <= y < h and 0 <= x < w and g[y][x] != b:
                            out[r][c] = M
                            n += 1
                            break
        return en_grille(out) if n else None
    return p


def cadre(g):
    h, w = H(g), W(g)
    return tuple(tuple(M if r in (0, h - 1) or c in (0, w - 1) else g[r][c]
                       for c in range(w)) for r in range(h))


def bord_objets(g):
    """Cellules de bord (non entourées) des objets multicolores -> M ; intérieur inchangé."""
    b = fond(g)
    h, w = H(g), W(g)
    out = [list(r) for r in g]
    n = 0
    for r in range(h):
        for c in range(w):
            if g[r][c] != b:
                for dy, dx in V4:
                    y, x = r + dy, c + dx
                    if not (0 <= y < h and 0 <= x < w) or g[y][x] == b:
                        out[r][c] = M
                        n += 1
                        break
    return en_grille(out) if n else None


# ---- rayons / gravité ----

def rayons(dirs):
    def p(g):
        b = fond(g)
        h, w = H(g), W(g)
        out = [list(r) for r in g]
        n = 0
        for r in range(h):
            for c in range(w):
                v = g[r][c]
                if v != b:
                    for dy, dx in dirs:
                        y, x = r + dy, c + dx
                        while 0 <= y < h and 0 <= x < w and g[y][x] == b:
                            out[y][x] = v
                            n += 1
                            y += dy
                            x += dx
        return en_grille(out) if n else None
    return p


def gravite(d):
    def p(g):
        b = fond(g)
        h, w = H(g), W(g)
        out = [[b] * w for _ in range(h)]
        if d in ('bas', 'haut'):
            for c in range(w):
                col = [g[r][c] for r in range(h) if g[r][c] != b]
                if d == 'bas':
                    for i, v in enumerate(reversed(col)):
                        out[h - 1 - i][c] = v
                else:
                    for i, v in enumerate(col):
                        out[i][c] = v
        else:
            for r in range(h):
                lig = [v for v in g[r] if v != b]
                if d == 'droite':
                    for i, v in enumerate(reversed(lig)):
                        out[r][w - 1 - i] = v
                else:
                    for i, v in enumerate(lig):
                        out[r][i] = v
        o = en_grille(out)
        return o if o != g else None
    return p


def gravite_objets(d):
    """Les objets multicolores 8-connexes tombent en bloc (rigides)."""
    dy, dx = {'bas': (1, 0), 'haut': (-1, 0), 'droite': (0, 1), 'gauche': (0, -1)}[d]

    def p(g):
        b = fond(g)
        h, w = H(g), W(g)
        objs = list(objets(g, 'm8'))
        if not objs or len(objs) > 30:
            return None
        objs.sort(key=lambda o: -(o.r1 * dy + o.c1 * dx) if (dy + dx) > 0 else (o.r0 * dy + o.c0 * dx))
        grid = [[b] * w for _ in range(h)]
        occ = set()
        res = []
        for o in objs:
            res.append(o)
        # on déplace en commençant par ceux qui sont le plus près du bord d'arrivée
        res.sort(key=lambda o: -max((r * dy + c * dx) for r, c in o.cells))
        for o in res:
            cells = [(r, c, g[r][c]) for r, c in o.cells]
            k = 0
            while True:
                ok = True
                for r, c, v in cells:
                    y, x = r + (k + 1) * dy, c + (k + 1) * dx
                    if not (0 <= y < h and 0 <= x < w) or (y, x) in occ:
                        ok = False
                        break
                if not ok:
                    break
                k += 1
            for r, c, v in cells:
                y, x = r + k * dy, c + k * dx
                occ.add((y, x))
                grid[y][x] = v
        o = en_grille(grid)
        return o if o != g else None
    return p


def decaler(dy, dx, rouler):
    def p(g):
        h, w = H(g), W(g)
        b = fond(g)
        out = []
        for r in range(h):
            lig = []
            for c in range(w):
                y, x = r - dy, c - dx
                if rouler:
                    lig.append(g[y % h][x % w])
                else:
                    lig.append(g[y][x] if 0 <= y < h and 0 <= x < w else b)
            out.append(tuple(lig))
        return tuple(out)
    return p


# ---- découpes en parties, logique ----

def _segments(n, seps):
    segs, d = [], None
    for i in range(n):
        if i in seps:
            if d is not None:
                segs.append((d, i))
                d = None
        elif d is None:
            d = i
    if d is not None:
        segs.append((d, n))
    return segs


@lru_cache(maxsize=100000)
def cellules_sep(g):
    """Découpe par lignes/colonnes séparatrices unies. -> tuple de grilles (même forme) ou None."""
    h, w = H(g), W(g)
    meilleur = None
    for s in sorted({v for r in g for v in r}):
        rows = {r for r in range(h) if all(v == s for v in g[r])}
        cols = {c for c in range(w) if all(g[r][c] == s for r in range(h))}
        if (not rows and not cols) or len(rows) == h or len(cols) == w:
            continue
        sr, sc = _segments(h, rows), _segments(w, cols)
        cells = [tuple(row[c0:c1] for row in g[r0:r1]) for r0, r1 in sr for c0, c1 in sc]
        if len(cells) >= 2 and len({(H(x), W(x)) for x in cells}) == 1:
            sc_ = (s != fond(g), len(cells))
            if meilleur is None or sc_ > meilleur[0]:
                meilleur = (sc_, tuple(cells))
    return None if meilleur is None else meilleur[1]


def cellules_eq(g, nr, nc):
    h, w = H(g), W(g)
    if h % nr or w % nc:
        return None
    hh, ww = h // nr, w // nc
    return tuple(tuple(row[j * ww:(j + 1) * ww] for row in g[i * hh:(i + 1) * hh])
                 for i in range(nr) for j in range(nc))


def parties(g, src):
    if src == 'sep':
        return cellules_sep(g)
    nr, nc = {'h2': (1, 2), 'v2': (2, 1), 'h3': (1, 3), 'v3': (3, 1), 'q4': (2, 2)}[src]
    return cellules_eq(g, nr, nc)


def _fond_parties(ps):
    cnt = Counter(v for p in ps for r in p for v in r)
    return max(cnt.items(), key=lambda kv: (kv[1], kv[0] == 0))[0]


def op_logique(src, op):
    def p(g):
        ps = parties(g, src)
        if not ps or len(ps) < 2 or len(ps) > 4:
            return None
        if op in ('anotb', 'bnota') and len(ps) != 2:
            return None
        b = 0 if any(0 in r for q in ps for r in q) else _fond_parties(ps)
        h, w = H(ps[0]), W(ps[0])
        out = []
        for r in range(h):
            lig = []
            for c in range(w):
                on = [q[r][c] != b for q in ps]
                if op == 'et':
                    v = all(on)
                elif op == 'ou':
                    v = any(on)
                elif op == 'xor':
                    v = sum(on) % 2 == 1 if len(on) == 2 else sum(on) == 1
                elif op == 'nor':
                    v = not any(on)
                elif op == 'nand':
                    v = not all(on)
                elif op == 'anotb':
                    v = on[0] and not on[1]
                else:
                    v = on[1] and not on[0]
                lig.append(M if v else b)
            out.append(tuple(lig))
        return tuple(out)
    return p


def op_superposer(src, inverse):
    def p(g):
        ps = parties(g, src)
        if not ps or len(ps) < 2 or len(ps) > 4:
            return None
        b = 0 if any(0 in r for q in ps for r in q) else _fond_parties(ps)
        ordre = list(reversed(ps)) if inverse else list(ps)
        h, w = H(ps[0]), W(ps[0])
        out = []
        for r in range(h):
            lig = []
            for c in range(w):
                v = b
                for q in ordre:
                    if q[r][c] != b:
                        v = q[r][c]
                        break
                lig.append(v)
            out.append(tuple(lig))
        return tuple(out)
    return p


def _nb_non_fond(q, b):
    return sum(1 for r in q for v in r if v != b)


def sel_partie(src, sel):
    def p(g):
        ps = parties(g, src)
        if not ps or len(ps) < 2:
            return None
        b = 0 if any(0 in r for q in ps for r in q) else _fond_parties(ps)
        if sel == 'plus_rempli':
            return _extreme_g(ps, lambda q: _nb_non_fond(q, b), True)
        if sel == 'moins_rempli':
            return _extreme_g(ps, lambda q: _nb_non_fond(q, b), False)
        if sel == 'plus_couleurs':
            return _extreme_g(ps, lambda q: len({v for r in q for v in r} - {b}), True)
        if sel == 'moins_couleurs':
            return _extreme_g(ps, lambda q: len({v for r in q for v in r} - {b}), False)
        if sel == 'unique':
            cnt = Counter(ps)
            u = [q for q in ps if cnt[q] == 1]
            return u[0] if len(u) == 1 and len(ps) >= 3 else None
        if sel == 'symetrique':
            u = [q for q in ps if q == fliph(q) or q == flipv(q)]
            return u[0] if len(u) == 1 else None
        if sel == 'asymetrique':
            u = [q for q in ps if not (q == fliph(q) or q == flipv(q))]
            return u[0] if len(u) == 1 else None
        if sel == 'premier':
            return ps[0]
        if sel == 'dernier':
            return ps[-1]
        return None
    return p


def _extreme_g(ps, cle, maxi):
    vals = [cle(q) for q in ps]
    t = max(vals) if maxi else min(vals)
    c = [q for q, v in zip(ps, vals) if v == t]
    return c[0] if len(c) == 1 else None


# ---- comptage ----

def dom_1x1(g):
    cnt = comptes(g)
    return ((cnt.most_common(1)[0][0],),) if cnt else None


def rare_1x1(g):
    cnt = comptes(g)
    if not cnt:
        return None
    return ((min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0],),)


def couleurs_col(g):
    cnt = comptes(g)
    if not cnt:
        return None
    return tuple((c,) for c, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])))


def couleurs_lig(g):
    c = couleurs_col(g)
    return None if c is None else (tuple(r[0] for r in c),)


def nb_objets_barre(kind, vertical, marq):
    def p(g):
        n = len(objets(g, kind))
        if n == 0 or n > 30:
            return None
        cnt = comptes(g)
        if not cnt:
            return None
        v = M if marq else cnt.most_common(1)[0][0]
        return tuple((v,) for _ in range(n)) if vertical else (tuple(v for _ in range(n)),)
    return p


# ---- opérateur par objet ----

def par_objet(kind, f):
    """Applique f (forme-préservante) à la sous-grille de chaque objet, puis repeint."""
    def p(g):
        b = fond(g)
        objs = objets(g, kind)
        if not objs or len(objs) > 40:
            return None
        out = [[b] * W(g) for _ in range(H(g))]
        for o in objs:
            sub = rogner(g, o, True)
            t = f(sub)
            if t is None or (H(t), W(t)) != (H(sub), W(sub)):
                return None
            for r in range(H(t)):
                for c in range(W(t)):
                    if t[r][c] != b:
                        out[o.r0 + r][o.c0 + c] = t[r][c]
        o2 = en_grille(out)
        return o2 if o2 != g else None
    return p


def construire_prims():
    P = []

    def ajouter(nom, f):
        P.append((nom, f))

    # géométrie
    for nom, f in (('rot90', rot90), ('rot180', rot180), ('rot270', rot270), ('fliph', fliph),
                   ('flipv', flipv), ('transp', transp), ('antitransp', antitransp)):
        ajouter(nom, f)
    # recadrage
    ajouter('crop_bbox', crop_bbox)
    ajouter('supprimer_fond_lc', supprimer_fond_lc)
    for kind in ('m8', 'c8', 'c4', 'm4'):
        for sel in SELECTEURS:
            for seul in (False, True):
                if kind == 'm4' and sel in ('couleur_freq', 'couleur_rare', 'couleur_unique'):
                    continue
                if kind[0] == 'm' and sel in ('couleur_freq', 'couleur_rare', 'couleur_unique'):
                    continue
                ajouter('crop_%s[%s%s]' % (sel, kind, '_seul' if seul else ''),
                        prim_crop_objet(kind, sel, seul))
    # échelle / pavage
    for k in (2, 3, 4, 5):
        ajouter('upscale%d' % k, upscale(k))
    for k in (2, 3, 4, 5):
        ajouter('downscale%d' % k, downscale(k))
    for k in (2, 3):
        ajouter('echantillon%d' % k, downscale_pick(k))
    for a, b in ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (3, 3), (4, 4), (1, 4), (4, 1)):
        ajouter('pavage%dx%d' % (a, b), tile(a, b))
    for nom, f in (('miroir_h', miroir_h), ('miroir_h_g', miroir_h2), ('miroir_v', miroir_v),
                   ('miroir_v_h', miroir_v2), ('miroir4', miroir4), ('miroir4b', miroir4b),
                   ('rot4', rot4), ('rot4b', rot4b)):
        ajouter(nom, f)
    ajouter('fractal', fractal(False))
    ajouter('fractal_inv', fractal(True))
    for axe in 'hv':
        for cote in (0, 1):
            ajouter('moitie_%s%d' % (axe, cote), moitie(axe, cote))
    ajouter('dedoublonner', dedoublonner)
    ajouter('dedoublonner_l', dedoublonner_l)
    ajouter('dedoublonner_c', dedoublonner_c)
    for modes in ('h', 'v', 'hv', 'r', 'ht', 'hvt'):
        ajouter('sym_union[%s]' % modes, sym_union(modes))
    # couleurs
    for nom, f in (('garder_dominante', garder_dominante), ('retirer_rare', retirer_rare),
                   ('retirer_dominante', retirer_dominante), ('echanger_2', echanger_2),
                   ('echanger_freq', echanger_freq), ('non_fond_en_M', non_fond_en_M)):
        ajouter(nom, f)
    # remplissage
    ajouter('remplir_trous', remplir_trous)
    ajouter('remplir_trous_objets', remplir_trous_objets)
    ajouter('remplir_bbox[m8]', remplir_bbox('m8'))
    ajouter('remplir_bbox[c8]', remplir_bbox('c8'))
    ajouter('contour4', contour_objets(False))
    ajouter('contour8', contour_objets(True))
    ajouter('cadre', cadre)
    ajouter('bord_objets', bord_objets)
    # rayons, gravité, décalages
    D = {'d': (0, 1), 'g': (0, -1), 'b': (1, 0), 'h': (-1, 0)}
    for nom, ds in (('droite', 'd'), ('gauche', 'g'), ('bas', 'b'), ('haut', 'h'), ('horiz', 'dg'),
                    ('vert', 'bh'), ('croix', 'dgbh')):
        ajouter('rayons_' + nom, rayons([D[x] for x in ds]))
    ajouter('rayons_diag', rayons(((1, 1), (1, -1), (-1, 1), (-1, -1))))
    ajouter('rayons_8', rayons(V8))
    for d in ('bas', 'haut', 'gauche', 'droite'):
        ajouter('gravite_' + d, gravite(d))
        ajouter('gravite_objets_' + d, gravite_objets(d))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ajouter('rouler(%d,%d)' % (dy, dx), decaler(dy, dx, True))
        ajouter('decaler(%d,%d)' % (dy, dx), decaler(dy, dx, False))
    # logique / parties
    for src in ('sep', 'h2', 'v2', 'q4', 'h3', 'v3'):
        for op in ('et', 'ou', 'xor', 'nor', 'anotb', 'bnota', 'nand'):
            ajouter('logique_%s[%s]' % (op, src), op_logique(src, op))
        ajouter('superposer[%s]' % src, op_superposer(src, False))
        ajouter('superposer_inv[%s]' % src, op_superposer(src, True))
        for sel in ('plus_rempli', 'moins_rempli', 'plus_couleurs', 'moins_couleurs', 'unique',
                    'symetrique', 'asymetrique', 'premier', 'dernier'):
            ajouter('partie_%s[%s]' % (sel, src), sel_partie(src, sel))
    # comptage
    for nom, f in (('dominante_1x1', dom_1x1), ('rare_1x1', rare_1x1), ('couleurs_col', couleurs_col),
                   ('couleurs_lig', couleurs_lig)):
        ajouter(nom, f)
    for kind in ('c8', 'c4'):
        for vert in (False, True):
            for marq in (False, True):
                ajouter('barre_nb_objets[%s,%s%s]' % (kind, 'v' if vert else 'h', ',M' if marq else ''),
                        nb_objets_barre(kind, vert, marq))
    # méta-opérateur par objet
    for kind in ('m8', 'c8'):
        for nom, f in (('fliph', fliph), ('flipv', flipv), ('rot180', rot180), ('transp', transp),
                       ('rot90', rot90)):
            ajouter('par_objet[%s](%s)' % (kind, nom), par_objet(kind, f))
    return P


PRIMS = construire_prims()
PRIM_FN = dict(PRIMS)


# --------------------------------------------------------------------------
# 4. Moteur : application sûre, composition, finisseur « table de couleurs »
# --------------------------------------------------------------------------

class Horloge:
    """Budget CPU (process_time) d'une tâche."""

    def __init__(self, budget):
        self.t0 = time.process_time()
        self.limite = self.t0 + budget

    def verifier(self):
        if time.process_time() > self.limite:
            raise Delai()

    def restant(self):
        return self.limite - time.process_time()

    def reduire(self, nouveau_restant):
        self.limite = min(self.limite, time.process_time() + nouveau_restant)


def appliquer(f, g):
    """Applique f à g ; None si échec/exception/grille invalide."""
    try:
        r = f(g)
    except Delai:
        raise
    except Exception:
        return None
    if not r or not r[0]:
        return None
    return r


def grille_valide(r):
    return bool(r) and bool(r[0]) and len({len(x) for x in r}) == 1 and len(r) <= 60 and len(r[0]) <= 60


def apprendre_cmap(etats, cibles):
    """Table de couleurs cohérente cellule à cellule, ou None."""
    m = {}
    for e, o in zip(etats, cibles):
        if e is None or len(e) != len(o) or len(e[0]) != len(o[0]):
            return None
        for re_, ro in zip(e, o):
            for a, b in zip(re_, ro):
                if m.setdefault(a, b) != b:
                    return None
    return m


def appliquer_cmap(cm, g):
    return tuple(tuple(cm.get(v, v) for v in r) for r in g)


def sortie_ok(r):
    return r is not None and grille_valide(r) and all(0 <= v <= 9 for x in r for v in x)


def composer(noms, cm=None):
    fs = [PRIM_FN[n] for n in noms]

    def prog(g):
        for f in fs:
            g = appliquer(f, g)
            if g is None:
                return None
        return appliquer_cmap(cm, g) if cm is not None else g
    return prog


def texte_prog(noms, cm=None):
    t = ' ; '.join(noms) if noms else 'identite'
    if cm:
        diff = {a: b for a, b in cm.items() if a != b}
        if diff:
            t += ' ; couleurs{%s}' % ','.join('%d>%d' % kv for kv in sorted(diff.items()))
    return t


def compat0(r0, o0):
    """Test rapide sur la 1re paire : même forme et (égalité ou table de couleurs possible)."""
    if len(r0) != len(o0) or len(r0[0]) != len(o0[0]):
        return False
    if r0 == o0:
        return True
    m = {}
    for a, b in zip(r0, o0):
        for x, y in zip(a, b):
            if m.setdefault(x, y) != y:
                return False
    return True


def finir(prog, etat, outs):
    """Vérifie un état complet (train+test) : exact, sinon avec table de couleurs."""
    n = len(outs)
    if all(etat[i] == outs[i] for i in range(n)):
        yield (texte_prog(prog), composer(prog))
        return
    cm = apprendre_cmap(etat[:n], outs)
    if cm is not None:
        yield (texte_prog(prog, cm), composer(prog, cm))


def calc_etat(f, etat):
    res = []
    for g in etat:
        r = appliquer(f, g)
        if r is None or not grille_valide(r):
            return None
        res.append(r)
    return tuple(res)


def recherche_dsl(train, tests, dl, t_niveau2):
    """Recherche en largeur de compositions de primitives (profondeur <= 3).

    Générateur de (texte, fn) vérifiés sur la 1re paire puis sur toutes.
    Émet None entre la fin des niveaux 1-2 et le niveau 3 (point de reprise).
    """
    ins = [p[0] for p in train]
    outs = [p[1] for p in train]
    n = len(ins)
    etat0 = tuple(ins) + tuple(tests)
    vus = {etat0}
    vus01 = set()
    yield from finir((), etat0, outs)
    niveau = [((), etat0)]
    for d in (1, 2, 3):
        if d == 3:
            yield None
        dernier = d == 3
        suivant = []
        for prog, etat in niveau:
            for nom, f in PRIMS:
                dl.verifier()
                if d <= 2 and time.process_time() > t_niveau2:
                    break
                r0 = appliquer(f, etat[0])
                if r0 is None or not grille_valide(r0):
                    continue
                if compat0(r0, outs[0]):
                    e2 = calc_etat(f, etat)
                    if e2 is not None:
                        yield from finir(prog + (nom,), e2, outs)
                if not dernier:
                    if r0 == etat[0] or len(r0) > 45 or len(r0[0]) > 45:
                        continue
                    cle = (r0, appliquer(f, etat[1]) if n > 1 else None)
                    if cle in vus01:
                        continue
                    vus01.add(cle)
                    e2 = calc_etat(f, etat)
                    if e2 is None or e2 in vus:
                        continue
                    vus.add(e2)
                    suivant.append((prog + (nom,), e2))
        niveau = suivant


# --------------------------------------------------------------------------
# 5. Règle locale cellule -> cellule apprise (table de voisinage)
# --------------------------------------------------------------------------

def _val(g, r, c):
    return g[r][c] if 0 <= r < len(g) and 0 <= c < len(g[0]) else -1


def f_offsets(offs):
    def f(g):
        return [[(g[r][c],) + tuple(_val(g, r + dy, c + dx) for dy, dx in offs)
                 for c in range(W(g))] for r in range(H(g))]
    return f


def f_roles(offs):
    def f(g):
        b = fond(g)
        res = []
        for r in range(H(g)):
            lig = []
            for c in range(W(g)):
                v = g[r][c]
                k = []
                for dy, dx in offs:
                    x = _val(g, r + dy, c + dx)
                    k.append(3 if x == -1 else 0 if x == b else 1 if x == v else 2)
                lig.append((v,) + tuple(k))
            res.append(lig)
        return res
    return f


def f_compte(offs):
    def f(g):
        b = fond(g)
        return [[(g[r][c], sum(1 for dy, dx in offs if _val(g, r + dy, c + dx) not in (-1, b)))
                 for c in range(W(g))] for r in range(H(g))]
    return f


def f_compte_col(offs):
    def f(g):
        res = []
        for r in range(H(g)):
            lig = []
            for c in range(W(g)):
                cnt = Counter(_val(g, r + dy, c + dx) for dy, dx in offs)
                lig.append((g[r][c], tuple(sorted(cnt.items()))))
            res.append(lig)
        return res
    return f


def _rayon(g, r, c, dy, dx, b):
    y, x = r + dy, c + dx
    h, w = len(g), len(g[0])
    while 0 <= y < h and 0 <= x < w:
        if g[y][x] != b:
            return g[y][x]
        y += dy
        x += dx
    return -1


def f_rayons(dirs, abstrait=False):
    def f(g):
        b = fond(g)
        res = []
        for r in range(H(g)):
            lig = []
            for c in range(W(g)):
                v = g[r][c]
                k = tuple(_rayon(g, r, c, dy, dx, b) for dy, dx in dirs)
                if abstrait:
                    k = tuple(3 if x == -1 else 1 if x == v else 2 for x in k)
                lig.append((v,) + k)
            res.append(lig)
        return res
    return f


def f_trous(g):
    t = trous_bg(g)
    return [[(g[r][c], (r, c) in t) for c in range(W(g))] for r in range(H(g))]


def f_rowcol(g):
    b = fond(g)
    rs = [frozenset(v for v in row if v != b) for row in g]
    cs = [frozenset(g[r][c] for r in range(H(g)) if g[r][c] != b) for c in range(W(g))]
    return [[(g[r][c], rs[r], cs[c]) for c in range(W(g))] for r in range(H(g))]


def f_mod(k):
    return lambda g: [[(g[r][c], r % k, c % k) for c in range(W(g))] for r in range(H(g))]


def f_sym(g):
    h, w = H(g), W(g)
    return [[(g[r][c], g[r][w - 1 - c], g[h - 1 - r][c], g[h - 1 - r][w - 1 - c]) for c in range(w)]
            for r in range(h)]


def f_comp(g):
    taille = {}
    for o in objets(g, 'c4'):
        for p in o.cells:
            taille[p] = (o.size, touche_bord(g, o))
    return [[(g[r][c], taille.get((r, c))) for c in range(W(g))] for r in range(H(g))]


def f_comp8(g):
    taille = {}
    for o in objets(g, 'c8'):
        for p in o.cells:
            taille[p] = (o.size, touche_bord(g, o))
    return [[(g[r][c], taille.get((r, c))) for c in range(W(g))] for r in range(H(g))]


def f_ligne(g):
    return [[(g[r][c], r) for c in range(W(g))] for r in range(H(g))]


def f_colonne(g):
    return [[(g[r][c], c) for c in range(W(g))] for r in range(H(g))]


def f_dist(g):
    h, w = H(g), W(g)
    return [[(g[r][c], min(r, c, h - 1 - r, w - 1 - c)) for c in range(w)] for r in range(h)]


def f_pos(g):
    return [[(g[r][c], r, c) for c in range(W(g))] for r in range(H(g))]


def f_diag(g):
    return [[(g[r][c], r - c) for c in range(W(g))] for r in range(H(g))]


def f_cellule_obj(g):
    """(couleur, rang de taille de l'objet 8-connexe monocolore, est le plus petit)."""
    objs = objets(g, 'c8')
    tailles = sorted({o.size for o in objs}, reverse=True)
    info = {}
    for o in objs:
        for p in o.cells:
            info[p] = (tailles.index(o.size), o.size == tailles[-1])
    return [[(g[r][c], info.get((r, c))) for c in range(W(g))] for r in range(H(g))]


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
N24 = tuple((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if (dy, dx) != (0, 0))
D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

FEATURES = [
    ('c', f_offsets(())),
    ('roles_n4', f_roles(N4)),
    ('n4', f_offsets(N4)),
    ('compte_n4', f_compte(N4)),
    ('roles_n8', f_roles(N8)),
    ('compte_n8', f_compte(N8)),
    ('n8', f_offsets(N8)),
    ('compte_col_n8', f_compte_col(N8)),
    ('trous', f_trous),
    ('rayons_h', f_rayons(((0, 1), (0, -1)))),
    ('rayons_v', f_rayons(((1, 0), (-1, 0)))),
    ('rayons4', f_rayons(D4)),
    ('rayons4_abs', f_rayons(D4, True)),
    ('rayons8', f_rayons(D8)),
    ('composante4', f_comp),
    ('composante8', f_comp8),
    ('objet_rang', f_cellule_obj),
    ('ligne_col_ens', f_rowcol),
    ('mod2', f_mod(2)),
    ('mod3', f_mod(3)),
    ('sym', f_sym),
    ('dist_bord', f_dist),
    ('diag', f_diag),
    ('ligne', f_ligne),
    ('colonne', f_colonne),
    ('n24', f_offsets(N24)),
    ('pos', f_pos),
]


def regle_locale(nom, feat, paires, tests):
    """Apprend feature -> couleur ; renvoie fn ou None. Clé inconnue : couleur inchangée."""
    table = {}
    for gi, go in paires:
        if (H(gi), W(gi)) != (H(go), W(go)):
            return None
        F = feat(gi)
        for r in range(H(gi)):
            for c in range(W(gi)):
                k = F[r][c]
                o = go[r][c]
                if table.setdefault(k, o) != o:
                    return None
    if all(k[0] == o for k, o in table.items()):
        return None  # identité : aucun intérêt

    def fn(g):
        F = feat(g)
        return tuple(tuple(table.get(F[r][c], g[r][c]) for c in range(W(g))) for r in range(H(g)))
    # cellules de test dont la clé est inconnue (repli : couleur inchangée)
    fn.inconnues = sum(1 for g in tests for row in feat(g) for k in row if k not in table)
    return fn


def appr_local(train, tests, dl):
    paires = [(p[0], p[1]) for p in train]
    if any((H(a), W(a)) != (H(b), W(b)) for a, b in paires):
        return
    cand = []
    for i, (nom, feat) in enumerate(FEATURES):
        dl.verifier()
        fn = regle_locale(nom, feat, paires, tests)
        if fn is not None:
            cand.append((fn.inconnues, i, nom, fn))
    cand.sort(key=lambda x: x[:2])  # d'abord les règles qui couvrent tout le test
    for _, _, nom, fn in cand:
        yield ('regle_locale[%s]' % nom, fn)


# --- règles locales « relatives » : sortie = S (inchangé) / couleur d'un voisin / absolue

def _rel_n(offs, nomsdir):
    def f(g):
        b = fond(g)
        res = []
        for r in range(H(g)):
            lig = []
            for c in range(W(g)):
                v = g[r][c]
                cand = {'S': v}
                k = [v == b]
                for (dy, dx), nm in zip(offs, nomsdir):
                    x = _val(g, r + dy, c + dx)
                    k.append(3 if x == -1 else 0 if x == b else 1 if x == v else 2)
                    cand[nm] = x
                lig.append((tuple(k), cand))
            res.append(lig)
        return res
    return f


def _rel_rays(dirs, noms):
    def f(g):
        b = fond(g)
        res = []
        for r in range(H(g)):
            lig = []
            for c in range(W(g)):
                v = g[r][c]
                cand = {'S': v}
                k = [v == b]
                rv = []
                for (dy, dx), nm in zip(dirs, noms):
                    x = _rayon(g, r, c, dy, dx, b)
                    cand[nm] = x
                    rv.append(x)
                    k.append(3 if x == -1 else 1 if x == v else 2)
                for i in range(0, len(rv) - 1, 2):
                    k.append(rv[i] == rv[i + 1])
                lig.append((tuple(k), cand))
            res.append(lig)
        return res
    return f


REL_FEATURES = [
    ('rel_rayons_hv', _rel_rays(D4, ('N', 'E', 'S_', 'O'))),
    ('rel_n4', _rel_n(N4, ('N', 'E', 'S_', 'O'))),
    ('rel_n8', _rel_n(N8, tuple('abcdefgh'))),
    ('rel_rayons8', _rel_rays(D8, tuple('abcdefgh'))),
]


def regle_relative(feat, paires):
    groupes = {}
    for gi, go in paires:
        if (H(gi), W(gi)) != (H(go), W(go)):
            return None
        F = feat(gi)
        for r in range(H(gi)):
            for c in range(W(gi)):
                k, cand = F[r][c]
                o = go[r][c]
                toks = {t for t, col in cand.items() if col == o}
                toks.add(('abs', o))
                if k in groupes:
                    groupes[k] &= toks
                    if not groupes[k]:
                        return None
                else:
                    groupes[k] = set(toks)
    ordre = ['S', 'N', 'E', 'S_', 'O', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    choix = {}
    for k, toks in groupes.items():
        t = None
        for nm in ordre:
            if nm in toks:
                t = nm
                break
        if t is None:
            t = next(iter(toks))
        choix[k] = t
    if all(t == 'S' for t in choix.values()):
        return None

    def fn(g):
        F = feat(g)
        out = []
        for r in range(H(g)):
            lig = []
            for c in range(W(g)):
                k, cand = F[r][c]
                t = choix.get(k, 'S')
                v = t[1] if isinstance(t, tuple) else cand[t]
                lig.append(g[r][c] if v == -1 else v)
            out.append(tuple(lig))
        return tuple(out)
    return fn


def appr_relative(train, tests, dl):
    paires = [(p[0], p[1]) for p in train]
    if any((H(a), W(a)) != (H(b), W(b)) for a, b in paires):
        return
    for nom, feat in REL_FEATURES:
        dl.verifier()
        fn = regle_relative(feat, paires)
        if fn is not None:
            yield ('regle_relative[%s]' % nom, fn)


# --------------------------------------------------------------------------
# 6. Recoloriage d'objets selon une propriété apprise
# --------------------------------------------------------------------------

def _props(g, o, objs):
    tailles = sorted({p.size for p in objs}, reverse=True)
    P = {
        'taille': o.size,
        'rang_taille': tailles.index(o.size),
        'rang_taille_inv': len(tailles) - 1 - tailles.index(o.size),
        'plus_grand': o.size == tailles[0],
        'plus_petit': o.size == tailles[-1],
        'hauteur': o.h,
        'largeur': o.w,
        'trous': nb_trous(g, o),
        'bord': touche_bord(g, o),
        'rectangle': o.size == o.h * o.w,
        'forme': o.forme(),
        'carre': o.h == o.w,
        'symetrique': _sym(o, g),
        'nb_couleurs': len({g[r][c] for r, c in o.cells}),
        'ligne_colonne': (o.h == 1 or o.w == 1),
    }
    if o.color is not None:
        P['couleur'] = o.color
        P['n_meme_couleur'] = sum(1 for p in objs if p.color == o.color)
        P['taille_couleur'] = (o.color, o.size)
        P['trous_couleur'] = (o.color, P['trous'])
    return P


def appr_objets(train, tests, dl):
    paires = [(p[0], p[1]) for p in train]
    if any((H(a), W(a)) != (H(b), W(b)) for a, b in paires):
        return
    for kind in ('c4', 'c8', 'm8', 'm4'):
        donnees = []
        ok = True
        for gi, go in paires:
            objs = objets(gi, kind)
            if not objs or len(objs) > 60:
                ok = False
                break
            couvert = set()
            lst = []
            for o in objs:
                outc = {go[r][c] for r, c in o.cells}
                if len(outc) != 1:
                    ok = False
                    break
                lst.append((o, outc.pop()))
                couvert.update(o.cells)
            if not ok:
                break
            for r in range(H(gi)):
                for c in range(W(gi)):
                    if (r, c) not in couvert and gi[r][c] != go[r][c]:
                        ok = False
            if not ok:
                break
            donnees.append((gi, objs, lst))
        if not ok:
            continue
        propss = [[_props(gi, o, objs) for o, _ in lst] for gi, objs, lst in donnees]
        noms = set.intersection(*[set(p[0].keys()) for p in propss])
        for nom in sorted(noms):
            dl.verifier()
            table = {}
            bon = True
            for (gi, objs, lst), props in zip(donnees, propss):
                for (o, oc), P in zip(lst, props):
                    cle = P[nom]
                    toks = {('abs', oc)}
                    if o.color is not None and oc == o.color:
                        toks.add('S')
                    if cle in table:
                        table[cle] &= toks
                        if not table[cle]:
                            bon = False
                            break
                    else:
                        table[cle] = set(toks)
                if not bon:
                    break
            if not bon:
                continue
            choix = {k: ('S' if 'S' in t else sorted(t)[0]) for k, t in table.items()}
            if all(v == 'S' for v in choix.values()):
                continue
            if len(choix) > 3 and len(choix) == sum(len(l) for _, _, l in donnees):
                continue  # table = mémorisation pure, pas de généralisation

            def mk(kind=kind, nom=nom, choix=choix):
                def fn(g):
                    objs = objets(g, kind)
                    out = [list(r) for r in g]
                    for o in objs:
                        P = _props(g, o, objs)
                        if nom not in P:
                            return None
                        t = choix.get(P[nom], 'S')
                        if t == 'S':
                            continue
                        for r, c in o.cells:
                            out[r][c] = t[1]
                    return en_grille(out)
                return fn
            yield ('recolorer_objets[%s,%s]' % (kind, nom), mk())


# --------------------------------------------------------------------------
# 7. Complétion de symétrie (cellules « masquées »)
# --------------------------------------------------------------------------

def _symetries(h, w, grand):
    """Applications (r,c)->(r',c') candidates (décalages d'axe bornés)."""
    S = []
    for s in range(w - 1 - grand, w + grand):
        if s >= 1:
            S.append(('h%d' % s, lambda r, c, s=s: (r, s - c)))
    for s in range(h - 1 - grand, h + grand):
        if s >= 1:
            S.append(('v%d' % s, lambda r, c, s=s: (s - r, c)))
    for a in range(-grand, grand + 1):
        S.append(('t%d' % a, lambda r, c, a=a: (c + a, r - a)))
    for s in range(h + w - 2 - grand, h + w - 2 + grand + 1):
        S.append(('a%d' % s, lambda r, c, s=s: (s - c, s - r)))
    for sr in range(h - 1 - grand, h + grand):
        for sc in range(w - 1 - grand, w + grand):
            if sr >= 1 and sc >= 1:
                S.append(('p%d_%d' % (sr, sc), lambda r, c, sr=sr, sc=sc: (sr - r, sc - c)))
    return S


def completer_symetrie(g, m, grand=3):
    h, w = H(g), W(g)
    inconnus = {(r, c) for r in range(h) for c in range(w) if g[r][c] == m}
    if not inconnus or len(inconnus) > h * w // 2:
        return None
    valides = []
    for nom, f in _symetries(h, w, grand):
        recouvre = 0
        ok = True
        for r in range(h):
            for c in range(w):
                if (r, c) in inconnus:
                    continue
                y, x = f(r, c)
                if 0 <= y < h and 0 <= x < w and (y, x) not in inconnus:
                    if g[y][x] != g[r][c]:
                        ok = False
                        break
                    recouvre += 1
            if not ok:
                break
        if ok and recouvre >= 0.3 * (h * w - len(inconnus)):
            valides.append(f)
    if not valides:
        return None
    out = [list(r) for r in g]
    reste = set(inconnus)
    for _ in range(4):
        nouveau = set()
        for (r, c) in list(reste):
            for f in valides:
                y, x = f(r, c)
                if 0 <= y < h and 0 <= x < w and (y, x) not in reste and out[y][x] != m:
                    out[r][c] = out[y][x]
                    nouveau.add((r, c))
                    break
        reste -= nouveau
        if not reste or not nouveau:
            break
    if reste:
        return None
    return en_grille(out)


def appr_symetrie(train, tests, dl):
    ins = [p[0] for p in train]
    outs = [p[1] for p in train]
    masques = [m for m in range(10)
               if all(any(m in r for r in g) for g in ins) and not any(any(m in r for r in o) for o in outs)]
    for m in masques:
        dl.verifier()
        yield ('completer_symetrie[masque=%d]' % m, lambda g, m=m: completer_symetrie(g, m))

        def rogne(g, m=m):
            f = completer_symetrie(g, m)
            if f is None:
                return None
            pts = [(r, c) for r in range(H(g)) for c in range(W(g)) if g[r][c] == m]
            return sous_grille(f, min(p[0] for p in pts), min(p[1] for p in pts),
                               max(p[0] for p in pts), max(p[1] for p in pts))
        yield ('completer_symetrie_puis_rogner[masque=%d]' % m, rogne)


# --------------------------------------------------------------------------
# 8. Orchestration
# --------------------------------------------------------------------------

def verifier_prog(fn, train, tests):
    """Sorties de test si fn reproduit TOUTES les sorties d'entraînement, sinon None."""
    for gi, go in train:
        r = appliquer(fn, gi)
        if r != go:
            return None
    res = []
    for g in tests:
        r = appliquer(fn, g)
        if not sortie_ok(r):
            return None
        res.append(r)
    return tuple(res)


def appr_constante(train, tests, dl):
    outs = {p[1] for p in train}
    if len(outs) == 1:
        o = next(iter(outs))
        yield ('constante', lambda g, o=o: o)


def resoudre(tache, budget=10.0):
    """Résout une tâche. Retourne dict(candidats=[(texte, sorties_test)], cpu, delai)."""
    train = [(en_grille(p['input']), en_grille(p['output'])) for p in tache['train']]
    tests = [en_grille(t['input']) for t in tache['test']]
    dl = Horloge(budget)
    cands = []
    vus = set()
    delai = False
    t_niv2 = dl.t0 + 0.45 * budget

    def offrir(texte, fn):
        sor = verifier_prog(fn, train, tests)
        if sor is None or sor in vus:
            return
        vus.add(sor)
        cands.append((texte, sor))
        if len(cands) == 1:
            dl.reduire(max(1.0, 0.3 * budget))

    try:
        for texte, fn in appr_constante(train, tests, dl):
            offrir(texte, fn)
        gen = recherche_dsl(train, tests, dl, t_niv2)
        for item in gen:
            if item is None:
                break
            offrir(*item)
            if len(cands) >= 2:
                break
        if len(cands) < 2:
            for appr in (appr_relative, appr_objets, appr_symetrie, appr_local):
                for texte, fn in appr(train, tests, dl):
                    offrir(texte, fn)
                    if len(cands) >= 2:
                        break
                if len(cands) >= 2:
                    break
        if len(cands) < 2:
            for item in gen:
                if item is None:
                    continue
                offrir(*item)
                if len(cands) >= 2:
                    break
    except Delai:
        delai = True
    return {'candidats': cands, 'cpu': time.process_time() - dl.t0, 'delai': delai}


# --------------------------------------------------------------------------
# 9. Évaluation sur un jeu de tâches
# --------------------------------------------------------------------------

def _travail(args):
    chemin, budget = args
    tid = os.path.basename(chemin)[:-5]
    with open(chemin) as f:
        tache = json.load(f)

    def _alarme(signum, frame):
        raise Delai()
    signal.signal(signal.SIGPROF, _alarme)
    signal.setitimer(signal.ITIMER_PROF, budget + 1.5)
    try:
        res = resoudre(tache, budget)
    except Delai:
        res = {'candidats': [], 'cpu': budget + 1.5, 'delai': True}
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
    # comparaison avec les sorties attendues (jamais lues par le solveur)
    attendus = [en_grille(t['output']) for t in tache['test']]
    c = res['candidats']
    ok = [tuple(attendus) == tuple(s) for _, s in c]
    return {
        'id': tid,
        'resolu1': bool(ok and ok[0]),
        'resolu2': any(ok[:2]),
        'cpu': round(res['cpu'], 3),
        'delai': res['delai'],
        'nb_candidats': len(c),
        'programmes': [t for t, _ in c],
        'ok': ok,
        '_sorties': c,
    }


def evaluer(split, budget=10.0, procs=2, ids=None):
    dossier = os.path.join(DOSSIER, 'data', split)
    fichiers = sorted(os.path.join(dossier, f) for f in os.listdir(dossier) if f.endswith('.json'))
    if ids:
        fichiers = [f for f in fichiers if os.path.basename(f)[:-5] in ids]
    args = [(f, budget) for f in fichiers]
    resultats = []
    if procs > 1:
        import multiprocessing as mp
        with mp.Pool(min(procs, 2)) as pool:
            for i, r in enumerate(pool.imap(_travail, args, chunksize=1)):
                r.pop('_sorties', None)
                resultats.append(r)
                if (i + 1) % 50 == 0:
                    print('  %d/%d  resolus2=%d' % (i + 1, len(args), sum(x['resolu2'] for x in resultats)),
                          file=sys.stderr, flush=True)
    else:
        for a in args:
            r = _travail(a)
            r.pop('_sorties', None)
            resultats.append(r)
    return resultats


def resume(resultats):
    n = len(resultats)
    cpu = [r['cpu'] for r in resultats]
    return {
        'n': n,
        'resolu_2_tentatives': sum(r['resolu2'] for r in resultats),
        'resolu_1_tentative': sum(r['resolu1'] for r in resultats),
        'cpu_moyen': round(sum(cpu) / max(n, 1), 3),
        'cpu_max': round(max(cpu), 3) if cpu else 0,
        'timeouts': sum(r['delai'] for r in resultats),
        'ids_resolus': sorted(r['id'] for r in resultats if r['resolu2']),
    }


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description='Solveur ARC par recherche de programmes')
    ap.add_argument('split', nargs='?', choices=['training', 'evaluation'])
    ap.add_argument('--budget', type=float, default=10.0)
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--tache')
    ap.add_argument('--split-tache', default='training')
    ap.add_argument('--sortie', help='fichier JSON de résultats détaillés')
    ap.add_argument('--ids', help="liste d'ids séparés par des virgules")
    a = ap.parse_args(argv)
    if a.tache:
        chemin = os.path.join(DOSSIER, 'data', a.split_tache, a.tache + '.json')
        r = _travail((chemin, a.budget))
        for t, s in r['_sorties']:
            print('PROGRAMME :', t)
            print('  sortie test[0] :')
            for row in s[0]:
                print('   ', ''.join(str(v) for v in row))
        if not r['_sorties']:
            print('aucun programme trouvé')
        print('1 tentative :', r['resolu1'], '| 2 tentatives :', r['resolu2'], '| cpu : %.2fs' % r['cpu'])
        return 0
    if not a.split:
        ap.print_help()
        return 1
    t0 = time.time()
    res = evaluer(a.split, a.budget, a.procs, set(a.ids.split(',')) if a.ids else None)
    rs = resume(res)
    print(json.dumps({k: v for k, v in rs.items() if k != 'ids_resolus'}, ensure_ascii=False))
    print('durée murale : %.0fs' % (time.time() - t0))
    if a.sortie:
        with open(a.sortie, 'w') as f:
            json.dump({'resume': rs, 'details': res}, f, ensure_ascii=False)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
