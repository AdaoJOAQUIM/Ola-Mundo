"""Contexte `h` fourni aux compétences : perception générique sur les seules observations officielles
(image, actions disponibles, levels_completed, state). Aucune information interne du jeu.

Tout est déterministe. Les coûts sont bornés (64x64 cellules, listes plafonnées).
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

MAX_OBJ = 120
MAX_LOG = 600


def _cle_action(a):
    return a if a[0] != "C" else ("C", a[1], a[2])


class H:
    def __init__(self):
        self.f = None
        self.p = None
        self.f0 = None
        self.dispo = ()
        self.niv = 0
        self.n = 0
        self.nn = 0
        self.reste = 0
        self.etat = ""
        self.der = None
        self.dchg = -1
        self.mem = {}
        self.glob = {}
        self.masque = None
        self.log = []
        self.fx = {}
        self.vus = set()
        self.nouveau = True
        self.niv_actions = []
        self.dernier_gain = -1
        self._cache_obj = (None, None)
        self._n0_niveau = 0

    # ---------------------------------------------------------- mise à jour (appelée par l'exécuteur)
    def maj(self, obs, masque, derniere_action):
        f = np.array(obs["frame"][-1], dtype=np.int8)
        niv = int(obs.get("levels_completed") or 0)
        nouveau_niveau = self.f is None or niv != self.niv
        if self.f is not None and niv > self.niv:
            self.niv_actions.append(int(obs["actions_taken"]) - self._n0_niveau)
            self.dernier_gain = int(obs["actions_taken"])
        self.masque = masque
        if nouveau_niveau:
            self.p = None
            self.f0 = f
            self.log = []
            self.fx = {}
            self.vus = set()
            self._n0_niveau = int(obs["actions_taken"]) if self.f is not None else 0
            self.dchg = -1
            self.der = None
        else:
            self.p = self.f
            m = self.chgmask(f)
            self.dchg = int(m.sum())
            self.der = derniere_action
            if derniere_action is not None:
                self.log.append((derniere_action, self.dchg))
                if len(self.log) > MAX_LOG:
                    self.log = self.log[-MAX_LOG:]
                e = self.fx.setdefault(_cle_action(derniere_action), [0, 0])
                e[0] += 1
                e[1] += 1 if self.dchg > 0 else 0
        self.f = f
        self.niv = niv
        self.n = int(obs["actions_taken"])
        self.nn = self.n - self._n0_niveau
        self.reste = int(obs.get("actions_remaining", 0) or 0)
        self.etat = str(obs.get("state"))
        self.dispo = tuple(int(a) for a in obs.get("available_actions", []) if int(a) != 0)
        k = self.hach(f)
        self.nouveau = k not in self.vus
        self.vus.add(k)
        return nouveau_niveau

    # ---------------------------------------------------------- constructeurs d'actions
    def A(self, n):
        return ("A", int(n))

    def C(self, x, y):
        return ("C", int(x), int(y))

    def R(self):
        return ("R",)

    # ---------------------------------------------------------- perception
    def hach(self, f=None):
        f = self.f if f is None else f
        if self.masque is not None and self.masque.shape == f.shape and self.masque.any():
            f = f.copy()
            f[self.masque] = 0
        return hashlib.blake2b(f.tobytes(), digest_size=10).digest()

    def dans(self, y, x):
        return 0 <= y < 64 and 0 <= x < 64

    def couleurs(self, f=None):
        f = self.f if f is None else f
        return np.bincount(f.ravel().astype(np.int64), minlength=16)

    def rect(self, y0, x0, y1, x1, f=None):
        f = self.f if f is None else f
        return f[max(0, y0):y1 + 1, max(0, x0):x1 + 1]

    def chgmask(self, f=None):
        """Cellules changées entre l'image précédente et `f` (courante par défaut), hors masque HUD."""
        f = self.f if f is None else f
        base = self.p if f is self.f and self.p is not None else None
        if base is None:
            base = self.f if f is not self.f and self.f is not None else None
        if base is None or base.shape != f.shape:
            return np.zeros((64, 64), dtype=bool)
        m = base != f
        if self.masque is not None and self.masque.shape == m.shape:
            m = m & ~self.masque
        return m

    def objets(self, f=None):
        """Composantes connexes (4-voisinage) par couleur, hors fond (couleur la plus fréquente) et hors masque.
        Chaque objet : dict c (couleur), n (taille), y/x (une cellule de l'objet proche du centre),
        cy/cx (centre), b (y0,x0,y1,x1). Triés : petits objets d'abord. Au plus MAX_OBJ."""
        f = self.f if f is None else f
        if self._cache_obj[0] is f:
            return self._cache_obj[1]
        h, w = f.shape
        cnt = np.bincount(f.ravel().astype(np.int64), minlength=16)
        fond = int(np.argmax(cnt))
        vu = np.zeros((h, w), dtype=bool)
        if self.masque is not None and self.masque.shape == f.shape:
            vu |= self.masque
        vu |= (f == fond)
        out = []
        fl = f.tolist()
        for y0 in range(h):
            for x0 in range(w):
                if vu[y0, x0]:
                    continue
                c = fl[y0][x0]
                pile = [(y0, x0)]
                vu[y0, x0] = True
                cells = []
                while pile:
                    y, x = pile.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not vu[ny, nx] and fl[ny][nx] == c:
                            vu[ny, nx] = True
                            pile.append((ny, nx))
                ys = [q[0] for q in cells]
                xs = [q[1] for q in cells]
                my, mx = sum(ys) / len(ys), sum(xs) / len(xs)
                cy, cx = min(cells, key=lambda q: (q[0] - my) ** 2 + (q[1] - mx) ** 2)
                out.append({"c": c, "n": len(cells), "y": cy, "x": cx, "cy": my, "cx": mx,
                            "b": (min(ys), min(xs), max(ys), max(xs))})
        out.sort(key=lambda o: (o["n"], o["y"], o["x"]))
        out = out[:MAX_OBJ]
        self._cache_obj = (f, out)
        return out

    def boites(self, mask):
        """Composantes 8-connexes d'un masque booléen : liste (y0,x0,y1,x1,n), grandes d'abord (max 30)."""
        h, w = mask.shape
        vu = np.zeros((h, w), dtype=bool)
        res = []
        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if vu[y0, x0]:
                continue
            pile = [(y0, x0)]
            vu[y0, x0] = True
            a, b, c, d, n = y0, x0, y0, x0, 0
            while pile:
                y, x = pile.pop()
                n += 1
                a, b, c, d = min(a, y), min(b, x), max(c, y), max(d, x)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not vu[ny, nx]:
                            vu[ny, nx] = True
                            pile.append((ny, nx))
            res.append((a, b, c, d, n))
        res.sort(key=lambda r: -r[4])
        return res[:30]

    def mouvements(self, rayon=12):
        """Pour chaque région changée (p -> f) : meilleure translation (dy,dx) des cellules non-fond qui
        ont disparu. Renvoie (y0,x0,y1,x1,dy,dx,score) avec score = part des cellules expliquées (0..1)."""
        if self.p is None:
            return []
        m = self.chgmask()
        if not m.any():
            return []
        p, f = self.p, self.f
        fond = int(np.argmax(np.bincount(p.ravel().astype(np.int64), minlength=16)))
        res = []
        for (y0, x0, y1, x1, n) in self.boites(self._dilate(m)):
            sub = m[y0:y1 + 1, x0:x1 + 1]
            ys, xs = np.nonzero(sub & (p[y0:y1 + 1, x0:x1 + 1] != fond))
            if len(ys) == 0:
                continue
            ys = ys + y0
            xs = xs + x0
            cols = p[ys, xs]
            meilleur = (0, 0, 0.0)
            for dy in range(-rayon, rayon + 1):
                for dx in range(-rayon, rayon + 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = ys + dy, xs + dx
                    ok = (ny >= 0) & (ny < 64) & (nx >= 0) & (nx < 64)
                    if not ok.any():
                        continue
                    sc = float((f[ny[ok], nx[ok]] == cols[ok]).sum()) / len(ys)
                    if sc > meilleur[2] + 1e-9:
                        meilleur = (dy, dx, sc)
            res.append((y0, x0, y1, x1, meilleur[0], meilleur[1], round(meilleur[2], 3)))
        return res

    @staticmethod
    def _dilate(m):
        d = m.copy()
        d[1:, :] |= m[:-1, :]
        d[:-1, :] |= m[1:, :]
        d[:, 1:] |= m[:, :-1]
        d[:, :-1] |= m[:, 1:]
        return d

    def essais(self, a):
        """[nb d'essais, nb ayant changé l'image] de l'action `a` dans ce niveau."""
        return self.fx.get(_cle_action(a), [0, 0])

    def chemin(self, graphe, depart, buts):
        """BFS dans {noeud: [(action, noeud2), ...]} ; renvoie la liste d'actions vers le but le plus proche
        (liste vide si depart est un but), ou None. Taille plafonnée à 20000 noeuds."""
        if depart in buts:
            return []
        prec = {depart: None}
        file = deque([depart])
        while file and len(prec) < 20000:
            k = file.popleft()
            for a, k2 in graphe.get(k, ()):
                if k2 in prec:
                    continue
                prec[k2] = (k, a)
                if k2 in buts:
                    ch = []
                    cur = k2
                    while prec[cur] is not None:
                        cur, a2 = prec[cur][0], prec[cur][1]
                        ch.append(a2)
                    return ch[::-1]
                file.append(k2)
        return None
