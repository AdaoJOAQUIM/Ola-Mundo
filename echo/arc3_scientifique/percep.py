"""Perception orientée objets pour Le Scientifique (aucun identifiant de jeu, aucun LLM).

Contenu :
  - composantes connexes, couleur de fond, caractéristiques d'une couleur ;
  - détecteur de bandes HUD (compteurs/minuteries monotones) ;
  - extraction de mouvement par translation (groupement par co-mouvement) ;
  - gabarit de sprite et repérage du gabarit dans une image (suivi inter-images).
"""
from __future__ import annotations

import numpy as np

H = W = 64


def fond(frame: np.ndarray) -> int:
    c, n = np.unique(frame, return_counts=True)
    return int(c[np.argmax(n)])


def composantes(frame: np.ndarray, masque=None, connexite8=False):
    """Composantes connexes par couleur (hors fond). Renvoie (liste d'objets, fond)."""
    h, w = frame.shape
    bg = fond(frame)
    vu = np.zeros((h, w), dtype=bool)
    if masque is not None:
        vu |= masque
    voisins = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connexite8:
        voisins += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    objets = []
    for y0 in range(h):
        for x0 in range(w):
            if vu[y0, x0]:
                continue
            c = int(frame[y0, x0])
            vu[y0, x0] = True
            if c == bg:
                continue
            pile = [(y0, x0)]
            cells = []
            while pile:
                y, x = pile.pop()
                cells.append((y, x))
                for dy, dx in voisins:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not vu[ny, nx] and frame[ny, nx] == c:
                        vu[ny, nx] = True
                        pile.append((ny, nx))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            my, mx = sum(ys) / len(ys), sum(xs) / len(xs)
            cy, cx = min(cells, key=lambda p: (p[0] - my) ** 2 + (p[1] - mx) ** 2)
            objets.append({"couleur": c, "taille": len(cells), "x": cx, "y": cy,
                           "bbox": (min(xs), min(ys), max(xs), max(ys))})
    return objets, bg


def caracteristiques_couleurs(frame: np.ndarray, masque=None) -> dict:
    """Primitives booléennes par couleur, utilisées par les gabarits de règles (prior appris).

    Aucune valeur de couleur n'est utilisée comme caractéristique : seulement forme, aire, position.
    """
    objets, bg = composantes(frame, masque)
    par = {}
    for o in objets:
        par.setdefault(o["couleur"], []).append(o)
    total = float(frame.size)
    cars = {}
    for c in np.unique(frame).tolist():
        obs = par.get(c, [])
        aire = int((frame == c).sum())
        if c == bg:
            cars[c] = {"aire_grande": True, "aire_petite": False, "une_compo": True, "multi_compo": False,
                       "bord": True, "rectangle": False, "fond": True, "carre_petit": False}
            continue
        bords = any(o["bbox"][0] == 0 or o["bbox"][1] == 0 or o["bbox"][2] == frame.shape[1] - 1
                    or o["bbox"][3] == frame.shape[0] - 1 for o in obs)
        rect = bool(obs) and all(
            o["taille"] >= 0.95 * (o["bbox"][2] - o["bbox"][0] + 1) * (o["bbox"][3] - o["bbox"][1] + 1)
            for o in obs)
        petit = bool(obs) and all(o["taille"] <= 36 for o in obs)
        cars[c] = {"aire_grande": aire / total >= 0.08, "aire_petite": aire / total < 0.01,
                   "une_compo": len(obs) <= 1, "multi_compo": len(obs) >= 4,
                   "bord": bords, "rectangle": rect, "fond": False, "carre_petit": petit}
    return cars


# ----------------------------------------------------------------------- HUD
class Hud:
    """Bandes (lignes/colonnes) qui changent presque à chaque transition sans jamais revenir en arrière :
    compteurs de pas, minuteries. Elles sont masquées de l'état et de l'extraction de mouvement."""

    SEUIL = 0.7
    TOL_REV = 0.1
    POINTS = (6, 12, 24, 48, 96, 192, 384)

    def __init__(self):
        self.masque = None
        self.n = 0
        self.lig = np.zeros(H, dtype=np.int32)
        self.col = np.zeros(W, dtype=np.int32)
        self.rev_l = np.zeros(H, dtype=np.int32)
        self.rev_c = np.zeros(W, dtype=np.int32)
        self.paires = {}

    def reinitialiser_vie(self):
        self.paires = {}

    def maj(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Intègre une transition ; renvoie True si le masque a changé."""
        ys, xs = np.nonzero(a != b)
        if len(ys) == 0:
            self.n += 1
            return self._recalcul()
        if np.bincount(ys).max() >= 10 or np.bincount(xs).max() >= 10:
            if len(ys) > 40:
                self.paires = {}
        rl, rc = set(), set()
        for y, x in zip(ys.tolist(), xs.tolist()):
            va, vb = int(a[y, x]), int(b[y, x])
            p = self.paires.setdefault((y, x), set())
            if (vb, va) in p:
                rl.add(y)
                rc.add(x)
            p.add((va, vb))
        for y in rl:
            self.rev_l[y] += 1
        for x in rc:
            self.rev_c[x] += 1
        self.lig[np.unique(ys)] += 1
        self.col[np.unique(xs)] += 1
        self.n += 1
        return self._recalcul()

    def _recalcul(self) -> bool:
        if self.n not in self.POINTS:
            return False
        m = np.zeros((H, W), dtype=bool)
        lignes = (self.lig >= self.SEUIL * self.n) & (self.rev_l <= self.TOL_REV * self.n)
        colonnes = (self.col >= self.SEUIL * self.n) & (self.rev_c <= self.TOL_REV * self.n)
        m[lignes, :] = True
        m[:, colonnes] = True
        # sécurité : un masque qui couvre plus de 30 % de l'écran n'est pas un HUD
        if m.mean() > 0.3:
            m[:] = False
        ancien = self.masque
        if (ancien is None and m.any()) or (ancien is not None and not np.array_equal(ancien, m)):
            self.masque = m
            return True
        return False


# ------------------------------------------------------------------ mouvement
def extraire_mouvement(a: np.ndarray, b: np.ndarray, hud=None, max_shift=14, min_cells=2):
    """Cherche la translation (dy,dx) qui explique le mieux les cellules changées.

    Score = min(#arrivées, #départs) : une vraie translation produit les deux ; une recoloration locale
    adjacente à une zone de même couleur n'en produit qu'un.
    Renvoie une liste de mouvements {dy,dx,cellules_a (Nx2), couleurs_a (N)} (au plus 2, décalages distincts).
    """
    ch = a != b
    if hud is not None:
        ch &= ~hud
    res = []
    for _ in range(2):
        ys, xs = np.nonzero(ch)
        n = len(ys)
        if n < 2 * min_cells:
            break
        meilleur = None
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                if dy == 0 and dx == 0:
                    continue
                sy, sx = ys - dy, xs - dx
                ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
                if ok.sum() < min_cells:
                    continue
                arr = np.zeros(n, dtype=bool)
                arr[ok] = a[sy[ok], sx[ok]] == b[ys[ok], xs[ok]]
                ty, tx = ys + dy, xs + dx
                ok2 = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
                dep = np.zeros(n, dtype=bool)
                dep[ok2] = b[ty[ok2], tx[ok2]] == a[ys[ok2], xs[ok2]]
                s = min(int(arr.sum()), int(dep.sum()))
                if meilleur is None or s > meilleur[0]:
                    meilleur = (s, dy, dx, arr, dep)
        if meilleur is None or meilleur[0] < min_cells or meilleur[0] < 0.25 * n:
            break
        s, dy, dx, arr, dep = meilleur
        src = set(zip((ys[arr] - dy).tolist(), (xs[arr] - dx).tolist()))
        src |= set(zip(ys[dep].tolist(), xs[dep].tolist()))
        cells = np.array(sorted(src), dtype=np.int32)
        res.append({"dy": dy, "dx": dx, "cellules_a": cells, "couleurs_a": a[cells[:, 0], cells[:, 1]].astype(np.int8),
                    "score": s})
        # retire les cellules expliquées
        for (y, x) in src:
            ch[y, x] = False
            if 0 <= y + dy < H and 0 <= x + dx < W:
                ch[y + dy, x + dx] = False
    return res


# -------------------------------------------------------------------- gabarit
class Gabarit:
    """Sprite : cellules relatives à l'ancre (coin haut-gauche de la boîte) avec leurs couleurs."""

    def __init__(self, cellules, couleurs):
        cellules = np.asarray(cellules, dtype=np.int32)
        self.y0 = int(cellules[:, 0].min())
        self.x0 = int(cellules[:, 1].min())
        self.ty = cellules[:, 0] - self.y0
        self.tx = cellules[:, 1] - self.x0
        self.tc = np.asarray(couleurs, dtype=np.int8)
        self.h = int(self.ty.max()) + 1
        self.w = int(self.tx.max()) + 1
        self.signature = (self.h, self.w, tuple(sorted(zip(self.ty.tolist(), self.tx.tolist(), self.tc.tolist()))))
        # couleur la plus rare du gabarit : sert à générer les ancres candidates
        cs, ns = np.unique(self.tc, return_counts=True)
        self.c_rare = int(cs[np.argmin(ns)])
        sel = np.nonzero(self.tc == self.c_rare)[0][0]
        self.ref = (int(self.ty[sel]), int(self.tx[sel]))

    def masque_en(self, ay, ax):
        m = np.zeros((H, W), dtype=bool)
        yy, xx = self.ty + ay, self.tx + ax
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        m[yy[ok], xx[ok]] = True
        return m

    def localiser(self, frame: np.ndarray, pres=None, seuil=0.9):
        """Ancre (ay,ax) la plus proche de `pres` dont le taux de concordance >= seuil, sinon None."""
        ys, xs = np.nonzero(frame == self.c_rare)
        if len(ys) == 0:
            return None
        ay = ys - self.ref[0]
        ax = xs - self.ref[1]
        ok = (ay >= -0) & (ax >= 0) & (ay + self.h <= H) & (ax + self.w <= W)
        ay, ax = ay[ok], ax[ok]
        if len(ay) == 0:
            return None
        yy = ay[:, None] + self.ty[None, :]
        xx = ax[:, None] + self.tx[None, :]
        taux = (frame[yy, xx] == self.tc[None, :]).mean(axis=1)
        bons = np.nonzero(taux >= seuil)[0]
        if len(bons) == 0:
            return None
        if pres is None:
            k = bons[np.argmax(taux[bons])]
        else:
            d = (ay[bons] - pres[0]) ** 2 + (ax[bons] - pres[1]) ** 2
            k = bons[np.argmin(d)]
        return int(ay[k]), int(ax[k])
