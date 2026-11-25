"""Hypothèses de but (a priori pragmatiques) et planification BFS dans les états prédits (poses des corps + entités consommées)."""
from __future__ import annotations

from collections import deque

import numpy as np

from .transitions import D4


class Contexte:
    """Scène figée pour la planification : terrain (couleurs sans les corps), solides, cible, entités consommables."""

    def __init__(self, terrain, corps, solide_cols, hud, cible=None, proprio=None, consommables=None, actions=()):
        self.terrain = terrain                    # (H, W) int, -1 = hors jeu (HUD)
        self.H, self.W = terrain.shape
        self.corps = corps
        self.lut = np.zeros(17, bool)
        for c in solide_cols:
            self.lut[c] = True
        self.lut[16] = True                        # -1 -> indice 16 par masque
        self.cible = cible                         # masque booléen (H, W) de la cible « toucher »
        self.proprio = proprio                     # (H, W) int : id d'entité consommable, -1 sinon
        self.consommables = consommables or {}
        self.actions = tuple(actions)
        self.terr_idx = np.where(terrain < 0, 16, terrain)

    def evaluer(self, pix):
        """0 bloqué ; 1 libre ; 2 libre et touche la cible."""
        r, c = pix[:, 0], pix[:, 1]
        if r.min() < 0 or c.min() < 0 or r.max() >= self.H or c.max() >= self.W:
            return 0
        cols = self.terr_idx[r, c]
        bl = self.lut[cols]
        touche = False
        if self.cible is not None:
            m = self.cible[r, c]
            touche = bool(m.any())
            if touche:
                bl = bl & ~m
        if bl.any():
            return 0
        return 2 if touche else 1

    def consomme(self, pix):
        if self.proprio is None:
            return ()
        ids = self.proprio[pix[:, 0].clip(0, self.H - 1), pix[:, 1].clip(0, self.W - 1)]
        ids = ids[ids >= 0]
        if not len(ids):
            return ()
        return tuple(sorted(set(int(i) for i in ids)))


def bfs(ctx, depart, test_but, cap=30000, interdits=()):
    """BFS sur (poses, consommés). test_but(poses, gone, touche) -> bool. Renvoie la liste d'actions ou None."""
    t0 = (tuple(depart), frozenset())
    if test_but(t0[0], t0[1], False):
        return []
    vus = {t0: None}
    file_ = deque([t0])
    n = 0
    while file_ and n < cap:
        s = file_.popleft()
        n += 1
        poses, gone = s
        for a in ctx.actions:
            if (poses, a) in interdits:
                continue
            nouv = []
            bouge = False
            touche = False
            g2 = set(gone)
            for b, p in zip(ctx.corps, poses):
                q = b.appliquer(a, p)
                if q is None:
                    nouv.append(p)
                    continue
                pix = b.pixels_pose(q)
                ev = ctx.evaluer(pix)
                if ev == 0:
                    nouv.append(p)
                    continue
                nouv.append(q)
                if q != p:
                    bouge = True
                if ev == 2:
                    touche = True
                g2.update(ctx.consomme(pix))
            if not bouge:
                continue
            ns = (tuple(nouv), frozenset(g2))
            if ns in vus:
                continue
            vus[ns] = (s, a)
            if test_but(ns[0], ns[1], touche):
                chemin = [a]
                cur = s
                while vus[cur] is not None:
                    pr, ac = vus[cur]
                    chemin.append(ac)
                    cur = pr
                return chemin[::-1]
            file_.append(ns)
    return None


# ------------------------------------------------------------------ hypothèses de but
class But:
    def __init__(self, nom, prior, test, cible=None, detail=""):
        self.nom, self.prior, self.test, self.cible, self.detail = nom, prior, test, cible, detail
        self.rejete = False
        self.essais = 0

    def __repr__(self):
        return f"But({self.nom} p={self.prior:.2f} {self.detail})"


def poses_alignees(corps, statiques):
    """But « motif = référence » : parties du corps dont une jumelle statique existe (même couleur et forme D4).
    Renvoie l'ensemble des poses (M, tr, tc) où chaque partie clé recouvre exactement une jumelle distincte, ou None."""
    parts = corps.parts or []
    if len(parts) < 2:
        return None
    cles = []
    for pr in parts:
        jum = [s for s in statiques if s.col == pr["col"] and s.kd4 == pr["kd4"]]
        if jum:
            cles.append((pr, jum))
    if not cles or len(cles) == len(parts):
        return None
    # la partie la plus lourde ne doit pas être une clé (sinon l'ambiguïté est totale)
    poses = set()
    pr0, jum0 = cles[0]
    for M in range(8):
        rel = pr0["pix"][:, :2] @ D4[M].T
        for s in jum0:
            rr, cc = np.nonzero(s.masque)
            cible0 = np.stack([rr + s.r, cc + s.c], 1)
            # translation : alignement des coins minimaux (formes égales à D4 près)
            t = cible0.min(0) - rel.min(0)
            if set(map(tuple, (rel + t).tolist())) != set(map(tuple, cible0.tolist())):
                continue
            ok = True
            for pr, jum in cles[1:]:
                rel2 = pr["pix"][:, :2] @ D4[M].T + t
                st2 = set(map(tuple, rel2.tolist()))
                if not any(set(map(tuple, np.argwhere(s2.masque) + np.array([s2.r, s2.c]))) == st2 for s2 in jum):
                    ok = False
                    break
            if ok:
                poses.add((M, int(t[0]), int(t[1])))
    return poses or None
