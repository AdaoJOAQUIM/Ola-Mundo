"""Modèle causal local : corps rigides à pose D4 + translation, règles d'action (translation / rotation autour d'un pivot),
hypothèses bayésiennes sur les couleurs (solide, mortelle, gagnante) avec longueur de description."""
from __future__ import annotations

import math

import numpy as np

from .transitions import D4, d4_mul, d4_inv, pixels


# ----------------------------------------------------------------------------- hypothèses sur les couleurs
class HypCouleurs:
    """Posterior sur les sous-ensembles H de couleurs : observation (S, positive) = « l'ensemble de couleurs S est touché ;
    l'effet a eu lieu ». Vraisemblance : effet <=> S inter H non vide (erreur eps). A priori : exp(-lam |H|) (longueur de description)."""

    def __init__(self, lam=1.5, eps=0.03):
        self.lam, self.eps = lam, eps
        self.obs = {}
        self._cache = None

    def ajouter(self, couleurs, positif):
        m = 0
        for c in couleurs:
            m |= 1 << int(c)
        if m == 0:
            return
        k = (m, bool(positif))
        self.obs[k] = self.obs.get(k, 0) + 1
        self._cache = None

    def _calc(self):
        if self._cache is not None:
            return self._cache
        if not self.obs:
            self._cache = []
            return self._cache
        univ = 0
        for (m, _p) in self.obs:
            univ |= m
        bits = [b for b in range(16) if univ >> b & 1]
        n = len(bits)
        idx = {b: i for i, b in enumerate(bits)}
        H = np.arange(1 << n, dtype=np.int64)
        ll = np.zeros(1 << n)
        for (m, p), cnt in self.obs.items():
            mm = 0
            for b in bits:
                if m >> b & 1:
                    mm |= 1 << idx[b]
            inter = (H & mm) != 0
            ll += cnt * np.where(inter == p, 0.0, math.log(self.eps))
        pop = np.zeros(1 << n)
        for i in range(n):
            pop += (H >> i) & 1
        lp = ll - self.lam * pop
        ordre = np.argsort(-lp)[:8]
        top = []
        for o in ordre:
            mask = 0
            for i in range(n):
                if int(o) >> i & 1:
                    mask |= 1 << bits[i]
            top.append((mask, float(lp[o])))
        self._cache = top
        return top

    def meilleurs(self):
        t = self._calc()
        return t

    def map_cols(self):
        t = self._calc()
        if not t:
            return set()
        return _cols(t[0][0])

    def map_union(self, tol=1e-6):
        t = self._calc()
        if not t:
            return set()
        s = set()
        for m, lp in t:
            if lp >= t[0][1] - tol:
                s |= _cols(m)
        return s


def _cols(m):
    return {b for b in range(16) if m >> b & 1}


# ----------------------------------------------------------------------------- corps rigides
class Corps:
    def __init__(self, pix, ident):
        self.pix = np.asarray(pix, dtype=int)          # (n, 3) r, c, col à la création (pose identité, t=0)
        self.id = ident
        self.rel = [self.pix[:, :2] @ D4[i].T for i in range(8)]
        self.cols0 = self.pix[:, 2].copy()              # couleurs d'origine (jamais modifiées)
        self.cols = self.pix[:, 2].copy()               # -1 = joker (couleur de sélection : change quand le corps est (dé)sélectionné)
        self.M, self.t = 0, (0, 0)
        self.ops = {}                                   # action -> ('T', d) | ('R', g, q)
        self.rot_tab = {}                               # (action, M) -> (M', (dr, dc)) : effet exact des rotations, appris pose par pose
        self.parts = None                               # [(masque indices)] sous-corps par couleur-composante, rempli à la création

    def canon(self, M, t):
        """Pose équivalente d'indice D4 minimal (les corps symétriques ont plusieurs indices pour la même empreinte)."""
        sigs = {}
        for m in range(8):
            pts = self.rel[m] - self.rel[m].min(0)
            sigs[m] = frozenset(map(tuple, np.concatenate([pts, self.cols[:, None]], 1).tolist()))
        for m in range(8):
            if sigs[m] == sigs[M]:
                if m == M:
                    return M, tuple(t)
                d = self.rel[M].min(0) - self.rel[m].min(0)
                return m, (int(t[0] + d[0]), int(t[1] + d[1]))
        return M, tuple(t)

    def monde(self, M=None, t=None):
        M = self.M if M is None else M
        t = self.t if t is None else t
        return self.rel[M] + np.array(t)

    def pose(self):
        return (self.M, self.t[0], self.t[1])

    def appliquer(self, a, pose):
        """Pose résultante de l'action a depuis `pose` (M, tr, tc), ou None si le corps n'a pas de règle pour a."""
        op = self.ops.get(a)
        if op is None:
            return None
        M, tr, tc = pose
        if op[0] == "T":
            return (M, tr + op[1][0], tc + op[1][1])
        _, g, q = op
        k = (a, M)
        if k in self.rot_tab:
            M2, dv = self.rot_tab[k]
            return (M2, tr + dv[0], tc + dv[1])
        if self.rot_tab and any(kk[0] == a for kk in self.rot_tab):
            pass
        M2 = d4_mul(g, M)
        dv = (D4[M] - D4[M2]) @ q
        return (M2, tr + int(round(dv[0])), tc + int(round(dv[1])))

    def pixels_pose(self, pose):
        return self.rel[pose[0]] + np.array([pose[1], pose[2]])


class Modele:
    """Règles apprises : corps, solides, mortelles, gagnantes, consommables, réversibilité."""

    def __init__(self):
        self.corps = []
        self.solide = HypCouleurs(lam=1.5)
        self.mortel = HypCouleurs(lam=2.0)
        self.gagnant = HypCouleurs(lam=2.0)
        self.consomme = set()          # (couleur, forme exacte) disparaissant au contact
        self.fill_cnt = {}
        self.vu_libre = set()          # couleurs vues sous l'empreinte d'un déplacement réussi
        self.n_surprises = 0
        self.ops_vus = {}              # action -> nb d'observations de son effet

    # ---- empreinte -> couleurs de la scène autour
    def couleurs_dest(self, frame, pix_dest, pix_src):
        """Couleurs sous l'empreinte destination, hors pixels de l'ancienne empreinte (dans l'image avant)."""
        H, W = frame.shape
        src = set(map(tuple, pix_src[:, :2].tolist()))
        cols = set()
        hors = False
        for r, c in pix_dest[:, :2].tolist():
            if (r, c) in src:
                continue
            if 0 <= r < H and 0 <= c < W:
                cols.add(int(frame[r, c]))
            else:
                hors = True
        return cols, hors

    def fill(self, defaut):
        if not self.fill_cnt:
            return defaut
        return max(self.fill_cnt.items(), key=lambda kv: kv[1])[0]

    def noter_fill(self, cols):
        for c in cols:
            self.fill_cnt[int(c)] = self.fill_cnt.get(int(c), 0) + 1
