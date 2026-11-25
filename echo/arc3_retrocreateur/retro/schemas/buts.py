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
        self.tip_exempt = None                    # (H, W) bool : pointes des autres corps (franchissables par les pointes seulement)
        self.tipcol = None

    def evaluer(self, pix, b=None):
        """0 bloqué ; 1 libre ; 2 libre et touche la cible."""
        r, c = pix[:, 0], pix[:, 1]
        if r.min() < 0 or c.min() < 0 or r.max() >= self.H or c.max() >= self.W:
            return 0
        cols = self.terr_idx[r, c]
        bl = self.lut[cols]
        if self.tip_exempt is not None and b is not None:
            tm = (b.cols == self.tipcol) & self.tip_exempt[r, c]
            bl = bl & ~tm
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
                ev = ctx.evaluer(pix, b)
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


# ------------------------------------------------------------------ assemblage : toutes les pointes appariées entre objets
def _comp_tips(pts_tip):
    """Coins haut-gauche des pointes (composantes 4-connexes) d'un ensemble de pixels."""
    s = set(map(tuple, pts_tip.tolist()))
    vu, res = set(), []
    for p in sorted(s):
        if p in vu:
            continue
        pile = [p]; vu.add(p); comp = []
        while pile:
            r, c = pile.pop(); comp.append((r, c))
            for q in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if q in s and q not in vu:
                    vu.add(q); pile.append(q)
        res.append(min(comp))
    return res


def couleur_pointe(objets):
    """Couleur « connecteur » : petite pièce (<= 40 pixels de cette couleur dans l'objet) présente dans >= 2 objets."""
    cnt = {}
    for o in objets:
        cols, n = np.unique(o["cols"], return_counts=True)
        for c, k in zip(cols.tolist(), n.tolist()):
            if c >= 0 and k <= 40:
                cnt[c] = cnt.get(c, 0) + 1
    if not cnt:
        return None
    c, k = max(cnt.items(), key=lambda kv: kv[1])
    return c if k >= 2 else None


def assembler(fixes, mobiles, tipcol, bornes=(1, 64, 0, 64), max_sol=40, ancre=None):
    """fixes : [{'pts','cols'}] (monde) ; mobiles : [{'id','rel':{M:(n,2)},'cols','Ms':[M..]}].
    Cherche des poses (M, tr, tc) des mobiles telles que toutes les pointes (couleur tipcol) soient appariées 2 à 2 et que les
    objets ne se recouvrent qu'aux pointes appariées. Renvoie une liste de solutions {id: (M, tr, tc)}."""
    r0, r1, c0, c1 = bornes
    base = []
    for f in fixes:
        tm = f["cols"] == tipcol
        base.append((f["pts"], tm))
    sols = []
    ancre_cur = [None]

    def libres(placed):
        out = []
        for (pp, tm) in placed:
            out += [tuple(x) for x in _comp_tips(pp[tm])]
        return out

    def rec(restants, placed, poses):
        if len(sols) >= max_sol:
            return
        if not restants:
            cnt = {}
            for pp, tm in placed:
                for p in map(tuple, pp[tm].tolist()):
                    cnt[p] = cnt.get(p, 0) + 1
            if cnt and all(v == 2 for v in cnt.values()):
                sols.append((dict(poses), ancre_cur[0]))
            return
        for k, b in enumerate(restants):
            tm0 = b["cols"] == tipcol
            for M in b["Ms"]:
                rel = b["rel"][M]
                tcs = _comp_tips(rel[tm0])
                libs = libres(placed)
                vus = set()
                for tc in tcs:
                    for u in libs:
                        t = (u[0] - tc[0], u[1] - tc[1])
                        if t in vus:
                            continue
                        vus.add(t)
                        q = rel + np.array(t)
                        if q[:, 0].min() < r0 or q[:, 0].max() >= r1 or q[:, 1].min() < c0 or q[:, 1].max() >= c1:
                            continue
                        sq = set(map(tuple, q.tolist())); tq = set(map(tuple, q[tm0].tolist()))
                        ok = True
                        for pp, tm in placed:
                            sp = set(map(tuple, pp.tolist())); tp = set(map(tuple, pp[tm].tolist()))
                            if not (sq & sp) <= (tq & tp):
                                ok = False
                                break
                        if ok:
                            rec(restants[:k] + restants[k + 1:], placed + [(q, tm0)], {**poses, b["id"]: (M, int(t[0]), int(t[1]))})

    if fixes:
        rec(list(mobiles), base, {})
    else:
        # sans objet fixe : chaque mobile à tour de rôle sert d'ancre (il ne bouge pas)
        for a in range(len(mobiles)):
            b = mobiles[a]
            if ancre is not None and b["id"] != ancre:
                continue
            ancre_cur[0] = b["id"]
            M0 = b["M0"]
            q = b["rel"][M0] + np.array(b["t0"])
            rec([m for i, m in enumerate(mobiles) if i != a], [(q, b["cols"] == tipcol)], {b["id"]: (M0, int(b["t0"][0]), int(b["t0"][1]))})
    return sols
