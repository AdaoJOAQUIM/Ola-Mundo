"""Agent à schémas causaux locaux : perception sans grille, sondage des actions, induction de règles (corps rigides,
solides / mortelles / gagnantes par posterior bayésien avec longueur de description), planification BFS dans le modèle,
découverte de but par hypothèses pragmatiques testées au moindre coût.

Contrat : `choisir(obs)` -> action {"name","data"} ou None (abandon : l'appelant bascule sur le dernier recours).
Aucun identifiant de jeu, aucun réseau, aucun modèle de langue.
"""
from __future__ import annotations

import random

import numpy as np

from .entites import composantes, couleur_fond
from .transitions import expliquer, pixels, trouver_d4, D4, d4_mul
from .modele import Corps, Modele, HypCouleurs
from .buts import Contexte, bfs, But, poses_alignees

MOUV = (1, 2, 3, 4, 5)


def _hud_masque(frame):
    """Bandeaux : composantes minces et longues collées au bord (compteurs, jauges). Renvoie un masque booléen."""
    H, W = frame.shape
    masque = np.zeros((H, W), bool)
    bg = couleur_fond(frame)
    for e in composantes(frame, bg):
        if e.h <= 2 and e.w >= 6 and (e.r <= 1 or e.r + e.h >= H - 1):
            masque[e.r:e.r + e.h, :] = True
        elif e.w <= 2 and e.h >= 6 and (e.c <= 1 or e.c + e.w >= W - 1):
            masque[:, e.c:e.c + e.w] = True
    return masque


class AgentSchema:
    def __init__(self, graine=0, options=None):
        self.rng = random.Random(graine)
        self.opt = dict(options or {})
        self.m = Modele()
        self.lv = None
        self.ops_globaux = {}
        self.buts_confirmes = []
        self.stats = {"sondes": 0, "plans": 0, "surprises": 0, "buts_testes": 0, "explor": 0, "abandon": 0}
        self.prev_frame = None
        self.prev_act = None
        self.dest_cols = None
        self.file = []
        self.lv_actions = 0
        self.trace = []
        self.log = []

    # ================================================================== niveau
    def _nouveau_niveau(self, frame, lv, frame0=None):
        self.lv = lv
        self.frame0 = (frame0 if frame0 is not None else frame).copy()
        self.bg = couleur_fond(frame)
        self.hud = _hud_masque(frame)
        if getattr(self, "hud_persist", None) is not None:
            self.hud = self.hud | self.hud_persist          # un bandeau détecté ailleurs reste un bandeau
        self.hud_persist = self.hud.copy()
        self.chg = np.zeros(frame.shape, int)
        self.corps = []
        self.sondes = []          # actions encore à sonder
        self.file = []
        self.buts = None
        self.but_cur = None
        self.rejetes = set()
        self.pose_vues = set()
        self.lv_actions = 0
        self.pre_gone = set()
        self.clics = []
        self.clics_faits = False
        self.reprobe = False
        self.n_reprobe = 0
        self.interdits = set()
        self.nul_de_suite = 0
        self.stat_ent = None
        self.cur = frame
        self.E = None
        self.n_transitions = 0
        self.dernier_but_pose = None
        self.apres_but = 0

    def _entites(self, frame):
        return composantes(frame, self.bg, ignorer=self.hud)

    # ================================================================== apprentissage
    def _maj_hud(self, prev, frame):
        """Bandeau dynamique : lignes / colonnes du bord qui changent presque à chaque coup (compteur, jauge)."""
        self.chg += (prev != frame)
        n = self.n_transitions
        if n < 4:
            return
        seuil = 0.7 * n
        H, W = frame.shape
        for r in (0, 1, H - 2, H - 1):
            if (self.chg[r, :] > 0).any() and (self.chg[r, :].sum() >= seuil):
                self.hud[r, :] = True
        for c in (0, 1, W - 2, W - 1):
            if (self.chg[:, c] > 0).any() and (self.chg[:, c].sum() >= seuil):
                self.hud[:, c] = True
        self.hud_persist |= self.hud

    def _fit(self, corps, pose, frame):
        pix = corps.pixels_pose(pose)
        H, W = frame.shape
        ok = (pix[:, 0] >= 0) & (pix[:, 0] < H) & (pix[:, 1] >= 0) & (pix[:, 1] < W)
        if not ok.any():
            return 0.0
        r, c = pix[ok, 0], pix[ok, 1]
        return float((frame[r, c] == corps.cols[ok]).sum()) / len(corps.cols)

    def _localiser(self, corps, frame):
        """Cherche la pose dont l'empreinte colle le mieux à l'image : 8 éléments D4 x translations ancrées sur un pixel rare."""
        best, bp = 0.0, None
        cols, cnt = np.unique(corps.cols, return_counts=True)
        # ancre : couleur la plus rare dans le corps
        ca = int(cols[np.argmin(cnt)])
        i0 = int(np.nonzero(corps.cols == ca)[0][0])
        cand = np.argwhere(frame == ca)
        if len(cand) > 1500:
            return None, 0.0
        H, W = frame.shape
        for M in range(8):
            rel = corps.rel[M]
            for q in cand:
                t = q - rel[i0]
                pix = rel + t
                ok = (pix[:, 0] >= 0) & (pix[:, 0] < H) & (pix[:, 1] >= 0) & (pix[:, 1] < W)
                if ok.sum() < 0.8 * len(pix):
                    continue
                f = float((frame[pix[ok, 0], pix[ok, 1]] == corps.cols[ok]).sum()) / len(pix)
                if f > best:
                    best, bp = f, (M, int(t[0]), int(t[1]))
                    if f >= 0.999:
                        return bp, best
        return bp, best

    def _creer_corps(self, Ea, idx):
        ents = [Ea[i] for i in idx]
        pix = pixels(ents)
        c = Corps(pix, len(self.corps))
        c.parts = []
        for e in ents:
            p = pixels([e])
            c.parts.append({"col": e.col, "kd4": e.kd4, "kex": e.kex, "pix": p, "aire": e.aire})
        self.corps.append(c)
        return c

    def _corps_pour(self, pix_monde):
        """Corps dont l'empreinte courante recouvre l'essentiel de `pix_monde` (r, c)."""
        s = set(map(tuple, pix_monde[:, :2].tolist()))
        best, bs = None, 0.0
        for c in self.corps:
            fp = set(map(tuple, c.monde().tolist()))
            inter = len(s & fp)
            sc = inter / max(1, min(len(s), len(fp)))
            if sc > bs:
                best, bs = c, sc
        return best if bs >= 0.5 else None

    def _apprendre(self, prev, a, frame):
        """Une transition (prev, action a, frame). Met à jour corps, règles, hypothèses de couleurs."""
        m = self.m
        self.n_transitions += 1
        Ea = self.E if self.E is not None else self._entites(prev)
        Eb = self._entites(frame)
        self.E = Eb
        ev = expliquer(Ea, Eb, prev, frame, self.hud)
        self.nul_de_suite = self.nul_de_suite + 1 if ev["type"] == "none" else 0
        self._maj_hud(prev, frame)
        # --- remise à zéro du niveau (game over) : l'image redevient l'image initiale
        if np.array_equal(frame, self.frame0) and not np.array_equal(prev, self.frame0):
            if self.dest_cols is not None:
                m.mortel.ajouter(self.dest_cols - {self.bg}, True)
            for c in self.corps:
                p, f = self._localiser(c, frame)
                if p is not None and f >= 0.85:
                    c.M, c.t = p[0], (p[1], p[2])
                else:
                    c.perdu = True
            self.stats["reset_vu"] = self.stats.get("reset_vu", 0) + 1
            self.file = []
            return
        if self.dest_cols is not None:
            m.mortel.ajouter(self.dest_cols - {self.bg}, False)
        # corps concerné par l'événement (avant mise à jour des poses)
        corps_ev = None
        if ev["type"] == "T":
            corps_ev = self._corps_pour(pixels([Ea[i] for i in ev["idx_a"]]))
        elif ev["type"] == "D4":
            corps_ev = self._corps_pour(ev["pa"])
        # --- corps connus : la pose prédite est-elle observée ?
        connus = [c for c in self.corps if a in c.ops]
        resolu = set()
        for c in connus:
            pred = c.appliquer(a, c.pose())
            f_pred = self._fit(c, pred, frame)
            f_old = self._fit(c, c.pose(), frame)
            if pred != c.pose() and f_pred >= 0.9 and f_pred >= f_old:
                cols, hors = m.couleurs_dest(prev, c.pixels_pose(pred), c.pixels_pose(c.pose()))
                if cols:
                    m.solide.ajouter(cols, False)
                    m.vu_libre |= cols
                rev = self._revele(prev, frame, c, pred)
                c.M, c.t = pred[0], (pred[1], pred[2])
                resolu.add(c.id)
            elif f_old >= 0.9 and f_pred < f_old + 1e-9 or (pred == c.pose() and f_old >= 0.9):
                # le corps n'a pas bougé alors qu'une règle l'y poussait : obstacle
                cols, hors = m.couleurs_dest(prev, c.pixels_pose(pred), c.pixels_pose(c.pose()))
                if cols and pred != c.pose():
                    m.solide.ajouter(cols, True)
                if pred != c.pose():
                    self.interdits.add((tuple(q.pose() for q in self.corps if q.ops), a))
                    self.file = []
                resolu.add(c.id)
            else:
                # surprise : localisation globale
                self.stats["surprises"] += 1
                p, f = self._localiser(c, frame)
                if p is not None and f >= 0.85:
                    c.M, c.t = p[0], (p[1], p[2])
                else:
                    c.perdu = True
        # --- nouvelles règles (les clics sont paramétriques : leurs effets sont des événements, pas des opérateurs)
        if a == 6:
            self.clics_effets = getattr(self, "clics_effets", [])
            self.clics_effets.append((ev["type"], len(ev.get("recol", []))))
            if ev["type"] != "none":
                self.reprobe = True            # un clic a changé la scène (sélection ?) : on re-sonde les déplacements
            return
        if ev["type"] in ("T", "D4") and (corps_ev is None or a not in corps_ev.ops):
            self._induire(Ea, Eb, ev, a, prev, frame, corps_ev)
        if ev["type"] in ("autre", "T"):
            for i in ev.get("gone", []):
                e = Ea[i]
                if e.aire > 300:
                    continue
                mine = set(map(tuple, (np.argwhere(e.masque) + np.array([e.r, e.c])).tolist()))
                for c in self.corps:
                    if mine & set(map(tuple, c.monde().tolist())):
                        m.consomme.add((e.col, e.kex))
        return

    def _revele(self, prev, frame, c, pred):
        """Couleurs révélées par le départ du corps (sol sous le corps) : pixels de l'ancienne empreinte hors nouvelle."""
        old = c.pixels_pose(c.pose())
        new = set(map(tuple, c.pixels_pose(pred).tolist()))
        cols = []
        for r, cc in old.tolist():
            if (r, cc) not in new and 0 <= r < frame.shape[0] and 0 <= cc < frame.shape[1]:
                cols.append(int(frame[r, cc]))
        if cols:
            self.m.noter_fill(cols)
        return cols

    def _induire(self, Ea, Eb, ev, a, prev, frame, c=None):
        m = self.m
        if ev["type"] == "T":
            idx = ev["idx_a"]
            if c is None:
                if len(self.corps) >= 5:
                    return
                c = self._creer_corps(Ea, idx)
                c.pose0 = c.pose()
            d = ev["d"]
            old = c.pixels_pose(c.pose())
            c.ops[a] = ("T", (int(d[0]), int(d[1])))
            self.ops_globaux[a] = ("T", (int(d[0]), int(d[1])))
            pred = c.appliquer(a, c.pose())
            cols, hors = m.couleurs_dest(prev, c.pixels_pose(pred), old)
            if cols:
                m.solide.ajouter(cols, False)
                m.vu_libre |= cols
            self._revele(prev, frame, c, pred)
            c.M, c.t = pred[0], (pred[1], pred[2])
        elif ev["type"] == "D4":
            pa = ev["pa"]
            if c is None:
                if len(self.corps) >= 5:
                    return
                idx = ev["idx_a"]
                c = self._creer_corps(Ea, idx)
                c.pose0 = c.pose()
            i, tt = ev["i"], np.array(ev["t"])
            M0, t0 = c.M, np.array(c.t)
            M1 = d4_mul(i, M0)
            t1 = D4[i] @ t0 + tt
            A = D4[M0] - D4[M1]
            b = (t1 - t0).astype(float)
            q = np.linalg.lstsq(A.astype(float), b, rcond=None)[0] if abs(np.linalg.det(A)) > 1e-9 else np.zeros(2)
            c.ops[a] = ("R", i, q)
            self.ops_globaux[a] = ("R", i)
            c.M, c.t = M1, (int(t1[0]), int(t1[1]))

    # ================================================================== buts
    def _terrain(self, frame):
        T = frame.astype(int).copy()
        fill = self.m.fill(self.bg)
        for c in self.corps:
            p = c.monde()
            ok = (p[:, 0] >= 0) & (p[:, 0] < T.shape[0]) & (p[:, 1] >= 0) & (p[:, 1] < T.shape[1])
            T[p[ok, 0], p[ok, 1]] = fill
        T[self.hud] = -1
        return T

    def _statiques(self, terrain):
        """Entités statiques = composantes du terrain (hors fond, hors HUD)."""
        T = terrain.copy()
        T[T < 0] = self.bg
        ents = composantes(T, self.bg)
        return ents

    def _construire_buts(self, frame, terrain, statiques):
        buts = []
        # 1. buts confirmés aux niveaux précédents
        for k in self.buts_confirmes[-2:]:
            pass
        # 2. motif = référence (clés de chaque corps)
        for c in self.corps:
            ps = poses_alignees(c, statiques)
            if ps:
                cpose = c.id
                def test(poses, gone, touche, ps=ps, cid=cpose):
                    return poses[cid] in ps
                buts.append(But("aligner", 0.8, test, detail=f"{len(ps)} poses"))
        # 3. gagnantes apprises
        gagn = self.m.gagnant.map_cols() if self.m.gagnant.obs and any(p for (_, p) in self.m.gagnant.obs) else set()
        if gagn:
            cible = np.isin(terrain, list(gagn))
            if cible.any():
                buts.append(But("couleur_gagnante", 0.9, lambda p, g, t: t, cible=cible, detail=str(sorted(gagn))))
        # 4. toucher une entité unique et saillante
        types = {}
        for e in statiques:
            types.setdefault((e.col, e.kd4), []).append(e)
        cand = []
        for e in statiques:
            n = len(types[(e.col, e.kd4)])
            if e.aire > 600:
                continue
            sc = (1.0 / n) * (1.0 if e.aire >= 2 else 0.3)
            cand.append((sc, e))
        cand.sort(key=lambda x: -x[0])
        for sc, e in cand[:6]:
            cible = np.zeros(terrain.shape, bool)
            cible[e.r:e.r + e.h, e.c:e.c + e.w] |= e.masque
            buts.append(But("toucher", 0.3 * sc, lambda p, g, t: t, cible=cible, detail=f"col{e.col}@{e.r},{e.c}"))
        # 5. tout consommer (si une règle de consommation est connue)
        return buts

    def _contexte(self, frame, but=None, consommables=None):
        T = self._terrain(frame)
        sol = self.m.solide.map_union()
        cible = but.cible if but is not None else None
        actions = sorted({a for c in self.corps for a in c.ops})
        return Contexte(T, self.corps, sol, self.hud, cible=cible, actions=actions), T

    # ================================================================== décision
    def choisir(self, obs, frame0=None):
        frame = np.asarray(obs["frame"][-1], dtype=np.int16)
        lv = obs["levels_completed"]
        avail = [a for a in obs.get("available_actions", []) if a != 0]
        nouveau = self.lv is None or lv != self.lv
        if nouveau:
            if self.lv is not None and lv > self.lv:
                self._victoire()
            self._nouveau_niveau(frame, lv, frame0)
            self.sondes = [a for a in MOUV if a in avail]
            # a priori entre niveaux : les actions déjà vues comme translations sont sondées d'abord
            self.sondes.sort(key=lambda a: (a not in self.ops_globaux, a))
        elif self.prev_act is not None:
            try:
                self._apprendre(self.prev_frame, self.prev_act, frame)
            except Exception:
                import traceback
                self.trace.append(traceback.format_exc())
                self.stats["exc"] = self.stats.get("exc", 0) + 1
                if self.stats["exc"] > 20:
                    return None
        self.cur = frame
        self.lv_actions += 1
        if self.lv_actions > self.opt.get("schema_max_niveau", 150) or self.nul_de_suite >= 14:
            return None
        self.dest_cols = None
        act = self._decider(frame, avail)
        if act is None:
            self.stats["abandon"] += 1
            return None
        self.prev_frame, self.prev_act = frame, act[0]
        # couleurs de destination prévues (pour mortel / gagnant)
        for c in self.corps:
            if act[0] in c.ops:
                pred = c.appliquer(act[0], c.pose())
                cols, _ = self.m.couleurs_dest(frame, c.pixels_pose(pred), c.pixels_pose(c.pose()))
                self.dest_cols = (self.dest_cols or set()) | cols
                break
        self.last_pose = [c.pose() for c in self.corps]
        if act[0] == 6:
            return {"name": "ACTION6", "data": {"x": int(act[1]), "y": int(act[2])}}
        return {"name": f"ACTION{act[0]}", "data": {}}

    def _victoire(self):
        m = self.m
        if self.dest_cols:
            m.gagnant.ajouter(self.dest_cols - {self.bg}, True)
        if self.but_cur is not None:
            self.buts_confirmes.append(self.but_cur.nom)
            self.stats.setdefault("confirmes", []).append(self.but_cur.nom)

    def _decider(self, frame, avail):
        if getattr(self, "reprobe", False) and not self.sondes and self.n_reprobe < 6:
            self.reprobe = False
            self.n_reprobe += 1
            self.sondes = [a for a in MOUV if a in avail]
        # 1. sondage des actions
        while self.sondes:
            a = self.sondes.pop(0)
            if self.ops_globaux.get(a, ("",))[0] == "T" and self.corps and a not in self.corps[0].ops:
                # a priori de niveau précédent : même translation
                for c in self.corps:
                    c.ops[a] = ("T", self.ops_globaux[a][1])
                continue
            self.stats["sondes"] += 1
            return (a,)
        # 2. file d'actions déjà planifiées
        corps = [c for c in self.corps if c.ops and not getattr(c, "perdu", False)]
        if not corps:
            return self._sans_corps(frame, avail)
        if self.file:
            return (self.file.pop(0),)
        return self._planifier(frame, avail)

    @staticmethod
    def _point_dans(e):
        """Pixel de l'entité le plus proche de son centre (le centre d'une forme creuse peut être dans le vide)."""
        pts = np.argwhere(e.masque)
        cr, cc = e.centre
        k = int(np.argmin((pts[:, 0] + e.r - cr) ** 2 + (pts[:, 1] + e.c - cc) ** 2))
        return int(pts[k, 1] + e.c), int(pts[k, 0] + e.r)

    def _sans_corps(self, frame, avail):
        """Aucun corps réactif trouvé : un clic sur chaque entité saillante (apprentissage), puis abandon."""
        if 6 in avail:
            if not self.clics and not getattr(self, "clics_faits", False):
                self.clics_faits = True
                E = self._entites(frame)
                E = [e for e in E if 2 <= e.aire <= 1500]
                E.sort(key=lambda e: -e.aire)
                self.clics = [self._point_dans(e) for e in E[:14]]
            if self.clics:
                x, y = self.clics.pop(0)
                return (6, x, y)
        return None

    def _planifier(self, frame, avail):
        m = self.m
        if self.stat_ent is None or self.buts is None:
            terrain = self._terrain(frame)
            self.stat_ent = self._statiques(terrain)
            self.buts = self._construire_buts(frame, terrain, self.stat_ent)
            for b in self.buts:
                b.cle = (b.nom, b.detail)
        poses = tuple(c.pose() for c in self.corps if c.ops)
        corps = [c for c in self.corps if c.ops]
        for c in corps:
            self.pose_vues.add((c.id,) + c.pose())
        # un but atteint sans victoire est falsifié
        if self.but_cur is not None and self.but_cur.test(poses, frozenset(), self.but_cur.cible is not None and self._sur_cible(self.but_cur, corps)):
            self.rejetes.add(self.but_cur.cle)
            self.but_cur = None
        meilleur = None
        for b in self.buts:
            if b.cle in self.rejetes:
                continue
            ctx, T = self._contexte(frame, b)
            ctx.corps = corps
            plan = bfs(ctx, poses, b.test, interdits=self.interdits)
            if plan is None:
                continue
            sc = b.prior / (1.0 + len(plan) / 20.0)
            if meilleur is None or sc > meilleur[0]:
                meilleur = (sc, b, plan)
        if meilleur is not None:
            _, b, plan = meilleur
            self.but_cur = b
            self.stats["plans"] += 1
            self.stats["buts_testes"] += 1
            self.log.append((self.lv, b.nom, b.detail, len(plan)))
            self.file = list(plan)
            return (self.file.pop(0),)
        # 3. exploration : pose jamais vue la plus proche
        ctx, T = self._contexte(frame)
        ctx.corps = corps
        vues = self.pose_vues

        def nouveau(poses_, gone, t, vues=vues, corps=corps):
            return any((c.id,) + p not in vues for c, p in zip(corps, poses_))
        plan = bfs(ctx, poses, nouveau, interdits=self.interdits)
        if plan:
            self.stats["explor"] += 1
            self.file = list(plan)
            return (self.file.pop(0),)
        return None

    def _sur_cible(self, but, corps):
        if but.cible is None:
            return False
        for c in corps:
            p = c.monde()
            ok = (p[:, 0] >= 0) & (p[:, 0] < but.cible.shape[0]) & (p[:, 1] >= 0) & (p[:, 1] < but.cible.shape[1])
            if but.cible[p[ok, 0], p[ok, 1]].any():
                return True
        return False
