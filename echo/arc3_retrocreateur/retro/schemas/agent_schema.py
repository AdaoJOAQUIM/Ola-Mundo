"""Agent à schémas causaux locaux : perception sans grille, sondage des actions, induction de règles (corps rigides,
solides / mortelles / gagnantes par posterior bayésien avec longueur de description), planification BFS dans le modèle,
découverte de but par hypothèses pragmatiques testées au moindre coût.

Contrat : `choisir(obs)` -> action {"name","data"} ou None (abandon : l'appelant bascule sur le dernier recours).
Aucun identifiant de jeu, aucun réseau, aucun modèle de langue.
"""
from __future__ import annotations

import itertools
import random

import numpy as np

from .entites import composantes, couleur_fond, composites
from .transitions import expliquer, pixels, trouver_d4, D4, d4_mul
from .modele import Corps, Modele, HypCouleurs
from .buts import Contexte, bfs, But, poses_alignees, assembler, couleur_pointe

MOUV = (1, 2, 3, 4, 5)


def _iou(a, b):
    y0, x0 = max(a[0], b[0]), max(a[1], b[1])
    y1, x1 = min(a[2], b[2]), min(a[3], b[3])
    if y1 <= y0 or x1 <= x0:
        return 0.0
    i = (y1 - y0) * (x1 - x0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(1, u)


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
        self.sel = None
        self.n_sel = 0
        self.clic_deja = set()
        self.click_corps = {}
        self.file_interrompu = False
        self.asm_cibles = {}
        self.asm_essais = 0
        self.asm_ancre = None
        self.visites = {}
        self.reprobes = {}
        self.epoque = 0
        self.effets = []
        self.resid_hist = []
        self.touches = {}
        self.region_effet = None
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

    def _score(self, corps, pix, frame):
        """Fraction de pixels du corps qui collent à l'image : couleurs exactes, jokers (couleur de sélection) acceptés
        seulement s'ils prennent une couleur unique différente du fond."""
        H, W = frame.shape
        ok = (pix[:, 0] >= 0) & (pix[:, 0] < H) & (pix[:, 1] >= 0) & (pix[:, 1] < W)
        if not ok.any():
            return 0.0
        cc = corps.cols[ok]
        vals = frame[pix[ok, 0], pix[ok, 1]]
        fixe = cc >= 0
        bon = float((vals[fixe] == cc[fixe]).sum())
        jok = ~fixe
        if jok.any():
            c0 = corps.cols0[ok][jok]
            vj = vals[jok]
            for g in np.unique(c0):
                m = c0 == g
                v, n = np.unique(vj[m], return_counts=True)
                k = int(np.argmax(n))
                if int(v[k]) != self.bg and n[k] >= 0.95 * m.sum():
                    bon += float(n[k])
        return bon / len(corps.cols)

    def _recoloration(self, corps, pose, frame):
        """À la pose prédite, une partie du corps a gardé sa couleur et le reste a changé de couleur de façon uniforme par groupe
        (le corps a été recoloré par un déclencheur) : la pose est bonne, les pixels changés deviennent des jokers."""
        pix = corps.pixels_pose(pose)
        H, W = frame.shape
        ok = (pix[:, 0] >= 0) & (pix[:, 0] < H) & (pix[:, 1] >= 0) & (pix[:, 1] < W)
        if ok.sum() < 0.95 * len(pix):
            return False
        vals = frame[pix[ok, 0], pix[ok, 1]]
        c0 = corps.cols[ok]
        fixe = c0 >= 0
        if fixe.sum() == 0:
            return False
        exact = (vals == c0) & fixe
        if exact.sum() < 0.35 * len(pix):
            return False
        for g in np.unique(c0[fixe]):
            m = fixe & (c0 == g) & (vals != c0)
            if m.sum() == 0:
                continue
            v, n = np.unique(vals[m], return_counts=True)
            if n.max() < 0.95 * m.sum():
                return False
        return True

    def _forme(self, corps, pose, frame):
        pix = corps.pixels_pose(pose)
        H, W = frame.shape
        ok = (pix[:, 0] >= 0) & (pix[:, 0] < H) & (pix[:, 1] >= 0) & (pix[:, 1] < W)
        if not ok.any():
            return 0.0
        return float((frame[pix[ok, 0], pix[ok, 1]] != self.bg).sum()) / len(pix)

    def _exact(self, corps, pose, frame):
        """Fraction de pixels aux couleurs d'origine (celles du corps quand il était commandé)."""
        pix = corps.pixels_pose(pose)
        H, W = frame.shape
        ok = (pix[:, 0] >= 0) & (pix[:, 0] < H) & (pix[:, 1] >= 0) & (pix[:, 1] < W)
        if not ok.any():
            return 0.0
        return float((frame[pix[ok, 0], pix[ok, 1]] == corps.cols0[ok]).sum()) / len(pix)

    def _sel_depuis_image(self, frame):
        """Sélection lue sur l'image : le corps dont les couleurs d'origine sont intactes est celui qui est commandé."""
        cs = [c for c in self.corps if c.ops and not getattr(c, "perdu", False)]
        if len(cs) < 2 or not any((c.cols < 0).any() for c in cs):
            return
        ex = [c for c in cs if self._exact(c, c.pose(), frame) >= 0.93]
        if len(ex) == 1:
            self.sel = ex[0].id
        elif len(ex) == 0:
            self.sel = None

    def _fit(self, corps, pose, frame):
        return self._score(corps, corps.pixels_pose(pose), frame)

    def _localiser(self, corps, frame):
        """Cherche la pose dont l'empreinte colle le mieux à l'image : 8 éléments D4 x translations ancrées sur un pixel rare."""
        best, bp = 0.0, None
        cols, cnt = np.unique(corps.cols[corps.cols >= 0], return_counts=True)
        if not len(cols):
            return None, 0.0
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
                f = self._score(corps, pix, frame)
                if f > best:
                    best, bp = f, (M, int(t[0]), int(t[1]))
                    if f >= 0.999:
                        return bp, best
        return bp, best

    def _localiser_relache(self, corps, frame):
        """Localisation tolérante aux changements de couleur (sélection) : même forme (aucun pixel de fond sous le corps) et au
        moins 25 % de pixels à la couleur d'origine ; les pixels restants deviendront des jokers."""
        cols, cnt = np.unique(corps.cols0, return_counts=True)
        ca = int(cols[np.argmin(cnt)])
        i0 = int(np.nonzero(corps.cols0 == ca)[0][0])
        cand = np.argwhere(frame == ca)
        if len(cand) > 1500:
            return None
        H, W = frame.shape
        best, bp = 0.0, None
        for M in range(8):
            rel = corps.rel[M]
            for q in cand:
                t = q - rel[i0]
                pix = rel + t
                ok = (pix[:, 0] >= 0) & (pix[:, 0] < H) & (pix[:, 1] >= 0) & (pix[:, 1] < W)
                if ok.sum() < len(pix):
                    continue
                vals = frame[pix[:, 0], pix[:, 1]]
                forme = float((vals != self.bg).mean())
                ex = float((vals == corps.cols0).mean())
                if forme >= 0.97 and ex >= 0.10 and ex + forme > best:
                    best, bp = ex + forme, (M, int(t[0]), int(t[1]))
        return bp

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
        """Une transition : met à jour corps / règles (_apprendre_corps), puis détecte un effet à distance (schéma « déclencheur »)."""
        av = {c.id: c.pose() for c in self.corps}
        self._apprendre_corps(prev, a, frame)
        try:
            self._effet_distant(prev, a, frame, av)
        except Exception:
            import traceback
            self.trace.append(traceback.format_exc())

    def _effet_distant(self, prev, a, frame, av):
        """Pixels qui changent hors des empreintes des corps et hors bandeaux : un déclencheur a agi sur une autre entité
        (registre, porte, motif…). On retient la zone, l'entité touchée et on ouvre une nouvelle époque (les buts écartés sont
        de nouveau testables : le monde a changé)."""
        if a not in MOUV or not self.corps:
            return
        D = (prev != frame) & ~self.hud
        H, W = D.shape
        for c in self.corps:
            for pose in (av.get(c.id, c.pose()), c.pose()):
                w = c.pixels_pose(pose)
                ok = (w[:, 0] >= 0) & (w[:, 0] < H) & (w[:, 1] >= 0) & (w[:, 1] < W)
                D[w[ok, 0], w[ok, 1]] = False
        n = int(D.sum())
        self.resid_hist = (self.resid_hist + [n >= 3])[-6:]
        if n < 3 or sum(self.resid_hist[:-1]) >= 3:
            return                                   # rien, ou bruit continu (animation, compteur) : pas un événement isolé
        ys, xs = np.nonzero(D)
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
        # entité statique touchée : recouvre la nouvelle empreinte (image avant)
        touche = None
        for c in self.corps:
            if c.pose() == av.get(c.id):
                continue
            new = set(map(tuple, c.monde().tolist()))
            best = None
            for e in self._entites(prev):
                if e.aire > 400 or e.aire < 2:
                    continue
                pts = set(map(tuple, (np.argwhere(e.masque) + np.array([e.r, e.c])).tolist()))
                if pts & new and not (pts <= set(map(tuple, c.pixels_pose(av.get(c.id, c.pose())).tolist()))):
                    if best is None or e.aire < best.aire:
                        best = e
            if best is not None:
                touche = (best.col, best.kd4, best.r, best.c)
        self.epoque += 1
        self.effets.append({"n": self.n_transitions, "bbox": bbox, "touche": touche, "pixels": n})
        self.region_effet = bbox
        if touche is not None:
            self.touches[touche] = self.touches.get(touche, 0) + 1
        self.rejetes = {k for k in self.rejetes if k[0] != "toucher"}
        self.interdits = set()               # un blocage observé avant le changement du monde n'est plus une preuve
        self.visites = {}
        self.buts = None
        self.stat_ent = None
        self.stats["effets_distants"] = self.stats.get("effets_distants", 0) + 1

    def _apprendre_corps(self, prev, a, frame):
        """Une transition (prev, action a, frame). Met à jour corps, règles, hypothèses de couleurs."""
        m = self.m
        self.n_transitions += 1
        Ea = self.E if self.E is not None else self._entites(prev)
        Eb = self._entites(frame)
        self.E = Eb
        ev = expliquer(Ea, Eb, prev, frame, self.hud)
        poses_av = {c.id: c.pose() for c in self.corps}
        self.nul_de_suite = self.nul_de_suite + 1 if ev["type"] == "none" else 0
        self._maj_hud(prev, frame)
        # --- remise à zéro du niveau (game over) : l'image redevient l'image initiale
        egal0 = lambda f: bool(((f == self.frame0) | self.hud).all())
        hud_on = lambda f: int(((f != self.bg) & self.hud).sum())
        # un retour à la disposition initiale est un game over seulement si la jauge s'est remplie (un simple aller-retour la vide)
        if egal0(frame) and not egal0(prev) and (hud_on(frame) > hud_on(prev) or not self.hud.any()
                                                  and not np.array_equal(prev, frame)
                                                  and np.array_equal(frame, self.frame0) and self.n_transitions > 8):
            # game over : le niveau repart de zéro (poses, sélection, éventuellement aussi les états cachés). On repart d'un niveau
            # neuf en gardant ce qui a été appris du jeu (translations, solides, mortelles, buts confirmés).
            if self.dest_cols is not None:
                m.mortel.ajouter(self.dest_cols - {self.bg}, True)
            self.stats["reset_vu"] = self.stats.get("reset_vu", 0) + 1
            n_reset = self.stats["reset_vu"]
            lv_act = self.lv_actions
            frame0, lv = self.frame0, self.lv
            self._nouveau_niveau(frame, lv, frame0)
            self.lv_actions = lv_act
            self.prev_act = None
            avail = [x for x in MOUV if x in self.avail_cur]
            self.sondes = sorted(avail, key=lambda x: (x not in self.ops_globaux, x))
            self.sel = None
            return
        if self.dest_cols is not None:
            m.mortel.ajouter(self.dest_cols - {self.bg}, False)
        # corps concerné par l'événement (avant mise à jour des poses)
        corps_ev = None
        rot_vu = None
        if ev["type"] == "T":
            corps_ev = self._corps_pour(pixels([Ea[i] for i in ev["idx_a"]]))
        elif ev["type"] == "D4":
            corps_ev = self._corps_pour(ev["pa"])
            if corps_ev is not None:
                i, tt = ev["i"], np.array(ev["t"])
                M0, t0 = corps_ev.M, np.array(corps_ev.t)
                M1 = d4_mul(i, M0)
                t1 = D4[i] @ t0 + tt
                M1c, t1c = corps_ev.canon(M1, (int(t1[0]), int(t1[1])))
                corps_ev.rot_tab[(a, M0)] = (M1c, (t1c[0] - int(t0[0]), t1c[1] - int(t0[1])))
                rot_vu = (corps_ev.id, M1c, t1c)
        # --- corps connus : la pose prédite est-elle observée ?
        connus = [c for c in self.corps if a in c.ops]
        if len(connus) > 1:
            # plusieurs corps répondent à la même action : un seul est commandé (sélection) ; les autres restent en place
            if corps_ev is not None:
                connus = [c for c in connus if c.id == corps_ev.id]
            elif self.sel is not None and any(c.id == self.sel for c in connus):
                connus = [c for c in connus if c.id == self.sel]
        resolu = set()
        if rot_vu is not None:
            for c in connus:
                if c.id == rot_vu[0]:
                    c.M, c.t = rot_vu[1], rot_vu[2]
                    resolu.add(c.id)
        for c in connus:
            if c.id in resolu:
                continue
            pred = c.appliquer(a, c.pose())
            f_pred = self._fit(c, pred, frame)
            f_old = self._fit(c, c.pose(), frame)
            if pred != c.pose() and f_pred < 0.9 and self._recoloration(c, pred, frame):
                w = c.pixels_pose(pred)
                H, W = frame.shape
                for k, (r, cc) in enumerate(w.tolist()):
                    if 0 <= r < H and 0 <= cc < W and c.cols[k] >= 0 and frame[r, cc] != c.cols[k]:
                        c.cols[k] = -1
                f_pred = self._fit(c, pred, frame)
            if pred != c.pose() and f_pred < 0.9 and f_old < 0.9 and self._forme(c, pred, frame) >= 0.97 \
                    and self._forme(c, c.pose(), frame) < 0.9:
                # même forme à la pose prédite, mais des pièces ont changé de couleur (pointes appariées) : jokers par pixel
                w = c.pixels_pose(pred)
                H, W = frame.shape
                for k, (r, cc) in enumerate(w.tolist()):
                    if 0 <= r < H and 0 <= cc < W and c.cols[k] >= 0 and frame[r, cc] != c.cols[k]:
                        c.cols[k] = -1
                f_pred = self._fit(c, pred, frame)
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
                vrai_blocage = ev["type"] == "none" or (self.sel is None and len(self.corps) == 1)
                if cols and pred != c.pose() and vrai_blocage:
                    m.solide.ajouter(cols, True)
                if pred != c.pose() and vrai_blocage:
                    self.interdits.add((tuple(q.pose() for q in self.corps if q.ops), a))
                    self.file = []
                    self.file_interrompu = True
                resolu.add(c.id)
            else:
                # surprise : localisation globale
                self.stats["surprises"] += 1
                self.file, self.file_interrompu = [], True
                p, f = self._localiser(c, frame)
                if p is not None and f >= 0.85:
                    if f >= 0.97 and (self.sel is None or self.sel == c.id or len(connus) == 1) and a in MOUV:
                        self._apprendre_pose(c, a, p)     # l'événement n'était pas explicable par entités : on apprend par la pose
                    c.M, c.t = c.canon(p[0], (p[1], p[2]))
                else:
                    c.perdu = True
        # --- corps dont l'action a n'a pas encore de règle mais dont la pose a manifestement changé : on le relocalise par son motif
        if a in MOUV and ev["type"] not in ("none",):
            cand = [c for c in self.corps if a not in c.ops and (self.sel is None or c.id == self.sel) and not getattr(c, "perdu", False)]
            for c in cand:
                if self._fit(c, c.pose(), frame) < 0.9:
                    p, f = self._localiser(c, frame)
                    if p is not None and f >= 0.97:
                        self._apprendre_pose(c, a, p)
                        c.M, c.t = c.canon(p[0], (p[1], p[2]))
                        self.ops_globaux.setdefault(a, c.ops[a][:2] if c.ops[a][0] == "T" else ("R", c.ops[a][1]))
        # --- nouvelles règles (les clics sont paramétriques : leurs effets sont des événements, pas des opérateurs)
        if a == 6:
            self._marquer_jokers(Ea, ev)
            if ev["type"] != "none" and self.click_corps.get(self.dernier_clic) is not None:
                self.sel = self.click_corps[self.dernier_clic]
            self.clics_effets = getattr(self, "clics_effets", [])
            self.clics_effets.append((ev["type"], len(ev.get("recol", []))))
            if ev["type"] != "none" and self.click_corps.get(self.dernier_clic) is None:
                self.reprobe = True            # un clic exploratoire a changé la scène (sélection ?) : on re-sonde les déplacements
            return
        if corps_ev is not None and ev["type"] in ("T", "D4"):
            self.sel = corps_ev.id
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

    def _marquer_jokers(self, Ea, ev):
        """Entités recolorées par un clic (sélection) : leurs pixels deviennent des jokers dans les corps qui les contiennent."""
        for i, j in ev.get("recol", []):
            e = Ea[i]
            ep = set(map(tuple, (np.argwhere(e.masque) + np.array([e.r, e.c])).tolist()))
            for c in self.corps:
                w = c.monde()
                idx = [k for k, p in enumerate(map(tuple, w.tolist())) if p in ep]
                if idx:
                    c.cols[idx] = -1

    def _apprendre_pose(self, c, a, p):
        """Règle d'action déduite de la pose observée (corps localisé par son motif) : translation ou rotation exacte."""
        M0, t0 = c.M, np.array(c.t)
        M1, t1c = c.canon(p[0], (p[1], p[2]))
        t1 = np.array(t1c)
        if M1 == M0:
            if a not in c.ops:
                c.ops[a] = ("T", (int(t1[0] - t0[0]), int(t1[1] - t0[1])))
        else:
            if a not in c.ops:
                g = None
                for gg in range(8):
                    if d4_mul(gg, M0) == M1:
                        g = gg
                        break
                c.ops[a] = ("R", g if g is not None else 1, np.zeros(2))
            c.rot_tab[(a, M0)] = (M1, (int(t1[0] - t0[0]), int(t1[1] - t0[1])))
        self.stats["pose_apprise"] = self.stats.get("pose_apprise", 0) + 1

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
            c.M, c.t = c.canon(M1, (int(t1[0]), int(t1[1])))

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
        # registre <-> porte : une entité de même gabarit et de même palette que la zone qui vient de changer est la cible probable
        pal_reg, dim_reg = None, None
        if self.region_effet is not None:
            y0, x0, y1, x1 = self.region_effet
            zone = frame[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2]
            pal_reg = set(np.unique(zone).tolist()) - {self.bg}
            dim_reg = (y1 - y0 + 1, x1 - x0 + 1)
        # objets composites (cadre + motif d'un glyphe, par exemple) : cibles de contact au même titre que les entités monochromes
        Tn = terrain.copy()
        Tn[Tn < 0] = self.bg
        comps = [cp for cp in composites(Tn, self.bg, aire_max=400) if 12 <= len(cp["pts"])
                 and cp["pts"][:, 0].min() > 1 and cp["pts"][:, 1].min() > 1
                 and cp["pts"][:, 0].max() < Tn.shape[0] - 2 and cp["pts"][:, 1].max() < Tn.shape[1] - 2]
        sigs = {}
        for cp in comps:
            h = int(cp["pts"][:, 0].max() - cp["pts"][:, 0].min() + 1)
            w = int(cp["pts"][:, 1].max() - cp["pts"][:, 1].min() + 1)
            cp["dim"] = (min(h, w), max(h, w))
            cp["pal"] = frozenset(np.unique(cp["cols"]).tolist())
            sigs[(cp["dim"], cp["pal"])] = sigs.get((cp["dim"], cp["pal"]), 0) + 1
        cand2 = []
        vus_bbox = []
        for sc, e in cand:
            cand2.append([sc, ("ent", e), (e.r, e.c, e.r + e.h, e.c + e.w)])
        for cp in comps:
            y0, x0 = int(cp["pts"][:, 0].min()), int(cp["pts"][:, 1].min())
            y1, x1 = int(cp["pts"][:, 0].max()) + 1, int(cp["pts"][:, 1].max()) + 1
            sc = 1.0 / sigs[(cp["dim"], cp["pal"])] * 1.1       # légèrement préférés : un objet entier plutôt qu'un morceau de couleur
            cand2.append([sc, ("comp", cp), (y0, x0, y1, x1)])
        for it in cand2:
            sc, (kind, o), bb = it
            bonus = 1.0
            if pal_reg is not None:
                reg = frame[bb[0]:bb[2], bb[1]:bb[3]]
                pal = set(np.unique(reg).tolist()) - {self.bg}
                if pal and pal_reg and len(pal & pal_reg) / max(1, len(pal | pal_reg)) >= 0.5:
                    bonus *= 2.5
            sig = (bb[0], bb[1])
            for k_, v_ in self.touches.items():
                if abs(k_[2] - bb[0]) <= 2 and abs(k_[3] - bb[1]) <= 2:
                    bonus *= 1.0 if v_ < 4 else 0.2
            it[0] = sc * bonus
        cand2.sort(key=lambda x: -x[0])
        retenus = []
        for sc, (kind, o), bb in cand2:
            if any(_iou(bb, r) > 0.6 for r in retenus):
                continue
            retenus.append(bb)
            cible = np.zeros(terrain.shape, bool)
            if kind == "ent":
                cible[o.r:o.r + o.h, o.c:o.c + o.w] |= o.masque
                det = f"col{o.col}@{o.r},{o.c}"
            else:
                cible[bb[0]:bb[2], bb[1]:bb[3]] = True
                det = f"obj@{bb[0]},{bb[1]}"
            buts.append(But("toucher", 0.3 * sc, lambda p, g, t: t, cible=cible, detail=det))
            if len(retenus) >= 9:
                break
        # 5. buts confirmés par une victoire à un niveau précédent : leur a priori est relevé (la suite du jeu les réutilise)
        for b in buts:
            if b.nom in self.buts_confirmes:
                b.prior = min(0.99, b.prior * 1.6 + 0.1)
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
        self.avail_cur = avail
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
        if not nouveau:
            self._sel_depuis_image(frame)
        if self.opt.get("debug_schema"):
            import sys
            print("   [fits]", [(c.id, c.pose(), round(self._fit(c, c.pose(), frame), 2)) for c in self.corps], "sel", self.sel, file=sys.stderr)
        self.lv_actions += 1
        cap = self.opt.get("schema_max_niveau", 150)
        if not self.buts_confirmes and self.stats["abandon"] >= 1:
            cap = min(cap, self.opt.get("schema_max_sans_succes", 40))   # le modèle n'a encore rien résolu dans ce jeu : crédit réduit
        if self.lv_actions > cap or self.nul_de_suite >= 14:
            self.stats["abandon_raison"] = "plafond" if self.lv_actions > cap else "immobile"
            return None
        self.dest_cols = None
        self.dernier_clic = None
        act = self._decider(frame, avail)
        if act is None:
            self.stats["abandon"] += 1
            self.stats.setdefault("abandon_raison", "decider")
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
            self.dernier_clic = (int(act[1]), int(act[2]))
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
            return self._jeton(self.file.pop(0))
        # actions sans effet connu (probablement bloquées à la pose du sondage) : on les re-sonde depuis une autre pose
        if corps:
            c0 = next((c for c in corps if c.id == self.sel), corps[0])
            for a in MOUV:
                if a in avail and all(a not in c.ops for c in corps) and self.reprobes.get(a, (None, 0))[1] < 4:
                    pose_a, k = self.reprobes.get(a, (None, 0))
                    if pose_a != tuple(c0.pose()):
                        self.reprobes[a] = (tuple(c0.pose()), k + 1)
                        self.stats["sondes"] += 1
                        return (a,)
        for c in corps:
            for a, op in c.ops.items():
                if op[0] == "R" and self.sel == c.id and (a, c.M) not in c.rot_tab and a in avail \
                        and len([k for k in c.rot_tab if k[0] == a]) < 4:
                    self.stats["sondes"] += 1
                    return (a,)          # on poursuit la rotation jusqu'à refermer le cycle : table exacte pose par pose
        r = self._phase_selection(frame, avail)
        if r is not None:
            return r
        return self._planifier(frame, avail)

    def _jeton(self, x):
        return x if isinstance(x, tuple) else (x,)

    def _point_corps(self, c, pose=None):
        w = c.monde() if pose is None else c.pixels_pose(pose)
        ctr = w.mean(0)
        k = int(np.argmin(((w - ctr) ** 2).sum(1)))
        return int(w[k, 1]), int(w[k, 0])

    def _phase_selection(self, frame, avail):
        """Objets composites jamais commandés : un clic (sélection ?) puis re-sondage des déplacements."""
        if 6 not in avail or self.n_sel >= 6 or not any(c.ops for c in self.corps):
            return None
        comps = composites(frame, self.bg, self.hud)
        cand = []
        for cp in comps:
            if len(cp["pts"]) < 30:
                continue
            sig = (int(cp["pts"][:, 0].min()), int(cp["pts"][:, 1].min()), len(cp["pts"]))
            if sig in self.clic_deja:
                continue
            ps = set(map(tuple, cp["pts"].tolist()))
            couvert = False
            for c in self.corps:
                fp = set(map(tuple, c.monde().tolist()))
                if len(ps & fp) >= 0.4 * min(len(ps), len(fp)):
                    couvert = True
                    break
            if not couvert:
                cand.append((len(cp["pts"]), sig, cp))
        if not cand:
            return None
        cand.sort(key=lambda x: -x[0])
        _, sig, cp = cand[0]
        self.clic_deja.add(sig)
        self.n_sel += 1
        ctr = cp["pts"].mean(0)
        k = int(np.argmin(((cp["pts"] - ctr) ** 2).sum(1)))
        x, y = int(cp["pts"][k, 1]), int(cp["pts"][k, 0])
        self.dernier_clic = (x, y)
        return (6, x, y)

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

    # ------------------------------------------------------------------ assemblage multi-corps
    def _Ms(self, c):
        Ms, front = {c.M}, [c.M]
        while front:
            M = front.pop()
            for a, op in c.ops.items():
                if op[0] == "R":
                    r = c.appliquer(a, (M, 0, 0))
                    M2 = r[0]
                    if M2 not in Ms:
                        Ms.add(M2); front.append(M2)
        return sorted(Ms)

    def _plan_assemblage(self, frame):
        corps = [c for c in self.corps if c.ops and not getattr(c, "perdu", False)]
        if not corps:
            return None
        comps = composites(frame, self.bg, self.hud)
        fixes = []
        for cp in comps:
            ps = set(map(tuple, cp["pts"].tolist()))
            if len(ps) < 20:
                continue
            if any(len(ps & set(map(tuple, c.monde().tolist()))) >= 0.4 * min(len(ps), len(c.cols)) for c in corps):
                continue
            fixes.append(cp)
        if not fixes and len(corps) < 2:
            return None
        objets = list(fixes) + [{"cols": c.cols0} for c in corps]
        tipcol = couleur_pointe(objets)
        if tipcol is None:
            return None
        mobiles = []
        for c in corps:
            Ms = self._Ms(c)
            mobiles.append({"id": c.id, "rel": {M: c.rel[M] for M in Ms}, "cols": c.cols0, "Ms": Ms, "M0": c.M, "t0": c.t})
        sols = assembler(fixes, mobiles, tipcol, ancre=self.asm_ancre if not fixes else None)
        if not sols and self.asm_ancre is not None and not fixes:
            sols = assembler(fixes, mobiles, tipcol)
        if not sols:
            self.stats["asm_vide"] = self.stats.get("asm_vide", 0) + 1
            return None
        par_id = {c.id: c for c in corps}

        def cout(so):
            sol = so[0]
            tot = 0
            for i, (M, tr, tc) in sol.items():
                c = par_id[i]
                tot += (abs(tr - c.t[0]) + abs(tc - c.t[1])) / 3.0 + (2 if M != c.M else 0)
            return tot
        sols.sort(key=cout)
        for sol, anc in sols[:6]:
            movers = [par_id[i] for i, p in sol.items() if p != par_id[i].pose()]
            if not movers:
                continue
            for ordre in itertools.islice(itertools.permutations(movers), 24):
                plan = self._naviguer(frame, ordre, sol, corps, fixes, tipcol)
                if plan is not None:
                    self.asm_cibles = dict(sol)
                    if anc is not None and self.asm_ancre is None:
                        self.asm_ancre = anc
                    self.stats["asm_plans"] = self.stats.get("asm_plans", 0) + 1
                    return plan
        self.stats["asm_inaccessible"] = self.stats.get("asm_inaccessible", 0) + 1
        return None

    def _naviguer(self, frame, ordre, sol, corps, fixes, tipcol):
        sim = {c.id: c.pose() for c in corps}
        jetons = []
        for b in ordre:
            T = self._terrain(frame)
            H, W = T.shape
            ex = np.zeros((H, W), bool)
            for fx in fixes:
                T[fx["pts"][:, 0], fx["pts"][:, 1]] = -1
                tk = fx["cols"] == tipcol
                ex[fx["pts"][tk, 0], fx["pts"][tk, 1]] = True
            for o in corps:
                if o.id == b.id:
                    continue
                pix = o.pixels_pose(sim[o.id])
                mk = (pix[:, 0] >= 0) & (pix[:, 0] < H) & (pix[:, 1] >= 0) & (pix[:, 1] < W)
                T[pix[mk, 0], pix[mk, 1]] = -1
                tk = mk & (o.cols0 == tipcol)
                ex[pix[tk, 0], pix[tk, 1]] = True
            ctx = Contexte(T, [b], self.m.solide.map_union(), self.hud, actions=sorted(b.ops))
            ctx.tip_exempt, ctx.tipcol = ex, tipcol
            cible = sol[b.id]
            chemin = bfs(ctx, (sim[b.id],), lambda poses, g, t, cible=cible: poses[0] == cible, interdits=self.interdits)
            if chemin is None:
                return None
            if self.sel != b.id:
                x, y = self._point_corps(b, sim[b.id])
                self.click_corps[(x, y)] = b.id
                jetons.append((6, x, y))
            jetons += chemin
            sim[b.id] = cible
        return jetons

    def _planifier(self, frame, avail):
        m = self.m
        cle = tuple(c.pose() for c in self.corps if c.ops)
        self.visites[cle] = self.visites.get(cle, 0) + 1
        if self.visites[cle] > 6:
            self.stats["boucle"] = self.stats.get("boucle", 0) + 1
            return None                      # on tourne en rond dans le même état : le modèle ne suffit pas, on abandonne le niveau
        if ("assemblage",) not in self.rejetes and (len(self.corps) >= 2 or any(c.ops for c in self.corps)):
            atteint = all(c.pose() == self.asm_cibles.get(c.id, c.pose()) for c in self.corps)
            if self.but_cur is not None and self.but_cur.nom == "assemblage" and not self.file_interrompu and \
                    (atteint or self.asm_essais >= 4):
                self.rejetes.add(("assemblage",))          # poses prévues atteintes sans victoire : but falsifié
                self.but_cur = None
            else:
                if self.but_cur is not None and self.but_cur.nom == "assemblage":
                    self.asm_essais += 1                   # le monde a réagi autrement (poussée, blocage) : on replanifie
                plan = self._plan_assemblage(frame)
                self.file_interrompu = False
                if plan:
                    b = But("assemblage", 0.95, lambda p, g, t: False, detail=f"{len(plan)} jetons")
                    b.cle = ("assemblage",)
                    self.but_cur = b
                    self.stats["buts_testes"] += 1
                    self.log.append((self.lv, "assemblage", b.detail, len(plan)))
                    self.file = list(plan)
                    return self._jeton(self.file.pop(0))
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
            if self.epoque == getattr(self.but_cur, "epoque0", self.epoque):
                self.rejetes.add(self.but_cur.cle)           # atteint, rien n'a changé, pas de victoire : but falsifié
            else:
                self.stats["declencheurs"] = self.stats.get("declencheurs", 0) + 1   # atteint et le monde a réagi : c'est un déclencheur
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
            if b.nom in ("toucher", "couleur_gagnante") and plan:
                plan = list(plan) + [plan[-1]]       # le contact en pixels précède souvent l'entrée : on pousse une fois de plus
            b.epoque0 = self.epoque
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
