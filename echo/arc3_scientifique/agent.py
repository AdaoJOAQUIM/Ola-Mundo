"""Le Scientifique : agent ARC-AGI-3 qui découvre les règles par expériences (sans LLM, sans réseau).

Contrat : ne parle jamais au jeu ; reçoit des dicts d'observation du courtier, renvoie une action
{"name":..., "data":{...}}. N'importe ni arc_agi ni arcengine. Aucun identifiant de jeu.

Boucle (EMPA / Go-Explore / DreamCoder simplifiés) :
  1. Perception : HUD, mouvement par translation, gabarit d'avatar, mémoire de scène.
  2. Théories : type de règle par couleur (pass/block/lethal/collect/win), postérieur bayésien (theorie.py),
     déplacement par action ; repli « effets de clic/état » par classes d'objets (Beta, gain BALD).
  3. Expériences cruciales : choix de la cible au gain d'information espéré (+ bonus de nouveauté =
     archive Go-Explore des positions visitées), pondéré par la distance (le RESET et les pas coûtent).
  4. Planification en imagination : BFS sur positions prédites dans la théorie MAP ; réplanification
     à chaque surprise (le plan est recalculé à chaque pas à partir de l'état réel).
  5. Inférence d'objectif : un niveau gagné fait monter la proba que les couleurs touchées soient « win » ;
     les niveaux suivants réutilisent la théorie confirmée.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, deque

import numpy as np

import percep
import theorie
from theorie import BLOCK, COLLECT, LETHAL, PASS, WIN, Croyances, Prior

SIMPLES = (1, 2, 3, 4, 5, 7)
GAMMA = 0.96
W_WIN = 6.0
BONUS_NOUV = 0.06
BONUS_COLLECT = 0.25
MAX_PROFONDEUR = 80
H = W = 64


def digamma(x: float) -> float:
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f / 252))


def _h2(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def bald(a: float, b: float) -> float:
    """Information mutuelle entre une issue binaire et le paramètre θ ~ Beta(a,b)."""
    s = a + b
    eh = digamma(s + 1) - (a / s) * digamma(a + 1) - (b / s) * digamma(b + 1)
    return max(0.0, _h2(a / s) - eh)


def _classe_taille(n: int) -> int:
    return int(math.log2(n)) if n > 0 else 0


class Scientifique:
    def __init__(self, graine: int = 0, prior: Prior | None = None, options: dict | None = None):
        self.rng = np.random.default_rng(graine)
        self.opt = {"explo": "eig", "plan": True, "generique": True}
        self.opt.update(options or {})
        self.croy = Croyances(prior)
        self.hud = percep.Hud()
        # avatar
        self.gab = None
        self.vec = {}                       # action -> Counter{(dy,dx): n}
        self.pos = None
        self.perdu = 0
        self.cand_av = Counter()
        self.essais_action = Counter()      # action -> nombre d'essais (découverte des commandes)
        # niveau courant
        self.lv = 0
        self.M = None                       # mémoire de scène (avatar effacé)
        self.visite = set()
        self.bloque = set()                 # (pos, action)
        self.pending = {}                   # cellule -> couleur d'entrée (dévoilement)
        self.cible = None
        # graphe générique
        self.graphe = {}
        self.cle = None
        self.hist = []
        self.macro = deque()
        self.derniere_sol = []
        self.gb = {"new": {}, "up": {}}
        self.prev = None
        self.dernier_reset = -99
        self.n_trans_utiles = 0
        self.stats = {"morts": 0, "niveaux": 0, "resets_agent": 0, "mode_avatar": 0, "mode_generique": 0,
                      "surprises": 0, "avatar_adopte": 0}
        self.journal_pred = []              # (issue prédite proba, issue observée) pour diagnostic/sommeil
        self.niveaux_actions = []

    # =========================================================== utilitaires
    def _cle_etat(self, frame):
        f = frame
        if self.hud.masque is not None and self.hud.masque.any():
            f = frame.copy()
            f[self.hud.masque] = 0
        return hashlib.blake2b(f.tobytes(), digest_size=12).digest()

    def _vec(self, a):
        c = self.vec.get(a)
        if not c:
            return None
        v, n = c.most_common(1)[0]
        return v if n >= 1 else None

    def _boite(self, pos):
        return pos[0], pos[1], pos[0] + self.gab.h, pos[1] + self.gab.w   # y0,x0,y1,x1 (exclusif)

    def _couleurs_entree(self, pos0, pos1):
        """Couleurs de la scène (mémoire, avatar effacé) dans box(pos1) \\ box(pos0)."""
        if self.M is None or self.gab is None:
            return []
        y0, x0, y1, x1 = self._boite(pos1)
        m = np.zeros((H, W), dtype=bool)
        m[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)] = True
        a0, b0, a1, b1 = self._boite(pos0)
        m[max(a0, 0):min(a1, H), max(b0, 0):min(b1, W)] = False
        if self.hud.masque is not None:
            m &= ~self.hud.masque
        if not m.any():
            return []
        return sorted(set(np.unique(self.M[m]).tolist()))

    def _init_scene(self, frame):
        """Début de vie/niveau : repère l'avatar, initialise la mémoire de scène."""
        self.pos = None
        self.pending = {}
        self.M = None
        if self.gab is None:
            return
        p = self.gab.localiser(frame, None)
        if p is None:
            return
        self.pos = p
        M = frame.copy()
        self._effacer_avatar(M, p)
        self.M = M

    def _effacer_avatar(self, M, p):
        y0, x0, y1, x1 = self._boite(p)
        ya, xa, yb, xb = max(y0 - 2, 0), max(x0 - 2, 0), min(y1 + 2, H), min(x1 + 2, W)
        anneau = np.ones((yb - ya, xb - xa), dtype=bool)
        anneau[y0 - ya:y1 - ya, x0 - xa:x1 - xa] = False
        vals = M[ya:yb, xa:xb][anneau]
        cs, ns = np.unique(vals, return_counts=True)
        fill = int(cs[np.argmax(ns)]) if len(cs) else 0
        mk = self.gab.masque_en(*p)
        M[mk] = fill

    # =========================================================== apprentissage
    def _transition(self, obs, frame, avail, lv, n):
        prev = self.prev
        A, act = prev["frame"], prev["act"]
        delta = n - prev["n"]
        if act["kind"] == "reset":
            self._nouvelle_vie(frame)
            return
        if lv > prev["lv"]:
            self._niveau_gagne(prev, act, frame, lv, n)
            return
        if delta >= 2 and lv == prev["lv"]:
            self.stats["morts"] += 1
            if act.get("entry"):
                self.croy.maj_contact(act["entry"], "died")
            self.hud.reinitialiser_vie()
            self._nouvelle_vie(frame)
            return
        if act["kind"] in ("simple", "clic"):
            self.hud.maj(A, frame)
            ka, kb = self._cle_etat(A), self._cle_etat(frame)
            change = ka != kb
            noeud = self.graphe.setdefault(ka, {"frame": A, "avail": prev["avail"], "tried": {}, "arms": None})
            existait = kb in self.graphe
            self.graphe.setdefault(kb, {"frame": frame, "avail": avail, "tried": {}, "arms": None})
            noeud["tried"][act["arm"]] = kb
            self._maj_classe(act, change, (not existait) and change, False)
            self.hist.append(act["arm"])
            if self.macro and not change:
                self.macro.clear()
            self.cle = kb
            if act["kind"] == "simple":
                if self.gab is not None:
                    self._maj_avatar(A, frame, act)
                else:
                    self._detecter_avatar(A, frame, act)
        else:
            self.cle = self._cle_etat(frame)

    def _nouvelle_vie(self, frame):
        self.hist = []
        self.macro.clear()
        self.cle = self._cle_etat(frame)
        self.graphe.setdefault(self.cle, {"frame": frame, "avail": self.prev["avail"], "tried": {}, "arms": None})
        self._init_scene(frame)
        self.perdu = 0

    def _niveau_gagne(self, prev, act, frame, lv, n):
        self.stats["niveaux"] += 1
        self.niveaux_actions.append(n)
        if act.get("entry"):
            self.croy.maj_contact(act["entry"], "win")
        self._maj_classe(act, True, True, True)
        sol = self.hist + [act["arm"]]
        self.derniere_sol = [a for a in sol]
        self.macro = deque(self.derniere_sol) if self.gab is None else deque()
        self.lv = lv
        self.graphe = {}
        self.visite = set()
        self.bloque = set()
        self.hist = []
        self.cible = None
        self.hud.reinitialiser_vie()
        self.cle = self._cle_etat(frame)
        self.graphe[self.cle] = {"frame": frame, "avail": prev["avail"], "tried": {}, "arms": None}
        self.croy.definir_cars(percep.caracteristiques_couleurs(frame, self.hud.masque))
        self._init_scene(frame)
        if self.pos is not None:
            self.visite.add(self.pos)

    # ---- classes d'arcs (repli générique)
    def _classe(self, arm):
        return arm if arm[0] == "A" else ("C", arm[3], arm[4])

    def _ab(self, q, cl):
        d = self.gb[q]
        if cl not in d:
            parent = ("C",) if cl[0] == "C" else cl
            pa = d.get(parent, [0.0, 0.0]) if cl != parent else [0.0, 0.0]
            m = (pa[0] + 0.5) / (pa[0] + pa[1] + 1.0)
            base = (0.6, 0.6) if q == "new" else (0.15, 1.5)
            d[cl] = [base[0] + 1.0 * m, base[1] + 1.0 * (1 - m)]
        return d[cl]

    def _maj_classe(self, act, change, nouveau, up):
        arm = act.get("arm")
        if arm is None:
            return
        cl = self._classe(arm)
        if change:
            self._ab("new", cl)[0 if nouveau else 1] += 1.0
            if cl[0] == "C":
                self._ab("new", ("C",))[0 if nouveau else 1] += 1.0
        else:
            self._ab("new", cl)[1] += 1.0
        self._ab("up", cl)[0 if up else 1] += 1.0
        if cl[0] == "C" and up:
            self._ab("up", ("C",))[0] += 1.0

    # ---- avatar
    def _detecter_avatar(self, A, B, act):
        movs = percep.extraire_mouvement(A, B, self.hud.masque)
        for m in movs:
            n = len(m["cellules_a"])
            if n < 4 or n > 300 or max(abs(m["dy"]), abs(m["dx"])) > 14:
                continue
            gab = percep.Gabarit(m["cellules_a"], m["couleurs_a"])
            cle = (gab.h, gab.w, tuple(sorted(Counter(m["couleurs_a"].tolist()).items())))
            self.cand_av[cle] += 1
            if self.cand_av[cle] < 2:           # confirmation : le même sprite doit bouger deux fois
                continue
            p = gab.localiser(B, (gab.y0 + m["dy"], gab.x0 + m["dx"]))
            q = gab.localiser(A, (gab.y0, gab.x0))
            if p is None or q is None:
                continue
            self.gab = gab
            self.vec.setdefault(act["a"], Counter())[(p[0] - q[0], p[1] - q[1])] += 1
            self.pos = p
            M = B.copy()
            self._effacer_avatar(M, p)
            self.M = M
            self.visite.add(p)
            self.stats["avatar_adopte"] += 1
            self.croy.definir_cars(percep.caracteristiques_couleurs(B, self.hud.masque))
            return

    def _maj_avatar(self, A, B, act):
        a = act["a"]
        pos0 = self.pos
        guess = self._vec(a) or (0, 0)
        near = (pos0[0] + guess[0], pos0[1] + guess[1]) if pos0 else None
        pos1 = self.gab.localiser(B, near)
        if pos0 is None or pos1 is None:
            self.perdu += 1
            if self.perdu >= 3:                  # le gabarit ne correspond plus : redécouverte
                self.gab = None
                self.M = None
                self.vec = {}
            else:
                self._init_scene(B)
            return
        self.perdu = 0
        v = (pos1[0] - pos0[0], pos1[1] - pos0[1])
        mB = self.gab.masque_en(*pos1)
        if v != (0, 0):
            self.vec.setdefault(a, Counter())[v] += 1
            entry = act.get("entry") if act.get("pos1") == pos1 else None
            if entry is None:
                entry = self._couleurs_entree(pos0, pos1)
            pred = self._issue_predite(act)
            if pred is not None:
                self.journal_pred.append((pred, "moved"))
            self.croy.maj_contact(entry, "moved")
            y0, x0, y1, x1 = self._boite(pos1)
            a0, b0, a1, b1 = self._boite(pos0)
            for y in range(max(y0, 0), min(y1, H)):
                for x in range(max(x0, 0), min(x1, W)):
                    if not (a0 <= y < a1 and b0 <= x < b1) and (y, x) not in self.pending:
                        self.pending[(y, x)] = int(self.M[y, x])
        else:
            if act.get("entry"):
                pred = self._issue_predite(act)
                if pred is not None:
                    self.journal_pred.append((pred, "blocked"))
                self.croy.maj_contact(act["entry"], "blocked")
                self.bloque.add((pos0, a))
        if pred_surprise := (act.get("p_bloque") is not None and ((v == (0, 0)) != (act["p_bloque"] > 0.5))):
            self.stats["surprises"] += 1
        # dévoilement : cellules d'entrée à présent découvertes
        if self.pending:
            par_c = {}
            for (y, x), c in list(self.pending.items()):
                if not mB[y, x] and not (self.hud.masque is not None and self.hud.masque[y, x]):
                    par_c.setdefault(c, []).append(int(B[y, x]) != c)
                    del self.pending[(y, x)]
            for c, lst in par_c.items():
                self.croy.maj_reveal(c, sum(lst) * 2 > len(lst))
        nm = ~mB
        self.M[nm] = B[nm]
        self.pos = pos1
        self.visite.add(pos1)

    def _issue_predite(self, act):
        p = act.get("p_issues")
        return p

    # =========================================================== décision
    def choisir(self, obs):
        frame = np.array(obs["frame"][-1], dtype=np.int8)
        avail = tuple(int(a) for a in obs.get("available_actions", []) if int(a) != 0)
        lv = obs.get("levels_completed") or 0
        n = obs["actions_taken"]
        if self.prev is None:
            self.lv = lv
            self.cle = self._cle_etat(frame)
            self.graphe[self.cle] = {"frame": frame, "avail": avail, "tried": {}, "arms": None}
            self.croy.definir_cars(percep.caracteristiques_couleurs(frame, None))
            self.prev = {"avail": avail}
        else:
            self._transition(obs, frame, avail, lv, n)
        act = self._decider(obs, frame, avail, n)
        self.prev = {"frame": frame, "act": act, "n": n, "lv": lv, "avail": avail}
        return act["dict"]

    def _mk_simple(self, a, entry=None, pos1=None, p_bloque=None, issues=None):
        return {"kind": "simple", "a": a, "arm": ("A", a), "dict": {"name": f"ACTION{a}", "data": {}},
                "entry": entry, "pos1": pos1, "p_bloque": p_bloque, "p_issues": issues}

    def _mk_clic(self, arm):
        return {"kind": "clic", "arm": arm, "dict": {"name": "ACTION6", "data": {"x": int(arm[1]), "y": int(arm[2])}}}

    def _mk_reset(self):
        self.stats["resets_agent"] += 1
        self.dernier_reset = self.prev.get("n", 0) if self.prev else 0
        return {"kind": "reset", "arm": None, "dict": {"name": "RESET", "data": {}}}

    def _decider(self, obs, frame, avail, n):
        simples = [a for a in avail if a in SIMPLES and a != 7]
        # 1. découverte des commandes : tester chaque action simple jusqu'à repérer l'avatar
        if self.gab is None and simples and self.opt.get("generique", True) is not None:
            for a in simples:
                if self.essais_action[a] < 2 and self.stats["avatar_adopte"] == 0:
                    self.essais_action[a] += 1
                    return self._mk_simple(a)
        # 2. mode avatar : planification en imagination
        if self.gab is not None and self.pos is not None and self.M is not None:
            act = self._decider_avatar(simples)
            if act is not None:
                self.stats["mode_avatar"] += 1
                return act
        # 3. repli : exploration générique d'états/clics
        self.stats["mode_generique"] += 1
        return self._decider_generique(obs, frame, avail, n)

    # ---------------------------------------------------------- mode avatar
    def _integrales(self):
        """Images intégrales par couleur sur la mémoire de scène (comptes en boîte O(1))."""
        ints = {}
        for c in np.unique(self.M).tolist():
            ints[c] = np.pad((self.M == c).astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return ints

    @staticmethod
    def _compte(ii, y0, x0, y1, x1):
        y0, x0, y1, x1 = max(y0, 0), max(x0, 0), min(y1, H), min(x1, W)
        if y1 <= y0 or x1 <= x0:
            return 0
        return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])

    def _entree_boite(self, ints, p0, p1):
        y0, x0, y1, x1 = self._boite(p1)
        a0, b0, a1, b1 = self._boite(p0)
        iy0, ix0, iy1, ix1 = max(y0, a0), max(x0, b0), min(y1, a1), min(x1, b1)
        res = []
        for c, ii in ints.items():
            tot = self._compte(ii, y0, x0, y1, x1)
            if tot == 0:
                continue
            inter = self._compte(ii, iy0, ix0, iy1, ix1) if (iy1 > iy0 and ix1 > ix0) else 0
            if tot - inter > 0:
                res.append(c)
        return res

    def _decider_avatar(self, simples):
        # actions dont le déplacement est encore inconnu : on les teste d'abord (expérience de calibrage)
        for a in simples:
            if self._vec(a) is None and self.essais_action[a] < 4:
                self.essais_action[a] += 1
                ent = None
                return self._mk_simple(a, entry=ent)
        acts = [(a, self._vec(a)) for a in simples if self._vec(a) is not None]
        if not acts:
            return None
        ints = self._integrales()
        hh, ww = self.gab.h, self.gab.w
        depart = self.pos
        # BFS en imagination
        dist = {depart: 0}
        parent = {}
        file = deque([depart])
        info = {}
        croy = self.croy
        profond = MAX_PROFONDEUR if self.opt["plan"] else 1
        cache_col = {}

        def proba(colors):
            k = tuple(colors)
            if k not in cache_col:
                cache_col[k] = croy.proba_issues(colors)
            return cache_col[k]

        while file:
            p = file.popleft()
            d = dist[p]
            if d >= profond:
                continue
            for a, v in acts:
                q = (p[0] + v[0], p[1] + v[1])
                if q in dist or q[0] < 0 or q[1] < 0 or q[0] + hh > H or q[1] + ww > W:
                    continue
                if (p, a) in self.bloque:
                    continue
                ent = self._entree_boite(ints, p, q)
                pi = proba(ent)
                if pi["blocked"] > 0.5 or pi["died"] > 0.4:
                    continue
                dist[q] = d + 1
                parent[q] = (p, a)
                info[q] = (ent, pi)
                file.append(q)
        if len(dist) <= 1:
            return None
        # valeur de chaque nœud atteignable
        meilleur, val_m = None, 0.0
        for q, (ent, pi) in info.items():
            d = dist[q]
            gain = croy.gain_info(ent)
            pw = 1.0 - (1.0 - pi["win"]) if False else croy.p_type(ent, WIN)
            coll = sum(croy.get(c)[COLLECT] for c in ent) * BONUS_COLLECT
            nov = BONUS_NOUV if q not in self.visite else 0.0
            if self.opt["explo"] == "alea":
                gain = float(self.rng.random()) * 0.3 if q not in self.visite else 0.0
                coll = 0.0
            v = gain + W_WIN * pw + coll + nov
            s = v * (GAMMA ** d)
            if s > val_m:
                meilleur, val_m = q, s
        if meilleur is None or val_m < 1e-4:
            return None
        # remonter le chemin jusqu'au premier pas
        q = meilleur
        while parent[q][0] != depart:
            q = parent[q][0]
        p0, a = parent[q]
        ent, pi = info[q]
        pb = pi["blocked"]
        return self._mk_simple(a, entry=ent, pos1=q, p_bloque=pb, issues=pi)

    # ---------------------------------------------------------- mode générique
    def _arms(self, noeud):
        if noeud["arms"] is not None:
            return noeud["arms"]
        arms = [("A", a) for a in SIMPLES if a in noeud["avail"]]
        if 6 in noeud["avail"]:
            objets, bg = percep.composantes(noeud["frame"], self.hud.masque)
            comptes = Counter()
            for o in objets:
                comptes[o["couleur"]] += o["taille"]
            objets.sort(key=lambda o: (comptes[o["couleur"]], o["taille"]))
            vus = set()
            for o in objets:
                if len(vus) >= 40:
                    break
                if (o["x"], o["y"]) in vus:
                    continue
                vus.add((o["x"], o["y"]))
                arms.append(("C", o["x"], o["y"], o["couleur"], _classe_taille(o["taille"])))
            if not vus:
                for y in range(8, 64, 16):
                    for x in range(8, 64, 16):
                        arms.append(("C", x, y, -1, 0))
        noeud["arms"] = arms
        return arms

    def _valeur_arc(self, arm):
        cl = self._classe(arm)
        a, b = self._ab("new", cl)
        au, bu = self._ab("up", cl)
        if self.opt["explo"] == "alea":
            return float(self.rng.random())
        return bald(a, b) + 6.0 * (au / (au + bu)) + 1e-4 * float(self.rng.random())

    @staticmethod
    def _cle_arc(arm):
        return arm if arm[0] == "A" else ("C", arm[1], arm[2])

    def _decider_generique(self, obs, frame, avail, n):
        noeud = self.graphe[self.cle]
        arms = self._arms(noeud)
        # rejeu de la solution du niveau précédent (macro)
        while self.macro:
            arm = self.macro.popleft()
            if arm[0] == "A" and arm[1] in avail:
                return self._mk_simple(arm[1])
            if arm[0] == "C" and 6 in avail:
                return self._mk_clic(arm)
        libres = [b for b in arms if self._cle_arc(b) not in noeud["tried"]]
        meilleur_loc = None
        val_loc = -1e9
        if libres:
            if self.opt["explo"] == "alea":
                meilleur_loc = libres[int(self.rng.integers(len(libres)))]
                val_loc = 0.0
            else:
                for b in libres:
                    v = self._valeur_arc(b)
                    if v > val_loc:
                        meilleur_loc, val_loc = b, v
        cible_val, cible_bras = -1e9, None
        if self.opt["plan"] and self.opt["explo"] != "alea":
            for k, (d, premier) in self._bfs(self.cle).items():
                if d == 0 or d > 40:
                    continue
                nd = self.graphe[k]
                lib = [b for b in self._arms(nd) if self._cle_arc(b) not in nd["tried"]]
                if not lib:
                    continue
                v = (GAMMA ** d) * max(self._valeur_arc(b) for b in lib)
                if v > cible_val:
                    cible_val, cible_bras = v, premier
        elif self.opt["plan"]:
            cands = [(k, d, pr) for k, (d, pr) in self._bfs(self.cle).items() if d > 0]
            cands = [(k, d, pr) for k, d, pr in cands if any(self._cle_arc(b) not in self.graphe[k]["tried"]
                                                             for b in self._arms(self.graphe[k]))]
            if cands:
                k, d, pr = cands[int(self.rng.integers(len(cands)))]
                cible_val, cible_bras = 0.0, pr
        choix = None
        if meilleur_loc is not None and val_loc >= cible_val:
            choix = meilleur_loc
        elif cible_bras is not None:
            choix = cible_bras
        else:
            ok = [b for b in noeud["tried"] if noeud["tried"][b] != self.cle]
            if ok:
                k = ok[int(self.rng.integers(len(ok)))]
                choix = k if k[0] == "A" else None
                if choix is None:
                    for b in arms:
                        if self._cle_arc(b) == k:
                            choix = b
            if choix is None:
                if n - self.dernier_reset > 3 and n >= 3:
                    return self._mk_reset()
                choix = arms[int(self.rng.integers(len(arms)))] if arms else ("A", 1)
        if choix[0] == "A":
            return self._mk_simple(choix[1])
        return self._mk_clic(choix)

    def _bfs(self, source):
        res = {source: (0, None)}
        file = deque([source])
        while file:
            k = file.popleft()
            d, premier = res[k]
            for arm_key, kn in self.graphe[k]["tried"].items():
                if kn == k or kn in res or kn not in self.graphe:
                    continue
                arm = arm_key
                if arm_key[0] == "C":
                    arm = None
                    for b in self._arms(self.graphe[k]):
                        if self._cle_arc(b) == arm_key:
                            arm = b
                    if arm is None:
                        arm = ("C", arm_key[1], arm_key[2], -1, 0)
                res[kn] = (d + 1, arm if premier is None else premier)
                file.append(kn)
        return res

    # =========================================================== export
    def exporter_etiquettes(self):
        return self.croy.exporter_etiquettes()
