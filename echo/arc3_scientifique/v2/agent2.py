"""Le Scientifique v2 : fusion de quatre idées LLM-free (réimplémentées), sans LLM ni réseau, sans id de jeu.

  1. Modèle d'effet d'action (CNN petit, en ligne) : P(l'image change | image, action ou pixel de clic) ;
     on échantillonne les actions en proportion de cette probabilité (effet.py).
  2. Graphe d'états (hachage des images stables, HUD masqué) : arcs = actions ; on étend la frontière en retournant
     aux états prometteurs par le plus court chemin connu, ou par RESET + rejeu quand c'est moins cher (graphe.py).
  3. Théorie MAP de v1 (avatar par translation) : quand elle planifie (mode découverte/avatar), on exécute son plan.
  4. Modèle de valeur sur des hypothèses de but génériques (tout effacer/collecter, motif de référence, transformation)
     (buts.py). Tout (modèle d'effet, valeur, statistiques d'actions, dernière solution) passe du niveau k au niveau k+1.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

import chemin  # noqa: F401  (chemins d'import)
import buts
import percep
from agent import Scientifique, SIMPLES, _classe_taille, bald
from effet import ModeleEffet
from graphe import Graphe, RESET, hacher

GAMMA = 0.97
LAM_V = 0.8
BONUS_PROF = 0.1
PERIODE_ENTRAIN = 10
MAX_CLICS = 40
POINTS_FIXES = [(16, 16), (48, 16), (16, 48), (48, 48)]


class Theorie(Scientifique):
    """v1 en tant que planificateur de théorie : ne renvoie que ses actions « découverte » et « avatar »."""

    def __init__(self, graine, prior=None):
        super().__init__(graine, prior, {"generique": True})
        self.mode = None

    def _decider(self, obs, frame, avail, n):
        simples = [a for a in avail if a in SIMPLES and a != 7]
        if self.gab is None and simples:
            for a in simples:
                if self.essais_action[a] < 2 and self.stats["avatar_adopte"] == 0:
                    self.essais_action[a] += 1
                    self.mode = "decouverte"
                    return self._mk_simple(a)
        if self.gab is not None and self.pos is not None and self.M is not None:
            act = self._decider_avatar(simples)
            if act is not None:
                self.stats["mode_avatar"] += 1
                self.mode = "avatar"
                return act
        if self.gab is not None:
            # avatar connu mais plan vide : v1 retombe sur son exploration générique (comportement v1 conservé)
            act = self._decider_generique(obs, frame, avail, n)
            self.mode = "generique_av"
            return act
        self.mode = "aucun"
        a0 = simples[0] if simples else 1
        return {"kind": "none", "arm": None, "dict": {"name": f"ACTION{a0}", "data": {}}}


def _dict(arc):
    if arc == RESET:
        return {"name": "RESET", "data": {}}
    if arc[0] == "A":
        return {"name": f"ACTION{arc[1]}", "data": {}}
    return {"name": "ACTION6", "data": {"x": int(arc[1]), "y": int(arc[2])}}


class Scientifique2:
    def __init__(self, graine=0, options=None, modele=None):
        self.opt = {"effet": True, "graphe": True, "theorie": True, "valeur": True, "macro": True,
                    "periode": PERIODE_ENTRAIN, "nette": 1.0, "th_alloc": 120, "th_bloc": 80}
        self.opt.update(options or {})
        self.rng = np.random.default_rng(graine)
        self.effet = (modele if modele is not None else ModeleEffet(graine, threads=1)) if self.opt["effet"] else None
        self.th = Theorie(graine) if self.opt["theorie"] else None
        self.val = buts.ModeleValeur(self.opt["valeur"])
        self.hud = percep.Hud()
        self.g = Graphe()
        self.racine = None
        self.cle = None
        self.prev = None
        self.plan = deque()
        self.plan_cle = deque()
        self.macro = deque()
        self._th_en_cours = False
        self.plan_cible = None
        self.hist = []              # arcs depuis le début de la vie courante
        self.chemin_noeuds = []     # nœuds de la vie courante
        self.lv = 0
        self.derniere_sol = []
        self.stat = {}              # classe d'arc -> [changé, nouveau, total] (statistiques d'actions, transférées)
        self.gagnantes = set()      # classes de l'action finale des niveaux gagnés
        self.th_stall = 0
        self.th_pause = 0
        self.niveau_debut_n = 0
        self.stats = {"morts": 0, "niveaux": 0, "resets": 0, "actions_theorie": 0, "actions_graphe": 0,
                      "actions_plan": 0, "actions_local": 0, "noeuds_max": 0, "entrainements": 0,
                      "reconstructions": 0, "macro": 0}
        self.niveaux_actions = []
        self.perte_eff = None

    # ============================================================ utilitaires
    def _hash(self, frame):
        return hacher(frame, self.hud.masque)

    def _classe_stat(self, arc, node):
        if arc[0] == "A":
            return arc
        cls = (node.info or {}).get("cls", {}).get(arc)
        return cls if cls is not None else ("C", -1, 0)

    def _stat_p(self, cl):
        s = self.stat.get(cl)
        if not s:
            return 0.5, 0.5
        return (s[0] + 0.5) / (s[2] + 1.0), (s[1] + 0.5) / (s[2] + 1.0)

    def _maj_stat(self, cl, change, nouveau=False):
        for c in ((cl, ("C",)) if cl[0] == "C" else (cl,)):
            s = self.stat.setdefault(c, [0.0, 0.0, 0.0])    # changé, nouveau nœud, total
            s[0] += float(change)
            s[1] += float(nouveau)
            s[2] += 1.0

    def _creer_noeud(self, key, frame, avail, parent=None):
        existe = key in self.g.noeuds
        n = self.g.ajouter(key, frame, avail, parent)
        if not existe:
            an = buts.analyser(frame, self.hud.masque)
            f = buts.caracteristiques(frame, an, self.racine, self.hud.masque)
            n.info = {"an": an, "f": f, "cls": {}}
            n.valeur = self.val.valeur(f)
            self.stats["noeuds_max"] = max(self.stats["noeuds_max"], len(self.g.noeuds))
        return n

    def _candidats(self, node):
        arcs = [("A", a) for a in node.avail if a in SIMPLES]
        if 6 in node.avail:
            objets = sorted(node.info["an"]["objets"], key=lambda o: o["taille"])
            cnt = Counter(o["couleur"] for o in objets)
            objets.sort(key=lambda o: (cnt[o["couleur"]], o["taille"]))
            vus = set()
            for o in objets:
                if len(vus) >= MAX_CLICS:
                    break
                k = (o["x"], o["y"])
                if k in vus:
                    continue
                vus.add(k)
                arc = ("C", o["x"], o["y"])
                arcs.append(arc)
                node.info["cls"][arc] = ("C", o["couleur"], _classe_taille(o["taille"]))
            for (x, y) in POINTS_FIXES:
                if (x, y) not in vus:
                    arc = ("C", x, y)
                    arcs.append(arc)
                    node.info["cls"][arc] = ("C", int(node.frame[y, x]), -1)
        return arcs

    def _probs(self, node, frais=False):
        """Probabilité d'effet de chaque arc candidat (CNN mélangé aux statistiques d'actions)."""
        if node.cands is None:
            node.cands = self._candidats(node)
        ver = self.effet.version if self.effet else 0
        if node.ver >= 0 and (not frais or (ver - node.ver) < 2):
            return node.probs
        if self.effet is None:
            node.probs = {a: 0.5 for a in node.cands}
            node.ver = 0
            return node.probs
        ps, pc = self.effet.predire(node.frame)
        w = min(0.85, self.effet.n_ex / (self.effet.n_ex + 150.0))
        pr = {}
        for a in node.cands:
            cnn = float(ps[a[1] - 1]) if a[0] == "A" else float(pc[a[2], a[1]])
            cl = self._classe_stat(a, node)
            pch, pnew = self._stat_p(cl)
            st = self.stat.get(cl, (0.0, 0.0, 0.0))
            info = 3.0 * bald(st[1] + 0.6, st[2] - st[1] + 0.6)
            pr[a] = info +  (w * cnn + (1 - w) * pch) * (0.4 + 0.6 * pnew) + (0.6 if cl in self.gagnantes else 0.0)
        node.probs = pr
        node.ver = ver
        return pr

    # ============================================================ transitions
    def _reconstruire(self):
        """Le masque HUD a changé : recalcule les clés et fusionne les nœuds."""
        self.stats["reconstructions"] += 1
        ancien = self.g.noeuds
        remap = {k: self._hash(n.frame) for k, n in ancien.items()}
        nouveau = Graphe()
        nouveau.racine = remap[self.g.racine]
        for k, n in ancien.items():
            kk = remap[k]
            if kk not in nouveau.noeuds:
                n.cle = kk
                nouveau.noeuds[kk] = n
                n.arcs = {a: (remap.get(t) if t is not None else None) for a, t in n.arcs.items()}
            else:
                tgt = nouveau.noeuds[kk]
                for a, t in n.arcs.items():
                    tgt.arcs.setdefault(a, remap.get(t) if t is not None else None)
        self.g = nouveau
        self.cle = remap.get(self.cle, self.cle)
        self._vider_plan()

    def _transition(self, obs, frame, avail, lv, n):
        p = self.prev
        arc = p["arc"]
        A = p["frame"]
        delta = n - p["n"]
        ka = p["cle"]
        if arc == RESET:
            self.cle = self._hash(frame)
            self._creer_noeud(self.cle, frame, avail)
            attendu = self.plan_cle.popleft() if self.plan_cle else None
            if attendu is not None and attendu != self.cle:
                self._vider_plan()
            elif not self.plan_cle:
                self.plan_cible = None
            self.hist = []
            self.chemin_noeuds = [self.cle]
            return
        if lv > p["lv"]:
            self._niveau_gagne(p, frame, avail, lv, n)
            return
        node_a = self.g.noeuds.get(ka)
        if delta >= 2 and lv == p["lv"]:
            self.stats["morts"] += 1
            if node_a is not None:
                self.g.marquer_mortel(ka, arc)
                self._apprendre(node_a, arc, False)
            self.hud.reinitialiser_vie()
            self.cle = self._hash(frame)
            self._creer_noeud(self.cle, frame, avail)
            self._nouvelle_vie()
            return
        # transition ordinaire
        masque_change = self.hud.maj(A, frame)
        if masque_change:
            self._reconstruire()
            ka = self._hash(A)
        kb = self._hash(frame)
        change = ka != kb
        if node_a is None or ka not in self.g.noeuds:
            node_a = self._creer_noeud(ka, A, p["avail"])
        existait = kb in self.g.noeuds
        self._creer_noeud(kb, frame, avail, ka)
        self.g.relier(ka, arc, kb)
        self.g.noeuds[kb].visites += 1
        self._apprendre(self.g.noeuds[ka], arc, change, nouveau=(not existait))
        self.hist.append(arc)
        self.chemin_noeuds.append(kb)
        self.cle = kb
        if self.plan_cle:
            attendu = self.plan_cle.popleft()
            if attendu is not None and attendu != kb:
                self._vider_plan()
            elif not self.plan_cle:
                self.plan_cible = None
        if self._th_en_cours:
            if existait:
                self.th_stall += 1
            else:
                self.th_stall = 0
            if self.th_stall >= 60:
                self.th_pause = 80
                self.th_stall = 0

    def _apprendre(self, node, arc, change, nouveau=False):
        cl = self._classe_stat(arc, node)
        if self.opt["effet"]:
            self._maj_stat(cl, change, nouveau)
            if arc[0] == "A":
                self.effet.ajouter(node.cle, node.frame, arc[1], float(change))
            else:
                self.effet.ajouter(node.cle, node.frame, 6, float(change), arc[2], arc[1])

    def _vider_plan(self):
        self.plan.clear()
        self.plan_cle.clear()
        self.plan_cible = None

    def _nouvelle_vie(self):
        self._vider_plan()
        self.hist = []
        self.chemin_noeuds = [self.cle]

    def _niveau_gagne(self, p, frame, avail, lv, n):
        arc = p["arc"]
        self.stats["niveaux"] += 1
        self.niveaux_actions.append(n)
        if self.opt["effet"] and p["cle"] in self.g.noeuds:
            self._apprendre(self.g.noeuds[p["cle"]], arc, True, nouveau=True)
            self.gagnantes.add(self._classe_stat(arc, self.g.noeuds[p["cle"]]))
        sol = self.hist + [arc]
        self.derniere_sol = sol
        # modèle de valeur : positifs = derniers états avant la victoire
        if self.opt["valeur"]:
            pos_keys = (self.chemin_noeuds[-6:]) if self.chemin_noeuds else []
            pos = [self.g.noeuds[k].info["f"] for k in pos_keys if k in self.g.noeuds]
            ps = set(pos_keys)
            neg_all = [k for k in self.g.noeuds if k not in ps]
            if len(neg_all) > 150:
                neg_all = [neg_all[i] for i in self.rng.choice(len(neg_all), 150, replace=False)]
            neg = [self.g.noeuds[k].info["f"] for k in neg_all]
            self.val.apprendre(pos, neg)
        self.lv = lv
        self.hud.reinitialiser_vie()
        self._nouveau_niveau(frame, avail, n)

    def _nouveau_niveau(self, frame, avail, n):
        self.racine = buts.Racine(frame, self.hud.masque)
        self.cle = self._hash(frame)
        self.g = Graphe()
        self.g.noeuds = {}
        self.g.racine = self.cle
        self._creer_noeud(self.cle, frame, avail)
        self.g.noeuds[self.cle].profondeur = 0
        self._nouvelle_vie()
        self.niveau_debut_n = n
        self.th_stall = 0
        self.th_pause = 0
        if self.opt["macro"] and self.derniere_sol and len(self.derniere_sol) <= 60 and (
                self.th is None or self.th.gab is None):
            self.macro = deque(self.derniere_sol)
        else:
            self.macro = deque()

    # ============================================================ décision
    def choisir(self, obs):
        frame = np.array(obs["frame"][-1], dtype=np.int8)
        avail = tuple(int(a) for a in obs.get("available_actions", []) if int(a) != 0)
        lv = obs.get("levels_completed") or 0
        n = obs["actions_taken"]
        self._th_en_cours = False
        if self.prev is None:
            self.lv = lv
            self.racine = buts.Racine(frame, None)
            self.cle = self._hash(frame)
            self.g = Graphe()
            self.g.racine = self.cle
            self._creer_noeud(self.cle, frame, avail)
            self.chemin_noeuds = [self.cle]
        else:
            self._th_en_cours = self.prev.get("th", False)
            self._transition(obs, frame, avail, lv, n)
        if self.effet is not None and self.effet.nouveaux >= self.opt["periode"]:
            self.perte_eff = self.effet.entrainer(pas=2)
            self.stats["entrainements"] += 1
        arc, par_th = self._decider(obs, frame, avail, n)
        self._th_en_cours = False
        # garder la théorie v1 cohérente avec l'action réellement jouée
        if self.th is not None:
            self.th.prev["act"] = self._act_v1(arc, par_th)
        self.prev = {"frame": frame, "arc": arc, "n": n, "lv": lv, "cle": self.cle, "avail": avail, "th": par_th}
        return _dict(arc)

    def _act_v1(self, arc, par_th):
        if par_th:
            return self.th.prev["act"]
        if arc == RESET:
            return self.th._mk_reset()
        if arc[0] == "A":
            return self.th._mk_simple(arc[1])
        return self.th._mk_clic(("C", arc[1], arc[2], -1, 0))

    def _decider(self, obs, frame, avail, n):
        # 1. la théorie v1 reçoit toujours l'observation (elle apprend) ; on suit son plan s'il est confiant
        arc_th = None
        if self.th is not None:
            d = self.th.choisir(obs)
            ecoule = n - self.niveau_debut_n
            alloue = ecoule < self.opt["th_alloc"] or ((ecoule - self.opt["th_alloc"]) // self.opt["th_bloc"]) % 2 == 1
            if self.th_pause > 0:
                self.th_pause -= 1
            elif not alloue:
                pass
            elif self.th.mode in ("decouverte", "avatar", "generique_av"):
                if d["name"] == "ACTION6":
                    arc_th = ("C", int(d["data"]["x"]), int(d["data"]["y"]))
                elif d["name"] == "RESET":
                    arc_th = RESET
                else:
                    arc_th = ("A", int(d["name"][6:]))
        if arc_th is not None and (arc_th == RESET or arc_th[0] == "C" or arc_th[1] in avail):
            self.stats["actions_theorie"] += 1
            self._vider_plan()
            return arc_th, True
        # 2. rejeu de la solution précédente (macro) tant qu'elle produit des changements
        while self.macro:
            arc = self.macro.popleft()
            if arc[0] == "A" and arc[1] in avail or arc[0] == "C" and 6 in avail:
                self.stats["macro"] += 1
                return arc, False
        # 3. plan en cours
        if self.plan:
            arc = self.plan.popleft()
            self.stats["actions_plan"] += 1
            return arc, False
        node = self.g.noeuds[self.cle]
        self._probs(node, frais=True)
        if not self.opt["graphe"]:
            return self._decider_sans_graphe(node), False
        return self._decider_graphe(node), False

    def _echantillon(self, node):
        libres = node.non_essayes()
        pr = node.probs
        if not libres:
            return None
        if self.effet is None:
            return libres[int(self.rng.integers(len(libres)))]
        p = (np.array([pr[a] for a in libres]) + 0.02) ** self.opt["nette"]
        p /= p.sum()
        return libres[int(self.rng.choice(len(libres), p=p))]

    def _decider_sans_graphe(self, node):
        arc = self._echantillon(node)
        self.stats["actions_local"] += 1
        if arc is None:
            self.stats["resets"] += 1
            return RESET
        return arc

    def _decider_graphe(self, node):
        dist, parent = self.g.chemins(self.cle, rayon=80)
        meilleur, s_best = None, -1.0
        s_loc = None
        for k, d in dist.items():
            nd = self.g.noeuds[k]
            if nd.cands is None:
                nd.cands = self._candidats(nd)
            libres = [a for a in nd.cands if a not in nd.arcs]
            if not libres:
                continue
            pr = self._probs(nd)
            pm = max(pr[a] for a in libres)
            s = (pm + LAM_V * nd.valeur + BONUS_PROF * min(nd.profondeur, 30) / 30.0) * (GAMMA ** d)
            if k == self.cle:
                s_loc = s
            if s > s_best:
                meilleur, s_best = k, s
        if meilleur is None:
            self.stats["resets"] += 1
            return RESET
        if meilleur == self.cle or (s_loc is not None and s_loc >= 0.9 * s_best):
            self.stats["actions_local"] += 1
            return self._echantillon(node) or RESET
        route = Graphe.route(parent, self.cle, meilleur)
        if not route:
            self.stats["actions_local"] += 1
            return self._echantillon(node) or RESET
        # clés attendues le long de la route (pour détecter les surprises)
        attendues = []
        k = self.cle
        for a in route:
            if a == RESET:
                k = self.g.racine
            else:
                k = self.g.noeuds[k].arcs.get(a)
            attendues.append(k)
        premier = route[0]
        self.plan = deque(route[1:])
        self.plan_cle = deque(attendues)
        self.plan_cible = meilleur
        if premier == RESET:
            self.stats["resets"] += 1
        self.stats["actions_graphe"] += 1
        return premier

    # ============================================================ fin de séance
    def entrainer_si_besoin(self, n):
        pass
