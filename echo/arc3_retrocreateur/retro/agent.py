"""Le Rétro-Créateur : agent. Perception -> inférence de programme -> expériences cruciales -> planification exacte.

Contrat : `choisir(obs) -> {"name", "data"}` ; ne parle jamais au jeu ; aucun identifiant de jeu ; aucun réseau.
"""
from __future__ import annotations

import math
import random

import numpy as np

from .inference import Ev, Modele
from .langage import Prog
from .planif import chercher, clics_pour
from .reseau import Types
from .scene import Scene

MOVES = (1, 2, 3, 4)


def _act_dict(act, geo):
    name, arg = act
    if name == "ACTION6":
        x, y = geo.pixel(arg)
        return {"name": "ACTION6", "data": {"x": int(x), "y": int(y)}}
    return {"name": name, "data": {}}


class RetroCreateur:
    def __init__(self, graine=0, options=None):
        self.rng = random.Random(graine)
        self.opt = dict(options or {})
        import os, json as _json
        if os.environ.get('RETRO_OPTS'):
            self.opt.update(_json.loads(os.environ['RETRO_OPTS']))
        self.types = Types()
        self.scene = Scene(self.types)
        self.modele = Modele(self.types)
        self.geo = None
        self.snap = None            # état perçu courant
        self.raw = None
        self.prev_snap = None
        self.prev_act = None
        self.lv = 0
        self.taken = 0
        self.avail = []
        self.essais = {}            # action déplacement -> nb d'essais (amorçage)
        self.M_ref = None
        self.niveau_debut = 0
        self.stats = {"plans": 0, "explorations": 0, "fallback": 0, "echec_perception": 0}
        self.v1 = None
        self.utilise_v1 = False
        self.vies_pas = 0
        self.guess_excl = []
        self.guess = None
        self.guess_essais = 0
        self.dernier_plan = None
        self.trace = []

    # ================================================================== perception
    def _niveau(self, frame):
        geo = self.scene.nouveau_niveau(frame)
        self.geo = geo
        if geo is None:
            self.snap = None
            return False
        self.raw = self.scene.lire(frame)
        self.M_ref = list(self.raw)
        self.guess = None
        self.prev_snap = None
        self.prev_act = None
        self.snap = self._construire_snap(self.raw, None)
        return True

    def _construire_snap(self, raw, prev):
        geo = self.geo
        sc = self.scene
        M = list(raw)
        for i in geo.ignore:
            M[i] = self.M_ref[i]
        pos, avc, tor = -1, 0, 0
        if sc.av_class is not None:
            cand = sc.cellules_avatar(raw)
            if not cand and prev is not None and prev[1] >= 0:
                # couleur d'avatar nouvelle : parties de la bonne forme sur des cases qui ont changé
                for p in range(geo.N):
                    if p in geo.ignore or raw[p] == self.raw_prev[p]:
                        continue
                    for col, cl in self.types.inv[raw[p]]:
                        if cl == sc.av_class:
                            cand.append((p, col))
                            break
                cand = [x for x in cand if isinstance(x, tuple)]
                if not cand:
                    # fusion avec une case de même couleur : unique case d'arrivée voisine dont la signature a changé
                    pr, pc = divmod(prev[1], geo.C)
                    arr = [p for p in range(geo.N) if p not in geo.ignore and p != prev[1] and raw[p] != self.raw_prev[p]
                           and raw[p] != 0 and abs(p // geo.C - pr) + abs(p % geo.C - pc) == 1]
                    if len(arr) == 1:
                        for col, cl in self.types.inv[raw[arr[0]]]:
                            if col in sc.av_colors or True:
                                sc.ajouter_part((col, cl))
                                cand = [(arr[0], col)]
                                break
                        if cand:
                            self.modele.avc_change = True if cand[0][1] not in sc.av_colors else self.modele.avc_change
                            sc.ajouter_couleur_av(cand[0][1])
                if cand:
                    best = min(cand, key=lambda x: abs(x[0] // geo.C - prev[1] // geo.C) + abs(x[0] % geo.C - prev[1] % geo.C))
                    sc.ajouter_couleur_av(best[1])
                    self.modele.avc_change = True
                    cand = [best[0]]
            if cand:
                if prev is not None and prev[1] >= 0:
                    # valides : avatar immobile (case inchangée) ou case d'arrivée (case changée) ;
                    # la case précédente modifiée n'est qu'un dernier recours (look-alike révélé)
                    bons = [p for p in cand if (p == prev[1]) == (raw[p] == self.raw_prev[p])]
                    ok = bons or [p for p in cand if p == prev[1] or raw[p] != self.raw_prev[p]]
                    if ok:
                        cand = ok
                    pr, pc = divmod(prev[1], geo.C)
                    cand.sort(key=lambda p: abs(p // geo.C - pr) + abs(p % geo.C - pc))
                elif len(cand) > 1:
                    c0 = sc.av_colors[0]
                    cand.sort(key=lambda p: 0 if (c0, sc.av_class) in self.types.inv[raw[p]] else 1)
                pos = cand[0]
                part = sc.partie_avatar(raw[pos])
                avc = part[0]
                if prev is not None and prev[1] == pos:
                    tor = prev[3]
                elif prev is not None and prev[1] >= 0:
                    tor = prev[0][pos] if prev[0][pos] != -1 else 0
                else:
                    tor = sc.reste(raw[pos], part)
                    if tor not in (0,) and prev is None:
                        tor = 0
                M[pos] = -1
        return (tuple(M), pos, avc, tor)

    # ================================================================== boucle
    def choisir(self, obs):
        frame = np.asarray(obs["frame"][-1], dtype=np.int16)
        self.avail = [a for a in obs.get("available_actions", []) if a != 0]
        lv = obs["levels_completed"]
        taken = obs["actions_taken"]
        if self.utilise_v1:
            return self._v1(obs)
        premier = self.geo is None and self.prev_act is None and self.snap is None and self.raw is None
        try:
            if premier:
                ok = self._niveau(frame)
                if not ok:
                    return self._fallback(obs, "pas de réseau")
            else:
                self._transition(frame, lv, taken)
            self.lv, self.taken = lv, taken
            if self.geo is None:
                return self._fallback(obs, "pas de réseau")
            act = self._decider()
        except Exception:                  # sécurité : jamais d'exception vers le harnais
            import traceback
            self.trace.append(traceback.format_exc())
            self.n_exc = getattr(self, "n_exc", 0) + 1
            if self.n_exc > 25 or self.geo is None:
                return self._fallback(obs, "exception")
            self.prev_snap, self.prev_act = self.snap, ("ACTION5", None)
            return self._act_de_secours(obs)
        self.prev_snap = self.snap
        self.prev_act = act
        self.raw_prev = self.raw
        if self.opt.get("verbose"):
            import sys
            g = self.geo
            print(f"[{taken}] niv{lv} {act[0]} {divmod(act[1], g.C) if act[1] is not None else ''} pos={divmod(self.snap[1], g.C) if self.snap[1] >= 0 else None}"
                  f" win={self.modele.prog.win} guess={self.guess}", file=sys.stderr)
        return _act_dict(act, self.geo)

    def _transition(self, frame, lv, taken):
        prev, act = self.prev_snap, self.prev_act
        geo0 = self.geo
        if prev is None or act is None:
            self._niveau(frame)
            return
        mort = (taken - self.taken) >= 2 and lv == self.lv
        gagne = lv > self.lv
        if mort or gagne:
            ev = Ev(prev, act, None, 2 if mort else 1, geo0)
            self.modele.observer(ev)
            self._niveau(frame)
            return
        raw = self.scene.lire(frame)
        # amorçage de l'avatar
        if self.scene.av_class is None and act[0] != "ACTION6":
            mob = self.scene.trouver_mobile(self.raw, raw)
            if mob:
                (col, cl), p, q = mob[0]
                self.scene.av_class = cl
                self.scene.av_colors = [col]
                self.scene.reset_av()
                part = (col, cl)
                s0M = list(self.raw)
                s0M[p] = -1
                tor0 = self.scene.reste(self.raw[p], part)
                prev = (tuple(s0M), p, col, tor0 if tor0 == 0 else 0)
                self.prev_snap = prev
                self.snap = prev
        self.raw_prev = self.raw
        new = self._construire_snap(raw, prev)
        self.raw = raw
        ev = Ev(prev, act, new, 0, geo0)
        chg = self.modele.observer(ev)
        if self.guess is not None and act[0] == "ACTION6":
            ref = self.modele.explore if new[1] >= 0 else self.modele.explore_click
            inex = [t for t in new[0] if t > 0 and t not in ref]
            if self.modele.prog.atome(self.guess, new[0], geo0) and not chg and not inex:
                self.guess_excl.append(self.guess)
                self.guess = None
        self.snap = new
        # complète le dictionnaire des directions : bijection actions <-> directions
        self._completer_dirs()

    def _completer_dirs(self):
        dm = self.modele.dirmap
        moves = [f"ACTION{k}" for k in MOVES if k in self.avail]
        if len(moves) == 4 and len(dm) == 3:
            manquant = [m for m in moves if m not in dm][0]
            d = ({0, 1, 2, 3} - set(dm.values())).pop()
            dm[manquant] = d
            self.modele.prog = self.modele.construire()

    # ================================================================== décision
    def _decider(self):
        mo = self.modele
        prog = mo.prog
        geo, s = self.geo, self.snap
        has_move = any(k in self.avail for k in MOVES)
        has_click = 6 in self.avail
        # 1. amorçage : apprendre les directions
        if has_move:
            inconnues = [f"ACTION{k}" for k in MOVES if k in self.avail and f"ACTION{k}" not in prog.dirmap]
            if inconnues and (s[1] >= 0 or self.scene.av_class is None) and len(prog.dirmap) < 3:
                # essaie l'action la moins tentée
                a = min(inconnues, key=lambda n: (self.essais.get(n, 0), n))
                self.essais[a] = self.essais.get(a, 0) + 1
                if self.essais[a] <= 6:
                    return (a, None)
        # 2. plan vers la victoire dans le programme MAP
        plan = self._plan_victoire(prog, geo, s)
        if plan:
            self.stats["plans"] += 1
            return plan[0]
        # 3. expérience cruciale
        act = self._experience(prog, geo, s, has_click)
        if act is not None:
            self.stats["explorations"] += 1
            return act
        # 4. dernier recours : action aléatoire parmi celles qui existent
        return self._hasard(prog, has_click)

    def _hasard(self, prog, has_click):
        opts = [(f"ACTION{k}", None) for k in MOVES if k in self.avail]
        if has_click and self.geo is not None:
            cells = [p for p, t in enumerate(self.snap[0]) if t > 0 and p not in self.geo.ignore]
            if cells:
                opts.append(("ACTION6", self.rng.choice(cells)))
        if not opts:
            return ("ACTION5", None)
        return self.rng.choice(opts)

    # ------------------------------------------------------------ plan
    def _plan_victoire(self, prog, geo, s):
        w = prog.win
        if w is None and prog.clk and s[1] < 0:
            w = self._deviner_victoire(prog, s)
            if w is not None:
                prog = prog.copie()
                prog.win = w
        if w is None:
            return None
        pre = []
        sx = s
        # préalable : panneaux requis par les apparences dérivées ou la victoire
        ats = []
        for _, _, a in prog.sync:
            ats += [x for x in a if x[0] in ("all", "pmatch")]
        if w[0] == "cond":
            ats += [x for x in w[1] if x[0] in ("all", "pmatch")]
        if ats and prog.clk:
            cl = clics_pour(prog, geo, s, ats)
            if cl is None:
                return None
            pre = cl
            for act in cl:
                r = prog.pas(geo, sx, act)
                if r is None:
                    break
                sx = r[0]
                if r[1] == 1:
                    return pre
        if w[0] == "cond":
            return pre or None
        if sx[1] < 0:
            return pre or None
        strict = {t for t in self._types_presents(sx) if t not in self.modele.explore and t > 0}
        res = chercher(prog, geo, sx, lambda s2, fl, tin, act: fl == 1, interdit=strict)
        if res is None:
            return None
        return pre + res[0]

    def _types_presents(self, s):
        return set(s[0])

    def _deviner_victoire(self, prog, s):
        """Victoire de panneau supposée avant tout succès : motif égal à la référence, sinon tout dans l'état majoritaire."""
        mo = self.modele
        ats = mo.atomes_candidats(prog, [s[0]])
        cands = [a for a in ats if a[0] in ("pmatch", "all")]
        cands.sort(key=lambda a: (0 if a[0] == "pmatch" else 1))
        # état majoritaire pour 'all'
        cnt = {}
        for t in s[0]:
            cnt[t] = cnt.get(t, 0) + 1
        out = []
        for a in cands:
            if a[0] == "all":
                out.append((-cnt.get(a[2], 0), a))
            else:
                out.append((-10 ** 6, a))
        out.sort(key=lambda x: x[0])
        if self.guess is not None and self.guess not in self.guess_excl:
            return ("cond", (self.guess,))
        for _, a in out:
            if a in self.guess_excl:
                continue
            self.guess = a
            return ("cond", (a,))
        return None

    # ------------------------------------------------------------ expériences
    def _experience(self, prog, geo, s, has_click):
        mo = self.modele
        M = s[0]
        cnt = {}
        for t in M:
            if t > 0:
                cnt[t] = cnt.get(t, 0) + 1
        best, best_sc = None, 0.0
        # (a) entrer sur un type inexploré : gain d'information espéré / coût du trajet
        if s[1] >= 0 and prog.dirmap:
            non_explores = [t for t in cnt if t not in mo.explore and t not in prog.kill]
            if non_explores:
                poids = {t: self._poids(t, cnt) for t in non_explores}
                cible = set(non_explores)
                res = chercher(prog, geo, s, lambda s2, fl, tin, act: tin in cible and not (act[0] == "ACTION6"),
                               cout=lambda tin: 1.0 if (tin in mo.explore or tin == 0) else 2.0,
                               cap=6000)
                if res is not None:
                    ch, s2, fl, tin = res
                    c = len(ch) + (1.0 if tin not in mo.explore else 0)
                    sc = poids[tin] / (c + 0.5)
                    best, best_sc = ch[0], sc
        # (b) cliquer sur un type inexploré : coût 1
        if has_click:
            inex = [t for t in cnt if t not in mo.clk and t not in mo.click_inerte
                    and t not in prog.solid and t not in prog.kill and t not in prog.push]
            if inex:
                poids = {t: self._poids_clic(t, cnt, M, geo) for t in inex}
                t = max(inex, key=lambda t: poids[t])
                cells = [p for p, x in enumerate(M) if x == t and p not in geo.ignore]
                sc = poids[t] / 1.5
                if cells and sc > best_sc:
                    best, best_sc = ("ACTION6", cells[0]), sc
        # (c) résoudre le panneau selon la meilleure hypothèse (ce qui ouvre une sortie verrouillée)
        if best is None and prog.clk and s[1] >= 0 and not any(x[0] in ("all", "pmatch") for _, _, a in prog.sync for x in a):
            w = self._deviner_victoire(prog, s)
            if w is not None:
                cl = clics_pour(prog, geo, s, list(w[1]))
                if cl:
                    sc = 1.3 / (len(cl) + 0.5)
                    if sc > best_sc:
                        best, best_sc = cl[0], sc
                elif cl is not None:
                    self.guess_excl.append(self.guess)
                    self.guess = None
        # (d) exploration épuisée sans but connu : tester des hypothèses de victoire (type, couleur d'avatar)
        if best is None and s[1] >= 0 and prog.win is None and not self.opt.get('no_d'):
            best = self._tester_victoire(prog, geo, s)
        return best

    def _tester_victoire(self, prog, geo, s):
        mo = self.modele
        cols = list(self.scene.av_colors)
        cands = []
        for t in set(s[0]):
            if t <= 0 or t in prog.solid or t in prog.kill or t in prog.push:
                continue
            for c in cols:
                if (t, c) in mo.entre_sans_gain:
                    continue
                tc = self.types.couleurs(t)
                sc = 1.0 if c in tc else 0.15
                cands.append((sc, t, c))
        cands.sort(reverse=True)
        strict = {t for t in set(s[0]) if t not in mo.explore and t > 0}
        for sc, t, c in cands[:6]:
            p2 = prog.copie()
            cls = None
            for col, cl in self.types.inv[t]:
                if col == c:
                    cls = cl
            if cls is not None and len(cols) > 1:
                p2.win = ("entermatch", cls)
            else:
                p2.win = ("enter", frozenset({t}))
            res = chercher(p2, geo, s, lambda s2, fl, tin, act: fl == 1, interdit=strict, cap=20000)
            if res:
                return res[0][0]
        return None

    def _poids_clic(self, t, cnt, M, geo):
        """A priori : les réseaux compacts de 4 à 10 cases semblables sont des panneaux cliquables ;
        les objets isolés et les longs murs le sont rarement."""
        cells = [p for p, x in enumerate(M) if x == t]
        n = len(cells)
        if n < 4:
            return 0.12
        if n > 10:
            return 0.05
        rs = [p // geo.C for p in cells]
        cs = [p % geo.C for p in cells]
        aire = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
        # voisins d'un autre type non vide : un panneau mêle deux états
        voisins = 0
        for p in cells:
            for q in geo.nb[p]:
                if q >= 0 and M[q] > 0 and M[q] != t:
                    voisins += 1
        if aire <= 2.6 * n and voisins >= n // 2:
            return 1.2
        return 0.2

    def _poids(self, t, cnt):
        n = cnt.get(t, 0)
        w = 1.0
        if n == 1:
            w *= 1.6
        elif n >= 6:
            w *= 0.35
        if self.types.inv[t] and any(cl[0] == 0 for _, cl in self.types.inv[t] if cl):
            w *= 0.5            # plein : mur probable
        return w

    def _act_de_secours(self, obs):
        """Après une exception interne : une action sûre qui n'engage pas l'inférence."""
        self.prev_act = None
        a = [k for k in MOVES if k in self.avail]
        if a:
            return {"name": f"ACTION{self.rng.choice(a)}", "data": {}}
        return {"name": "ACTION5", "data": {}}

    # ------------------------------------------------------------ repli
    def _fallback(self, obs, raison):
        self.stats["fallback"] += 1
        self.stats["echec_perception"] += 1
        self.trace.append(raison)
        return self._v1(obs)

    def _v1(self, obs):
        if self.v1 is None:
            import importlib.util
            import sys
            from .v1 import V1
            sys.path.insert(0, str(V1))
            try:
                spec = importlib.util.spec_from_file_location("retro_v1_agent", V1 / "agent.py")
                m = importlib.util.module_from_spec(spec)
                sys.modules["retro_v1_agent"] = m
                spec.loader.exec_module(m)
                self.v1 = m.Scientifique(0)
            finally:
                sys.path.remove(str(V1))
        self.utilise_v1 = True
        return self.v1.choisir(obs)
