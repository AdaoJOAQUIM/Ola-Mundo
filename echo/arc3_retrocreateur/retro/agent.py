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
from .compteur import Compteur
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
        self.hist = {}
        self.n_dec = 0
        self.compteur = Compteur()
        self.used = 0
        self.pile = []
        self.frame = None
        self.V0 = None
        self.obs_E = {}
        self.cap_niveau = {}
        self.lv_cur = 0
        self.n_energie = 0

    # ================================================================== perception
    def _niveau(self, frame):
        geo = self.scene.nouveau_niveau(frame)
        self.geo = geo
        self.frame = frame
        self.used = 0
        self.pile = []
        if geo is not None:
            geo.cap = self.cap_niveau.get(self.lv_cur)
        if self.compteur.bande is None and geo is not None and getattr(self.scene, "hud_rows", None):
            rows = np.array(self.scene.hud_rows)
            vals, cnts = np.unique(frame[rows], return_counts=True)
            m = [(c, v) for v, c in zip(vals.tolist(), cnts.tolist()) if v != geo.net.bg]
            if m:
                self.compteur.bande = rows
                self.compteur.ce = max(m)[1]
        L0 = self.compteur.longueur(frame)
        if L0 is not None:
            self.V0 = max(self.V0 or 0, L0)
            self.obs_E.setdefault(self.lv_cur, []).append((0, L0))
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

    def _construire_snap(self, raw, prev, geo=None, raw_prev=None, M_ref=None):
        geo = geo or self.geo
        M_ref = M_ref if M_ref is not None else self.M_ref
        raw_prev = raw_prev if raw_prev is not None else getattr(self, 'raw_prev', None)
        sc = self.scene
        M = list(raw)
        for i in geo.ignore:
            M[i] = M_ref[i]
        pos, avc, tor = -1, 0, 0
        if sc.av_class is not None:
            cand = sc.cellules_avatar(raw)
            if not cand and prev is not None and prev[1] >= 0:
                # couleur d'avatar nouvelle : parties de la bonne forme sur des cases qui ont changé
                for p in range(geo.N):
                    if p in geo.ignore or raw[p] == raw_prev[p]:
                        continue
                    for col, cl in self.types.inv[raw[p]]:
                        if cl == sc.av_class:
                            cand.append((p, col))
                            break
                cand = [x for x in cand if isinstance(x, tuple)]
                if not cand:
                    # fusion avec une case de même couleur : unique case d'arrivée voisine dont la signature a changé
                    pr, pc = divmod(prev[1], geo.C)
                    arr = [p for p in range(geo.N) if p not in geo.ignore and p != prev[1] and raw[p] != raw_prev[p]
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
                    bons = [p for p in cand if (p == prev[1]) == (raw[p] == raw_prev[p])]
                    ok = bons or [p for p in cand if p == prev[1] or raw[p] != raw_prev[p]]
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
            self.lv_cur = lv
            if premier:
                ok = self._niveau(frame)
                if not ok:
                    return self._fallback(obs, "pas de réseau")
            else:
                self._transition(frame, lv, taken)
            self.lv, self.taken = lv, taken
            if self.geo is None:
                return self._fallback(obs, "pas de réseau")
            if self.geo is not None and self.snap is not None and len(self.hist) < 6000:
                self.hist[self.n_dec] = (self.geo, self.snap, self.raw, self.M_ref, tuple(self.pile), self.used, lv)
            self.n_dec += 1
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
            if mort and self.modele.stats.get("energie", 0) > self.n_energie:
                self.cap_niveau[self.lv_cur] = self.used + 1
            self.n_energie = self.modele.stats.get("energie", 0)
            self._niveau(frame)
            return
        raw = self.scene.lire(frame)
        # amorçage de l'avatar
        if self.scene.av_class is None and act[0] in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
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
        if act[0] == "ACTION7":
            if self.pile and self.modele.undo is None:
                ok = new[:3] == self.pile[-1][:3]
                self.modele.undo = bool(ok)
                self.modele.prog = self.modele.construire()
            if self.pile and self.modele.undo and new[:3] == self.pile[-1][:3]:
                self.pile.pop()
                self.used = max(0, self.used - 1)
            self.snap = new
            self.frame = frame
            return
        effectif = new[:3] != prev[:3]
        if effectif:
            self.pile.append(prev)
            self.used += 1
            self.compteur.maj(self.frame, frame)
            L = self.compteur.longueur(frame)
            if L is not None and self.V0:
                self.obs_E.setdefault(self.lv_cur, []).append((self.used, L))
                self._maj_cap()
        self.frame = frame
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

    def _maj_cap(self):
        obs = self.obs_E.get(self.lv_cur, [])
        if len(obs) < 3 or not self.V0:
            return
        v0 = self.V0
        kmax = max(k for k, _ in obs)
        best, be = None, 10 ** 9
        for E in range(max(2, kmax + 1), 400):
            err = sum(abs(L - (v0 * (E - k)) // E) for k, L in obs)
            if err < be:
                best, be = E, err
        if best is not None and be <= 1:
            self.cap_niveau[self.lv_cur] = best
            if self.geo is not None:
                self.geo.cap = best

    def _completer_dirs(self):
        dm = self.modele.dirmap
        moves = [f"ACTION{k}" for k in MOVES if k in self.avail]
        if len(moves) == 4 and len(dm) == 3:
            manquant = [m for m in moves if m not in dm][0]
            d = ({0, 1, 2, 3} - set(dm.values())).pop()
            dm[manquant] = d
            self.modele.prog = self.modele.construire()

    # ================================================================== validation (utilisée par l'évaluateur)
    def _cellule_de_pixel(self, geo, x, y):
        n = geo.net
        q = (x - n.px) // n.c - geo.gx0
        r = (y - n.py) // n.c - geo.gy0
        if 0 <= q < geo.C and 0 <= r < geo.R:
            return r * geo.C + q
        return -1

    def etats_visites(self):
        """États visités exploitables pour la validation (avatar localisé s'il y en a un dans ce monde)."""
        av = self.scene.av_class is not None
        return sorted(k for k, v in self.hist.items() if (v[1][1] >= 0) or not av)

    def valider(self, k, actions, frames, drapeaux):
        """Rejoue dans le programme MAP final une séquence d'actions (dicts de l'env) à partir de l'état visité k et la
        compare aux images réellement observées. Renvoie (exact, indice_du_premier_écart ou None)."""
        if k not in self.hist:
            return False, -1
        geo, s0, raw0, M_ref, pile0, u0, lv0 = self.hist[k]
        prog = self.modele.prog
        geo.cap = self.cap_niveau.get(lv0, geo.cap)
        sc = self.scene
        ancien = sc.geo
        sc.geo = geo
        sc.reset_av()
        try:
            cur, vrai_prev, raw_prev = s0, s0, raw0
            pile, u = list(pile0), u0
            for j, (a, fr, fl) in enumerate(zip(actions, frames, drapeaux)):
                name = a["name"]
                arg = None
                if name == "ACTION6":
                    d = a.get("data") or {}
                    arg = self._cellule_de_pixel(geo, d.get("x", -1), d.get("y", -1))
                if name == "ACTION7":
                    r = None
                    if prog.undo and pile:
                        pred = pile.pop()
                        u = max(0, u - 1)
                        r = (pred, 0, 0)
                    else:
                        pred = cur
                    if fl != 0:
                        return False, j
                else:
                    r4 = prog.pas_u(geo, cur, u, (name, arg))
                    r = None if r4 is None else r4[:3]
                    if r4 is not None:
                        u = r4[3]
                    if fl != 0:
                        return (r is not None and r[1] == fl), j
                    if r is not None and r[1] != 0:
                        return False, j
                    pred = r[0] if r is not None else cur
                    if r is not None and pred[:3] != cur[:3]:
                        pile.append(cur)
                raw = sc.lire(fr)
                vrai = self._construire_snap(raw, vrai_prev, geo, raw_prev, M_ref)
                if pred[:3] != vrai[:3]:
                    self.dernier_ecart = {"j": j, "act": (name, arg), "pos": (cur[1], pred[1], vrai[1]), "avc": (pred[2], vrai[2]),
                                          "cases": [(i, pred[0][i], vrai[0][i]) for i in range(geo.N) if pred[0][i] != vrai[0][i]][:8],
                                          "cap": geo.cap, "u": u}
                    return False, j
                cur, vrai_prev, raw_prev = pred, vrai, raw
            return True, None
        finally:
            sc.geo = ancien
            sc.reset_av()

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
        # 1bis. une action d'annulation est-elle disponible ? un seul essai, après un coup effectif
        if 7 in self.avail and mo.undo is None and self.pile and not self.opt.get("sans_undo"):
            return ("ACTION7", None)
        # 2. plan vers la victoire dans le programme MAP
        plan = self._plan_victoire(prog, geo, s)
        if plan:
            # curiosité bon marché : tester un type inconnu adjacent, ou ré-entrer un déclencheur ambigu (un coup de plus)
            rest = (geo.cap - self.used) if geo.cap else None
            if has_move and s[1] >= 0 and not self.opt.get("sans_curiosite") and (rest is None or rest > len(plan) + 2):
                c = self._curiosite(prog, geo, s, len(plan))
                if c is not None:
                    return c
            self.stats["plans"] += 1
            return plan[0]
        # 3. expérience cruciale
        act = self._experience(prog, geo, s, has_click)
        if act is not None:
            self.stats["explorations"] += 1
            return act
        # 4. dernier recours : action aléatoire parmi celles qui existent
        return self._hasard(prog, has_click)

    def _curiosite(self, prog, geo, s, n_plan):
        """Expérience peu coûteuse pendant un plan : bosse sur un type inconnu voisin ; ré-entrée d'un déclencheur dont
        l'effet (recolor / toggle) est encore ambigu. Gain d'information / coût > seuil."""
        mo = self.modele
        for name, d in prog.dirmap.items():
            q = geo.nb[s[1]][d]
            if q < 0 or q in geo.ignore:
                continue
            T = s[0][q]
            if T > 0 and T not in mo.explore and T not in prog.kill and T not in prog.push:
                return (name, None)
        for cle, cr in mo.creneaux.items():
            if cle[0] != "evt" or len(cr.alts) < 2 or max(cr.post) > 0.8 or cle[1] <= 0:
                continue
            T0 = cle[1]
            if T0 not in s[0]:
                continue
            res = chercher(prog, geo, s, lambda s2, fl, tin, act: tin == T0, cap=4000)
            if res is not None and len(res[0]) <= 6:
                return res[0][0]
        return None

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
