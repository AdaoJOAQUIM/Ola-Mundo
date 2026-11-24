"""Inférence bayésienne de programme, pilotée par les événements.

Chaque transition observée (état avant, action, état après, drapeau) est rejouée dans le programme MAP.
Si elle n'est pas prédite exactement, les écarts PROPOSENT des règles (généralisées par type) :
  - déplacement observé -> direction de l'action ; blocage -> type solide ; entrée inattendue -> plus solide ;
  - case quittée qui ne retrouve pas son contenu -> effet « set » du type d'origine (ramasser = set 0) ;
  - blocs qui bougent -> poussable (+ gravité si la chute dépasse la poussée) ;
  - toutes les cases d'un type A deviennent B -> créneau ambigu {recolor, toggle, apparence dérivée} ;
  - clic -> bascule voisine ('flip') ou cycle ('next') ; mort -> type mortel ; victoire -> règle de victoire.
Les créneaux ambigus portent plusieurs alternatives notées par : exactitude de prédiction sur tous les événements
(rejeu) x a priori (longueur de description, indices pragmatiques). Les K meilleurs programmes (déviations d'un
seul créneau autour du MAP) sont conservés pour le choix des expériences.
"""
from __future__ import annotations

import math

from .langage import Prog

EPS = 1e-3


class Ev:
    __slots__ = ("s0", "act", "s1", "flag", "geo", "cle")

    def __init__(self, s0, act, s1, flag, geo):
        self.s0, self.act, self.s1, self.flag, self.geo = s0, act, s1, flag, geo
        self.cle = (geo.cle, s0, act, flag, s1 if flag == 0 else None)


def ecart(prog, ev):
    """Nombre d'écarts entre la prédiction du programme et l'événement observé (0 = exact)."""
    r = prog.pas(ev.geo, ev.s0, ev.act)
    if ev.flag != 0:
        return 0 if (r is not None and r[1] == ev.flag) else 1
    s0, s1 = ev.s0, ev.s1
    if r is None:
        return 0 if s0[:3] == s1[:3] and s0[0] == s1[0] else 1 + sum(a != b for a, b in zip(s0[0], s1[0]))
    s2, fl, _ = r
    if fl != 0:
        return 1
    n = sum(a != b for a, b in zip(s2[0], s1[0]))
    n += (s2[1] != s1[1]) + (s2[2] != s1[2])
    return n


class Creneau:
    """Choix ambigu : liste d'alternatives (nom, prior) ; `appliquer(prog, nom)` construit la règle."""

    def __init__(self, cle, alts):
        self.cle = cle
        self.alts = alts                # [(nom, log_prior)]
        self.choix = 0
        self.post = [0.0] * len(alts)


class Modele:
    def __init__(self, types):
        self.types = types
        self.dirmap = {}
        self.solid, self.push, self.kill = set(), set(), set()
        self.fx = {}                  # type -> liste d'effets certains
        self.clk = {}                 # type -> spec de clic
        self.grav = None
        self.win_obs = []             # [(type, couleur_avatar, classe)] entrées gagnantes
        self.win_click = []           # [événements gagnants par clic]
        self.creneaux = {}            # cle -> Creneau
        self.events = []
        self.vus = set()
        self.explore = set()          # types dont le comportement à l'entrée est connu
        self.click_inerte = set()
        self.prog = Prog(types)
        self.stats = {"surprises": 0, "evenements": 0, "reevals": 0}
        self.avc_change = False
        self.win_nonobs = set()

    # ------------------------------------------------------------ construction
    def construire(self, choix=None):
        p = Prog(self.types)
        p.dirmap = dict(self.dirmap)
        p.solid, p.push, p.kill = set(self.solid), set(self.push), set(self.kill)
        p.fx = {t: tuple(v) for t, v in self.fx.items()}
        p.clk = dict(self.clk)
        p.grav = self.grav
        for cle, cr in self.creneaux.items():
            nom = cr.alts[(choix or {}).get(cle, cr.choix)][0]
            self._appliquer(p, cle, nom)
        w = self._victoire(p)
        p.win = w
        return p

    def _appliquer(self, p, cle, nom):
        k = cle[0]
        if k == "evt":
            _, T0, A, B = cle
            if nom == "recolor":
                p.fx[T0] = tuple(p.fx.get(T0, ())) + (("recolor", A, B),)
            elif nom == "toggle":
                p.fx[T0] = tuple(p.fx.get(T0, ())) + (("toggle", A, B),)
            elif nom == "sync":
                ats = self.apprendre_cond(A, B, p)
                if ats is not None:
                    p.sync.append((A, B, tuple(ats)))
        elif k == "win":
            pass

    def _victoire(self, p):
        if self.win_click:
            ats = self.apprendre_win_cond(p)
            if ats:
                return ("cond", tuple(ats))
        if self.win_obs:
            for cr in [self.creneaux.get(("winmode",))]:
                if cr is not None and cr.alts[cr.choix][0] == "entermatch":
                    cls = {c for _, _, c in self.win_obs}
                    if len(cls) == 1:
                        return ("entermatch", next(iter(cls)))
            return ("enter", frozenset(t for t, _, _ in self.win_obs) - self.win_nonobs)
        return None

    # ------------------------------------------------------------ atomes et apprentissage de conditions
    def familles_panneau(self, prog=None):
        """Types touchés par les clics (famille de panneau) et types « référence » de mêmes couleurs."""
        Z = set()
        for t, ck in self.clk.items():
            Z.add(t)
            if ck[0] == "flip":
                Z.update((ck[2], ck[3]))
            else:
                Z.update(ck[2].keys())
                Z.update(ck[2].values())
        return Z

    def atomes_candidats(self, prog, etats):
        ats = []
        vus = {}
        for M in etats:
            for t in set(M):
                if t > 0:
                    vus[t] = vus.get(t, 0) + 1
        evapore = set()
        for ev in self.events:
            if ev.s1 is None or ev.flag:
                continue
            a = set(ev.s0[0])
            b = set(ev.s1[0])
            evapore |= {t for t in a - b if t > 0}
        for t in sorted(evapore):
            ats.append(("gone", t))
        Z = self.familles_panneau()
        if Z:
            Zf = frozenset(Z)
            for S in sorted(Z):
                ats.append(("all", Zf, S))
            cols = set()
            for t in Z:
                cols.update(self.types.couleurs(t))
            R = frozenset(t for t in vus if t not in Z and t > 0
                          and self.types.couleurs(t) and set(self.types.couleurs(t)) <= cols
                          and t not in evapore)
            if R:
                ats.append(("pmatch", Zf, R))
        return ats

    def apprendre_cond(self, A, B, prog):
        """Plus petite conjonction d'atomes vraie sur tous les états où B est affiché et fausse sur au moins un
        atome des états où A est affiché."""
        pos, neg = [], []
        for ev in self.events:
            for s in (ev.s0, ev.s1):
                if s is None:
                    continue
                M = s[0]
                if B in M:
                    pos.append(M)
                elif A in M:
                    neg.append(M)
        if not pos and not neg:
            return ()
        ats = self.atomes_candidats(prog, [m for m in pos + neg])
        geo = self.events[-1].geo if self.events else None
        ok = [a for a in ats if all(prog.atome(a, M, geo) for M in pos)]
        if not neg:
            return ()
        # conjonction maximale : tout atome vrai sur tous les exemples ouverts et faux sur au moins un exemple fermé
        res = [a for a in ok if any(not prog.atome(a, M, geo) for M in neg)]
        restant = [M for M in neg if all(prog.atome(a, M, geo) for a in res)]
        return res if not restant else None

    def apprendre_win_cond(self, prog):
        geo = self.win_click[0].geo
        pos = []
        for ev in self.win_click:
            r = prog.pas(ev.geo, ev.s0, ev.act)
            # victoire par clic : l'après-état est celui où les cases ont changé ; sans effet connu, on n'apprend rien
            if r is not None:
                pos.append(r[0][0])
        if not pos:
            return None
        neg = [ev.s1[0] for ev in self.events if ev.flag == 0 and ev.s1 is not None and ev.act[0] == "ACTION6"]
        neg += [ev.s0[0] for ev in self.events if ev.flag == 0]
        ats = [a for a in self.atomes_candidats(prog, pos + neg) if a[0] in ("all", "pmatch")]
        ok = [a for a in ats if all(prog.atome(a, M, geo) for M in pos)]
        if not ok:
            return None
        restant = [M for M in neg if all(prog.atome(a, M, geo) for a in ok)]
        # préférence pragmatique : motif égal à la référence avant « tout dans le même état »
        ok.sort(key=lambda a: (0 if a[0] == "pmatch" else 1))
        res = []
        restant = list(neg)
        while restant:
            best, bs = None, 0
            for a in ok:
                s = sum(1 for M in restant if not prog.atome(a, M, geo))
                if s > bs:
                    best, bs = a, s
            if best is None:
                break
            res.append(best)
            restant = [M for M in restant if prog.atome(best, M, geo)]
        return res or ok[:1]

    # ------------------------------------------------------------ observation
    def observer(self, ev):
        """Ajoute un événement ; propose des règles s'il n'est pas prédit. Renvoie True si le programme a changé."""
        if ev.cle in self.vus:
            return False
        self.vus.add(ev.cle)
        self.events.append(ev)
        self.stats["evenements"] += 1
        self._marquer_explore(ev)
        if ev.act[0] == "ACTION6" and ev.flag == 0 and ev.s1 is not None and ev.s1[0] == ev.s0[0]:
            T = ev.s0[0][ev.act[1]]
            if T not in self.clk:
                self.click_inerte.add(T)
        if ecart(self.prog, ev) == 0:
            return False
        self.stats["surprises"] += 1
        changed = False
        for _ in range(8):
            if not self._expliquer(ev):
                break
            changed = True
            self.prog = self.construire()
            if ecart(self.prog, ev) == 0:
                break
        if changed:
            self._stabiliser()
        return changed

    def _marquer_explore(self, ev):
        name = ev.act[0]
        d = self.dirmap.get(name)
        if d is None or ev.s0[1] < 0 or name == "ACTION6":
            return
        q = ev.geo.nb[ev.s0[1]][d]
        if q < 0 or q in ev.geo.ignore:
            return
        T = ev.s0[0][q]
        if T > 0:
            self.explore.add(T)

    def _stabiliser(self):
        """Rejoue tous les événements ; les non expliqués relancent des propositions ; choisit les alternatives."""
        self.stats["reevals"] += 1
        for _ in range(6):
            self.prog = self.construire()
            self._choisir_creneaux()
            self.prog = self.construire()
            mauvais = [e for e in self.events if ecart(self.prog, e) != 0]
            prog_chg = False
            for e in mauvais[-6:]:
                if self._expliquer(e):
                    prog_chg = True
                    self.prog = self.construire()
            if not prog_chg:
                break
        self.prog = self.construire()

    def _choisir_creneaux(self):
        for cle, cr in self.creneaux.items():
            if len(cr.alts) < 2:
                continue
            sc = []
            for i, (nom, lp) in enumerate(cr.alts):
                p = self.construire({cle: i})
                err = sum(1 for e in self.events if ecart(p, e) != 0)
                sc.append(lp - 6.0 * err - 0.3 * p.taille())
            m = max(sc)
            ex = [math.exp(x - m) for x in sc]
            z = sum(ex)
            cr.post = [x / z for x in ex]
            cr.choix = max(range(len(sc)), key=lambda i: sc[i])

    def top_k(self, K=6):
        """Le MAP et ses déviations d'un seul créneau, avec poids relatifs."""
        res = [(0.0, self.prog)]
        for cle, cr in self.creneaux.items():
            for i in range(len(cr.alts)):
                if i == cr.choix:
                    continue
                res.append((math.log(max(cr.post[i], 1e-9) / max(cr.post[cr.choix], 1e-9)),
                            self.construire({cle: i})))
        res.sort(key=lambda x: -x[0])
        return res[:K]

    # ------------------------------------------------------------ propositions
    def _expliquer(self, ev):
        name, arg = ev.act
        geo = ev.geo
        if ev.flag == 2:
            return self._expl_mort(ev)
        if name == "ACTION6":
            return self._expl_clic(ev)
        return self._expl_move(ev)

    def _expl_mort(self, ev):
        name, _ = ev.act
        d = self.dirmap.get(name)
        if d is None:
            return False
        q = ev.geo.nb[ev.s0[1]][d]
        if q < 0:
            return False
        T = ev.s0[0][q]
        if T in self.push:
            return False
        if T in self.kill:
            return False
        self.kill.add(T)
        self.solid.discard(T)
        self.explore.add(T)
        return True

    def _expl_move(self, ev):
        name, _ = ev.act
        geo = ev.geo
        s0, s1 = ev.s0, ev.s1
        M0 = s0[0]
        pos0 = s0[1]
        if pos0 < 0:
            return False
        if ev.flag == 1:
            return self._expl_win_move(ev)
        pos1 = s1[1]
        if name not in self.dirmap:
            if pos1 != pos0 and pos1 >= 0:
                r0, c0 = divmod(pos0, geo.C)
                r1, c1 = divmod(pos1, geo.C)
                v = (r1 - r0, c1 - c0)
                if v in geo.dirs:
                    self.dirmap[name] = geo.dirs.index(v)
                    return True
            return False
        d = self.dirmap[name]
        q = geo.nb[pos0][d]
        if q < 0 or q in geo.ignore:
            return False
        T0 = M0[q]
        changed = False
        if pos1 == pos0:           # bloqué
            if T0 not in self.push and T0 not in self.solid and T0 > 0 and T0 != -1:
                self.solid.add(T0)
                self.explore.add(T0)
                changed = True
            elif T0 == 0 and q in geo.outer:
                changed = False
            return changed
        if pos1 != q:
            return False
        M1 = s1[0]
        if T0 in self.solid:
            self.solid.discard(T0)
            changed = True
        self.explore.add(T0)
        # case quittée : contenu révélé
        tor = s0[3]
        if M1[pos0] != -1 and M1[pos0] != self.prog.revele(tor) and M0[pos0] == -1:
            X = M1[pos0]
            old = [e for e in self.fx.get(tor, []) if e[0] != "set"]
            self.fx[tor] = old + [("set", X)]
            changed = True
        # blocs poussés
        q2 = geo.nb[q][d]
        poussee = False
        if q2 >= 0 and T0 > 0:
            # la case q contenait T0 ; si elle est vide d'après M1 (avatar dessus) et un T0 est apparu plus loin
            nouveaux = [i for i in range(geo.N) if M0[i] == 0 and M1[i] == T0 and i not in geo.ignore]
            if len(nouveaux) == 1:
                z = nouveaux[0]
                if T0 not in self.push:
                    self.push.add(T0)
                    self.solid.discard(T0)
                    changed = True
                poussee = True
                if z != q2:
                    rq, cq = divmod(q2, geo.C)
                    rz, cz = divmod(z, geo.C)
                    v = (rz - rq, cz - cq)
                    g = None
                    for k, (dr, dc) in enumerate(geo.dirs):
                        if (dr == 0 and v[0] == 0 and dc * v[1] > 0) or (dc == 0 and v[1] == 0 and dr * v[0] > 0):
                            g = k
                    if g is not None and self.grav != g:
                        self.grav = g
                        changed = True
        # avatar : couleur
        if s1[2] != s0[2]:
            if ("avc", s1[2]) not in self.fx.get(T0, []):
                self.fx[T0] = [e for e in self.fx.get(T0, []) if e[0] != "avc"] + [("avc", s1[2])]
                changed = True
        # autres changements de cases : groupes (A -> B)
        grp = {}
        for i in range(geo.N):
            if i == pos0 or i == q or i in geo.ignore:
                continue
            a, b = M0[i], M1[i]
            if a != b and a != -1 and b != -1:
                if poussee and (a == 0 or b == 0) and (a == T0 or b == T0):
                    continue
                if poussee and a == 0 and b == T0:
                    continue
                grp.setdefault((a, b), []).append(i)
        for (a, b), cells in list(grp.items()):
            tot = sum(1 for t in M0 if t == a)
            if len(cells) != tot:
                continue                 # seulement une partie des cases A : non géré
            if a == 0 or b == 0:
                continue
            if (b, a) in grp and len(grp[(b, a)]) == sum(1 for t in M0 if t == b):
                cle = ("evt", T0, min(a, b), max(a, b))
                if cle not in self.creneaux or True:
                    self._ajouter_creneau_certain(T0, a, b)
                    changed = True
                continue
            cle = ("evt", T0, a, b)
            if cle not in self.creneaux and ("evt", T0, b, a) not in self.creneaux:
                collecte = any(e[0] == "set" and e[1] == 0 for e in self.fx.get(T0, [])) or True
                self.creneaux[cle] = Creneau(cle, [("recolor", math.log(0.5)), ("sync", math.log(0.35)),
                                                   ("toggle", math.log(0.15))])
                changed = True
        return changed

    def _ajouter_creneau_certain(self, T0, a, b):
        cle = ("evt", T0, a, b)
        cr = Creneau(cle, [("toggle", 0.0)])
        self.creneaux[cle] = cr

    def _expl_win_move(self, ev):
        name, _ = ev.act
        d = self.dirmap.get(name)
        if d is None:
            return False
        q = ev.geo.nb[ev.s0[1]][d]
        T0 = ev.s0[0][q]
        if T0 in self.push:
            return False
        avc = ev.s0[2]
        cls = None
        for col, cl in self.types.inv[T0]:
            if col == avc:
                cls = cl
        entree = (T0, avc, cls)
        if entree in self.win_obs:
            return False
        self.win_obs.append(entree)
        self.explore.add(T0)
        self.win_nonobs.discard(T0)
        if self.avc_change and cls is not None:
            cr = Creneau(("winmode",), [("entermatch", math.log(0.6)), ("enter", math.log(0.4))])
            self.creneaux[("winmode",)] = cr
        return True

    def _expl_clic(self, ev):
        geo = ev.geo
        s0, s1 = ev.s0, ev.s1
        p = ev.act[1]
        if p is None or p < 0:
            return False
        if ev.flag == 1:
            if ev not in self.win_click:
                self.win_click.append(ev)
                return True
            return False
        T = s0[0][p]
        M0, M1 = s0[0], s1[0]
        ch = [i for i in range(geo.N) if M0[i] != M1[i] and i not in geo.ignore and M0[i] != -1 and M1[i] != -1]
        if not ch:
            if T not in self.click_inerte and T not in self.clk:
                self.click_inerte.add(T)
                self.explore.add(T)
                return True
            return False
        self.click_inerte.discard(T)
        self.explore.add(T)
        r0, c0 = divmod(p, geo.C)
        # famille de panneau : types du changement de la case cliquée (+ familles déjà connues)
        fam = set()
        if M0[p] != M1[p]:
            fam.update((M0[p], M1[p]))
        fam |= self.familles_panneau() & {M0[i] for i in ch}
        fam |= {M1[i] for i in ch if M0[i] in fam}
        offs, pairs, derives = set(), set(), {}
        for i in ch:
            a, b = M0[i], M1[i]
            if a in fam and b in fam:
                r, c = divmod(i, geo.C)
                offs.add((r - r0, c - c0))
                pairs.add((a, b))
            else:
                derives.setdefault((a, b), []).append(i)
        changed = False
        for (a, b), cells in derives.items():
            if len(cells) == sum(1 for t in M0 if t == a) and a > 0 and b > 0:
                cle = ("evt", -2, a, b)
                if cle not in self.creneaux:
                    self.creneaux[cle] = Creneau(cle, [("sync", 0.0)])
                    changed = True
        if not pairs:
            return changed
        if offs == {(0, 0)}:
            a, b = next(iter(pairs))
            old = self.clk.get(T)
            nm = dict(old[2]) if old and old[0] == "next" else {}
            nm[a] = b
            self.clk[T] = ("next", ((0, 0),), nm)
            self._fermer_cycles(T)
            for t in list(nm):
                if t != T:
                    self.clk[t] = self.clk[T]
            for t in nm.values():
                if t not in self.clk:
                    self.clk[t] = self.clk[T]
            return True
        offs4 = {(-1, 0), (1, 0), (0, -1), (0, 1)}
        offs2 = set(offs)
        if offs2 & offs4:
            offs2 |= offs4 | {(0, 0)}
        ts = set()
        for a, b in pairs:
            ts.update((a, b))
        if len(ts) == 2:
            A, B = sorted(ts)
            spec = ("flip", tuple(sorted(offs2)), A, B)
            if self.clk.get(T) != spec or self.clk.get(A) != spec or self.clk.get(B) != spec:
                for t in (A, B):
                    self.clk[t] = spec
                self.clk[T] = spec
                return True
        return changed

    def _fermer_cycles(self, T):
        spec = self.clk[T]
        nm = dict(spec[2])
        # chaîne a->b->c : si c n'a pas de suivant, on suppose le cycle fermé (c -> a) ; a est la tête
        destinations = set(nm.values())
        tetes = [a for a in nm if a not in destinations]
        fins = [b for b in destinations if b not in nm]
        if len(tetes) == 1 and len(fins) == 1:
            nm[fins[0]] = tetes[0]
        elif not tetes and not fins:
            pass
        self.clk[T] = ("next", ((0, 0),), nm)
        for t in nm:
            self.clk[t] = self.clk[T]
