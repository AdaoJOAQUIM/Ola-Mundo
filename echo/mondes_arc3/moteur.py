"""Moteur de règles d'un niveau + oracle A* (preuve de solvabilité, optimum).

État abstrait (tuple hachable) :
    (pieces, sel, blocs, gemmes, cles, interr, couleur, panneau, t)
Actions abstraites : 0..3 = haut/bas/gauche/droite, 4 = attendre, 5+c = clic sur la case c.
Le moteur est pur : `pas(etat, a)` -> None (sans effet) ou (nouvel_etat, drapeau)
drapeau : 0 normal, 1 niveau gagné, 2 mort.
"""
from __future__ import annotations

import heapq
import itertools
from math import gcd

DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))   # (dx, dy) : haut, bas, gauche, droite
ATTENTE = 4
CLIC0 = 5
I_PCS, I_SEL, I_BLK, I_GEM, I_KEY, I_SW, I_COL, I_PAN, I_T = range(9)


class Niveau:
    """Données statiques d'un niveau (positions = y*n + x)."""

    def __init__(self, n):
        self.n = n
        self.solid = frozenset()       # murs + zones réservées (panneau, référence)
        self.pieces = ()               # positions de départ
        self.col0 = 0                  # couleur initiale (mode couleurs, une seule pièce)
        self.piece_cols = ()           # couleur de chaque pièce (mode sélection)
        self.exits = ()                # ((pos, couleur), ...)
        self.gems = ()
        self.hazards = frozenset()
        self.keys = ()                 # ((pos, kid), ...)
        self.doors = {}                # pos -> kid
        self.blocks = ()
        self.targets = frozenset()
        self.tele = {}                 # pos -> pos partenaire
        self.switches = {}             # pos -> groupe
        self.swalls = {}               # pos -> (groupe, inv) fermé quand bit(groupe)==inv
        self.paint = {}                # pos -> couleur
        self.enemies = ()              # trajets : tuple de tuples de positions (cycle complet)
        self.panel = None              # dict(kind, cells, ref, K, init, goal)
        self.goal = "sortie"           # sortie | cibles | panel
        self.grav = None               # None ou indice de direction
        self.selection = False
        self.couleurs = False
        self.energy = None
        self.opt = None
        self.sol = None

    def presentes(self):
        """Mécaniques effectivement présentes dans ce niveau."""
        s = set()
        if self.goal != "panel":
            s.add("deplacement")
            if self.solid - self._panel_cells():
                s.add("murs")
        if self.keys and self.doors:
            s.add("cles_portes")
        if self.blocks:
            s.add("blocs_poussables")
        if self.goal == "cibles" and self.targets:
            s.add("cibles")
        if self.gems:
            s.add("collectibles")
        if self.hazards:
            s.add("dangers")
        if self.tele:
            s.add("teleporteurs")
        if self.switches and self.swalls:
            s.add("interrupteurs")
        if self.couleurs and (self.paint or self.selection):
            s.add("couleurs")
        if self.panel and self.panel["kind"] == "lumieres":
            s.add("lumieres")
        if self.panel and self.panel["kind"] == "motif":
            s.add("motif")
        if self.selection:
            s.add("selection")
        if self.grav is not None and self.blocks:
            s.add("gravite")
        if self.energy is not None:
            s.add("energie")
        if self.enemies:
            s.add("ennemis")
        return s

    def _panel_cells(self):
        if not self.panel:
            return frozenset()
        return frozenset(self.panel["cells"]) | frozenset(self.panel.get("ref", ()))


class Moteur:
    def __init__(self, lv: Niveau):
        self.lv = lv
        n = lv.n
        self.n = n
        N = n * n
        self.nb = []
        for p in range(N):
            y, x = divmod(p, n)
            row = []
            for dx, dy in DIRS:
                xx, yy = x + dx, y + dy
                row.append(yy * n + xx if 0 <= xx < n and 0 <= yy < n else -1)
            self.nb.append(tuple(row))
        self.solid = lv.solid
        self.door = lv.doors
        self.swall = lv.swalls
        self.gempos = {p: i for i, p in enumerate(lv.gems)}
        self.keypos = {p: k for p, k in lv.keys}
        self.paint = lv.paint
        self.tele = lv.tele
        self.sw_at = lv.switches
        self.hazard = lv.hazards
        self.has_player = lv.goal != "panel"
        self.epos = ()
        self.period = 1
        epath = set()
        if lv.enemies:
            per = 1
            for tr in lv.enemies:
                per = per * len(tr) // gcd(per, len(tr))
            self.period = per
            self.epos = tuple(tuple(tr[t % len(tr)] for tr in lv.enemies) for t in range(per))
            for tr in lv.enemies:
                epath.update(tr)
        self.epath = frozenset(epath)
        # cases où un bloc peut se trouver : sol nu ou cible
        interdit = set(lv.solid) | set(lv.hazards) | set(lv.tele) | set(lv.switches) | set(lv.swalls) \
            | set(lv.doors) | set(lv.gems) | set(p for p, _ in lv.keys) | set(p for p, _ in lv.exits) \
            | set(lv.paint) | self.epath
        self.bfree = [p not in interdit for p in range(N)]
        self.exitcols = {}
        for p, c in lv.exits:
            self.exitcols[p] = c
        self.full_gem = (1 << len(lv.gems)) - 1
        self.targets = lv.targets
        pn = lv.panel
        if pn:
            self.pidx = {p: i for i, p in enumerate(pn["cells"])}
            self.k = pn["k"]
            k = self.k
            self.lmask = []
            if pn["kind"] == "lumieres":
                for i in range(k * k):
                    a, b = divmod(i, k)
                    m = 1 << i
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if 0 <= a + da < k and 0 <= b + db < k:
                            m |= 1 << ((a + da) * k + b + db)
                    self.lmask.append(m)
            self.K = pn.get("K", 2)
        self.ncells = N

    # ------------------------------------------------------------ utilitaires
    def etat0(self):
        lv = self.lv
        pan = lv.panel["init"] if lv.panel else 0
        blk = tuple(sorted(lv.blocks))
        if lv.grav is not None and blk:
            blk = self.gravite(blk, lv.pieces)
        return (tuple(lv.pieces), 0, blk, 0, 0, 0, lv.col0, pan, 0)

    def ferme(self, q, sw):
        g, inv = self.swall[q]
        return ((sw >> g) & 1) == inv

    def gravite(self, blk, pcs):
        g = self.lv.grav
        dx, dy = DIRS[g]
        bl = list(blk)
        n = self.n
        while True:
            bouge = False
            ordre = sorted(bl, key=lambda p: -((p % n) * dx + (p // n) * dy))
            occ = set(bl)
            for b in ordre:
                q = self.nb[b][g]
                if q >= 0 and self.bfree[q] and q not in occ and q not in pcs:
                    occ.discard(b)
                    occ.add(q)
                    bouge = True
            bl = list(occ)
            if not bouge:
                break
        return tuple(sorted(bl))

    def panneau_resolu(self, pan):
        return pan == self.lv.panel["goal"]

    def verrous_ok(self, s):
        if self.lv.gems and s[I_GEM] != self.full_gem:
            return False
        if self.lv.panel and self.lv.goal != "panel" and not self.panneau_resolu(s[I_PAN]):
            return False
        return True

    def gagne(self, s):
        lv = self.lv
        if not self.verrous_ok(s):
            return False
        if lv.goal == "panel":
            return self.panneau_resolu(s[I_PAN])
        if lv.goal == "cibles":
            return all(t in s[I_BLK] for t in lv.targets)
        pcs = s[I_PCS]
        if lv.selection:
            for i, p in enumerate(pcs):
                c = self.exitcols.get(p)
                if c is None or c != lv.piece_cols[i]:
                    return False
            return True
        c = self.exitcols.get(pcs[0])
        if c is None:
            return False
        return (not lv.couleurs) or c == s[I_COL]

    # ------------------------------------------------------------ transition
    def pas(self, s, a):
        lv = self.lv
        pcs, sel, blk, gem, key, sw, col, pan, t = s
        if a < 4:
            if not self.has_player:
                return None
            p = pcs[sel]
            q = self.nb[p][a]
            if q < 0 or q in self.solid:
                return None
            if q in self.swall and self.ferme(q, sw):
                return None
            d = self.door.get(q)
            if d is not None and not (key >> d) & 1:
                return None
            if q in pcs:
                return None
            pousse = False
            if q in blk:
                q2 = self.nb[q][a]
                if q2 < 0 or not self.bfree[q2] or q2 in blk or q2 in pcs:
                    return None
                blk = tuple(sorted(q2 if b == q else b for b in blk))
                pousse = True
            mort = q in self.hazard
            gi = self.gempos.get(q)
            if gi is not None:
                gem |= 1 << gi
            kid = self.keypos.get(q)
            if kid is not None:
                key |= 1 << kid
            g = self.sw_at.get(q)
            if g is not None:
                sw ^= 1 << g
            c = self.paint.get(q)
            if c is not None:
                col = c
            newp = q
            tp = self.tele.get(q)
            if tp is not None and tp not in pcs:
                newp = tp
            pcs2 = pcs[:sel] + (newp,) + pcs[sel + 1:]
            if pousse and lv.grav is not None:
                blk = self.gravite(blk, pcs2)
            old = pcs
            pcs = pcs2
        elif a == ATTENTE:
            if not lv.enemies:
                return None
            old = pcs
            mort = False
        else:
            c = a - CLIC0
            old = pcs
            mort = False
            pn = lv.panel
            if pn and c in self.pidx:
                i = self.pidx[c]
                if pn["kind"] == "lumieres":
                    pan ^= self.lmask[i]
                else:
                    pan = pan[:i] + ((pan[i] + 1) % self.K,) + pan[i + 1:]
            elif lv.selection and c in pcs and pcs.index(c) != sel:
                sel = pcs.index(c)
            else:
                return None
            s2 = (pcs, sel, blk, gem, key, sw, col, pan, t)
            return s2, (1 if self.gagne(s2) else 0)
        # tic : ennemis (seuls déplacements et attente font avancer le temps)
        t2 = t
        if lv.enemies:
            t2 = (t + 1) % self.period
            ancien = self.epos[t]
            nouveau = self.epos[t2]
            for i, ep in enumerate(nouveau):
                if ep in pcs:
                    mort = True
                    break
                for j, p0 in enumerate(old):
                    if p0 == ep and pcs[j] == ancien[i] and ancien[i] != ep:
                        mort = True
                        break
        s2 = (pcs, sel, blk, gem, key, sw, col, pan, t2)
        if mort:
            return s2, 2
        if self.gagne(s2):
            return s2, 1
        return s2, 0

    # ------------------------------------------------------------ oracle
    def actions(self, s):
        acts = []
        if self.has_player:
            acts.extend((0, 1, 2, 3))
        if self.lv.enemies:
            acts.append(ATTENTE)
        pn = self.lv.panel
        if pn:
            acts.extend(CLIC0 + c for c in pn["cells"])
        if self.lv.selection:
            for i, p in enumerate(s[I_PCS]):
                if i != s[I_SEL]:
                    acts.append(CLIC0 + p)
        return acts

    def heur(self, s):
        lv = self.lv
        h = 0
        pn = lv.panel
        if pn:
            if pn["kind"] == "motif":
                K = self.K
                h += sum((g - c) % K for c, g in zip(s[I_PAN], pn["goal"]))
            elif s[I_PAN] != pn["goal"]:
                h += 1
        if lv.goal == "cibles":
            rest = sum(1 for t in lv.targets if t not in s[I_BLK])
            h += min(rest, 1) if lv.grav is not None else rest
        elif lv.goal == "sortie" and not lv.tele:
            n = self.n
            ex = [p for p, c in lv.exits]
            if lv.selection:
                for i, p in enumerate(s[I_PCS]):
                    tg = [e for e in ex if self.exitcols[e] == lv.piece_cols[i]]
                    if p not in tg:
                        h += min(abs(p % n - e % n) + abs(p // n - e // n) for e in tg)
            else:
                p = s[I_PCS][0]
                if p not in ex:
                    h += min(abs(p % n - e % n) + abs(p // n - e // n) for e in ex)
        return h

    def resoudre(self, cap=150000):
        """A* (heuristique admissible) : (optimum, actions abstraites) ou None (insoluble / cap)."""
        s0 = self.etat0()
        if self.gagne(s0):
            return None
        g = {s0: 0}
        parent = {s0: None}
        cpt = itertools.count()
        tas = [(self.heur(s0), 0, next(cpt), s0, None)]
        exp = 0
        while tas:
            f, gs, _, s, but = heapq.heappop(tas)
            if but is not None:          # nœud-but : premier extrait = optimal
                chemin = [but]
                x = s
                while parent[x] is not None:
                    x, aa = parent[x]
                    chemin.append(aa)
                chemin.reverse()
                return gs, chemin
            if gs > g.get(s, 1 << 30):
                continue
            exp += 1
            if exp > cap:
                return None
            for a in self.actions(s):
                r = self.pas(s, a)
                if r is None:
                    continue
                s2, fl = r
                if fl == 2:
                    continue
                if fl == 1:
                    heapq.heappush(tas, (gs + 1, gs + 1, next(cpt), s, a))
                    continue
                g2 = gs + 1
                if g2 < g.get(s2, 1 << 30):
                    g[s2] = g2
                    parent[s2] = (s, a)
                    heapq.heappush(tas, (g2 + self.heur(s2), g2, next(cpt), s2, None))
        return None

    def bfs_optimum(self, cap=400000):
        """BFS exact (contrôle indépendant de l'A*), pour les tests."""
        s0 = self.etat0()
        vu = {s0}
        front = [s0]
        prof = 0
        while front and len(vu) < cap:
            prof += 1
            nxt = []
            for s in front:
                for a in self.actions(s):
                    r = self.pas(s, a)
                    if r is None:
                        continue
                    s2, fl = r
                    if fl == 2:
                        continue
                    if fl == 1:
                        return prof
                    if s2 not in vu:
                        vu.add(s2)
                        nxt.append(s2)
            front = nxt
        return None


def solution_panneau(pn):
    """Séquence minimale de clics (actions abstraites) amenant le panneau à son but."""
    if pn["kind"] == "motif":
        K = pn["K"]
        out = []
        for i, (c, g) in enumerate(zip(pn["init"], pn["goal"])):
            out.extend([CLIC0 + pn["cells"][i]] * ((g - c) % K))
        return out
    k = pn["k"]
    lm = []
    for i in range(k * k):
        a, b = divmod(i, k)
        m = 1 << i
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= a + da < k and 0 <= b + db < k:
                m |= 1 << ((a + da) * k + b + db)
        lm.append(m)
    s0, g = pn["init"], pn["goal"]
    par = {s0: None}
    front = [s0]
    while front and g not in par:
        nxt = []
        for s in front:
            for i, m in enumerate(lm):
                s2 = s ^ m
                if s2 not in par:
                    par[s2] = (s, i)
                    nxt.append(s2)
        front = nxt
    if g not in par:
        return None
    out = []
    x = g
    while par[x] is not None:
        x, i = par[x]
        out.append(CLIC0 + pn["cells"][i])
    out.reverse()
    return out


def oracle(lv, cap=30000):
    """Optimum et solution d'un niveau. Panneau et navigation sont indépendants (les clics ne font pas avancer le temps des ennemis) : on décompose, l'optimum est alors la somme exacte."""
    if lv.panel:
        pa = solution_panneau(lv.panel)
        if pa is None:
            return None
        if lv.goal == "panel":
            return (len(pa), pa) if pa else None
        import copy
        lv2 = copy.copy(lv)
        lv2.panel = None
        r = Moteur(lv2).resoudre(cap)
        if r is None:
            return None
        return len(pa) + r[0], pa + r[1]
    return Moteur(lv).resoudre(cap)
