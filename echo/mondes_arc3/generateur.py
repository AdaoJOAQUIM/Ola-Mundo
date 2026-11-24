"""Générateur procédural de mondes (jeux) ARC-AGI-3-like.

Un MONDE (seed, difficulté) fixe des règles cachées : but, mécaniques actives, correspondance
ACTION1..4 -> directions, couleurs et formes des rôles, direction de la gravité, nombre de
niveaux (3 à 8). Les NIVEAUX d'un monde partagent ces règles et montent en difficulté ; chacun est
prouvé soluble par l'oracle A* de moteur.py (optimum = proxy d'efficacité humaine).
Tout est déterministe : chaque niveau tire sa propre graine "niv|seed|d|k".
"""
from __future__ import annotations

import random

from moteur import Moteur, Niveau, oracle

MECANIQUES = (
    "deplacement", "murs", "cles_portes", "blocs_poussables", "collectibles", "dangers",
    "teleporteurs", "interrupteurs", "couleurs", "lumieres", "selection", "motif", "gravite",
    "energie", "ennemis", "cibles", "annuler", "attente",
)
# mécaniques dont la présence se lit dans les données du niveau
MECA_NIVEAU = set(MECANIQUES) - {"annuler"}

SORTIE_OPTS = ["cles_portes", "collectibles", "dangers", "teleporteurs", "interrupteurs", "couleurs",
               "blocs_poussables", "lumieres", "motif", "ennemis", "energie", "annuler", "selection"]
CIBLES_OPTS = ["dangers", "ennemis", "energie", "annuler", "interrupteurs", "cles_portes",
               "collectibles", "gravite"]

ROLES = ["mur", "joueur", "sortie", "gemme", "danger", "cle0", "cle1", "bloc", "cible", "tele0", "tele1",
         "inter", "swall", "ennemi", "on", "off", "c0", "c1", "c2", "sel", "hud_on", "hud_off", "energie"]
FORMES = ["plein", "anneau", "croix", "diamant", "point", "damier"]


class Monde:
    def __init__(self, seed: int, difficulte: int = 1):
        self.seed = int(seed)
        self.d = int(difficulte)
        w = random.Random(f"monde|{self.seed}|{self.d}")
        self.nl = w.randint(3, 8)
        r = w.random()
        self.goal = "sortie" if r < 0.55 else ("cibles" if r < 0.8 else "panel")
        meca = set()
        nopt = self.d + 1 + w.randint(0, 1)
        if self.goal == "sortie":
            pool = list(SORTIE_OPTS)
            w.shuffle(pool)
            for m in pool:
                if len(meca) >= nopt:
                    break
                if m == "lumieres" and "motif" in meca or m == "motif" and "lumieres" in meca:
                    continue
                if m == "selection" and "couleurs" in meca or m == "couleurs" and "selection" in meca:
                    continue
                if m == "selection" and meca & {"ennemis", "teleporteurs", "blocs_poussables", "interrupteurs", "cles_portes"}:
                    continue
                if m in ("ennemis", "teleporteurs", "blocs_poussables", "interrupteurs", "cles_portes") and "selection" in meca:
                    continue
                meca.add(m)
            if "blocs_poussables" in meca and w.random() < 0.45:
                meca.add("gravite")
        elif self.goal == "cibles":
            meca |= {"blocs_poussables", "cibles"}
            pool = list(CIBLES_OPTS)
            w.shuffle(pool)
            meca |= set(pool[:nopt])
        else:
            meca.add(w.choice(["lumieres", "motif"]))
            for m in ("energie", "annuler"):
                if w.random() < 0.35 + 0.1 * self.d:
                    meca.add(m)
        if "selection" in meca:
            meca.add("couleurs")
        if self.goal != "panel":
            meca |= {"deplacement", "murs"}
        if "ennemis" in meca:
            meca.add("attente")
        self.meca = meca
        # règles cachées
        self.dirperm = list(range(4))
        w.shuffle(self.dirperm)                     # ACTION(k+1) -> direction dirperm[k]
        self.grav = w.randrange(4) if "gravite" in meca else None
        self.verrou_lampe_cible = w.choice(["allume", "eteint"]) if False else "allume"
        # rendu
        self.bg = w.choice([0, 0, 0, 5, 10])
        pool = [c for c in range(16) if c != self.bg]
        w.shuffle(pool)
        self.col = {}
        besoin = ["mur", "hud_on", "hud_off", "sel"]
        if self.goal == "sortie":
            besoin.append("sortie")
        if "couleurs" not in meca:
            besoin.append("joueur")
        for m, rs in (("cles_portes", ("cle0", "cle1")), ("blocs_poussables", ("bloc",)), ("cibles", ("cible",)),
                      ("collectibles", ("gemme",)), ("dangers", ("danger",)), ("teleporteurs", ("tele0", "tele1")),
                      ("interrupteurs", ("inter", "swall")), ("couleurs", ("c0", "c1", "c2")),
                      ("motif", ("c0", "c1", "c2")), ("lumieres", ("on", "off")), ("energie", ("energie",)),
                      ("ennemis", ("ennemi",))):
            if m in meca:
                besoin.extend(rs)
        vus = []
        for r_ in besoin:
            if r_ not in vus:
                vus.append(r_)
        autres = [r_ for r_ in ROLES if r_ not in vus]
        for i, role in enumerate(vus + autres):
            self.col[role] = pool[i % len(pool)]
        # couleurs de HUD dédiées et distinctes de bg
        formes = list(FORMES)
        self.forme = {}
        for role in ROLES:
            self.forme[role] = w.choice(formes)
        self.forme["cle0"], self.forme["cle1"] = "croix", "croix"
        self.forme["joueur"] = w.choice(["plein", "diamant"])
        self.forme["ennemi"] = w.choice(["croix", "damier"])
        self.forme["danger"] = w.choice(["croix", "damier", "diamant"])
        self.forme["sortie"] = "anneau"
        self.forme["cible"] = "point"
        self.forme["bloc"] = "damier" if self.forme["joueur"] != "damier" else "plein"
        self._niveaux = {}

    # ------------------------------------------------------------ interface
    def actions_disponibles(self):
        a = []
        if self.goal != "panel":
            a += [1, 2, 3, 4]
        if "ennemis" in self.meca:
            a.append(5)
        if self.goal == "panel" or "lumieres" in self.meca or "motif" in self.meca or "selection" in self.meca:
            a.append(6)
        if "annuler" in self.meca:
            a.append(7)
        return a

    def action_vers_dir(self, k):
        """ACTION k (1..4) -> direction abstraite 0..3."""
        return self.dirperm[k - 1]

    def niveau(self, k: int) -> Niveau:
        if k not in self._niveaux:
            self._niveaux[k] = _cache_lire(self, k) or _cache_ecrire(self, k, gen_niveau(self, k))
        return self._niveaux[k]

    def niveaux(self):
        return [self.niveau(k) for k in range(self.nl)]

    def optimum(self):
        return [lv.opt for lv in self.niveaux()]

    @property
    def game_id(self):
        return f"mx{self.seed:05d}d{self.d}"


# ---------------------------------------------------------------------------------------------
# Cache disque optionnel (accélère les évaluations répétées) ; invalidé si le code change.
def _version():
    import hashlib
    from pathlib import Path
    h = hashlib.sha1()
    for nom in ("generateur.py", "moteur.py"):
        h.update((Path(__file__).resolve().parent / nom).read_bytes())
    return h.hexdigest()[:10]


def _chemin_cache(W, k):
    import os
    from pathlib import Path
    if os.environ.get("MONDES_SANS_CACHE"):
        return None
    d = Path(__file__).resolve().parent / "runs" / "cache" / _version()
    return d / f"mx{W.seed}d{W.d}_{k}.pkl"


def _cache_lire(W, k):
    import pickle
    p = _chemin_cache(W, k)
    if p is not None and p.exists():
        try:
            return pickle.loads(p.read_bytes())
        except Exception:
            return None
    return None


def _cache_ecrire(W, k, lv):
    import pickle
    p = _chemin_cache(W, k)
    if p is not None:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp%d" % __import__("os").getpid())
            tmp.write_bytes(pickle.dumps(lv))
            tmp.replace(p)
        except Exception:
            pass
    return lv


class _Echec(Exception):
    pass


def difficulte_niveau(W: Monde, k: int) -> float:
    return W.d * 0.55 + 2.4 * k / max(1, W.nl - 1)


def bornes(W: Monde, k: int):
    sup = max(0, W.d - 3)          # paliers 4 et 5 : niveaux plus longs (calibrage vers les jeux réels)
    if W.goal == "panel":
        return 2 + k // 3 + sup, 5 + 2 * k + 2 * W.d
    return 3 + 2 * k + 4 * sup, min(90, 16 + 6 * k + 10 * (W.d - 1))


def gen_niveau(W: Monde, k: int) -> Niveau:
    rng = random.Random(f"niv|{W.seed}|{W.d}|{k}")
    L = difficulte_niveau(W, k)
    lo, hi = bornes(W, k)
    meilleur = None
    for tent in range(240):
        relax = tent // 60
        try:
            lv = _tenter_panel(W, rng, L) if W.goal == "panel" else _tenter(W, rng, L, relax)
        except _Echec:
            continue
        # présence de toutes les mécaniques déclarées
        requis = (W.meca & MECA_NIVEAU)
        if W.goal == "panel":
            requis = {m for m in requis if m in ("lumieres", "motif", "energie")}
        manque = requis - lv.presentes() - ({"energie"} if "energie" in W.meca else set()) \
            - ({"attente"} if "ennemis" in lv.presentes() else set())
        if manque:
            continue
        r = oracle(lv, cap=8000)
        if r is None:
            continue
        opt, chemin = r
        if opt < lo and tent < 150:
            continue
        if opt > hi and tent < 150:
            continue
        lv.opt, lv.sol = opt, chemin
        if "energie" in W.meca:
            lv.energy = int(opt * (1.85 - 0.11 * L) + 0.999) + 2
        return lv
    raise RuntimeError(f"génération impossible monde={W.seed} d={W.d} niveau={k}")


# ---------------------------------------------------------------------------------------------
def _tenter_panel(W: Monde, rng, L):
    kind = "lumieres" if "lumieres" in W.meca else "motif"
    if kind == "lumieres":
        k = 3 if L < 3.2 else 4
        n = k + 2
        cells = [(1 + a, 1 + b) for a in range(k) for b in range(k)]
        goal = (1 << (k * k)) - 1
        init = goal
        j = 2 + int(L)
        idx = list(range(k * k))
        Mo = _lmasks(k)
        for i in rng.sample(idx, min(j, len(idx))):
            init ^= Mo[i]
        if init == goal:
            raise _Echec
        lv = Niveau(n)
        lv.goal = "panel"
        lv.panel = dict(kind="lumieres", cells=tuple(y * n + x for x, y in cells), k=k, K=2, ref=(),
                        init=init, goal=goal)
    else:
        k = 3 if L < 3.2 else 4
        K = 2 if L < 2.3 else 3
        n = 2 * k + 3
        cells = [(1 + b, 1 + a) for a in range(k) for b in range(k)]
        refs = [(k + 2 + b, 1 + a) for a in range(k) for b in range(k)]
        goal = tuple(rng.randrange(K) for _ in range(k * k))
        j = min(k * k - 1, 2 + int(L))
        init = list(goal)
        for i in rng.sample(range(k * k), j):
            init[i] = (init[i] + rng.randrange(1, K)) % K
        if tuple(init) == goal:
            raise _Echec
        lv = Niveau(n)
        lv.goal = "panel"
        lv.panel = dict(kind="motif", cells=tuple(y * n + x for x, y in cells), k=k, K=K,
                        ref=tuple(y * n + x for x, y in refs), init=tuple(init), goal=goal)
    lv.solid = frozenset()
    return lv


def _lmasks(k):
    out = []
    for i in range(k * k):
        a, b = divmod(i, k)
        m = 1 << i
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= a + da < k and 0 <= b + db < k:
                m |= 1 << ((a + da) * k + b + db)
        out.append(m)
    return out


# ---------------------------------------------------------------------------------------------
def _tenter(W: Monde, rng, L, relax):
    M = W.meca
    goal = W.goal
    n = min(10, 6 + int(L * 0.8))
    kind = "lumieres" if "lumieres" in M else ("motif" if "motif" in M else None)
    if kind:
        n = max(n, 9)
    # orientation aléatoire du repère canonique (u = axe de la chaîne de salles)
    transp = rng.random() < 0.5
    flipu = rng.random() < 0.5
    flipv = rng.random() < 0.5

    def mp(u, v):
        if flipu:
            u = n - 1 - u
        if flipv:
            v = n - 1 - v
        x, y = (v, u) if transp else (u, v)
        return y * n + x

    # zones réservées
    reserve = set()
    P = []
    R = []
    if kind:
        for a in range(3):
            for b in range(3):
                P.append((n - 3 + a, b))
                reserve.add((n - 3 + a, b))
                if kind == "motif":
                    R.append((n - 3 + a, 4 + b))
                    reserve.add((n - 3 + a, 4 + b))
    # portails (salles/portes)
    types = []
    if "cles_portes" in M:
        types.append("porte")
    if "interrupteurs" in M:
        types.append("inter")
    if "teleporteurs" in M:
        types.append("tele")
    if "ennemis" in M:
        types.append("ennemi")
    if "blocs_poussables" in M and goal == "sortie":
        types.append("bloc")
    rng.shuffle(types)
    ulo, uhi = 2, (n - 4 if kind else n - 3)
    maxg = (uhi - ulo) // 3 + 1
    if goal == "cibles":
        want = min(len(types), 1 + int(L >= 2.5))
    else:
        want = len(types) + (1 if L >= 3 else 0)
    ng = max(0, min(maxg, want, 3 if relax == 0 else 2))
    if types and ng == 0:
        ng = 1
    gates_t = (types + ["libre"] * 3)[:ng]
    # positions u des portes, espacées d'au moins 2
    cand = list(range(ulo, uhi + 1))
    gus = None
    for _ in range(20):
        c = sorted(rng.sample(cand, ng)) if ng <= len(cand) else None
        if c is not None and all(c[i + 1] - c[i] >= 2 for i in range(len(c) - 1)):
            gus = c
            break
    if gus is None:
        raise _Echec
    gvs = [rng.randrange(1, n - 1) for _ in gus]
    extras = [t for t in types if t not in gates_t]

    walls = set()
    occ = set(reserve)
    for gu, gv, t in zip(gus, gvs, gates_t):
        for v in range(n):
            if (gu, v) in reserve:
                continue
            if v == gv and t != "tele":
                continue
            walls.add((gu, v))
        occ.add((gu, gv))
        if t == "ennemi":
            m = 2 if (L >= 3 and 2 <= gv <= n - 3) else 1
            for dv in range(-m, m + 1):
                occ.add((gu - 1, gv + dv))
    # murs épars
    nw = int(n * n * (0.03 + 0.012 * L) / (1 + relax * 0.5))
    for _ in range(nw):
        c = (rng.randrange(n), rng.randrange(n))
        if c not in occ and c not in walls:
            walls.add(c)

    def libre(pred=None):
        cs = [(u, v) for u in range(n) for v in range(n)
              if (u, v) not in occ and (u, v) not in walls and (pred is None or pred(u, v))]
        if not cs:
            raise _Echec
        return cs

    def prendre(pred=None):
        c = rng.choice(libre(pred))
        occ.add(c)
        return c

    def piece_de(u):
        return sum(1 for g in gus if g < u)

    r0 = lambda u, v: piece_de(u) == 0 and (u not in gus)
    rl = lambda u, v: piece_de(u) == len(gus) and (u not in gus)

    lv = Niveau(n)
    lv.goal = goal
    lv.selection = "selection" in M
    lv.couleurs = "couleurs" in M
    lv.grav = W.grav
    items = {}
    # départ
    if lv.selection:
        npc = 2 + int(L >= 3.8 and relax == 0)
        starts = [prendre(r0) for _ in range(npc)]
        lv.piece_cols = tuple(range(npc))
    else:
        starts = [prendre(r0)]
    # sortie(s)
    exits = []
    if goal == "sortie":
        if lv.selection:
            for i in range(len(starts)):
                exits.append((prendre(rl), i))
        elif lv.couleurs:
            K = 2 if L < 2.8 else 3
            for c in range(1, K):
                exits.append((prendre(rl), c))
            lv.col0 = 0
            items["paint"] = []
            for c in range(1, K):
                items["paint"].append((prendre(lambda u, v: piece_de(u) <= len(gus)), c))
            if L >= 3 and K == 3:
                items["paint"].append((prendre(), 0))
        else:
            exits.append((prendre(rl), 0))
    # portes
    gem_cells, key_cells, doors, tele, switches, swalls, enemies, blocks = [], [], {}, {}, {}, {}, [], []
    kid = 0
    swg = 0
    for gu, gv, t in zip(gus, gvs, gates_t):
        near = lambda u, v, gu=gu: u < gu
        if t == "porte":
            doors[(gu, gv)] = kid
            key_cells.append((prendre(near), kid))
            kid += 1
        elif t == "inter":
            swalls[(gu, gv)] = (swg, 0 if rng.random() < 0.8 else 1)
            switches[prendre(near)] = swg
            swg += 1
        elif t == "tele":
            a = prendre(lambda u, v, gu=gu: u < gu)
            b = prendre(lambda u, v, gu=gu: u > gu)
            tele[a] = b
            tele[b] = a
        elif t == "ennemi":
            m = 2 if (L >= 3 and 2 <= gv <= n - 3) else 1
            tr = [(gu - 1, gv + dv) for dv in range(-m, m + 1)]
            for c in tr:
                walls.discard(c)
            cyc = tr + tr[-2:0:-1]
            r = rng.randrange(len(cyc))
            cyc = cyc[r:] + cyc[:r]
            enemies.append(tuple(cyc))
        elif t == "bloc":
            blocks.append((gu, gv))
    # extras hors portail
    for t in extras:
        if t == "porte":
            c = prendre()
            doors[c] = kid
            key_cells.append((prendre(), kid))
            kid += 1
        elif t == "inter":
            c = prendre()
            swalls[c] = (swg, 0)
            switches[prendre()] = swg
            swg += 1
        elif t == "tele":
            a, b = prendre(), prendre()
            tele[a] = b
            tele[b] = a
        elif t == "ennemi":
            for _ in range(30):
                u0, v0 = rng.randrange(n - 2), rng.randrange(n)
                tr = [(u0 + i, v0) for i in range(3)]
                if all(c not in occ and c not in walls for c in tr):
                    for c in tr:
                        occ.add(c)
                    enemies.append(tuple(tr + [tr[1]]))
                    break
            else:
                raise _Echec
        elif t == "bloc":
            blocks.append(prendre())
    if "collectibles" in M:
        for _ in range(1 if relax >= 2 else min(4, 1 + int(L) // 2)):
            gem_cells.append(prendre())
    hz = []
    if "dangers" in M:
        sx = starts
        for _ in range(1 if relax >= 2 else min(5, 1 + int(L))):
            hz.append(prendre(lambda u, v: all(abs(u - a) + abs(v - b) > 1 for a, b in sx)
                              and (u, v) not in [(g, w_) for g, w_ in zip(gus, gvs)]))
    targets = []
    if goal == "cibles":
        nb = 1 if relax >= 2 else min(3, 1 + int(L / 2.2))
        for _ in range(nb):
            ro = rng.randrange(len(gus) + 1)
            pr = lambda u, v, ro=ro: piece_de(u) == ro and u not in gus and 1 <= u <= n - 2 and 1 <= v <= n - 2
            blocks.append(prendre(pr))
            if W.grav is not None:
                def pr2(u, v, ro=ro):
                    if piece_de(u) != ro or u in gus:
                        return False
                    p = mp(u, v)
                    x, y = p % n, p // n
                    dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[W.grav]
                    xx, yy = x + dx, y + dy
                    if not (0 <= xx < n and 0 <= yy < n):
                        return True
                    for (uu, vv) in walls:
                        if mp(uu, vv) == yy * n + xx:
                            return True
                    return False
                targets.append(prendre(pr2))
            else:
                targets.append(prendre(pr))
    elif "blocs_poussables" in M and not blocks:
        blocks.append(prendre(lambda u, v: 1 <= u <= n - 2 and 1 <= v <= n - 2))
    # panneau
    if kind:
        k = 3
        goalp = (1 << 9) - 1 if kind == "lumieres" else None
        if kind == "lumieres":
            init = goalp
            Mo = _lmasks(3)
            for i in rng.sample(range(9), 2 + int(L >= 3)):
                init ^= Mo[i]
            if init == goalp:
                raise _Echec
            pan = dict(kind="lumieres", cells=tuple(mp(u, v) for u, v in P), k=3, K=2, ref=(),
                       init=init, goal=goalp)
        else:
            K = 2 if L < 2.5 else 3
            goalp = tuple(rng.randrange(K) for _ in range(9))
            init = list(goalp)
            for i in rng.sample(range(9), 2 + int(L >= 3)):
                init[i] = (init[i] + rng.randrange(1, K)) % K
            if tuple(init) == goalp:
                raise _Echec
            pan = dict(kind="motif", cells=tuple(mp(u, v) for u, v in P), k=3, K=K,
                       ref=tuple(mp(u, v) for u, v in R), init=tuple(init), goal=goalp)
        lv.panel = pan
    # assemblage en coordonnées réelles
    lv.solid = frozenset([mp(u, v) for u, v in walls] + [mp(u, v) for u, v in reserve])
    lv.pieces = tuple(mp(u, v) for u, v in starts)
    lv.exits = tuple((mp(u, v), c) for (u, v), c in exits)
    lv.gems = tuple(sorted(mp(u, v) for u, v in gem_cells))
    lv.hazards = frozenset(mp(u, v) for u, v in hz)
    lv.keys = tuple((mp(u, v), kk) for (u, v), kk in key_cells)
    lv.doors = {mp(u, v): kk for (u, v), kk in doors.items()}
    lv.blocks = tuple(sorted(mp(u, v) for u, v in blocks))
    lv.targets = frozenset(mp(u, v) for u, v in targets)
    lv.tele = {mp(*a): mp(*b) for a, b in tele.items()}
    lv.switches = {mp(u, v): g for (u, v), g in switches.items()}
    lv.swalls = {mp(u, v): gi for (u, v), gi in swalls.items()}
    lv.paint = {mp(u, v): c for (u, v), c in items.get("paint", [])}
    lv.enemies = tuple(tuple(mp(u, v) for u, v in tr) for tr in enemies)
    if lv.grav is not None and not lv.blocks:
        lv.grav = None
    return lv
