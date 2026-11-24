"""Planification exacte dans un programme : BFS / Dijkstra sur les états (M, pos, avc), macro-actions de panneau."""
from __future__ import annotations

import heapq
import itertools
import time
from collections import deque


def chercher(prog, geo, s0, but, actions=None, cout=None, cap=60000, interdit=None, duree=0.6):
    """Plus court chemin (en coût) de s0 à un état satisfaisant `but(s2, flag, tin, act)` ; None si introuvable.
    cout(tin) -> coût du pas (défaut 1) ; interdit : types traités comme solides pour ce plan.
    Renvoie (liste d'actions, s_final, flag, tin)."""
    if actions is None:
        mv = [(n, None) for n in prog.dirmap]
        if prog.select:
            def acts_f(s):
                return mv + [("ACTION6", p) for p, t in enumerate(s[0]) if t < -1]
        else:
            acts_f = lambda s: mv
    else:
        acts_f = lambda s: actions
    limite = time.process_time() + duree
    sol0 = None
    if interdit:
        sol0 = prog.solid
        prog.solid = prog.solid | set(interdit)
    try:
        if cout is None:
            vu = {s0: None}
            q = deque([s0])
            n = 0
            while q:
                s = q.popleft()
                for act in acts_f(s):
                    r = prog.pas(geo, s, act)
                    if r is None:
                        continue
                    s2, fl, tin = r
                    if fl == 2:
                        continue
                    if but(s2, fl, tin, act):
                        ch = [act]
                        x = s
                        while vu[x] is not None:
                            x, a = vu[x]
                            ch.append(a)
                        ch.reverse()
                        return ch, s2, fl, tin
                    if s2 not in vu:
                        vu[s2] = (s, act)
                        q.append(s2)
                n += 1
                if n > cap or (n & 255 == 0 and time.process_time() > limite):
                    return None
            return None
        cpt = itertools.count()
        g = {s0: 0}
        par = {s0: None}
        tas = [(0, next(cpt), s0, None)]
        n = 0
        while tas:
            c, _, s, fin = heapq.heappop(tas)
            if fin is not None:
                act, s2, fl, tin = fin
                ch = [act]
                x = s
                while par[x] is not None:
                    x, a = par[x]
                    ch.append(a)
                ch.reverse()
                return ch, s2, fl, tin
            if c > g.get(s, 1 << 30):
                continue
            n += 1
            if n > cap or (n & 255 == 0 and time.process_time() > limite):
                return None
            for act in acts_f(s):
                r = prog.pas(geo, s, act)
                if r is None:
                    continue
                s2, fl, tin = r
                if fl == 2:
                    continue
                c2 = c + cout(tin)
                if but(s2, fl, tin, act):
                    heapq.heappush(tas, (c2, next(cpt), s, (act, s2, fl, tin)))
                    continue
                if c2 < g.get(s2, 1 << 30):
                    g[s2] = c2
                    par[s2] = (s, act)
                    heapq.heappush(tas, (c2, next(cpt), s2, None))
        return None
    finally:
        if sol0 is not None:
            prog.solid = sol0


def _cibles(prog, M0, atomes):
    """Pour chaque case de panneau, l'ensemble des types visés par les atomes de panneau ; None si impossible."""
    cl = prog.types.couleurs
    cib = {}
    for at in atomes:
        if at[0] == "all":
            Z, S = at[1], at[2]
            for p, t in enumerate(M0):
                if t in Z:
                    cib[p] = {S} if p not in cib else cib[p] & {S}
        elif at[0] == "pmatch":
            P, R = at[1], at[2]
            lp = [p for p, t in enumerate(M0) if t in P]
            lr = [p for p, t in enumerate(M0) if t in R]
            if not lp or len(lp) != len(lr):
                return None
            for a, b in zip(lp, lr):
                cs = set(cl(M0[b]))
                vis = {t for t in prog.types_panneau if set(cl(t)) == cs}
                cib[a] = vis if a not in cib else cib[a] & vis
    return cib


def clics_pour(prog, geo, s0, atomes, cap=300000):
    """Suite minimale de clics qui rend vrais tous les atomes de panneau (cycles indépendants : calcul direct ;
    bascules voisines : BFS sur bits ; sinon BFS générique)."""
    atomes = [a for a in atomes if a[0] in ("all", "pmatch")]
    M0 = s0[0]
    if not atomes or prog.conds(atomes, M0, geo):
        return []
    types_p = set()
    for t, ck in prog.clk.items():
        types_p.add(t)
        if ck[0] == "flip":
            types_p.update((ck[2], ck[3]))
        else:
            types_p.update(ck[2].values())
    prog.types_panneau = types_p
    cib = _cibles(prog, M0, atomes)
    if cib is None:
        return None
    cases = sorted(p for p, t in enumerate(M0) if t in prog.clk and p not in geo.ignore)
    cks = [prog.clk[M0[p]] for p in cases]
    if cases and all(c[0] == "next" and c[1] == ((0, 0),) for c in cks):
        out = []
        for p in cases:
            if p not in cib:
                continue
            t = M0[p]
            nm = {**prog.clk[t][2], **(geo.nm or {})}
            k = 0
            while t not in cib[p]:
                t = nm.get(t)
                k += 1
                if t is None or k > 12:
                    return None
            out.extend([("ACTION6", p)] * k)
        return out
    flips = cases and all(c[0] == "flip" for c in cks)
    if flips and len(cases) <= 20:
        ck = cks[0]
        A, B = ck[2], ck[3]
        idx = {p: i for i, p in enumerate(cases)}
        masks = []
        for p in cases:
            m = 0
            r, c = divmod(p, geo.C)
            for dr, dc in prog.clk[M0[p]][1]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < geo.R and 0 <= cc < geo.C and rr * geo.C + cc in idx:
                    m |= 1 << idx[rr * geo.C + cc]
            masks.append(m)
        init = sum(1 << i for i, p in enumerate(cases) if M0[p] == B)
        want = 0
        for i, p in enumerate(cases):
            ts = cib.get(p)
            if ts is None:
                want |= (init >> i & 1) << i
            elif B in ts and A not in ts:
                want |= 1 << i
            elif A in ts and B not in ts:
                pass
            else:
                want |= (init >> i & 1) << i
        par = {init: None}
        q = deque([init])
        while q and want not in par:
            s = q.popleft()
            for i, m in enumerate(masks):
                s2 = s ^ m
                if s2 not in par:
                    par[s2] = (s, i)
                    q.append(s2)
        if want not in par:
            return None
        out = []
        x = want
        while par[x] is not None:
            x, i = par[x]
            out.append(("ACTION6", cases[i]))
        out.reverse()
        return out
    acts = [("ACTION6", p) for p in cases]
    vu = {s0: None}
    q = deque([s0])
    n = 0
    while q:
        s = q.popleft()
        for act in acts:
            r = prog.pas(geo, s, act)
            if r is None:
                continue
            s2 = r[0]
            if s2 in vu:
                continue
            vu[s2] = (s, act)
            if prog.conds(atomes, s2[0], geo):
                ch = [act]
                x = s
                while vu[x] is not None:
                    x, a = vu[x]
                    ch.append(a)
                ch.reverse()
                return ch
            q.append(s2)
        n += 1
        if n > cap:
            return None
    return None
