"""Couverture déterministe : exploration systématique par graphe d'états (empreintes d'images hors HUD).
Action : essayer une flèche non encore essayée dans l'état courant ; sinon suivre le plus court chemin connu
vers l'état le plus proche qui a encore une flèche non essayée. Remplace le tirage aléatoire du cerveau
par une couverture sans revisite inutile."""
NOM = "couverture"
BUDGET = 100
COOLDOWN = 3


def flechesdispo(h):
    return [a for a in (1, 2, 3, 4) if a in h.dispo]


def maj_graphe(h):
    m = h.mem
    mk = 0
    if h.masque is not None:
        mk = int(h.masque.sum())
    if m.get("mk") != mk or "g" not in m:
        m["mk"] = mk
        m["g"] = {}
        m["essaye"] = set()
        m["prec"] = None
    cur = h.hach()
    if m.get("n_maj") == h.n:
        return cur
    m["n_maj"] = h.n
    d = h.der
    if m["prec"] is not None and d is not None and d[0] == "A" and d[1] <= 4:
        g = m["g"]
        g.setdefault(m["prec"], [])
        if (d[1], cur) not in g[m["prec"]]:
            g[m["prec"]].append((d[1], cur))
        m["essaye"].add((m["prec"], d[1]))
    m["prec"] = cur
    return cur


def frontieres(h):
    m = h.mem
    fl = flechesdispo(h)
    res = set()
    for k in m["g"]:
        for a in fl:
            if (k, a) not in m["essaye"]:
                res.add(k)
                break
    return res


def activation(h):
    fl = flechesdispo(h)
    maj_graphe(h)
    if len(fl) < 2 or 6 in h.dispo:
        return False
    return h.nn > 3


def agir(h):
    m = h.mem
    cur = maj_graphe(h)
    fl = flechesdispo(h)
    m["g"].setdefault(cur, [])
    for a in fl:
        if (cur, a) not in m["essaye"]:
            return h.A(a)
    buts = frontieres(h)
    ch = h.chemin(m["g"], cur, buts)
    if ch is None or len(ch) == 0:
        return h.A(fl[0])
    return h.A(ch[0])


def terminer(h):
    m = h.mem
    cur = maj_graphe(h)
    m["g"].setdefault(cur, [])
    fl = flechesdispo(h)
    for a in fl:
        if (cur, a) not in m["essaye"]:
            return False
    buts = frontieres(h)
    if len(buts) == 0:
        return True
    ch = h.chemin(m["g"], cur, buts)
    return ch is None
