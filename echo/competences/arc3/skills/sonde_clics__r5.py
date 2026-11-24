"""Recherche de cibles de clic : au début d'un niveau, clique une fois chaque objet de l'image initiale
(objets rares d'abord, puis petits), pour apprendre vite quels objets répondent."""
NOM = "sonde_clics"
BUDGET = 40
COOLDOWN = 3


def liste_cibles(h):
    ob = h.objets(h.f0)
    cnt = {}
    for o in ob:
        cnt[o["c"]] = cnt.get(o["c"], 0) + o["n"]
    ob = [o for o in ob if o["n"] <= 600]
    classe = {}
    ob2 = []
    for o in ob:
        k = (o["c"], o["n"])
        classe[k] = classe.get(k, 0) + 1
        ob2.append((classe[k], cnt[o["c"]], o["n"], o["y"], o["x"]))
    ob = [(t[0] * 1000 + t[1], t[2], t[3], t[4]) for t in sorted(ob2)]
    ob = [{"c": 0, "n": t[1], "y": t[2], "x": t[3]} for t in ob]
    vus = set()
    res = []
    for o in ob:
        k = (o["x"], o["y"])
        if k not in vus and len(res) < 40:
            vus.add(k)
            res.append(k)
    return res


def activation(h):
    if 6 not in h.dispo or h.nn > 150:
        return False
    m = h.mem
    if "cibles" not in m:
        m["cibles"] = liste_cibles(h)
        m["i"] = 0
    return m["i"] < len(m["cibles"])


def agir(h):
    m = h.mem
    k = m["cibles"][m["i"]]
    m["i"] = m["i"] + 1
    return h.C(k[0], k[1])


def terminer(h):
    return h.mem.get("i", 0) >= len(h.mem.get("cibles", []))
