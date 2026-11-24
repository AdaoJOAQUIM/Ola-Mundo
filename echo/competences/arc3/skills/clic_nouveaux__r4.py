"""Clic sur ce qui vient d'apparaître : après une interaction, les objets présents dans l'image courante
mais absents de l'image de départ du niveau (nouvelle position ou couleur) sont cliqués une fois chacun."""
NOM = "clic_nouveaux"
BUDGET = 12
COOLDOWN = 2


def nouveaux(h):
    m = h.mem
    if "dej" not in m:
        m["dej"] = set()
        m["init"] = set([(o["c"], o["y"], o["x"]) for o in h.objets(h.f0)])
    res = []
    for o in h.objets():
        k = (o["c"], o["y"], o["x"])
        if k not in m["init"] and (o["x"], o["y"]) not in m["dej"] and 2 <= o["n"] <= 300:
            res.append((o["n"], o["y"], o["x"]))
    return sorted(res)


def activation(h):
    if 6 not in h.dispo or h.nn < 3:
        return False
    return len(nouveaux(h)) > 0


def agir(h):
    m = h.mem
    r = nouveaux(h)
    q = r[0]
    m["dej"].add((q[2], q[1]))
    return h.C(q[2], q[1])


def terminer(h):
    return len(nouveaux(h)) == 0
