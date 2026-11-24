"""Sélection puis déplacement : quand clics et flèches existent, clique un objet (rares d'abord) puis
pousse chaque flèche trois fois (l'objet sélectionné bouge-t-il ? atteint-il sa place ?)."""
NOM = "clic_puis_fleches"
BUDGET = 110
COOLDOWN = 6


def activation(h):
    m = h.mem
    if 6 not in h.dispo or h.nn < 8:
        return False
    fl = [a for a in (4, 3, 2, 1) if a in h.dispo]
    if len(fl) == 0:
        return False
    if "plan" not in m:
        ob = h.objets(h.f0)
        cnt = {}
        for o in ob:
            cnt[o["c"]] = cnt.get(o["c"], 0) + o["n"]
        t = sorted([(cnt[o["c"]], o["n"], o["y"], o["x"]) for o in ob if 3 <= o["n"] <= 400])
        plan = []
        for q in t[:9]:
            plan.append(h.C(q[3], q[2]))
            for a in fl:
                plan.append(h.A(a))
                plan.append(h.A(a))
                plan.append(h.A(a))
        m["plan"] = plan
        m["k"] = 0
    return m["k"] < len(m["plan"])


def agir(h):
    m = h.mem
    a = m["plan"][m["k"]]
    m["k"] = m["k"] + 1
    return a


def terminer(h):
    return h.mem.get("k", 0) >= len(h.mem.get("plan", []))
