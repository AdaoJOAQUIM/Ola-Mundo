"""Paires de clics : parmi les objets qui ont répondu à un clic, essaie toutes les paires ordonnées
après un RESET (RESET, clic a, clic b). Cherche les niveaux qui demandent deux clics dans le bon ordre."""
NOM = "paires_clics"
BUDGET = 100
COOLDOWN = 4


def repondants(h):
    gros = {}
    for e in h.log:
        a = e[0]
        if a[0] == "C" and e[1] > gros.get((a[1], a[2]), 0):
            gros[(a[1], a[2])] = e[1]
    r = [(-gros[k], k[0], k[1]) for k in gros if gros[k] > 0]
    r = sorted(r)[:5]
    return [(t[1], t[2]) for t in r]


def activation(h):
    m = h.mem
    if 6 not in h.dispo or h.nn < 12:
        return False
    if "plan" not in m:
        r = repondants(h)
        plan = []
        for a in r:
            for b in r:
                if a != b:
                    plan.append(h.R())
                    plan.append(h.C(a[0], a[1]))
                    plan.append(h.C(b[0], b[1]))
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
