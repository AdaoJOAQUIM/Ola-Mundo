"""Paires de clics : parmi les objets qui ont répondu à un clic, essaie toutes les paires ordonnées
après un RESET (RESET, clic a, clic b). Cherche les niveaux qui demandent deux clics dans le bon ordre."""
NOM = "paires_clics"
BUDGET = 100
COOLDOWN = 4


def repondants(h):
    r = []
    for k in h.fx:
        if k[0] == "C" and h.fx[k][1] > 0:
            r.append((k[1], k[2]))
    return sorted(r)[:8]


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
