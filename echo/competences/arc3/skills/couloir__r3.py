"""Course en ligne droite : tant qu'un déplacement simple fait bouger l'image et mène à un état nouveau,
on le répète (explore les couloirs sans rejouer la loterie du cerveau à chaque pas)."""
NOM = "couloir"
BUDGET = 8
COOLDOWN = 0


def bouge(h):
    d = h.der
    if d is None or d[0] != "A" or d[1] > 4 or len(h.log) < 4:
        return False
    plancher = min([x[1] for x in h.log])
    return h.dchg > plancher + 2


def activation(h):
    return bouge(h) and h.nouveau and h.nn > 10


def agir(h):
    return h.der


def terminer(h):
    return (not bouge(h)) or (not h.nouveau)
