"""Répétition d'un clic efficace : si un clic vient de produire une image jamais vue, on reclique au même
endroit (cycles de basculement / rotation), jusqu'au retour à une image déjà vue."""
NOM = "repetition_clic"
BUDGET = 6
COOLDOWN = 2


def activation(h):
    d = h.der
    return d is not None and d[0] == "C" and h.dchg > 0 and h.nouveau


def agir(h):
    return h.der


def terminer(h):
    d = h.der
    return d is None or d[0] != "C" or h.dchg == 0 or (not h.nouveau)
