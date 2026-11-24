"""Juge épinglé du sommeil. NE JAMAIS MODIFIER après son premier commit (SHA-256 dans empreintes.json).

Un gabarit de règle (prédicat sur primitives de forme -> distribution des 5 types de règle) n'est gardé que s'il
améliore la log-vraisemblance de prédiction sur des étiquettes de jeux TENUS DE CÔTÉ (validation croisée
« un jeu laissé de côté »), par rapport au prior de base seul.
"""
import math

PRIMITIVES = ("aire_grande", "aire_petite", "une_compo", "multi_compo", "bord", "rectangle", "fond", "carre_petit")
LISSAGE = 1.0
SUPPORT_MIN = 3
SEUIL_GAIN = 0.01          # nats par étiquette, en moyenne sur les plis
SEUIL_PLIS = 0.6           # part des plis (avec support) où le gabarit ne dégrade pas


def _lisse(comptes):
    s = sum(comptes) + LISSAGE * 5
    return [(c + LISSAGE) / s for c in comptes]


def _base(labels):
    c = [0] * 5
    for l in labels:
        c[l["type"]] += 1
    return _lisse(c)


def _match(pred, cars):
    return all(cars.get(f) == v for f, v in pred.items())


def _loi(base, gabarits, cars):
    if not gabarits:
        return base
    lb = [math.log(x) for x in base]
    acc = [0.0] * 5
    k = 0
    for pred, probs in gabarits:
        if _match(pred, cars):
            for t in range(5):
                acc[t] += math.log(probs[t]) - lb[t]
            k += 1
    if k == 0:
        return base
    lp = [lb[t] + acc[t] / k for t in range(5)]
    m = max(lp)
    e = [math.exp(x - m) for x in lp]
    s = sum(e)
    return [x / s for x in e]


def ajuster(labels, preds):
    """Gabarits (pred, probs) ajustés sur `labels`, ceux dont le support >= SUPPORT_MIN."""
    out = []
    for pred in preds:
        c = [0] * 5
        for l in labels:
            if _match(pred, l["cars"]):
                c[l["type"]] += 1
        if sum(c) >= SUPPORT_MIN:
            out.append((pred, _lisse(c)))
    return out


def evaluer(labels_par_jeu, preds):
    """Validation croisée un-jeu-dehors : gain de log-vraisemblance (nats/étiquette) de base+preds vs base."""
    jeux = sorted(labels_par_jeu)
    gains, poids, positifs, valides = [], [], 0, 0
    for j in jeux:
        test = labels_par_jeu[j]
        train = [l for k in jeux if k != j for l in labels_par_jeu[k]]
        if not test or not train:
            continue
        base = _base(train)
        gab = ajuster(train, preds)
        touche = [l for l in test if any(_match(p, l["cars"]) for p, _ in gab)]
        if not touche:
            continue
        g = sum(math.log(_loi(base, gab, l["cars"])[l["type"]]) - math.log(base[l["type"]]) for l in touche)
        gains.append(g)
        poids.append(len(touche))
        valides += 1
        positifs += g >= 0
    if not poids:
        return {"gain_moyen": 0.0, "plis_ok": 0.0, "n_plis": 0}
    return {"gain_moyen": sum(gains) / sum(poids), "plis_ok": positifs / valides, "n_plis": valides}


def accepte(res):
    return res["n_plis"] >= 2 and res["gain_moyen"] > SEUIL_GAIN and res["plis_ok"] >= SEUIL_PLIS
