"""Théories de règles et postérieur bayésien (sans LLM).

Langage de règles (par couleur d'objet, au contact de l'avatar) :
    pass     : traversable, rien ne se passe
    block    : bloque le déplacement
    lethal   : contact mortel (game over)
    collect  : l'objet disparaît / change au contact
    win      : le contact termine le niveau (objectif « atteindre X »)
Une théorie du monde = (déplacement par action) x (type de règle par couleur) ; le postérieur est factorisé
par couleur, avec mise à jour conjointe exacte sur les couleurs présentes dans la zone de contact
(explication par élimination), puis projection champ-moyen.

Vraisemblance : « précision de prédiction avec bruit » — une issue observée qui contredit la prédiction
déterministe d'une affectation n'a pas probabilité 0 mais EPS.
Prior : fréquences de types de règles, conditionnées par des primitives de forme (gabarits), apprises sur DEV.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

TYPES = ("pass", "block", "lethal", "collect", "win")
PASS, BLOCK, LETHAL, COLLECT, WIN = range(5)
EPS = 0.03
ISSUES = ("moved", "blocked", "died", "win")


def lik_issue(issue: str, types) -> float:
    a = set(types)
    if issue == "blocked":
        ok = BLOCK in a
    elif issue == "died":
        ok = LETHAL in a
    elif issue == "win":
        ok = WIN in a
    else:  # moved sans événement
        ok = not (a & {BLOCK, LETHAL, WIN})
    return 1.0 - EPS if ok else EPS


def lik_reveal(changed: bool, t: int) -> float:
    """Après le départ de l'avatar : la couleur sous lui a-t-elle changé ?"""
    if t == WIN:
        return 0.5
    if t == COLLECT:
        return 1.0 - EPS if changed else EPS
    return EPS if changed else 1.0 - EPS


def entropie(p) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


class Prior:
    """Prior hiérarchique : base (fréquences de types de règles) + gabarits conditionnés par des primitives."""

    def __init__(self, base=None, gabarits=None):
        self.base = np.array(base if base is not None else [0.2] * 5, dtype=float)
        self.base /= self.base.sum()
        self.gabarits = gabarits or []        # liste de (predicat: dict primitive->bool, probs[5])

    def loi(self, cars: dict | None) -> np.ndarray:
        if not cars or not self.gabarits:
            return self.base.copy()
        lb = np.log(self.base)
        acc = np.zeros(5)
        k = 0
        for pred, probs in self.gabarits:
            if all(cars.get(f) == v for f, v in pred.items()):
                acc += np.log(np.asarray(probs, dtype=float)) - lb
                k += 1
        if k == 0:
            return self.base.copy()
        lp = lb + acc / k
        p = np.exp(lp - lp.max())
        return p / p.sum()

    @staticmethod
    def depuis_json(d):
        if not d:
            return Prior()
        return Prior(d.get("base"), [(g["pred"], g["probs"]) for g in d.get("gabarits", [])])

    def vers_json(self):
        return {"base": self.base.tolist(),
                "gabarits": [{"pred": p, "probs": list(map(float, q))} for p, q in self.gabarits]}


class Croyances:
    """Postérieur sur le type de règle de chaque couleur."""

    def __init__(self, prior: Prior | None = None):
        self.prior = prior or Prior()
        self.p = {}              # couleur -> vecteur de probas
        self.cars = {}           # couleur -> primitives
        self.n_reveal = {}       # couleur -> nombre d'observations de dévoilement
        self.n_contact = {}

    def definir_cars(self, cars: dict):
        for c, v in cars.items():
            self.cars[c] = v

    def get(self, c) -> np.ndarray:
        if c not in self.p:
            self.p[c] = self.prior.loi(self.cars.get(c))
        return self.p[c]

    # -- énumération conjointe sur les couleurs incertaines ------------------------------
    def _joint(self, colors, poids_fn, max_inc=5):
        colors = list(colors)
        fixes, inc = {}, []
        for c in colors:
            v = self.get(c)
            if v.max() > 0.995:
                fixes[c] = int(v.argmax())
            else:
                inc.append(c)
        if len(inc) > max_inc:
            inc.sort(key=lambda c: -entropie(self.get(c)))
            for c in inc[max_inc:]:
                fixes[c] = int(self.get(c).argmax())
            inc = inc[:max_inc]
        marg = {c: np.zeros(5) for c in inc}
        Z = 0.0
        for combo in itertools.product(range(5), repeat=len(inc)):
            aff = dict(fixes)
            pr = 1.0
            for c, t in zip(inc, combo):
                aff[c] = t
                pr *= self.get(c)[t]
            w = pr * poids_fn(aff)
            Z += w
            for c, t in zip(inc, combo):
                marg[c][t] += w
        return inc, marg, Z

    def maj_contact(self, colors, issue: str):
        colors = list(colors)
        if not colors:
            return
        inc, marg, Z = self._joint(colors, lambda aff: lik_issue(issue, aff.values()))
        if Z <= 0:
            return
        for c in inc:
            v = marg[c] / Z
            self.p[c] = np.maximum(v, 1e-6) / np.maximum(v, 1e-6).sum()
        for c in colors:
            self.n_contact[c] = self.n_contact.get(c, 0) + 1

    def maj_reveal(self, c, changed: bool):
        v = self.get(c) * np.array([lik_reveal(changed, t) for t in range(5)])
        v = np.maximum(v, 1e-9)
        self.p[c] = v / v.sum()
        self.n_reveal[c] = self.n_reveal.get(c, 0) + 1

    # -- prédictions ----------------------------------------------------------------
    def proba_issues(self, colors) -> dict:
        colors = list(colors)
        res = {}
        for issue in ISSUES:
            inc, _, Z = self._joint(colors, lambda aff, i=issue: lik_issue(i, aff.values()))
            res[issue] = Z
        s = sum(res.values())
        return {k: v / s for k, v in res.items()}

    def gain_info(self, colors) -> float:
        """Gain d'information espéré d'une expérience « entrer en contact avec ces couleurs » :
        entropie de l'issue prédite + incertitude résiduelle sur le type des couleurs non encore résolues."""
        colors = list(colors)
        if not colors:
            return 0.0
        pi = self.proba_issues(colors)
        g = entropie(list(pi.values()))
        for c in colors:
            if self.n_reveal.get(c, 0) < 2:
                g += 0.3 * entropie(self.get(c))
        return g

    def p_type(self, colors, t: int) -> float:
        """P(au moins une des couleurs est du type t)."""
        q = 1.0
        for c in colors:
            q *= 1.0 - self.get(c)[t]
        return 1.0 - q

    def map_types(self) -> dict:
        return {c: TYPES[int(v.argmax())] for c, v in self.p.items()}

    def exporter_etiquettes(self, seuil=0.6):
        """(couleur, type, primitives) pour les couleurs dont le type est établi par l'expérience."""
        out = []
        for c, v in self.p.items():
            if self.n_contact.get(c, 0) >= 1 and v.max() >= seuil:
                out.append({"type": int(v.argmax()), "cars": self.cars.get(c, {})})
        return out
