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


def issue_predite(types) -> str:
    """Issue déterministe prédite par une affectation de types aux couleurs touchées."""
    a = set(types)
    if LETHAL in a:
        return "died"
    if BLOCK in a:
        return "blocked"
    if WIN in a:
        return "win"
    return "moved"


def lik_issue(issue: str, types) -> float:
    """P(issue observée | affectation) : prédiction déterministe + bruit EPS sur chaque autre issue."""
    return 1.0 - 3 * EPS if issue_predite(types) == issue else EPS


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


_LIK_CACHE = {}


def tenseur_lik(k: int, issue: str) -> np.ndarray:
    """Tenseur 5^k : P(issue | types des k couleurs touchées)."""
    key = (k, issue)
    if key not in _LIK_CACHE:
        t = np.zeros((5,) * k)
        for idx in itertools.product(range(5), repeat=k):
            t[idx] = lik_issue(issue, idx)
        _LIK_CACHE[key] = t
    return _LIK_CACHE[key]


class Croyances:
    """Postérieur sur le type de règle de chaque couleur, par propagation de croyance (BP) sur les
    motifs d'observation (couleurs touchées, issue) : inférence indépendante de l'ordre des observations,
    avec explication par élimination (une couleur déjà établie bloquante absorbe l'effet)."""

    MAX_K = 4

    def __init__(self, prior: Prior | None = None):
        self.prior = prior or Prior()
        self.p = {}              # couleur -> vecteur de probas (croyance)
        self.cars = {}           # couleur -> primitives
        self.n_reveal = {}
        self.n_contact = {}
        self.motifs = {}         # (couleurs triées, issue) -> nombre
        self.msg = {}            # (motif, couleur) -> message (5,)
        self.rev = {}            # couleur -> (n_changé, n_inchangé)
        self.par_couleur = {}    # couleur -> set de motifs

    def definir_cars(self, cars: dict):
        for c, v in cars.items():
            if c not in self.cars:
                self.cars[c] = v

    def prior_c(self, c):
        return self.prior.loi(self.cars.get(c))

    def get(self, c) -> np.ndarray:
        if c not in self.p:
            self.p[c] = self.prior_c(c)
        return self.p[c]

    # -- BP ----------------------------------------------------------------------------
    def _creer_motif(self, colors, issue):
        cs = tuple(sorted(set(colors)))
        # écarte les couleurs déjà établies sans effet (passables certaines) : réduit la dimension
        gardees = tuple(c for c in cs if not (self.get(c)[PASS] > 0.995))
        if not gardees:
            return None
        if len(gardees) > self.MAX_K:
            gardees = tuple(sorted(sorted(gardees, key=lambda c: -entropie(self.get(c)))[:self.MAX_K]))
        return (gardees, issue)

    def _recalculer(self, couleurs, sweeps=3):
        dirty = set(couleurs)
        for _ in range(sweeps):
            motifs = set()
            for c in dirty:
                motifs |= self.par_couleur.get(c, set())
            if not motifs:
                break
            nouveau = set()
            for m in motifs:
                cs, issue = m
                k = len(cs)
                L = tenseur_lik(k, issue) ** self.motifs[m]
                cav = []
                for c in cs:
                    q = self.get(c) / np.maximum(self.msg.get((m, c), np.ones(5)), 1e-12)
                    cav.append(q / q.sum())
                for i, c in enumerate(cs):
                    T = L
                    for j in range(k):
                        if j != i:
                            shape = [1] * k
                            shape[j] = 5
                            T = T * cav[j].reshape(shape)
                    ax = tuple(j for j in range(k) if j != i)
                    mi = T.sum(axis=ax) if ax else T
                    mi = np.maximum(mi, 1e-12)
                    mi = mi / mi.sum()
                    old = self.msg.get((m, c), np.ones(5))
                    new_q = self.get(c) / np.maximum(old, 1e-12) * mi
                    new_q = np.maximum(new_q, 1e-9)
                    new_q /= new_q.sum()
                    if np.abs(new_q - self.p[c]).max() > 1e-4:
                        nouveau.add(c)
                    self.p[c] = new_q
                    self.msg[(m, c)] = mi
            dirty = nouveau
            if not dirty:
                break

    def maj_contact(self, colors, issue: str):
        colors = list(colors)
        m = self._creer_motif(colors, issue)
        for c in colors:
            self.n_contact[c] = self.n_contact.get(c, 0) + 1
            self.get(c)
        if m is None:
            return
        self.motifs[m] = self.motifs.get(m, 0) + 1
        for c in m[0]:
            self.par_couleur.setdefault(c, set()).add(m)
        self._recalculer(m[0])

    def maj_reveal(self, c, changed: bool):
        v = self.get(c)
        l = np.array([lik_reveal(changed, t) for t in range(5)])
        # la vraisemblance de dévoilement multiplie directement la croyance (facteur unaire)
        v = np.maximum(v * l, 1e-9)
        self.p[c] = v / v.sum()
        self.n_reveal[c] = self.n_reveal.get(c, 0) + 1

    # -- prédictions ----------------------------------------------------------------
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
        Z = 0.0
        for combo in itertools.product(range(5), repeat=len(inc)):
            aff = dict(fixes)
            pr = 1.0
            for c, t in zip(inc, combo):
                aff[c] = t
                pr *= self.get(c)[t]
            Z += pr * poids_fn(aff)
        return inc, None, Z

    def proba_issues(self, colors) -> dict:
        """Distribution prédictive sur les issues (moved/blocked/died/win), moyennée sur le postérieur."""
        colors = list(colors)
        res = {}
        for issue in ISSUES:
            inc, _, Z = self._joint(colors, lambda aff, i=issue: lik_issue(i, aff.values()))
            res[issue] = Z
        s = sum(res.values())
        return {k: v / s for k, v in res.items()}

    def gain_info(self, colors) -> float:
        """Gain d'information espéré d'une expérience « entrer en contact avec ces couleurs » :
        information mutuelle I(théorie ; issue) = H(issue) - E_théorie[H(issue | théorie)],
        plus l'incertitude résiduelle sur « disparaît/reste » des couleurs jamais dévoilées."""
        colors = list(colors)
        if not colors:
            return 0.0
        pi = self.proba_issues(colors)
        h_cond = entropie([1 - 3 * EPS, EPS, EPS, EPS])      # bruit irréductible d'une affectation donnée
        g = max(0.0, entropie(list(pi.values())) - h_cond)
        for c in colors:
            if self.n_reveal.get(c, 0) < 2:
                v = self.get(c)
                g += 0.3 * entropie([v[COLLECT], 1 - v[COLLECT]]) if v.max() < 0.995 else 0.0
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
        """(type, primitives) pour les couleurs dont le type est établi par l'expérience."""
        out = []
        for c, v in self.p.items():
            if self.n_contact.get(c, 0) >= 1 and v.max() >= seuil:
                out.append({"type": int(v.argmax()), "cars": self.cars.get(c, {})})
        return out
