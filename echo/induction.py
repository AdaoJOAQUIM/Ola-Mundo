#!/usr/bin/env python3
"""Induction pour Écho : des règles que personne n'a écrites, à partir d'exemples.

Bibliothèque standard uniquement. Aucun réseau de neurones, aucun poids appris,
aucune descente de gradient : on COMPTE des exemples et des contre-exemples.

Principe
--------
Le `Raisonneur` (raisonneur.py) est une logique déductive : tout ce qu'il affirme
est prouvé, et la preuve est revérifiée. L'induction est d'une autre nature : à
partir d'individus connus (« Médor est un chien », « Médor est un aboyeur »...)
on PROPOSE des généralisations (« tout chien est un aboyeur »). Elles peuvent
être fausses (cygne noir). Pour cette raison :

  * les hypothèses restent dans cet objet (`Inducteur.hyps`) et ne sont JAMAIS
    insérées dans la base du raisonneur comme faits ou règles certains ;
  * une réponse qui s'appuie sur une hypothèse est toujours étiquetée
    « Probablement oui/non (confiance X %) » avec les hypothèses utilisées ;
  * une réponse certaine vient d'abord du raisonneur (preuve vérifiée).

Preuves et mesures d'incertitude
--------------------------------
Pour deux classes (A, B), après fermeture déductive de la base :
  s = individus connus de A dont on sait qu'ils sont B,
  c = individus connus de A dont on sait qu'ils ne sont PAS B (contre-exemples),
  u = individus connus de A pour lesquels B est inconnu.
« Tout A est B » est proposée si s >= s_min et c == 0.
« Aucun A n'est B » est proposée symétriquement (preuves négatives : s' = c,
contre-exemples : c' = s).

Confiance : règle de succession de Laplace, (s + 1) / (s + 2). C'est la moyenne
a posteriori d'une loi Beta(1, 1) (prior uniforme) mise à jour par s succès et
c = 0 échec : Beta(s + 1, c + 1), de moyenne (s + 1) / (s + c + 2). On fournit
aussi la borne inférieure à 5 % de cette loi Beta (`resume_beta`). ATTENTION :
c'est la probabilité qu'un NOUVEL individu de A soit B sous l'hypothèse que les
individus sont échangeables ; la confiance n'est pas la probabilité que la règle
soit universellement vraie, et elle n'est pas garantie calibrée (voir
resultats_induction.md).

Chaînage : une réponse peut enchaîner des hypothèses et des règles certaines.
La confiance du chemin est le PRODUIT des confiances des hypothèses utilisées
(indépendance supposée, simplification assumée) ; on garde le meilleur chemin.

Révision des croyances
----------------------
Les connaissances données ne font que croître (le raisonneur refuse toute phrase
contradictoire) : donc c ne peut qu'augmenter. Dès qu'un contre-exemple arrive,
`ajouter` recalcule les hypothèses : celles qui sont réfutées sont retirées
immédiatement et le rapport les liste (avec le contre-exemple et ses phrases).
Une hypothèse réfutée ne revient jamais. Une hypothèse peut aussi disparaître
parce qu'elle devient CERTAINE (règle ensuite donnée ou déductible) : le rapport
la distingue (« promue »).

Boucle de point fixe et terminaison
-----------------------------------
`boucle()` itère : déduction (fermeture D de la base) -> induction H_i -> faits
provisoires F_i = F_{i-1} U Cons(H_i, D U F_{i-1}) (une étape de chaînage par
itération) -> ré-induction... jusqu'à ce que (H, F) ne changent plus.

Anti-auto-renforcement : par défaut H_i est induite de D SEULEMENT. Les faits
provisoires ne comptent ni comme exemples ni comme contre-exemples : sinon une
hypothèse produirait des « exemples » qui la confirmeraient (cercle vicieux, la
confiance monterait sans aucune nouvelle observation). L'option
`soutien_provisoire=True` active ce comportement fautif, uniquement pour le
mesurer dans le banc d'essai.

Théorème (terminaison). Soit I l'ensemble fini des individus et K celui des
classes de la base (finis : la boucle n'ajoute aucune phrase). Les faits
provisoires sont des éléments de I x {+, -} x K, soit au plus 2|I||K| éléments.
(1) F_i contient F_{i-1} par construction (suite croissante) et reste dans un
ensemble fini : elle ne peut croître strictement qu'au plus 2|I||K| fois, donc il
existe i0 avec F_i = F_{i0} pour tout i >= i0.
(2) H_i est une fonction de D (mode par défaut) ou de (D, F_{i-1}) (mode
naïf). Dès que F_{i-1} = F_{i-2}, on a H_i = H_{i-1}. Donc pour i >= i0 + 2,
(H_i, F_i) = (H_{i-1}, F_{i-1}) : le test d'arrêt est satisfait.
Donc au plus 2|I||K| + 2 itérations. En mode par défaut H est constante : on a
H_1 = H_2 = ... et il y a au moins 2 itérations (la seconde constate le point
fixe), au plus la longueur de la plus longue chaîne d'inférence + 2.
Un plafond `iter_max` est appliqué en plus (filet de sécurité) ; si jamais il
était atteint, le rapport le dit (`plafond_atteint`) au lieu de le cacher.
"""

import argparse
import heapq
import math
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import raisonneur as R

S_MIN_DEFAUT = 3
ITER_MAX_DEFAUT = 50


# ======================================================================
# 1. Incertitude : Laplace et Beta(1, 1)
# ======================================================================

def laplace(s, c=0):
    """Règle de succession de Laplace avec s succès, c échecs : (s+1)/(s+c+2)."""
    return (s + 1) / (s + c + 2)


def _cdf_beta(x, a, b):
    """Fonction de répartition de Beta(a, b) (a, b >= 1) par Simpson."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    n = 400
    h = x / n

    def f(t):
        if t <= 0.0 or t >= 1.0:
            return 0.0 if (a > 1 and t <= 0.0) or (b > 1 and t >= 1.0) else 1.0
        return math.exp((a - 1) * math.log(t) + (b - 1) * math.log1p(-t) - lb)

    tot = f(0) + f(x)
    for i in range(1, n):
        tot += (4 if i % 2 else 2) * f(i * h)
    return tot * h / 3


def quantile_beta(q, a, b):
    """Quantile d'ordre q de Beta(a, b) par dichotomie."""
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _cdf_beta(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def resume_beta(s, c=0):
    """Résumé de la loi a posteriori Beta(s+1, c+1) (prior uniforme)."""
    a, b = s + 1, c + 1
    if c == 0:
        borne = 0.05 ** (1.0 / (s + 1))       # forme close : x^(s+1) = 0,05
    else:
        borne = quantile_beta(0.05, a, b)
    return {"alpha": a, "beta": b, "moyenne": a / (a + b),
            "borne_inf_5pct": borne}


# ======================================================================
# 2. Objets de résultat
# ======================================================================

@dataclass
class Hypothese:
    genre: str            # "tout" : « tout A est B » ; "aucun" : « aucun A n'est B »
    a: str
    b: str
    s: int                # exemples qui la soutiennent
    c: int                # contre-exemples (0 pour une hypothèse active)
    u: int                # individus de A pour lesquels B est inconnu
    exemples: list = field(default_factory=list)   # [(individu, frozenset(phrases))]

    @property
    def cle(self):
        return (self.genre, self.a, self.b)

    @property
    def confiance(self):
        return laplace(self.s, self.c)

    @property
    def phrases(self):
        out = set()
        for _, ph in self.exemples:
            out |= ph
        return sorted(out)

    def beta(self):
        return resume_beta(self.s, self.c)


@dataclass
class Resultat:
    """Réponse de l'inducteur. `certain` n'est vrai QUE pour une preuve vérifiée."""
    statut: str                  # certain | probable | inconnu | incompris
    valeur: str                  # oui | non | inconnu | incompris
    texte: str
    confiance: float = None      # None si certain (preuve) ou inconnu
    hypotheses: list = field(default_factory=list)
    reponse_certaine: object = None
    details: list = field(default_factory=list)

    @property
    def certain(self):
        return self.statut == "certain"

    def __str__(self):
        return self.texte


@dataclass
class Rapport:
    """Effet d'une phrase ajoutée sur les hypothèses."""
    insertion: object
    nouvelles: list = field(default_factory=list)
    retractees: list = field(default_factory=list)   # [(Hypothese, contre-exemples)]
    promues: list = field(default_factory=list)      # devenues certaines
    txt: object = None                               # fonction d'affichage d'une hypothèse

    def __str__(self):
        return self.message

    @property
    def message(self):
        lignes = [str(self.insertion)]
        for h, ce in self.retractees:
            cites = "; ".join(f"{nom} (phrases n°{', '.join(map(str, ph))})"
                              for nom, ph in ce)
            lignes.append(f"  Hypothèse retirée (réfutée) : {self._txt(h)}"
                          f" -- contre-exemple : {cites}")
        for h in self.promues:
            lignes.append(f"  Hypothèse devenue certaine (retirée de la liste) : {self._txt(h)}")
        for h in self.nouvelles:
            lignes.append(f"  Nouvelle hypothèse : {self._txt(h)} "
                          f"(confiance {pct(h.confiance)})")
        return "\n".join(lignes)

    def _txt(self, h):
        return self.txt(h) if self.txt else f"{h.genre} {h.a} {h.b}"


@dataclass
class RapportBoucle:
    iterations: int
    converge: bool
    plafond_atteint: bool
    nb_hypotheses: int
    nb_faits_provisoires: int
    nb_conflits: int
    historique: list = field(default_factory=list)   # [(i, |H|, |F|)]
    hypotheses: dict = field(default_factory=dict)
    faits_provisoires: set = field(default_factory=set)


def pct(x):
    return f"{int(round(100 * x))} %"


# ======================================================================
# 3. L'inducteur
# ======================================================================

class _Fermeture:
    """Fermeture déductive de la base : classes connues de chaque individu."""

    def __init__(self, base):
        self.rev = defaultdict(list)
        self.classes = set()
        for a, lst in base.regles.items():
            self.classes.add(a)
            for b, n in lst:
                self.rev[b].append((a, n))
                self.classes.add(b)
        for a, b, _ in base.disj:
            self.classes.update((a, b))
        self.individus = base.individus()
        self.pos, self.neg = {}, {}
        for x in self.individus:
            self.pos[x], self.neg[x] = self._fermer(base, x)
            self.classes.update(self.pos[x])
            self.classes.update(self.neg[x])
        # accessibilité certaine entre classes (transitivité des règles)
        self.atteint = {}
        for a in self.classes:
            vu, file = set(), deque([a])
            while file:
                c = file.popleft()
                for b, _ in base.regles.get(c, ()):
                    if b not in vu:
                        vu.add(b)
                        file.append(b)
            self.atteint[a] = vu
        self.disj = [(a, b) for a, b, _ in base.disj]

    def _fermer(self, base, x):
        pos, file = {}, deque()
        for c, n in base.est_pos.get(x, ()):
            if c not in pos:
                pos[c] = frozenset([n])
                file.append(c)
        while file:
            c = file.popleft()
            for b, n in base.regles.get(c, ()):
                if b not in pos:
                    pos[b] = pos[c] | {n}
                    file.append(b)
        neg, file = {}, deque()

        def poser(c, ph):
            if c not in neg:
                neg[c] = ph
                file.append(c)

        for c, n in base.est_neg.get(x, ()):
            poser(c, frozenset([n]))
        for a, b, n in base.disj:
            for p, q in ((a, b), (b, a)):
                if p in pos:
                    poser(q, pos[p] | {n})
        while file:
            d = file.popleft()
            for c, n in self.rev.get(d, ()):
                poser(c, neg[d] | {n})
        return pos, neg

    def tout_certain(self, a, b):
        return a == b or b in self.atteint.get(a, ())

    def aucun_certain(self, a, b):
        ta = self.atteint.get(a, set()) | {a}
        tb = self.atteint.get(b, set()) | {b}
        return any((p in ta and q in tb) or (q in ta and p in tb)
                   for p, q in self.disj)


class Inducteur:
    """Couche d'induction au-dessus d'un `Raisonneur` (qu'elle ne modifie pas)."""

    def __init__(self, raisonneur=None, s_min=S_MIN_DEFAUT,
                 iter_max=ITER_MAX_DEFAUT, soutien_provisoire=False,
                 couverture_min=0.0):
        self.rais = raisonneur if raisonneur is not None else R.Raisonneur()
        self.s_min = s_min
        self.iter_max = iter_max
        self.soutien_provisoire = soutien_provisoire
        # Garde optionnelle (0 = désactivée, comportement de la spécification) :
        # exiger que s / |A| >= couverture_min, car l'absence d'information n'est
        # pas un contre-exemple (« tout animal est un chat » avec seulement des chats vus).
        self.couverture_min = couverture_min
        self._ferm, self._ferm_n = None, -1
        self._graphe_cache, self._graphe_cle = None, None
        self.hyps = {}                 # (genre, a, b) -> Hypothese  (JAMAIS des faits)
        self.refutees = {}             # (genre, a, b) -> (Hypothese, contre-exemples)
        self.hyps = self.induire()

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------
    @property
    def base(self):
        return self.rais.base

    def texte_hypothese(self, h):
        b = self.base
        if h.genre == "tout":
            return f"tout {b.cls.get(h.a, h.a)} est {b.attribut(h.b)}"
        return f"aucun {b.cls.get(h.a, h.a)} n'est {b.attribut(h.b)}"

    # ------------------------------------------------------------------
    # Déduction : fermeture (mise en cache selon le nombre de phrases)
    # ------------------------------------------------------------------
    def fermeture(self):
        n = len(self.base.phrases)
        if self._ferm is None or self._ferm_n != n:
            self._ferm, self._ferm_n = _Fermeture(self.base), n
        return self._ferm

    # ------------------------------------------------------------------
    # Induction
    # ------------------------------------------------------------------
    def induire(self, extra_pos=None, extra_neg=None):
        """Calcule l'ensemble des hypothèses actives à partir des faits donnés
        (fermés par déduction). `extra_*` : faits provisoires, utilisés SEULEMENT
        en mode `soutien_provisoire` (auto-renforcement, mesuré dans le banc)."""
        f = self.fermeture()
        S, C = Counter(), Counter()
        individus = list(f.individus)
        pos_x, neg_x = {}, {}
        for x in individus:
            pos_x[x] = set(f.pos[x])
            neg_x[x] = set(f.neg[x])
        for (x, c) in (extra_pos or ()):
            pos_x.setdefault(x, set()).add(c)
        for (x, c) in (extra_neg or ()):
            neg_x.setdefault(x, set()).add(c)
        n_a = Counter()
        for x in pos_x:
            P, N = pos_x[x], neg_x.get(x, set())
            for a in P:
                n_a[a] += 1
                for b in P:
                    if b != a:
                        S[(a, b)] += 1
                for b in N:
                    if b != a:
                        C[(a, b)] += 1
        hyps = {}
        for (a, b), s in S.items():
            if (s >= self.s_min and C[(a, b)] == 0 and s >= self.couverture_min * n_a[a]
                    and not f.tout_certain(a, b)):
                hyps[("tout", a, b)] = self._hyp("tout", a, b, s, 0, n_a[a], f,
                                                 pos_x, neg_x)
        for (a, b), s in C.items():
            if (s >= self.s_min and S[(a, b)] == 0 and s >= self.couverture_min * n_a[a]
                    and not f.aucun_certain(a, b)):
                hyps[("aucun", a, b)] = self._hyp("aucun", a, b, s, 0, n_a[a], f,
                                                  pos_x, neg_x)
        return hyps

    def _hyp(self, genre, a, b, s, c, n_a, f, pos_x, neg_x):
        ex = []
        for x in pos_x:
            if a not in pos_x[x]:
                continue
            if genre == "tout" and b in pos_x[x]:
                ph = f.pos.get(x, {}).get(a, frozenset()) | f.pos.get(x, {}).get(b, frozenset())
                ex.append((x, ph))
            elif genre == "aucun" and b in neg_x.get(x, ()):
                ph = f.pos.get(x, {}).get(a, frozenset()) | f.neg.get(x, {}).get(b, frozenset())
                ex.append((x, ph))
        return Hypothese(genre, a, b, s, c, n_a - s - c, ex)

    def _contre_exemples(self, genre, a, b):
        f = self.fermeture()
        out = []
        for x in f.individus:
            if a not in f.pos[x]:
                continue
            if genre == "tout" and b in f.neg[x]:
                out.append((self.base.nom(x), sorted(f.pos[x][a] | f.neg[x][b])))
            elif genre == "aucun" and b in f.pos[x]:
                out.append((self.base.nom(x), sorted(f.pos[x][a] | f.pos[x][b])))
        return out

    # ------------------------------------------------------------------
    # Ajout de phrase + révision des croyances
    # ------------------------------------------------------------------
    def rafraichir(self, insertion=None):
        """Recalcule les hypothèses ; retire immédiatement les hypothèses réfutées."""
        f = self.fermeture()
        nouv = self.induire()
        rap = Rapport(insertion, txt=self.texte_hypothese)
        for cle, h in self.hyps.items():
            if cle in nouv:
                continue
            ce = self._contre_exemples(*cle)
            certain = (f.tout_certain(h.a, h.b) if h.genre == "tout"
                       else f.aucun_certain(h.a, h.b))
            if ce:
                self.refutees[cle] = (h, ce)
                rap.retractees.append((h, ce))
            elif certain:
                rap.promues.append(h)
        for cle, h in nouv.items():
            if cle not in self.hyps:
                rap.nouvelles.append(h)
        self.hyps = nouv
        return rap

    def ajouter(self, texte, rafraichir=True):
        """Ajoute une phrase au raisonneur puis révise les hypothèses."""
        ins = self.rais.ajouter(texte)
        if ins.statut == "ajoutee" and rafraichir:
            return self.rafraichir(ins)
        return Rapport(ins, txt=self.texte_hypothese)

    def traiter(self, texte):
        a = R.analyser(texte)
        if a.genre == "question":
            return self.repondre(texte)
        return self.ajouter(texte)

    # ------------------------------------------------------------------
    # Graphe d'inférence : règles certaines (poids 1) + hypothèses (poids = confiance)
    # ------------------------------------------------------------------
    def _graphe(self):
        cle = (len(self.base.phrases), tuple(sorted(self.hyps)))
        if self._graphe_cle == cle:
            return self._graphe_cache
        g = defaultdict(list)
        for a, lst in self.base.regles.items():
            for b, n in lst:
                g[("P", a)].append((("P", b), 1.0, None, n))
                g[("N", b)].append((("N", a), 1.0, None, n))
        for a, b, n in self.base.disj:
            g[("P", a)].append((("N", b), 1.0, None, n))
            g[("P", b)].append((("N", a), 1.0, None, n))
        for h in self.hyps.values():
            w = h.confiance
            if h.genre == "tout":
                g[("P", h.a)].append((("P", h.b), w, h, None))
                g[("N", h.b)].append((("N", h.a), w, h, None))
            else:
                g[("P", h.a)].append((("N", h.b), w, h, None))
                g[("P", h.b)].append((("N", h.a), w, h, None))
        self._graphe_cache, self._graphe_cle = g, cle
        return g

    def _meilleurs_chemins(self, departs):
        """Dijkstra sur le produit des confiances (max). `departs` : états certains."""
        g = self._graphe()
        best, parent, tas, cpt = {}, {}, [], 0
        for e in departs:
            best[e] = 1.0
            parent[e] = None
            heapq.heappush(tas, (-1.0, cpt, e))
            cpt += 1
        while tas:
            negw, _, e = heapq.heappop(tas)
            w = -negw
            if w < best.get(e, 0) - 1e-15:
                continue
            for e2, p, h, n in g.get(e, ()):
                w2 = w * p
                if w2 > best.get(e2, 0.0) + 1e-15:
                    best[e2] = w2
                    parent[e2] = (e, h, n)
                    heapq.heappush(tas, (-w2, cpt, e2))
                    cpt += 1
        return best, parent

    @staticmethod
    def _hyps_du_chemin(parent, e):
        hs, pas = [], []
        while parent.get(e) is not None:
            e0, h, n = parent[e]
            if h is not None and all(h.cle != k.cle for k in hs):
                hs.append(h)
            pas.append((e0, e, h, n))
            e = e0
        hs.reverse()
        pas.reverse()
        return hs, pas

    # ------------------------------------------------------------------
    # Questions
    # ------------------------------------------------------------------
    def repondre(self, texte):
        """1) raisonneur seul (preuve vérifiée) ; 2) preuve + hypothèses, étiqueté
        « Probablement » ; 3) « Je ne sais pas »."""
        rep = self.rais.repondre(texte)
        if rep.verifie and rep.valeur in ("oui", "non"):
            return Resultat("certain", rep.valeur, rep.reponse, None, [], rep)
        a = R.analyser(texte)
        if a.genre != "question" or a.forme[0] != "q_est":
            return Resultat("incompris" if rep.valeur == "incompris" else "inconnu",
                            rep.valeur if rep.valeur in ("incompris", "inconnu") else "inconnu",
                            rep.reponse, None, [], rep)
        _, genre, s, c, pos = a.forme
        f = self.fermeture()
        if genre == "ind":
            if s not in f.pos:
                return Resultat("inconnu", "inconnu", rep.reponse, None, [], rep)
            departs = [("P", k) for k in f.pos[s]] + [("N", k) for k in f.neg[s]]
        else:   # question sur une règle : un individu « type » de la classe s
            departs = [("P", s)]
        best, parent = self._meilleurs_chemins(departs)
        wp, wn = best.get(("P", c), 0.0), best.get(("N", c), 0.0)
        # un état certain (poids 1 sans hypothèse) aurait été prouvé par le raisonneur
        hp, pp = self._hyps_du_chemin(parent, ("P", c)) if wp else ([], [])
        hn, pn = self._hyps_du_chemin(parent, ("N", c)) if wn else ([], [])
        ok_p = wp > 0 and hp
        ok_n = wn > 0 and hn
        if ok_p and ok_n:
            txt = ("Je ne sais pas (les hypothèses se contredisent : "
                   + " ; ".join(self.texte_hypothese(h) for h in hp + hn) + ")")
            return Resultat("inconnu", "inconnu", txt, None, hp + hn, rep)
        if not (ok_p or ok_n):
            return Resultat("inconnu", "inconnu", rep.reponse, None, [], rep)
        if ok_p:
            hs, w, pas, vrai = hp, wp, pp, True
        else:
            hs, w, pas, vrai = hn, wn, pn, False
        valeur = "oui" if vrai == bool(pos) else "non"
        morceaux = []
        for h in hs:
            n_ex = len(h.exemples)
            ph = ", ".join(str(n) for n in h.phrases)
            morceaux.append(f"{self.texte_hypothese(h)}, {n_ex} exemple"
                            f"{'s' if n_ex > 1 else ''} (phrases n°{ph})")
        mot = "hypothèse" if len(hs) == 1 else "hypothèses"
        txt = (f"Probablement {valeur} (confiance {pct(w)}), {mot} : "
               + " ; ".join(morceaux))
        det = [f"chemin : {e0[0]}({e0[1]}) -> {e1[0]}({e1[1]})"
               + (f" par l'hypothèse « {self.texte_hypothese(h)} »" if h
                  else f" par la règle certaine de la phrase n°{n}")
               for e0, e1, h, n in pas]
        return Resultat("probable", valeur, txt, w, hs, rep, det)

    # ------------------------------------------------------------------
    # Boucle de point fixe (voir la preuve de terminaison dans le docstring du module)
    # ------------------------------------------------------------------
    def _consequences(self, hyps, f, Fpos, Fneg):
        """Une étape de chaînage : règles certaines + hypothèses, appliquées aux
        faits donnés et provisoires. Retourne les NOUVEAUX faits provisoires."""
        nouv = set()
        regles = [(a, b) for a, lst in self.base.regles.items() for b, _ in lst]
        aucun = list(f.disj)
        for h in hyps.values():
            (regles if h.genre == "tout" else aucun).append((h.a, h.b))
        tous = set(f.individus) | {x for x, _ in Fpos} | {x for x, _ in Fneg}
        for x in tous:
            P = set(f.pos.get(x, ())) | {c for y, c in Fpos if y == x}
            N = set(f.neg.get(x, ())) | {c for y, c in Fneg if y == x}
            for a, b in regles:
                if a in P and b not in P:
                    nouv.add((x, "+", b))
                if b in N and a not in N:
                    nouv.add((x, "-", a))
            for a, b in aucun:
                for p, q in ((a, b), (b, a)):
                    if p in P and q not in N:
                        nouv.add((x, "-", q))
        return nouv

    def boucle(self):
        f = self.fermeture()
        F = set()                # faits provisoires (x, signe, classe)
        H_prec, hist = None, []
        for i in range(1, self.iter_max + 1):
            if self.soutien_provisoire:
                H = self.induire({(x, c) for x, s_, c in F if s_ == "+"},
                                 {(x, c) for x, s_, c in F if s_ == "-"})
            else:
                H = self.induire()
            Fpos = {(x, c) for x, s_, c in F if s_ == "+"}
            Fneg = {(x, c) for x, s_, c in F if s_ == "-"}
            F_new = F | self._consequences(H, f, Fpos, Fneg)
            hist.append((i, len(H), len(F_new)))
            stable = (H_prec is not None and set(H) == set(H_prec) and F_new == F)
            F, H_prec = F_new, H
            if stable:
                return self._rapport_boucle(i, True, hist, H, F, f)
        return self._rapport_boucle(self.iter_max, False, hist, H_prec, F, f)

    @staticmethod
    def _rapport_boucle(i, conv, hist, H, F, f):
        pos = {(x, c) for x, s, c in F if s == "+"}
        neg = {(x, c) for x, s, c in F if s == "-"}
        conflits = len(pos & neg)
        for x, c in pos:
            if c in f.neg.get(x, ()):
                conflits += 1
        for x, c in neg:
            if c in f.pos.get(x, ()):
                conflits += 1
        return RapportBoucle(i, conv, not conv, len(H), len(F), conflits, hist,
                             dict(H), set(F))


# ======================================================================
# 4. Démonstration
# ======================================================================

def _demo():
    def titre(t):
        print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

    ind = Inducteur()

    def dire(phrase):
        print(f"> {phrase}")
        r = ind.traiter(phrase)
        print("  " + str(r).replace("\n", "\n  "))
        return r

    titre("1. Émergence : une règle que personne n'a écrite")
    for p in ["Médor est un chien.", "Médor est un aboyeur.",
              "Rex est un chien.", "Rex est un aboyeur.",
              "Fido est un chien.", "Fido est un aboyeur.",
              "Pluto est un chien."]:
        dire(p)
    print("\nBase du raisonneur seul (hypothèses NON insérées) :")
    print("  phrases connues :", len(ind.base.phrases),
          "; règles certaines :", sum(len(v) for v in ind.base.regles.values()))
    print("  raisonneur seul ->", ind.rais.repondre("Pluto est-il un aboyeur ?"))
    r = dire("Pluto est-il un aboyeur ?")
    print(f"  certain ? {r.certain}  (statut = {r.statut})")

    titre("2. Révision : un contre-exemple retire l'hypothèse")
    dire("Rintintin est un chien.")
    dire("Rintintin n'est pas un aboyeur.")
    dire("Pluto est-il un aboyeur ?")

    titre("3. Négatif : « aucun » induit de preuves négatives")
    for p in ["Mistigri est un chat.", "Mistigri n'est pas un aboyeur.",
              "Felix est un chat.", "Felix n'est pas un aboyeur.",
              "Tom est un chat.", "Tom n'est pas un aboyeur.",
              "Minou est un chat."]:
        dire(p)
    dire("Minou est-il un aboyeur ?")

    titre("4. Une règle certaine reste prioritaire (réponse prouvée)")
    dire("Tout chat est un animal.")
    dire("Tout chien est un animal.")
    dire("Minou est-il un animal ?")

    titre("5. Boucle de point fixe (déduire -> induire -> appliquer)")
    rb = ind.boucle()
    print(f"  itérations : {rb.iterations} ; convergé : {rb.converge} ; "
          f"plafond atteint : {rb.plafond_atteint}")
    print(f"  hypothèses : {rb.nb_hypotheses} ; faits provisoires : "
          f"{rb.nb_faits_provisoires} ; conflits : {rb.nb_conflits}")
    for i, nh, nf in rb.historique:
        print(f"    itération {i} : |H| = {nh}, |F| = {nf}")
    print("  (conflits : « tout animal est un chat » et « tout animal est un chien » "
          "sont toutes deux proposées,\n   car l'absence d'information n'est pas un "
          "contre-exemple ; les réponses concernées s'abstiennent.)")
    for x, s, c in sorted(rb.faits_provisoires)[:6]:
        verbe = "est" if s == "+" else "n'est pas"
        print(f"    provisoire : {ind.base.nom(x)} "
              f"{verbe} {ind.base.attribut(c)}")


def _interactif(ind):
    print("Écho / induction. Phrases ou questions ; « boucle » ; Ctrl-D pour sortir.")
    for ligne in sys.stdin:
        ligne = ligne.strip()
        if not ligne:
            continue
        if ligne == "boucle":
            rb = ind.boucle()
            print(f"itérations {rb.iterations}, convergé {rb.converge}, "
                  f"|H| {rb.nb_hypotheses}, |F| {rb.nb_faits_provisoires}")
            continue
        print(ind.traiter(ligne))


def main(argv=None):
    p = argparse.ArgumentParser(description="Induction (hypothèses, confiance, révision)")
    p.add_argument("--demo", action="store_true", help="démonstration")
    p.add_argument("--s-min", type=int, default=S_MIN_DEFAUT,
                   help="nombre minimal d'exemples pour proposer une hypothèse")
    a = p.parse_args(argv)
    if a.demo:
        _demo()
        return 0
    _interactif(Inducteur(s_min=a.s_min))
    return 0


if __name__ == "__main__":
    sys.exit(main())
