#!/usr/bin/env python3
"""Écho — aiguilleur à coût minimal : « le module le moins cher qui suffit ».

Python standard uniquement. Même idée que l'aiguillage des tâches simples vers
de petits modèles et des tâches difficiles vers de gros modèles, mais ici avec
des modules symboliques dont le coût CPU est MESURÉ (``time.process_time``).

Modules, du moins cher au plus cher :
    0. cache des réponses VÉRIFIÉES précédentes (question normalisée exacte) ;
    1. ``Raisonneur`` : déduction, état du monde, arithmétique, insertion de faits ;
    2. ``Planificateur`` : objectifs « Objectif : ... doit ... » (plans vérifiés) ;
    3. Écho (``echo.ecrire``) : « Écris ... » / « Continue : ... », texte libre
       explicitement étiqueté « texte généré, non vérifié ».

Pour chaque requête on essaie le module applicable le moins cher ; on n'accepte
qu'un résultat VÉRIFIÉ (réponse du raisonneur dont la preuve est acceptée par le
vérificateur, plan vérifié) ou, pour la génération, un texte non vide étiqueté
non vérifié. Sinon on escalade. Une abstention (« Je ne sais pas ») du
raisonneur n'est PAS escaladée vers la génération : inventer du texte à la place
d'une réponse serait justement l'hallucination qu'on veut éviter.

Cache : clé = (question normalisée, version de la base). La version est le nombre
de phrases acceptées : toute nouvelle phrase invalide les réponses sur le monde.
Seules les questions d'arithmétique (indépendantes du monde) survivent aux
insertions. La normalisation du cache est volontairement bon marché
(minuscules, espaces et ponctuation finale) : la normalisation complète du
raisonneur (sans accents) coûte autant qu'une réponse simple.

Budget : chaque requête a un budget CPU. Il est vérifié AVANT chaque module et
à l'intérieur de la recherche du planificateur et de la génération (par blocs
de 40 caractères). Le raisonneur n'est pas interruptible : s'il dépasse le
budget, la réponse est rendue avec ``depassement_cpu=True``. Si le budget est
épuisé avant qu'un module ait pu répondre : « budget dépassé ».

Énergie : ESTIMATION = temps CPU x 10 W (hypothèse : 10 W par cœur occupé),
jamais une mesure.

Stratégies (pour la comparaison, voir banc_aiguilleur.py) :
    "aiguilleur"            : le moins cher d'abord + cache (défaut) ;
    "sans_cache"            : le moins cher d'abord, sans cache ;
    "toujours_tout"         : TOUS les modules sur TOUTES les requêtes (sans
                              cache), on garde le meilleur résultat vérifié ;
    "toujours_tout_syntaxe" : tous les modules dont la forme syntaxique convient.
"""

import argparse
import os
import re
import sys
import time
import zlib
from dataclasses import dataclass, field

import echo
import planificateur as pf
import raisonneur as rz

PUISSANCE_W = 10.0          # hypothèse : 10 W par cœur occupé (ESTIMATION)
BUDGET_CPU_DEFAUT = 5.0     # secondes CPU par requête
TAILLE_BLOC_GEN = 40        # caractères par bloc de génération (granularité du budget)
ETIQUETTE_GEN = "[texte généré, non vérifié]"

NOMS_MODULES = {0: "cache", 1: "raisonneur", 2: "planificateur", 3: "écho (génération)"}

_RE_GEN = re.compile(r"^\s*(?:[ée]cris|ecris|continue)\b\s*:?\s*(?P<r>.*)$", re.IGNORECASE | re.DOTALL)
_RE_GEN_VIDE = re.compile(r"^(?:un|une)\s+(?:texte|phrase|paragraphe|histoire)\b[^:]*:?\s*", re.IGNORECASE)
_RE_OBJECTIF = re.compile(r"^\s*objectif\b", re.IGNORECASE)
_RE_CALC = re.compile(r"^\s*combien\b", re.IGNORECASE)


def cle_cache(texte):
    """Question normalisée de façon bon marché : minuscules, espaces, ponctuation finale."""
    return " ".join(texte.lower().split()).rstrip(" ?.!;")


@dataclass
class Resultat:
    requete: str
    reponse: str
    module: int = None             # 0..3 ou None (aucun module n'a répondu)
    statut: str = "incompris"      # verifie | non_verifie | insertion | abstention |
                                   # impossible | incompris | budget_depasse
    valeur: object = None
    verifie: bool = False
    etiquette: str = ""
    cpu_s: float = 0.0
    energie_j: float = 0.0         # ESTIMATION : cpu_s x 10 W
    depassement_cpu: bool = False
    essais: list = field(default_factory=list)   # [(module, issue)]
    detail: object = None          # Reponse / Insertion / ResultatPlan d'origine

    @property
    def nom_module(self):
        return NOMS_MODULES.get(self.module, "aucun")

    def __str__(self):
        return self.reponse


class Aiguilleur:
    def __init__(self, strategie="aiguilleur", budget_cpu=BUDGET_CPU_DEFAUT,
                 chemin_memoire=None, raisonneur=None, planificateur=None,
                 n_gen=200, n_gen_toujours_tout=40):
        if strategie not in ("aiguilleur", "sans_cache", "toujours_tout",
                             "toujours_tout_syntaxe"):
            raise ValueError(f"stratégie inconnue : {strategie}")
        self.strategie = strategie
        self.budget_cpu = budget_cpu
        self.r = raisonneur or rz.Raisonneur()
        self.plan = planificateur or pf.Planificateur()
        self.chemin_memoire = chemin_memoire or os.path.join(echo.ICI, "memoire.xz")
        self.n_gen = n_gen
        self.n_gen_toujours_tout = n_gen_toujours_tout
        self._T = None             # bibliothèque d'Écho, chargée PARESSEUSEMENT
        self.cache = {}
        self.avec_cache = strategie == "aiguilleur"
        self.compteurs = {"requetes": 0, "cache_hits": 0, "cache_miss": 0}

    # ------------------------------------------------------------------
    def _version(self, texte):
        return None if _RE_CALC.match(texte) else len(self.r.base.phrases)

    # -- module 0 --------------------------------------------------------
    def _m0(self, texte):
        return self.cache.get((cle_cache(texte), self._version(texte)))

    def _stocker(self, texte, res):
        self.cache[(cle_cache(texte), self._version(texte))] = res

    # -- module 1 --------------------------------------------------------
    def _m1(self, texte):
        """Retourne (Resultat partiel ou None, issue)."""
        sortie = self.r.traiter(texte)
        if isinstance(sortie, rz.Reponse):
            v = sortie.valeur
            if sortie.verifie and v not in ("inconnu", "incompris"):
                ok, _ = rz.verifier(sortie.preuve, self.r.base)    # double contrôle
                if ok:
                    return Resultat(texte, sortie.reponse, 1, "verifie", v, True,
                                    "réponse déduite, preuve vérifiée", detail=sortie), "verifie"
            if v == "incompris":
                return None, "incompris"
            return Resultat(texte, sortie.reponse, 1, "abstention", v, False,
                            "abstention honnête du raisonneur", detail=sortie), "abstention"
        # Insertion
        if sortie.statut in ("ajoutee", "doublon"):
            return Resultat(texte, sortie.message, 1, "insertion", sortie.statut, False,
                            "phrase insérée (cohérence contrôlée)", detail=sortie), "insertion"
        if sortie.statut == "refusee":
            preuves_ok = all(c.get("verifie") for c in sortie.contradictions)
            return Resultat(texte, sortie.message, 1, "insertion", "refusee", preuves_ok,
                            "contradiction détectée"
                            + (", preuves vérifiées" if preuves_ok else ""),
                            detail=sortie), "refusee"
        return None, "incompris"

    # -- module 2 --------------------------------------------------------
    def _m2(self, texte, deadline):
        res = self.plan.planifier(self.r, texte, limite_cpu=deadline)
        if res.statut == "plan" and res.verifie:
            return Resultat(texte, res.message, 2, "verifie", res.cout, True,
                            "plan vérifié (rejeu dans un raisonneur neuf + preuve)",
                            detail=res), "verifie"
        if res.statut == "impossible":
            return Resultat(texte, res.message, 2, "impossible", None, False,
                            "conclusion du planificateur (non certifiée par le vérificateur)",
                            detail=res), "impossible"
        if res.statut == "budget":
            if res.raison == "budget CPU dépassé":
                return Resultat(texte, "Budget dépassé (CPU) pendant la planification.", 2,
                                "budget_depasse", None, False, "", detail=res), "budget_depasse"
            return Resultat(texte, res.message, 2, "abstention", None, False,
                            "budget de nœuds épuisé (ni plan ni preuve d'impossibilité)",
                            detail=res), "abstention"
        if res.statut == "non_verifie":
            return None, "plan non vérifié (refusé)"
        return None, "incompris"

    # -- module 3 --------------------------------------------------------
    def _bibliotheque(self):
        if self._T is None:
            self._T = echo.charger(self.chemin_memoire)
        return self._T

    def _m3(self, texte, deadline, n, forcer_amorce=None):
        if time.process_time() > deadline:
            return None, "budget_depasse"
        T = self._bibliotheque()
        if forcer_amorce is None:
            m = _RE_GEN.match(texte)
            amorce = _RE_GEN_VIDE.sub("", m.group("r")).strip()
        else:
            amorce = forcer_amorce
        graine = zlib.crc32(cle_cache(texte).encode("utf-8"))
        sortie, fait = amorce, 0
        while fait < n:
            if time.process_time() > deadline:
                return Resultat(texte, "Budget dépassé (CPU) pendant la génération.", 3,
                                "budget_depasse", None, False, ""), "budget_depasse"
            pas = min(TAILLE_BLOC_GEN, n - fait)
            nouveau = echo.ecrire(T, sortie, n=pas, graine=graine + fait)
            if len(nouveau) <= len(sortie):
                break
            fait += len(nouveau) - len(sortie)
            sortie = nouveau
        if not sortie.strip():
            return None, "génération vide"
        return Resultat(texte, f"{ETIQUETTE_GEN} {sortie}", 3, "non_verifie", sortie, False,
                        ETIQUETTE_GEN), "non_verifie"

    # ------------------------------------------------------------------
    def traiter(self, texte):
        """Traite une requête : retourne un Resultat (module, CPU, énergie estimée)."""
        t0 = time.process_time()
        deadline = t0 + self.budget_cpu
        self.compteurs["requetes"] += 1
        if self.strategie.startswith("toujours_tout"):
            res = self._tout(texte, deadline)
        else:
            res = self._moins_cher(texte, deadline)
        res.cpu_s = time.process_time() - t0
        res.energie_j = res.cpu_s * PUISSANCE_W
        res.depassement_cpu = res.depassement_cpu or res.cpu_s > self.budget_cpu
        return res

    def _moins_cher(self, texte, deadline):
        essais = []

        def fini(res):
            res.essais = essais
            return res

        # module 0
        if self.avec_cache:
            hit = self._m0(texte)
            if hit is not None:
                self.compteurs["cache_hits"] += 1
                essais.append((0, "succès"))
                return fini(Resultat(texte, hit.reponse, 0, hit.statut, hit.valeur,
                                     hit.verifie, "réponse vérifiée en cache (module d'origine : "
                                     f"{hit.nom_module})", detail=hit.detail))
            self.compteurs["cache_miss"] += 1
            essais.append((0, "absent"))
        # module 1
        if time.process_time() > deadline:
            essais.append((1, "budget_depasse"))
            return fini(Resultat(texte, "Budget dépassé (CPU) avant tout traitement.",
                                 None, "budget_depasse"))
        res, issue = self._m1(texte)
        essais.append((1, issue))
        if res is not None:
            if self.avec_cache and res.statut == "verifie":
                self._stocker(texte, res)
            return fini(res)
        # module 2
        if _RE_OBJECTIF.match(texte):
            if time.process_time() > deadline:
                essais.append((2, "budget_depasse"))
                return fini(Resultat(texte, "Budget dépassé (CPU) avant la planification.",
                                     None, "budget_depasse"))
            res, issue = self._m2(texte, deadline)
            essais.append((2, issue))
            if res is not None:
                if self.avec_cache and res.statut == "verifie":
                    self._stocker(texte, res)
                return fini(res)
        # module 3
        if _RE_GEN.match(texte):
            res, issue = self._m3(texte, deadline, self.n_gen)
            essais.append((3, issue))
            if res is not None:
                return fini(res)
            if issue == "budget_depasse":
                return fini(Resultat(texte, "Budget dépassé (CPU) avant la génération.",
                                     None, "budget_depasse"))
        return fini(Resultat(texte, rz.INCOMPRIS + " (aucun module ne sait traiter cette requête).",
                             None, "incompris"))

    # -- « toujours tout » -------------------------------------------------
    def _tout(self, texte, deadline):
        """Lance tous les modules (ou ceux dont la forme convient) et garde le meilleur."""
        syntaxe = self.strategie == "toujours_tout_syntaxe"
        essais, candidats = [], []
        r1, i1 = self._m1(texte)
        essais.append((1, i1))
        if r1 is not None:
            candidats.append(r1)
        if not syntaxe or _RE_OBJECTIF.match(texte):
            if time.process_time() <= deadline:
                r2, i2 = self._m2(texte, deadline)
                essais.append((2, i2))
                if r2 is not None:
                    candidats.append(r2)
        est_gen = bool(_RE_GEN.match(texte))
        if not syntaxe or est_gen:
            if est_gen:
                n, amorce = self.n_gen, None
            else:
                n, amorce = self.n_gen_toujours_tout, texte
            r3, i3 = self._m3(texte, deadline, n, amorce)
            essais.append((3, i3))
            if r3 is not None and est_gen:
                candidats.append(r3)    # un texte non vérifié ne répond qu'aux requêtes « Écris »
        # meilleur : vérifié > impossible > insertion > génération > abstention
        ordre = {"verifie": 0, "impossible": 1, "insertion": 2, "non_verifie": 3,
                 "abstention": 4, "budget_depasse": 5}
        if candidats:
            res = min(candidats, key=lambda c: ordre.get(c.statut, 9))
        else:
            res = Resultat(texte, rz.INCOMPRIS + " (aucun module ne sait traiter cette requête).",
                           None, "incompris")
        res.essais = essais
        return res


# ======================================================================
# Démo et ligne de commande
# ======================================================================

def _afficher(res):
    print(f"  réponse : {res.reponse}")
    print(f"  module  : {res.module if res.module is not None else '-'} ({res.nom_module}) ; "
          f"statut : {res.statut}"
          + ("" if not res.etiquette else f" ; {res.etiquette}"))
    print(f"  CPU : {res.cpu_s * 1e6:.1f} µs ; énergie ESTIMÉE (10 W x CPU) : "
          f"{res.energie_j * 1e6:.2f} µJ ; essais : {res.essais}")


def _demo():
    a = Aiguilleur()
    requetes = [
        "Marie va dans la cuisine.", "Marie prend la pomme.", "Jean va au jardin.",
        "Où est la pomme ?", "Où est la pomme ?",            # 2e fois : cache
        "Combien font 12 fois 7 ?", "Combien font 12 fois 7 ?",
        "Objectif : la pomme doit être au jardin.",
        "Objectif : la pomme doit être dans la cave.",
        "Qui a le stylo ?",                                   # abstention honnête
        "Quelle est la capitale de la France ?",              # hors langage contrôlé
        "Écris : It was a dark and stormy night",
    ]
    for q in requetes:
        print(f"\n> {q}")
        _afficher(a.traiter(q))
    print("\nBudget CPU minuscule (1 µs) :")
    petit = Aiguilleur(budget_cpu=1e-6)
    print("> Où est la pomme ?")
    _afficher(petit.traiter("Où est la pomme ?"))
    print(f"\nCompteurs : {a.compteurs}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Écho : aiguilleur à coût minimal.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--budget", type=float, default=BUDGET_CPU_DEFAUT,
                    help="budget CPU par requête (secondes)")
    a = ap.parse_args(argv)
    if a.demo:
        _demo()
        return 0
    ai = Aiguilleur(budget_cpu=a.budget)
    print("Aiguilleur Écho (ligne vide pour quitter).")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        _afficher(ai.traiter(q))
    return 0


if __name__ == "__main__":
    sys.exit(main())
