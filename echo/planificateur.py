#!/usr/bin/env python3
"""Écho — planificateur d'objectifs (A*) au-dessus du raisonneur.

Python standard uniquement. Aucun réseau de neurones, aucun poids appris.

Idée : on donne un objectif en français contrôlé (« Objectif : la pomme doit
être au jardin. ») et le monde courant d'un ``Raisonneur`` (son historique
d'événements). Le planificateur cherche une suite d'actions de COÛT MINIMAL qui
atteint l'objectif, puis VÉRIFIE le plan en le rejouant dans un raisonneur
tout neuf : chaque phrase du plan doit être acceptée, puis la question
correspondant à l'objectif doit recevoir la bonne réponse, avec une preuve
acceptée par le vérificateur indépendant du raisonneur. Seuls les plans
vérifiés sont renvoyés.

Actions (chacune produit une phrase de français contrôlé) :
    aller   : « X va dans la cuisine. »
    prendre : « X prend la pomme. »   (X est là où est l'objet, personne ne le porte)
    poser   : « X pose la pomme. »    (X porte l'objet)
    donner  : « X donne la pomme à Y. » (X porte l'objet, X et Y au même endroit)

Coûts : par action (défaut 1) et multiplicateur par personne (défaut 1), ce
qui rend l'optimalité non triviale (faire marcher la personne « rapide »).

Modèle d'état et simplification (justifiée)
-------------------------------------------
L'état est : lieu de chaque personne (ou inconnu), et pour l'objet du but,
qui le porte ou où il est posé. Les AUTRES objets ne sont pas représentés :
ils n'aident jamais à atteindre le but (porter ou non un autre objet ne
change aucune précondition des actions), donc les ignorer ne change ni
l'existence ni le coût optimal d'un plan.

Heuristique A* (admissible)
---------------------------
Soit ca, cp, cq, cd les coûts minimaux (tous acteurs confondus) d'aller,
prendre, poser, donner. h(état) est une borne inférieure du coût restant :
  * « X dans L » : 0 si déjà vrai ; sinon coût exact d'un « aller » de X
    (au moins une action aller de X est nécessaire, et une suffit).
  * « l'objet O est dans L » : 0 si vrai. Si O est posé ailleurs : il faut au
    moins une action « prendre » (cp) et, car le lieu d'un objet ne change
    que lorsqu'un porteur se déplace, au moins un « aller » (ca) : h = cp+ca.
    Si O est porté par quelqu'un qui n'est pas en L : h = ca.
  * « X a O » : 0 si vrai. Si O est posé : au moins un « prendre » : h = cp.
    Si O est porté par Y != X : le transfert passe par « donner » (cd) ou par
    « poser » puis « prendre » (cq+cp) : h = min(cd, cq+cp).
Chaque cas ne compte que des actions qui DOIVENT apparaître dans tout plan
(et chacune au moins à son coût minimal) : h ne surestime jamais, donc
l'heuristique est admissible. A* avec ré-ouverture des nœuds (on réinsère un
état si on le retrouve avec un g plus petit) renvoie alors un plan de coût
optimal, sans qu'il faille la cohérence de h.

Trois issues honnêtes : plan trouvé ; « Objectif impossible » (monde connu
épuisé ou entité inconnue, avec la raison) ; « budget de nœuds épuisé »
(on ne sait pas : différent de impossible).
"""

import argparse
import heapq
import itertools
import re
import sys
from dataclasses import dataclass, field

import raisonneur as rz

COUTS_DEFAUT = {"aller": 1.0, "prendre": 1.0, "poser": 1.0, "donner": 1.0}
BUDGET_DEFAUT = 20000


# ======================================================================
# Objectifs
# ======================================================================

@dataclass
class Objectif:
    genre: str          # "objet_lieu" | "personne_lieu" | "possede"
    sujet: str          # clé de l'objet (objet_lieu, possede : l'objet) ou de la personne
    cible: str          # clé du lieu, ou clé de la personne qui doit avoir l'objet
    texte: str = ""


_RE_OBJECTIF = re.compile(
    r"^objectif\s*:\s*(?P<s>.+?)\s+doit\s+(?P<v>etre|avoir|se trouver)\s+(?P<c>.+)$")
_RE_PREP_LIEU = re.compile(r"^(?:dans|au|aux|a|en|chez|vers|sur)\s+(?P<l>.+)$")


def _cle_nom(segment):
    """Clé d'un nom (article retiré, accents retirés), comme dans raisonneur.py."""
    n = rz.normaliser(segment.strip())
    _, reste, _ = rz._decouper_article(n, n)
    return reste.strip()


def analyser_objectif(texte):
    """Texte -> dict {genre, sujet_brut, cible_brute, verbe} ou None si incompris.

    Les clés ne sont pas encore résolues (on ne sait pas si « la pomme » est un
    objet ou une personne) : voir ``Planificateur._resoudre``.
    """
    n = rz.normaliser(texte).strip()
    n = n.rstrip(" .!;\t\r\n")
    m = _RE_OBJECTIF.match(n)
    if not m:
        return None
    verbe = "etre" if m.group("v") in ("etre", "se trouver") else "avoir"
    return {"sujet": _cle_nom(m.group("s")), "verbe": verbe,
            "cible_brute": m.group("c").strip()}


# ======================================================================
# Monde : état extrait de l'historique d'un Raisonneur
# ======================================================================

@dataclass
class Monde:
    personnes: list = field(default_factory=list)       # clés triées
    lieux: list = field(default_factory=list)           # clés triées
    objets: list = field(default_factory=list)          # clés triées
    lieu_personne: dict = field(default_factory=dict)   # clé -> lieu ou None
    porteur: dict = field(default_factory=dict)         # objet -> personne ou None
    lieu_pose: dict = field(default_factory=dict)       # objet -> lieu (posé) ou None
    surface_lieu: dict = field(default_factory=dict)    # lieu -> "dans la cuisine"
    noms: dict = field(default_factory=dict)            # personne -> "Marie"
    surface_objet: dict = field(default_factory=dict)   # objet -> "la pomme"


def extraire_monde(raisonneur):
    """Rejoue l'historique d'événements (code propre, indépendant du démonstrateur)."""
    b = raisonneur.base
    mo = Monde()
    pers, objs, lieux = set(), set(), set()
    for _, f in b.evenements:
        typ = f[1]
        if typ == "va":
            pers.add(f[2])
            lieux.add(f[3])
            mo.lieu_personne[f[2]] = f[3]
        else:
            pers.add(f[2])
            objs.add(f[3])
            if typ == "donne":
                pers.add(f[4])
            mo.lieu_personne.setdefault(f[2], None)
            if typ == "donne":
                mo.lieu_personne.setdefault(f[4], None)
            # (le lieu courant est celui du moment : on le rejoue ci-dessous)
    # second passage dans l'ordre pour les lieux d'objets
    lieu = {}
    for _, f in b.evenements:
        typ = f[1]
        if typ == "va":
            lieu[f[2]] = f[3]
        elif typ == "prend":
            mo.porteur[f[3]] = f[2]
            mo.lieu_pose.pop(f[3], None)
        elif typ == "donne":
            mo.porteur[f[3]] = f[4]
            mo.lieu_pose.pop(f[3], None)
        elif typ == "pose":
            mo.porteur[f[3]] = None
            mo.lieu_pose[f[3]] = lieu.get(f[2])
    # lieux déclarés : « Le grenier est un lieu. » -> règle ("regle", "grenier", "lieu")
    for cls, sup in ((c, s) for c, lst in b.regles.items() for s, _ in lst):
        if sup == "lieu":
            lieux.add(cls)
            if cls not in b.lieux:
                det = b.det.get(cls, "")
                art = "la" if det == "une" else "le" if det == "un" else None
                nom = b.cls.get(cls, cls)
                mo.surface_lieu[cls] = f"dans {art} {nom}" if art else f"à {nom}"
    mo.personnes = sorted(pers)
    mo.objets = sorted(objs)
    mo.lieux = sorted(lieux)
    for k in mo.personnes:
        mo.lieu_personne[k] = lieu.get(k)
        mo.noms[k] = b.noms.get(k, k.capitalize())
    for k in mo.lieux:
        mo.surface_lieu.setdefault(k, b.lieux.get(k, f"à {k}"))
    for k in mo.objets:
        mo.surface_objet[k] = b.objets.get(k, k)
    return mo


# ======================================================================
# Résultat
# ======================================================================

@dataclass
class ResultatPlan:
    statut: str                  # "plan" | "impossible" | "budget" | "non_verifie" | "incompris"
    message: str
    actions: list = field(default_factory=list)    # tuples (verbe, ...clés)
    phrases: list = field(default_factory=list)
    cout: float = 0.0
    noeuds: int = 0
    raison: str = ""
    verifie: bool = False
    verification: str = ""
    objectif: Objectif = None

    def __str__(self):
        return self.message


# ======================================================================
# Le planificateur
# ======================================================================

class Planificateur:
    """A* sur l'état du monde d'un Raisonneur."""

    def __init__(self, couts=None, multiplicateurs=None, budget=BUDGET_DEFAUT):
        self.couts = dict(COUTS_DEFAUT)
        if couts:
            self.couts.update(couts)
        for k, v in self.couts.items():
            if not v > 0:
                raise ValueError(f"coût de « {k} » doit être > 0 (A* exige des coûts positifs)")
        self.mult = {rz.normaliser(k): float(v) for k, v in (multiplicateurs or {}).items()}
        for k, v in self.mult.items():
            if not v > 0:
                raise ValueError(f"multiplicateur de {k} doit être > 0")
        self.budget = budget

    # -- coûts ----------------------------------------------------------
    def cout_action(self, verbe, personne):
        return self.couts[verbe] * self.mult.get(personne, 1.0)

    # -- résolution de l'objectif --------------------------------------
    def _resoudre(self, brut, mo):
        """Retourne (Objectif, None) ou (None, raison_impossible) ou (None, None) si incompris."""
        s, verbe, cb = brut["sujet"], brut["verbe"], brut["cible_brute"]
        if verbe == "etre":
            m = _RE_PREP_LIEU.match(rz.normaliser(cb))
            if not m:
                return None, None
            lieu = _cle_nom(m.group("l"))
            if not rz._cle_ok(lieu) or not rz._cle_ok(s):
                return None, None
            if s in mo.personnes:
                genre = "personne_lieu"
            elif s in mo.objets:
                genre = "objet_lieu"
            else:
                return None, f"entité inconnue : « {s} » n'apparaît dans aucun événement"
            if lieu not in mo.lieux:
                return None, f"lieu inconnu : « {lieu} » n'est mentionné nulle part"
            return Objectif(genre, s, lieu), None
        # avoir : sujet = personne, cible = objet
        obj = _cle_nom(cb)
        if not rz._cle_ok(obj) or not rz._cle_ok(s):
            return None, None
        if s not in mo.personnes:
            return None, f"personne inconnue : « {s} » n'apparaît dans aucun événement"
        if obj not in mo.objets:
            return None, f"objet inconnu : « {obj} » n'apparaît dans aucun événement"
        return Objectif("possede", obj, s), None

    # -- point d'entrée ---------------------------------------------------
    def planifier(self, raisonneur, texte_objectif, verifier_plan=True):
        brut = analyser_objectif(texte_objectif)
        if brut is None:
            return ResultatPlan("incompris", "Je ne comprends pas cet objectif "
                                "(forme attendue : « Objectif : X doit être dans L. » "
                                "ou « Objectif : X doit avoir O. »).",
                                raison="objectif incompris")
        mo = extraire_monde(raisonneur)
        obj, impossible = self._resoudre(brut, mo)
        if obj is None and impossible is None:
            return ResultatPlan("incompris", "Je ne comprends pas cet objectif.",
                                raison="objectif incompris")
        if obj is None:
            return ResultatPlan("impossible", f"Objectif impossible : {impossible}.",
                                raison=impossible)
        obj.texte = texte_objectif.strip()
        res = self._astar(mo, obj)
        res.objectif = obj
        if res.statut != "plan":
            return res
        if verifier_plan:
            self._verifier(raisonneur, mo, obj, res)
        return res

    # -- A* ---------------------------------------------------------------
    def _astar(self, mo, obj):
        P, L = mo.personnes, mo.lieux
        ip = {p: i for i, p in enumerate(P)}
        il = {l: i for i, l in enumerate(L)}
        nP = len(P)
        o = obj.sujet if obj.genre in ("objet_lieu", "possede") else None

        # état initial
        ploc = tuple(il[mo.lieu_personne[p]] if mo.lieu_personne.get(p) in il else -1
                     for p in P)
        holder = 0
        if o is not None:
            if mo.porteur.get(o) is not None:
                holder = ip[mo.porteur[o]]
            else:
                lp = mo.lieu_pose.get(o)
                if lp is None:
                    return ResultatPlan(
                        "impossible",
                        f"Objectif impossible : le lieu de « {o} » est inconnu "
                        f"(posé par quelqu'un dont le lieu n'a jamais été donné).",
                        raison="lieu de l'objet inconnu")
                holder = -1 - il[lp]
        init = (ploc, holder)

        # buts
        if obj.genre == "objet_lieu":
            cible = il[obj.cible]

            def but(s):
                ploc_, h = s
                return (ploc_[h] if h >= 0 else -1 - h) == cible
        elif obj.genre == "personne_lieu":
            cible = il[obj.cible]
            ix = ip[obj.sujet]

            def but(s):
                return s[0][ix] == cible
        else:
            ix = ip[obj.cible]

            def but(s):
                return s[1] == ix

        # coûts minimaux pour l'heuristique
        def cmin(verbe):
            return min(self.cout_action(verbe, p) for p in P)
        ca, cp, cq, cd = cmin("aller"), cmin("prendre"), cmin("poser"), cmin("donner")

        def h(s):
            ploc_, hold = s
            if obj.genre == "personne_lieu":
                if ploc_[ix] == cible:
                    return 0.0
                return self.cout_action("aller", obj.sujet)
            if obj.genre == "objet_lieu":
                if hold < 0:
                    return 0.0 if -1 - hold == cible else cp + ca
                return 0.0 if ploc_[hold] == cible else ca
            if hold == ix:
                return 0.0
            if hold < 0:
                return cp
            return min(cd, cq + cp)

        def voisins(s):
            ploc_, hold = s
            for i in range(nP):
                p = P[i]
                # aller
                for k in range(len(L)):
                    if ploc_[i] != k:
                        nl = ploc_[:i] + (k,) + ploc_[i + 1:]
                        yield (("aller", p, L[k]), self.cout_action("aller", p), (nl, hold))
                if o is None:
                    continue
                if hold < 0:
                    # prendre : X là où est l'objet, objet posé
                    if ploc_[i] == -1 - hold and ploc_[i] >= 0:
                        yield (("prendre", p, o), self.cout_action("prendre", p), (ploc_, i))
                elif hold == i:
                    # poser : lieu de X connu
                    if ploc_[i] >= 0:
                        yield (("poser", p, o), self.cout_action("poser", p),
                               (ploc_, -1 - ploc_[i]))
                    # donner à un autre, même lieu connu
                    for j in range(nP):
                        if j != i and ploc_[j] == ploc_[i] and ploc_[i] >= 0:
                            yield (("donner", p, o, P[j]), self.cout_action("donner", p),
                                   (ploc_, j))

        compteur = itertools.count()
        h0 = h(init)
        tas = [(h0, h0, next(compteur), 0.0, init)]
        meilleur = {init: 0.0}
        parent = {init: None}
        fermes_expansions = 0
        EPS = 1e-9
        while tas:
            f, _, _, g, s = heapq.heappop(tas)
            if g > meilleur[s] + EPS:
                continue        # entrée périmée
            if but(s):
                actions = []
                while parent[s] is not None:
                    s, a = parent[s][0], parent[s][1]
                    actions.append(a)
                actions.reverse()
                cout = sum(self.cout_action(a[0], a[1]) for a in actions)
                return ResultatPlan("plan", "", actions=actions, cout=cout,
                                    noeuds=fermes_expansions)
            if fermes_expansions >= self.budget:
                return ResultatPlan(
                    "budget",
                    f"Budget de {self.budget} nœuds épuisé : je ne sais pas si l'objectif "
                    f"est atteignable (ce n'est PAS une preuve d'impossibilité).",
                    noeuds=fermes_expansions, raison="budget de nœuds épuisé")
            fermes_expansions += 1
            for a, c, t in voisins(s):
                ng = g + c
                if ng + EPS < meilleur.get(t, float("inf")):
                    meilleur[t] = ng
                    parent[t] = (s, a)
                    ht = h(t)
                    heapq.heappush(tas, (ng + ht, ht, next(compteur), ng, t))
        return ResultatPlan(
            "impossible",
            "Objectif impossible : tout l'espace d'états du monde connu a été exploré "
            "sans atteindre le but.",
            noeuds=fermes_expansions, raison="espace d'états épuisé")

    # -- phrases ----------------------------------------------------------
    def phrase(self, action, mo):
        v = action[0]
        nom = mo.noms.get(action[1], action[1].capitalize())
        if v == "aller":
            return f"{nom} va {mo.surface_lieu[action[2]]}."
        obj = mo.surface_objet.get(action[2], action[2])
        if v == "prendre":
            return f"{nom} prend {obj}."
        if v == "poser":
            return f"{nom} pose {obj}."
        return f"{nom} donne {obj} à {mo.noms.get(action[3], action[3].capitalize())}."

    # -- vérification -------------------------------------------------------
    def _question(self, obj, mo):
        if obj.genre == "personne_lieu":
            return f"Où est {mo.noms.get(obj.sujet, obj.sujet.capitalize())} ?"
        o = mo.surface_objet.get(obj.sujet, obj.sujet)
        if not re.match(r"^(?:le|la|les|un|une|des|du|l')", rz.normaliser(o)):
            o = "le " + o
        return f"Où est {o} ?" if obj.genre == "objet_lieu" else f"Qui a {o} ?"

    def _verifier(self, raisonneur, mo, obj, res):
        """Rejoue historique + plan dans un raisonneur NEUF, puis pose la question du but."""
        res.phrases = [self.phrase(a, mo) for a in res.actions]
        frais = rz.Raisonneur()
        for t in raisonneur.base.textes():
            ins = frais.ajouter(t)
            if ins.statut not in ("ajoutee", "doublon"):
                return self._echec(res, f"rejeu de l'historique impossible : {ins.message}")
        for ph in res.phrases:
            ins = frais.ajouter(ph)
            if ins.statut != "ajoutee":
                return self._echec(res, f"phrase du plan non acceptée « {ph} » : {ins.message}")
        q = self._question(obj, mo)
        rep = frais.repondre(q)
        attendu = obj.cible if obj.genre != "possede" else obj.cible
        if rep.valeur != attendu:
            return self._echec(res, f"{q} -> « {rep.reponse} » (attendu : {attendu})")
        ok, raison = rz.verifier(rep.preuve, frais.base)
        if not (rep.verifie and ok):
            return self._echec(res, f"preuve rejetée par le vérificateur : {raison}")
        res.verifie = True
        res.verification = f"{q} -> {rep.reponse} (preuve vérifiée : {raison})"
        res.message = self._message_plan(res)

    def _echec(self, res, raison):
        res.statut = "non_verifie"
        res.verifie = False
        res.verification = raison
        res.message = f"Plan trouvé mais NON vérifié, donc non retourné : {raison}"
        res.phrases = []
        res.actions = []
        return res

    def _message_plan(self, res):
        if not res.phrases:
            return (f"Objectif déjà atteint (plan vide, coût 0). "
                    f"Vérifié : {res.verification}")
        lignes = [f"Plan ({len(res.phrases)} action(s), coût {res.cout:g}, "
                  f"{res.noeuds} nœud(s) développé(s)) :"]
        lignes += [f"  {i}. {p}" for i, p in enumerate(res.phrases, 1)]
        lignes.append(f"Vérifié : {res.verification}")
        return "\n".join(lignes)


# ======================================================================
# Démo et ligne de commande
# ======================================================================

def _demo():
    def titre(t):
        print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)

    r = rz.Raisonneur()
    for ph in ["Marie va dans la cuisine.", "Jean va au jardin.", "Paul va dans le salon.",
               "Marie prend la pomme.", "Marie pose la pomme.", "Jean prend le livre.",
               "Le grenier est un lieu."]:
        r.ajouter(ph)
    # « Jean prend le livre » avant d'aller quelque part : son lieu est donc connu (jardin)
    titre("Monde de départ")
    print("Marie dans la cuisine (pomme posée là), Jean au jardin (porte le livre), "
          "Paul dans le salon.")
    pl = Planificateur()
    for obj in ["Objectif : la pomme doit être au jardin.",
                "Objectif : Marie doit avoir le livre.",
                "Objectif : Jean doit être dans la cuisine.",
                "Objectif : la pomme doit être dans la cuisine.",
                "Objectif : Paul doit être dans la cave.",
                "Objectif : la pomme doit être dans le grenier."]:
        titre(obj)
        print(pl.planifier(r, obj))
    titre("Coûts par personne : Marie lente (x5), Jean rapide, Paul rapide")
    pl2 = Planificateur(multiplicateurs={"Marie": 5})
    obj = "Objectif : la pomme doit être au jardin."
    print(pl2.planifier(r, obj))
    titre("Budget de nœuds trop petit (1)")
    print(Planificateur(budget=1).planifier(r, "Objectif : la pomme doit être au jardin."))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Écho : planificateur d'objectifs (A*).")
    ap.add_argument("--demo", action="store_true", help="affiche une démonstration")
    ap.add_argument("--histoire", help="fichier de phrases décrivant le monde (une par ligne)")
    ap.add_argument("--objectif", help="objectif, ex. « Objectif : la pomme doit être au jardin. »")
    ap.add_argument("--budget", type=int, default=BUDGET_DEFAUT)
    a = ap.parse_args(argv)
    if a.demo:
        _demo()
        return 0
    if a.histoire and a.objectif:
        r = rz.Raisonneur()
        with open(a.histoire, encoding="utf-8") as f:
            r.charger([l.strip() for l in f if l.strip()])
        print(Planificateur(budget=a.budget).planifier(r, a.objectif))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
