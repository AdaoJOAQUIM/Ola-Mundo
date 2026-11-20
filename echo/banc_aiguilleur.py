#!/usr/bin/env python3
"""Banc d'essai de l'aiguilleur et du planificateur.

Deux bancs, graine fixe, vérité terrain calculée ICI avec du code indépendant
des modules testés (simulateur du monde, fermeture des règles, entiers Python,
recherche à coût uniforme exhaustive sur l'état COMPLET du monde) :

1. Charge mixte (~400 requêtes) rejouée avec trois stratégies :
   (a) aiguilleur : le moins cher d'abord + cache ;
   (b) « toujours tout » : tous les modules sur toutes les requêtes, sans
       cache, on garde le meilleur résultat vérifié (variante b' : seulement
       les modules dont la forme syntaxique convient) ;
   (c) sans cache : le moins cher d'abord, sans cache.
2. Planificateur : 200 mondes aléatoires, buts aléatoires (dont impossibles),
   comparaison du coût A* avec une recherche à coût uniforme indépendante.

Attention : charges et mondes sont écrits par nous, dans le français contrôlé
que le système comprend. Les résultats montrent la correction et le coût des
modules sur ces charges, pas une intelligence générale. Énergie : ESTIMATION
(temps CPU x 10 W), jamais une mesure.

Usage : python3 banc_aiguilleur.py [--graine 2024] [--n 400] [--repetitions 5]
        [--sans-toujours-tout] [--mondes 200]
Sorties : resultats_objectif_cout.md et resultats_objectif_cout.json
"""

import argparse
import heapq
import itertools
import json
import os
import platform
import random
import re
import statistics
import time
from collections import Counter, defaultdict

import aiguilleur as ag
import planificateur as pf
import raisonneur as rz

ICI = os.path.dirname(os.path.abspath(__file__))
PUISSANCE_W = 10.0

# ----------------------------------------------------------------------
# Vocabulaire du banc
# ----------------------------------------------------------------------
PERSONNES = ["Marie", "Jean", "Paul", "Anne"]
LIEUX = {"cuisine": "dans la cuisine", "jardin": "au jardin", "salon": "dans le salon",
         "cave": "à la cave", "garage": "au garage"}
OBJETS = {"pomme": "la pomme", "livre": "le livre", "cle": "la clé", "ballon": "le ballon",
          "stylo": "le stylo"}
INCONNUS = {"lieu": "garage", "objet": "stylo", "personne": "Pierre"}   # jamais dans les événements


def surf_lieu(k):
    return LIEUX[k]


# ----------------------------------------------------------------------
# Simulateur du monde (indépendant du raisonneur)
# ----------------------------------------------------------------------
class Sim:
    """État du monde selon NOS règles (pas celles du raisonneur) :
    lieu des personnes, qui porte quoi, où est posé chaque objet."""

    def __init__(self, personnes, lieux, objets):
        self.P, self.L, self.O = list(personnes), list(lieux), list(objets)
        self.loc = {p: None for p in self.P}
        self.hold = {}          # objet -> personne ou None (objets vus seulement)
        self.sol = {}           # objet -> lieu ou None (si posé / non porté)
        self.pers_vues, self.lieux_vus = set(), set()
        self.histoire = []      # phrases

    def vus_objets(self):
        return set(self.hold)

    def copie(self):
        c = Sim(self.P, self.L, self.O)
        c.loc, c.hold, c.sol = dict(self.loc), dict(self.hold), dict(self.sol)
        c.pers_vues, c.lieux_vus = set(self.pers_vues), set(self.lieux_vus)
        c.histoire = list(self.histoire)
        return c

    def valides(self):
        acts = []
        for p in self.P:
            for l in self.L:
                if self.loc[p] != l:
                    acts.append(("va", p, l))
            for o in self.O:
                if o in self.hold and self.hold[o] == p:
                    acts.append(("pose", p, o))
                    for q in self.P:
                        if q != p and self.loc[p] is not None and self.loc[q] == self.loc[p]:
                            acts.append(("donne", p, o, q))
                elif o not in self.hold:
                    acts.append(("prend", p, o))          # première apparition
                elif self.hold[o] is None and self.sol[o] is not None and self.loc[p] == self.sol[o]:
                    acts.append(("prend", p, o))
        return acts

    def appliquer(self, a):
        t, p = a[0], a[1]
        self.pers_vues.add(p)
        if t == "va":
            self.loc[p] = a[2]
            self.lieux_vus.add(a[2])
        elif t == "prend":
            self.hold[a[2]] = p
            self.sol.pop(a[2], None)
        elif t == "pose":
            self.hold[a[2]] = None
            self.sol[a[2]] = self.loc[p]
        elif t == "donne":
            self.pers_vues.add(a[3])
            self.hold[a[2]] = a[3]
            self.sol.pop(a[2], None)
        self.histoire.append(self.phrase(a))

    @staticmethod
    def phrase(a):
        t = a[0]
        if t == "va":
            return f"{a[1]} va {LIEUX[a[2]]}."
        if t == "prend":
            return f"{a[1]} prend {OBJETS[a[2]]}."
        if t == "pose":
            return f"{a[1]} pose {OBJETS[a[2]]}."
        return f"{a[1]} donne {OBJETS[a[2]]} à {a[3]}."

    def tirer(self, rng):
        acts = self.valides()
        poids = {"va": 4, "prend": 3, "pose": 2, "donne": 2}
        return rng.choices(acts, weights=[poids[a[0]] for a in acts])[0]

    # -- vérité terrain sur l'état -----------------------------------------
    def lieu_objet(self, o):
        if o not in self.hold:
            return None
        h = self.hold[o]
        return self.loc[h] if h else self.sol.get(o)


# ----------------------------------------------------------------------
# Plans : lecture de NOS phrases, rejeu et coût (indépendants du planificateur)
# ----------------------------------------------------------------------
_SURF_LIEU = {v: k for k, v in LIEUX.items()}
_SURF_OBJ = {v: k for k, v in OBJETS.items()}


def lire_phrase(ph):
    """'Marie va au jardin.' -> ('va','Marie','jardin') ; None si illisible."""
    ph = ph.strip().rstrip(".")
    m = re.match(r"^(\w+) va (.+)$", ph)
    if m and m.group(2) in _SURF_LIEU:
        return ("va", m.group(1), _SURF_LIEU[m.group(2)])
    m = re.match(r"^(\w+) (prend|pose) (.+)$", ph)
    if m and m.group(3) in _SURF_OBJ:
        return (m.group(2), m.group(1), _SURF_OBJ[m.group(3)])
    m = re.match(r"^(\w+) donne (.+) à (\w+)$", ph)
    if m and m.group(2) in _SURF_OBJ:
        return ("donne", m.group(1), _SURF_OBJ[m.group(2)], m.group(3))
    return None


def cout_acte(a, couts, mult):
    nom = {"va": "aller", "prend": "prendre", "pose": "poser", "donne": "donner"}[a[0]]
    return couts[nom] * mult.get(a[1], 1.0)


def buts_ok(sim, but):
    t = but[0]
    if t == "objet_lieu":
        return sim.lieu_objet(but[1]) == but[2]
    if t == "personne_lieu":
        return sim.loc[but[1]] == but[2]
    return sim.hold.get(but[1]) == but[2]          # possede : (objet, personne)


def rejouer_plan(sim0, phrases, but, couts, mult):
    """(valide, atteint, coût) en rejouant NOS règles sur une copie du monde."""
    s = sim0.copie()
    cout = 0.0
    for ph in phrases:
        a = lire_phrase(ph)
        if a is None or a not in s.valides():
            return False, False, cout
        cout += cout_acte(a, couts, mult)
        s.appliquer(a)
    return True, buts_ok(s, but), cout


# ----------------------------------------------------------------------
# Oracle : recherche à coût uniforme sur l'état COMPLET (tous les objets)
# ----------------------------------------------------------------------
def oracle(sim, but, couts, mult, plafond=250000):
    """Retourne ("plan", coût) | ("impossible", None) | ("plafond", None).

    État : (lieux des personnes, état de chaque objet vu). On ne retire AUCUN
    objet : c'est précisément ce qu'on vérifie de l'hypothèse du planificateur.
    Objet posé sans lieu connu : immobile (personne ne peut prouver y être).
    """
    t = but[0]
    P = [p for p in sim.P if p in sim.pers_vues]
    L = sorted(sim.lieux_vus)
    O = sorted(sim.vus_objets())
    # entités inconnues
    if t == "personne_lieu":
        if but[1] not in P or but[2] not in L:
            return "impossible", None
    elif t == "objet_lieu":
        if but[1] not in O or but[2] not in L:
            return "impossible", None
    else:
        if but[1] not in O or but[2] not in P:
            return "impossible", None
    ip = {p: i for i, p in enumerate(P)}
    il = {l: i for i, l in enumerate(L)}
    io = {o: i for i, o in enumerate(O)}
    ploc = tuple(il[sim.loc[p]] if sim.loc[p] in il else -1 for p in P)
    GELE = -1000                                   # posé en lieu inconnu : immobile
    ob = []
    for o in O:
        h = sim.hold[o]
        if h is not None:
            ob.append(ip[h])
        elif sim.sol.get(o) in il:
            ob.append(-1 - il[sim.sol[o]])
        else:
            ob.append(GELE)
    init = (ploc, tuple(ob))

    def atteint(s):
        pl, oo = s
        if t == "personne_lieu":
            return pl[ip[but[1]]] == il[but[2]]
        if t == "objet_lieu":
            v = oo[io[but[1]]]
            if v == GELE:
                return False
            return (pl[v] if v >= 0 else -1 - v) == il[but[2]]
        return oo[io[but[1]]] == ip[but[2]]

    def cc(nom, p):
        return couts[nom] * mult.get(P[p], 1.0)

    dist = {init: 0.0}
    tas = [(0.0, 0, init)]
    cpt = itertools.count(1)
    n = 0
    while tas:
        g, _, s = heapq.heappop(tas)
        if g > dist[s] + 1e-9:
            continue
        if atteint(s):
            return "plan", g
        n += 1
        if n > plafond:
            return "plafond", None
        pl, oo = s
        for i in range(len(P)):
            for k in range(len(L)):
                if pl[i] != k:
                    t2 = (pl[:i] + (k,) + pl[i + 1:], oo)
                    g2 = g + cc("aller", i)
                    if g2 + 1e-9 < dist.get(t2, 1e18):
                        dist[t2] = g2
                        heapq.heappush(tas, (g2, next(cpt), t2))
            for j, v in enumerate(oo):
                if v == GELE:
                    continue
                if v < 0 and pl[i] >= 0 and -1 - v == pl[i]:
                    new, nom = i, "prendre"
                elif v == i:
                    # poser
                    if pl[i] >= 0:
                        t2 = (pl, oo[:j] + (-1 - pl[i],) + oo[j + 1:])
                        g2 = g + cc("poser", i)
                        if g2 + 1e-9 < dist.get(t2, 1e18):
                            dist[t2] = g2
                            heapq.heappush(tas, (g2, next(cpt), t2))
                    # donner
                    for q in range(len(P)):
                        if q != i and pl[q] == pl[i] and pl[i] >= 0:
                            t2 = (pl, oo[:j] + (q,) + oo[j + 1:])
                            g2 = g + cc("donner", i)
                            if g2 + 1e-9 < dist.get(t2, 1e18):
                                dist[t2] = g2
                                heapq.heappush(tas, (g2, next(cpt), t2))
                    continue
                else:
                    continue
                t2 = (pl, oo[:j] + (new,) + oo[j + 1:])
                g2 = g + cc(nom, i)
                if g2 + 1e-9 < dist.get(t2, 1e18):
                    dist[t2] = g2
                    heapq.heappush(tas, (g2, next(cpt), t2))
    return "impossible", None


# ----------------------------------------------------------------------
# Ontologie de la charge (fermeture calculée par nous)
# ----------------------------------------------------------------------
# (singulier, pluriel avec article défini, pluriel avec « des »)
CLASSES = {"chien": "chiens", "chat": "chats", "mammifère": "mammifères",
           "animal": "animaux", "poisson": "poissons", "insecte": "insectes",
           "plante": "plantes"}
REGLES = [("chien", "mammifère"), ("chat", "mammifère"), ("mammifère", "animal"),
          ("poisson", "animal")]
INDIVIDUS = {"Médor": "chien", "Félix": "chat", "Nemo": "poisson", "Rex": "chien"}


def fermeture(x_classes, regles):
    """Toutes les classes atteintes depuis `x_classes` par les règles."""
    vus, pile = set(x_classes), list(x_classes)
    while pile:
        c = pile.pop()
        for a, b in regles:
            if a == c and b not in vus:
                vus.add(b)
                pile.append(b)
    return vus


def art(classe):
    return "un" if classe not in ("espèce",) else "une"


# ----------------------------------------------------------------------
# Calcul arithmétique (entiers Python, précédence écrite à la main)
# ----------------------------------------------------------------------
def calcul(termes, ops):
    """termes [a,b,c], ops ['plus','fois'] -> a + b*c avec la précédence usuelle."""
    t, o = list(termes), list(ops)
    i = 0
    while i < len(o):
        if o[i] == "fois":
            t[i] = t[i] * t.pop(i + 1)
            o.pop(i)
        else:
            i += 1
    r = t[0]
    for op, x in zip(o, t[1:]):
        r = r + x if op == "plus" else r - x
    return r


# ----------------------------------------------------------------------
# Charge mixte
# ----------------------------------------------------------------------
def construire_charge(graine=2024, n=400):
    rng = random.Random(graine)
    sim = Sim(PERSONNES, [k for k in LIEUX if k != "garage"],
              [o for o in OBJETS if o != "stylo"])
    reqs = []
    regles_ins, indiv_ins, negs = [], {}, set()       # connaissances insérées
    arith_pool = []
    for _ in range(14):
        k = rng.choice([2, 2, 3])
        termes = [rng.choice([rng.randint(2, 99), rng.randint(100, 99999),
                              rng.randint(10**6, 10**11)]) for _ in range(k)]
        ops = [rng.choice(["plus", "moins", "fois"]) for _ in range(k - 1)]
        arith_pool.append((termes, ops))
    gen_prompts = ["Écris : It was a dark and stormy night", "Continue : The sea was calm",
                   "Écris : Once upon a time", "Continue : In the beginning",
                   "Écris : The old man said", "Continue : She walked into the garden"]
    gen_a_faire = list(gen_prompts)
    inconnu_q = ["Qui a le stylo ?", "Où est le stylo ?", "Où est Pierre ?",
                 "Pierre est-il un animal ?", "Médor est-il une plante ?",
                 "Félix est-il un insecte ?"]
    illisibles = ["Quelle est la capitale de la France ?", "Pourquoi le ciel est bleu ?",
                  "Il pleut beaucoup depuis hier soir.", "Marie et Jean vont au parc.",
                  "Combien de pommes a Jean ?", "Où habite Marie ?",
                  "Elle prend la pomme.", "Dis-moi une blague."]

    def add(texte, cat, attendu=None, **extra):
        reqs.append(dict(texte=texte, categorie=cat, attendu=attendu, **extra))

    def etat_vers_questions():
        qs = []
        for o in sorted(sim.vus_objets()):
            lieu = sim.lieu_objet(o)
            qs.append((f"Où est {OBJETS[o]} ?", "q_monde", lieu if lieu else "abstention"))
            h = sim.hold[o]
            qs.append((f"Qui a {OBJETS[o]} ?", "q_monde", (h.lower() if h else "personne")))
        for p in sorted(sim.pers_vues):
            qs.append((f"Où est {p} ?", "q_monde", sim.loc[p] if sim.loc[p] else "abstention"))
        return qs

    def logique_questions():
        qs = []
        for ind, cl0 in indiv_ins.items():
            clos = fermeture({cl0}, regles_ins)
            for c in CLASSES:
                if c in clos:
                    qs.append((f"{ind} est-il {art(c)} {c} ?", "q_logique", "oui"))
                elif (ind, c) in negs:
                    qs.append((f"{ind} est-il {art(c)} {c} ?", "q_logique", "non"))
                elif c in ("insecte", "plante"):
                    qs.append((f"{ind} est-il {art(c)} {c} ?", "q_logique", "abstention"))
        return qs

    def but_aleatoire():
        """Retourne (texte, but) ; but None si impossible par entité inconnue."""
        r = rng.random()
        P = sorted(sim.pers_vues) or ["Marie"]
        Ok = sorted(sim.vus_objets()) or ["pomme"]
        Lk = sorted(sim.lieux_vus) or ["cuisine"]
        if r < 0.12:                         # impossible : lieu jamais mentionné
            o = rng.choice(Ok)
            return f"Objectif : {OBJETS[o]} doit être {LIEUX['garage']}.", ("objet_lieu", o, "garage")
        if r < 0.20:                         # impossible : objet inconnu
            return f"Objectif : {OBJETS['stylo']} doit être {LIEUX[rng.choice(Lk)]}.", \
                   ("objet_lieu", "stylo", rng.choice(Lk))
        if r < 0.26:                         # impossible : personne inconnue
            return f"Objectif : Pierre doit être {LIEUX[rng.choice(Lk)]}.", \
                   ("personne_lieu", "Pierre", rng.choice(Lk))
        if r < 0.50:
            o, l = rng.choice(Ok), rng.choice(Lk)
            return f"Objectif : {OBJETS[o]} doit être {LIEUX[l]}.", ("objet_lieu", o, l)
        if r < 0.75:
            p, l = rng.choice(P), rng.choice(Lk)
            return f"Objectif : {p} doit être {LIEUX[l]}.", ("personne_lieu", p, l)
        o, p = rng.choice(Ok), rng.choice(P)
        return f"Objectif : {p} doit avoir {OBJETS[o]}.", ("possede", o, p)

    def verite_but(but):
        if but[0] in ("objet_lieu", "possede") and but[1] not in sim.vus_objets() \
                or (but[0] == "personne_lieu" and but[1] not in sim.pers_vues) \
                or (but[0] == "possede" and but[2] not in sim.pers_vues) \
                or (but[0] != "possede" and but[2] not in sim.lieux_vus):
            return {"possible": False, "cout": None, "oracle": "impossible"}
        etat, c = oracle(sim, but, pf.COUTS_DEFAUT, {})
        return {"possible": etat == "plan" if etat != "plafond" else None, "cout": c,
                "oracle": etat}

    ronde = 0
    # amorce : une partie de l'ontologie dès le début
    while len(reqs) < n:
        ronde += 1
        # 1. connaissances
        if ronde <= 3 or rng.random() < 0.15:
            cands = [("regle", r) for r in REGLES if r not in regles_ins] + \
                    [("indiv", i) for i in INDIVIDUS if i not in indiv_ins]
            rng.shuffle(cands)
            for typ, v in cands[:3 if ronde <= 2 else 1]:
                if typ == "regle":
                    a, b = v
                    add(f"Tous les {CLASSES[a]} sont des {CLASSES[b]}.", "insertion", "ajoutee")
                    regles_ins.append(v)
                else:
                    add(f"{v} est {art(INDIVIDUS[v])} {INDIVIDUS[v]}.", "insertion", "ajoutee")
                    indiv_ins[v] = INDIVIDUS[v]
        if indiv_ins and rng.random() < 0.15:
            ind = rng.choice(sorted(indiv_ins))
            c = rng.choice(["insecte", "plante"])
            if (ind, c) not in negs:
                add(f"{ind} n'est pas {art(c)} {c}.", "insertion", "ajoutee")
                negs.add((ind, c))
        for _ in range(rng.randint(3, 6)):
            a = sim.tirer(rng)
            sim.appliquer(a)
            add(Sim.phrase(a), "insertion", "ajoutee")
        # contradiction rare
        if rng.random() < 0.25:
            tenus = [(o, h) for o, h in sim.hold.items() if h]
            if tenus:
                o, h = rng.choice(tenus)
                q = rng.choice([p for p in PERSONNES if p != h])
                add(f"{q} prend {OBJETS[o]}.", "insertion", "refusee")
            elif indiv_ins:
                ind = rng.choice(sorted(indiv_ins))
                clos = sorted(fermeture({indiv_ins[ind]}, regles_ins))
                add(f"{ind} n'est pas {art(clos[0])} {clos[0]}.", "insertion", "refusee")
        # 2. questions (avec répétitions)
        qs = etat_vers_questions() + logique_questions()
        rng.shuffle(qs)
        choix = qs[:rng.randint(3, 5)]
        for q in choix:
            add(*q)
        for q in rng.sample(choix, min(len(choix), rng.randint(0, 3))):
            add(*q)
        if choix and rng.random() < 0.3:
            for _ in range(2):
                add(*rng.choice(choix))
        if rng.random() < 0.5:
            add(rng.choice(inconnu_q), "q_inconnue", "abstention")
        if rng.random() < 0.35:
            add(rng.choice(illisibles), "illisible", "incompris")
        # 3. arithmétique
        for _ in range(rng.randint(1, 2)):
            termes, ops = rng.choice(arith_pool)
            txt = str(termes[0]) + "".join(f" {op} {x}" for op, x in zip(ops, termes[1:]))
            add(f"Combien font {txt} ?", "calcul", calcul(termes, ops))
        # 4. objectifs
        for _ in range(rng.randint(1, 2)):
            texte, but = but_aleatoire()
            v = verite_but(but)
            add(texte, "objectif", v, but=but, sim=sim.copie())
            if rng.random() < 0.4:
                add(texte, "objectif", v, but=but, sim=sim.copie())
        # 5. génération, espacée
        if gen_a_faire and ronde % 4 == 0:
            add(gen_a_faire.pop(0), "generation", "texte")
    reqs = reqs[:n]
    # place les générations restantes (s'il en manque)
    return reqs


# ----------------------------------------------------------------------
# Évaluation d'un résultat contre la vérité terrain
# ----------------------------------------------------------------------
def evaluer(req, res):
    """Retourne (juste, hallucination, note)."""
    cat, att = req["categorie"], req["attendu"]
    s = res.statut
    if cat == "insertion":
        juste = s == "insertion" and res.valeur == att
        return juste, False, ""
    if cat in ("q_monde", "q_logique", "q_inconnue"):
        if att == "abstention":
            return s == "abstention", s == "verifie", ""
        if s == "verifie":
            ok = str(res.valeur) == att
            return ok, not ok, ""
        return False, False, "abstention à tort" if s == "abstention" else s
    if cat == "calcul":
        if s == "verifie":
            ok = str(res.valeur) == str(att)
            return ok, not ok, ""
        return False, False, s
    if cat == "illisible":
        return s == "incompris", s in ("verifie", "insertion"), ""
    if cat == "generation":
        ok = (s == "non_verifie" and res.reponse.startswith(ag.ETIQUETTE_GEN)
              and len(res.reponse) > len(ag.ETIQUETTE_GEN) + 1)
        return ok, False, ""
    if cat == "objectif":
        v = att
        if v["oracle"] == "impossible":
            return s == "impossible", s == "verifie", "impossible attendu"
        if s == "impossible":
            return False, False, "impossible déclaré à tort"
        if s != "verifie":
            return False, False, s
        phr = res.detail.phrases
        valide, atteint, cout = rejouer_plan(req["sim"], phr, req["but"], pf.COUTS_DEFAUT, {})
        if not (valide and atteint):
            return False, True, "plan faux au rejeu indépendant"
        if v["oracle"] == "plan":
            ok = abs(cout - v["cout"]) < 1e-9
            return ok, False, "" if ok else f"non optimal ({cout} vs {v['cout']})"
        return True, False, "optimalité non vérifiée (plafond de l'oracle)"
    return False, False, "?"


def lancer_strategie(charge, strategie, **kw):
    a = ag.Aiguilleur(strategie=strategie, n_gen=100, **kw)
    lignes = []
    for rq in charge:
        res = a.traiter(rq["texte"])
        lignes.append(res)
    return a, lignes


def synthese(charge, lignes, a):
    cpu = [r.cpu_s for r in lignes]
    cats = defaultdict(lambda: [0, 0, 0.0])   # cat -> [n, justes, cpu]
    halluc = 0
    notes = Counter()
    mods = Counter()
    for rq, r in zip(charge, lignes):
        ok, h, note = evaluer(rq, r)
        c = cats[rq["categorie"]]
        c[0] += 1
        c[1] += ok
        c[2] += r.cpu_s
        halluc += h
        if not ok:
            notes[(rq["categorie"], note, r.statut)] += 1
        mods[r.module if r.module is not None else "aucun"] += 1
    cpu_gen = sum(r.cpu_s for rq, r in zip(charge, lignes) if rq["categorie"] == "generation")
    juste_tot = sum(v[1] for v in cats.values())
    return {
        "n": len(charge),
        "cpu_total_s": sum(cpu),
        "cpu_moyen_us": 1e6 * sum(cpu) / len(cpu),
        "cpu_hors_generation_s": sum(cpu) - cpu_gen,
        "cpu_moyen_hors_generation_us": 1e6 * (sum(cpu) - cpu_gen) /
        max(1, len(cpu) - sum(1 for rq in charge if rq["categorie"] == "generation")),
        "cpu_generation_s": cpu_gen,
        "energie_estimee_j": PUISSANCE_W * sum(cpu),
        "energie_estimee_hors_generation_j": PUISSANCE_W * (sum(cpu) - cpu_gen),
        "justes": juste_tot,
        "exactitude": juste_tot / len(charge),
        "hallucinations": halluc,
        "depassements_budget": sum(1 for r in lignes if r.statut == "budget_depasse"),
        "par_categorie": {k: {"n": v[0], "justes": v[1], "cpu_moyen_us": 1e6 * v[2] / v[0]}
                          for k, v in sorted(cats.items())},
        "modules": {str(k): v for k, v in sorted(mods.items(), key=lambda kv: str(kv[0]))},
        "cache": dict(a.compteurs),
        "erreurs": [{"categorie": k[0], "note": k[1], "statut": k[2], "n": v}
                    for k, v in notes.items()],
    }


def repetitions_utiles(charge):
    """Combien de requêtes répètent une requête identique vue avec la même version de la base."""
    vus, rep = set(), 0
    version = 0
    for rq in charge:
        if rq["categorie"] == "insertion":
            version += 1       # approximation : les refus ne changent pas la version
            continue
        cle = (ag.cle_cache(rq["texte"]),
               None if rq["texte"].lower().startswith("combien") else version)
        if cle in vus:
            rep += 1
        vus.add(cle)
    return rep


# ----------------------------------------------------------------------
# Banc du planificateur
# ----------------------------------------------------------------------
def banc_planificateur(graine, n_mondes):
    rng = random.Random(graine + 1)
    noms_l = ["cuisine", "jardin", "salon", "cave", "garage"]
    noms_o = ["pomme", "livre", "cle", "ballon"]
    stats = Counter()
    nodes, cpus, ecarts = [], [], []
    details_cout = Counter()
    instances = []
    for k in range(n_mondes):
        npers, nl, no = rng.randint(2, 4), rng.randint(2, 5), rng.randint(1, 4)
        P = PERSONNES[:npers]
        lieux = noms_l[:nl]
        objets = noms_o[:no]
        sim = Sim(P, lieux, objets)
        # historique : chaque personne bouge d'abord avec proba 0.85, puis marche aléatoire
        for p in P:
            if rng.random() < 0.85:
                sim.appliquer(("va", p, rng.choice(lieux)))
        for _ in range(rng.randint(2, 12)):
            sim.appliquer(sim.tirer(rng))
        # coûts
        couts = dict(pf.COUTS_DEFAUT)
        mult = {}
        if rng.random() < 0.5:
            mult = {p: rng.choice([1, 1, 2, 3]) for p in P}
        if rng.random() < 0.3:
            couts.update({"prendre": 2, "donner": 3})
        # but
        r = rng.random()
        Ok, Pk, Lk = sorted(sim.vus_objets()), sorted(sim.pers_vues), sorted(sim.lieux_vus)
        if r < 0.15 or not Ok or not Lk or not Pk:
            typ_imp = rng.choice(["lieu", "objet", "personne"])
            if typ_imp == "lieu" or not Ok:
                l_inc = [l for l in noms_l if l not in sim.lieux_vus] or ["grenier"]
                o = rng.choice(Ok or ["pomme"])
                l = rng.choice(l_inc)
                texte = f"Objectif : {OBJETS.get(o, 'la pomme')} doit être {LIEUX.get(l, 'au grenier')}."
                but = ("objet_lieu", o, l)
            elif typ_imp == "objet":
                texte = "Objectif : le stylo doit être " + LIEUX[rng.choice(Lk or ['cuisine'])] + "."
                but = ("objet_lieu", "stylo", "cuisine")
            else:
                texte = "Objectif : Pierre doit être " + LIEUX[rng.choice(Lk or ['cuisine'])] + "."
                but = ("personne_lieu", "Pierre", "cuisine")
            attendu_imp = True
        else:
            attendu_imp = False
            q = rng.random()
            if q < 0.4:
                o, l = rng.choice(Ok), rng.choice(Lk)
                texte, but = f"Objectif : {OBJETS[o]} doit être {LIEUX[l]}.", ("objet_lieu", o, l)
            elif q < 0.7:
                p, l = rng.choice(Pk), rng.choice(Lk)
                texte, but = f"Objectif : {p} doit être {LIEUX[l]}.", ("personne_lieu", p, l)
            else:
                o, p = rng.choice(Ok), rng.choice(Pk)
                texte, but = f"Objectif : {p} doit avoir {OBJETS[o]}.", ("possede", o, p)
        # oracle indépendant (état complet) -- inconnues traitées dans l'oracle
        try:
            etat_o, cout_o = oracle(sim, but, couts, mult)
        except KeyError:
            etat_o, cout_o = "impossible", None
        # raisonneur + planificateur
        rr = rz.Raisonneur()
        refus = [rr.ajouter(ph).statut for ph in sim.histoire]
        assert all(s in ("ajoutee", "doublon") for s in refus), refus
        pl = pf.Planificateur(couts=couts, multiplicateurs=mult, budget=200000)
        t0 = time.process_time()
        res = pl.planifier(rr, texte)
        cpus.append(time.process_time() - t0)
        stats["instances"] += 1
        nodes.append(res.noeuds)
        cat = "impossible_attendu" if etat_o == "impossible" else "possible"
        stats[f"oracle_{etat_o}"] += 1
        if res.statut == "plan":
            stats["plans"] += 1
            stats["plans_verifies"] += bool(res.verifie)
            valide, atteint, cout = rejouer_plan(sim, res.phrases, but, couts, mult)
            stats["plans_valides_rejeu_independant"] += bool(valide and atteint)
            if etat_o == "plan":
                stats["comparables"] += 1
                ok = abs(res.cout - cout_o) < 1e-9
                stats["optimaux"] += ok
                if not ok:
                    ecarts.append((k, res.cout, cout_o))
                details_cout[(len(res.actions) == 0)] += 1
            elif etat_o == "impossible":
                stats["plan_alors_que_impossible"] += 1
            else:
                stats["plan_oracle_plafonne"] += 1
        elif res.statut == "impossible":
            stats["impossibles_declares"] += 1
            if etat_o == "impossible":
                stats["impossibles_corrects"] += 1
            elif etat_o == "plan":
                stats["impossible_a_tort"] += 1
        elif res.statut == "budget":
            stats["budget_epuise"] += 1
        else:
            stats[f"statut_{res.statut}"] += 1
        instances.append({"k": k, "personnes": npers, "lieux": nl, "objets": no,
                          "but": texte, "oracle": etat_o, "cout_oracle": cout_o,
                          "statut": res.statut, "cout_astar": res.cout if res.statut == "plan" else None,
                          "noeuds": res.noeuds, "verifie": res.verifie})
    return {
        "n": stats["instances"], "stats": dict(stats),
        "noeuds_moyen": statistics.mean(nodes), "noeuds_max": max(nodes),
        "cpu_total_s": sum(cpus), "cpu_moyen_ms": 1e3 * statistics.mean(cpus),
        "cpu_max_ms": 1e3 * max(cpus),
        "ecarts": ecarts, "instances": instances,
    }


# ----------------------------------------------------------------------
# Rapport
# ----------------------------------------------------------------------
def fr(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def pct(a, b):
    return f"{100 * a / b:.1f} %".replace(".", ",") + f" ({a}/{b})" if b else "sans objet"


def ecrire_rapport(meta, charge, strat, plan):
    noms = {"aiguilleur": "(a) aiguilleur : le moins cher + cache",
            "toujours_tout": "(b) « toujours tout » (tous les modules, toutes les requêtes)",
            "toujours_tout_syntaxe": "(b') « toujours tout » limité aux formes adaptées",
            "sans_cache": "(c) le moins cher, sans cache"}
    L = []
    L.append("# Résultats : objectifs (planificateur) et aiguillage à coût minimal\n")
    L.append(f"Généré par `banc_aiguilleur.py` (graine {meta['graine']}, Python {meta['python']}, "
             f"{meta['plateforme']}). Python standard uniquement : aucun réseau de neurones, aucun poids.\n")
    L.append("## Avertissement\n")
    L.append("La charge et les mondes sont **générés par nous**, dans le français contrôlé que le "
             "système comprend ; la vérité terrain vient de code du banc indépendant des modules testés "
             "(simulateur du monde, fermeture des règles, entiers Python, recherche à coût uniforme "
             "sur l'état complet). Ces chiffres montrent le coût et la correction des modules sur ces "
             "charges, **pas** une intelligence générale. **Énergie = ESTIMATION** : temps CPU "
             "(`time.process_time`) x 10 W, hypothèse de 10 W par cœur occupé ; aucun wattmètre.\n")
    L.append("## 1. Charge mixte et stratégies\n")
    comp = Counter(r["categorie"] for r in charge)
    L.append(f"Charge : **{len(charge)} requêtes**, composition : " +
             ", ".join(f"{k} {v}" for k, v in sorted(comp.items())) + ". "
             f"Requêtes répétant exactement une requête déjà vue avec le même état de la base "
             f"(borne haute des succès du cache) : **{meta['repetitions']}**.\n")
    L.append("Définitions : (a) cache (question normalisée exacte + version de la base) puis raisonneur, "
             "planificateur, génération, en s'arrêtant au premier résultat accepté ; (c) pareil sans cache ; "
             "(b) **tous** les modules lancés sur **toutes** les requêtes (génération incluse, 40 caractères "
             "d'amorce par requête non générative), sans cache, on garde le meilleur résultat vérifié ; "
             "(b') tous les modules dont la forme syntaxique convient (planificateur seulement pour "
             "« Objectif », génération seulement pour « Écris/Continue »).\n")
    L.append(f"Mesures : médiane de {meta['repetitions_runs']} exécutions complètes pour (a), (c), (b') ; "
             f"(b) : {meta['runs_b']} exécution(s) (très coûteuse en CPU). Le temps de chargement de "
             f"`memoire.xz` (paresseux) est compté dans la première requête de génération.\n")
    L.append("### Coût total\n")
    L.append("| Stratégie | CPU total (s) | CPU moyen / requête (µs) | CPU hors génération (ms) | "
             "CPU moyen hors génération (µs) | Énergie estimée totale (J) | Énergie estimée hors génération (mJ) |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ("aiguilleur", "sans_cache", "toujours_tout_syntaxe", "toujours_tout"):
        if k not in strat:
            continue
        s = strat[k]
        L.append(f"| {noms[k]} | {fr(s['cpu_total_s'], 3)} | {fr(s['cpu_moyen_us'], 1)} | "
                 f"{fr(1e3 * s['cpu_hors_generation_s'], 2)} | {fr(s['cpu_moyen_hors_generation_us'], 1)} | "
                 f"{fr(s['energie_estimee_j'], 3)} | {fr(1e3 * s['energie_estimee_hors_generation_j'], 2)} |")
    L.append("")
    L.append("**Lecture honnête** : la génération (Écho) coûte des centaines de milliers de fois plus "
             "qu'une réponse du raisonneur ; le total est donc dominé par elle dès qu'on la lance, et c'est "
             "exactement ce que « toujours tout » fait. La colonne « hors génération » isole l'effet du cache "
             "et de l'ordre des modules sur le reste de la charge.\n")
    L.append("### Exactitude (vérité terrain indépendante)\n")
    L.append("| Stratégie | Justes | Exactitude | Hallucinations (réponse affirmée fausse) | Dépassements de budget |")
    L.append("|---|---|---|---|---|")
    for k in ("aiguilleur", "sans_cache", "toujours_tout_syntaxe", "toujours_tout"):
        if k not in strat:
            continue
        s = strat[k]
        L.append(f"| {noms[k]} | {s['justes']}/{s['n']} | {pct(s['justes'], s['n'])} | "
                 f"{s['hallucinations']} | {s['depassements_budget']} |")
    L.append("")
    L.append("« Juste » : insertion acceptée/refusée comme attendu ; réponse exacte avec preuve vérifiée ; "
             "abstention quand aucune réponse n'existe ; refus des phrases illisibles ; objectif : "
             "plan vérifié, rejoué par le banc avec ses propres règles, de **coût égal à l'optimum de "
             "l'oracle**, ou « impossible » quand l'oracle le dit ; génération : texte non vide étiqueté "
             "« non vérifié » (on ne juge pas sa qualité).\n")
    a = strat["aiguilleur"]
    L.append("### Par catégorie (stratégie (a))\n")
    L.append("| Catégorie | Requêtes | Justes | CPU moyen (µs) |")
    L.append("|---|---|---|---|")
    for k, v in a["par_categorie"].items():
        L.append(f"| {k} | {v['n']} | {v['justes']} | {fr(v['cpu_moyen_us'], 1)} |")
    L.append("")
    if a["erreurs"]:
        L.append("Erreurs de (a) : " + "; ".join(
            f"{e['categorie']}/{e['note']}/{e['statut']} x{e['n']}" for e in a["erreurs"]) + "\n")
    else:
        L.append("Aucune erreur pour (a) sur cette charge.\n")
    L.append("### Qui a répondu (nombre de requêtes par module)\n")
    L.append("| Stratégie | 0 cache | 1 raisonneur | 2 planificateur | 3 génération | aucun (abstention/refus) |")
    L.append("|---|---|---|---|---|---|")
    for k in ("aiguilleur", "sans_cache", "toujours_tout_syntaxe", "toujours_tout"):
        if k not in strat:
            continue
        m = strat[k]["modules"]
        L.append(f"| {noms[k]} | {m.get('0', 0)} | {m.get('1', 0)} | {m.get('2', 0)} | "
                 f"{m.get('3', 0)} | {m.get('aucun', 0)} |")
    L.append("")
    c = a["cache"]
    L.append(f"Cache de (a) : {c['cache_hits']} succès, {c['cache_miss']} échecs de recherche "
             f"(taux de succès {pct(c['cache_hits'], c['cache_hits'] + c['cache_miss'])}). "
             f"Colonne « aucun » : requêtes qu'aucun module n'a su traiter (phrases illisibles refusées). "
             f"Les abstentions honnêtes (« Je ne sais pas ») sont comptées sous le module 1 ; elles ne sont "
             f"pas mises en cache, d'où un écart avec la borne haute des répétitions ci-dessus.\n")
    # gain
    if "sans_cache" in strat:
        sc = strat["sans_cache"]
        g = 1 - a["cpu_hors_generation_s"] / sc["cpu_hors_generation_s"]
        L.append(f"**Gain du cache** (hors génération) : {fr(100 * g, 1)} % de CPU en moins que (c) "
                 f"({fr(1e3 * a['cpu_hors_generation_s'], 2)} ms contre "
                 f"{fr(1e3 * sc['cpu_hors_generation_s'], 2)} ms).\n")
    if "toujours_tout" in strat:
        tt = strat["toujours_tout"]
        L.append(f"**Aiguilleur contre « toujours tout »** : {fr(a['cpu_total_s'], 3)} s contre "
                 f"{fr(tt['cpu_total_s'], 1)} s de CPU, soit un facteur "
                 f"{fr(tt['cpu_total_s'] / a['cpu_total_s'], 1)} ; exactitude "
                 f"{pct(a['justes'], a['n'])} contre {pct(tt['justes'], tt['n'])}. "
                 f"L'économie vient du fait que la génération n'est jamais lancée sans que la requête "
                 f"la demande ; elle ne dit rien de la qualité d'un modèle de langage neuronal.\n")
    # planificateur
    s = plan["stats"]
    n = plan["n"]
    L.append("## 2. Planificateur (A*) sur 200 mondes aléatoires\n")
    L.append(f"{n} mondes : 2 à 4 personnes, 2 à 5 lieux, 1 à 4 objets, historique aléatoire de 2 à 12 "
             f"événements après un premier déplacement (85 % des personnes), buts aléatoires dont environ "
             f"15 % d'impossibles volontaires ; la moitié des mondes ont des multiplicateurs de coût par "
             f"personne (1, 2 ou 3), 30 % des coûts d'action modifiés (prendre = 2, donner = 3). "
             f"L'oracle est une recherche à coût uniforme sur l'état **complet** (tous les objets, "
             f"plafond {250000} nœuds).\n")
    L.append("| Mesure | Valeur |")
    L.append("|---|---|")
    pos = s.get("oracle_plan", 0)
    imp = s.get("oracle_impossible", 0)
    L.append(f"| Instances faisables selon l'oracle | {pos} |")
    L.append(f"| Instances impossibles selon l'oracle | {imp} |")
    L.append(f"| Oracle à plafond (non comparable) | {s.get('oracle_plafond', 0)} |")
    L.append(f"| Plans trouvés (parmi les faisables) | {pct(s.get('plans', 0) - s.get('plan_alors_que_impossible', 0) - s.get('plan_oracle_plafonne', 0), pos)} |")
    L.append(f"| Plans vérifiés par le raisonneur frais + vérificateur | {pct(s.get('plans_verifies', 0), s.get('plans', 0))} |")
    L.append(f"| Plans valides au rejeu indépendant du banc | {pct(s.get('plans_valides_rejeu_independant', 0), s.get('plans', 0))} |")
    L.append(f"| Optimalité (coût A* = coût oracle) | {pct(s.get('optimaux', 0), s.get('comparables', 0))} |")
    L.append(f"| Impossibles détectés correctement | {pct(s.get('impossibles_corrects', 0), imp)} |")
    L.append(f"| « Impossible » déclaré à tort | {s.get('impossible_a_tort', 0)} |")
    L.append(f"| Plan renvoyé alors que l'oracle dit impossible | {s.get('plan_alors_que_impossible', 0)} |")
    L.append(f"| Budget de nœuds épuisé | {s.get('budget_epuise', 0)} |")
    L.append(f"| Plans non vérifiés (refusés) | {s.get('statut_non_verifie', 0)} |")
    L.append(f"| Nœuds développés, moyenne / maximum | {fr(plan['noeuds_moyen'], 2)} / {plan['noeuds_max']} |")
    L.append(f"| CPU total (s) | {fr(plan['cpu_total_s'], 3)} |")
    L.append(f"| CPU moyen / maximum par instance (ms) | {fr(plan['cpu_moyen_ms'], 2)} / {fr(plan['cpu_max_ms'], 2)} |")
    L.append(f"| Énergie estimée totale (mJ) | {fr(1e3 * PUISSANCE_W * plan['cpu_total_s'], 2)} |")
    L.append("")
    if plan["ecarts"]:
        L.append("Écarts d'optimalité : " + ", ".join(
            f"monde {k} : A* {a_} contre oracle {o}" for k, a_, o in plan["ecarts"]) + "\n")
    else:
        L.append("Aucun écart d'optimalité sur les instances comparables.\n")
    L.append("## 3. Limites (à lire)\n")
    for ligne in LIMITES:
        L.append(f"- {ligne}")
    L.append("")
    return "\n".join(L)


LIMITES = [
    "**Banc écrit par nous.** Charges et mondes utilisent les tournures que le parseur connaît ; "
    "100 % ici ne dit rien d'une charge réelle ni de textes naturels.",
    "**La génération n'est pas « vérifiée ».** Écho produit du texte anglais littéraire ancien "
    "(sa bibliothèque), sans sens ; l'aiguilleur se contente de l'étiqueter « texte généré, non "
    "vérifié ». La comparaison de coût avec « toujours tout » met en scène un module lent de façon "
    "volontairement naïve (il est lancé sur tout) : c'est le point de l'expérience, pas une mesure "
    "de ce que ferait un vrai gros modèle.",
    "**Pas de comparaison avec un modèle neuronal.** Aucun LLM n'a été mesuré ; aucun gain "
    "d'énergie n'est revendiqué par rapport à un LLM. L'énergie est une estimation (CPU x 10 W).",
    "**Le cache n'aide que sur les répétitions exactes** (au caractère près après normalisation "
    "bon marché) et se vide de fait à chaque nouvelle phrase pour les questions sur le monde. Pour les "
    "réponses très bon marché (dizaines de µs), le gain est petit ; il est net sur les plans et les calculs.",
    "**Budget CPU coopératif** : vérifié avant chaque module, dans A* et par blocs de 40 caractères "
    "de génération ; le raisonneur lui-même n'est pas interruptible (un dépassement est signalé, "
    "pas empêché).",
    "**Une abstention n'est pas escaladée** : « Je ne sais pas » reste « Je ne sais pas » ; aucun "
    "module plus puissant n'est appelé pour deviner.",
    "**Planificateur : modèle du monde minimal.** Une personne a au plus un lieu, pas de portes, pas de "
    "capacités de transport, pas de temps. Les autres objets sont ignorés dans A* (ils n'aident jamais) ; "
    "l'oracle les garde tous, et les deux coïncident sur les instances comparées, ce qui valide "
    "l'hypothèse sur ces instances seulement.",
    "**Nos règles d'actions sont plus strictes que celles du raisonneur** (il accepterait « X prend O » "
    "sans que X soit là où est O). On exige la co-localisation : un plan est donc valide pour le "
    "raisonneur et pour nos règles.",
    "**« Impossible » n'est pas certifié par le vérificateur** : le vérificateur ne prouve que des "
    "conclusions positives. L'impossibilité vient soit d'une entité inconnue, soit de l'épuisement de "
    "l'espace d'états (donc dépend du modèle d'actions ci-dessus).",
    "**Plafond de l'oracle** : les instances où l'oracle dépasse son plafond ne sont pas comparées "
    "(comptées à part). L'optimalité n'est donc établie que sur des instances petites.",
    "**Vérificateur partageant l'analyseur** : comme pour le raisonneur, la vérification garantit que "
    "la conclusion découle des phrases telles que lues, pas que les phrases ont été bien lues.",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graine", type=int, default=2024)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--mondes", type=int, default=200)
    ap.add_argument("--sans-toujours-tout", action="store_true",
                    help="saute la stratégie (b), très coûteuse (~4 minutes de CPU)")
    a = ap.parse_args(argv)

    charge = construire_charge(a.graine, a.n)
    print(f"charge : {len(charge)} requêtes, {Counter(r['categorie'] for r in charge)}")
    strat = {}
    plan_strats = [("aiguilleur", a.repetitions), ("sans_cache", a.repetitions),
                   ("toujours_tout_syntaxe", a.repetitions)]
    if not a.sans_toujours_tout:
        plan_strats.append(("toujours_tout", 1))
    for nom, reps in plan_strats:
        essais = []
        for _ in range(reps):
            ai, lignes = lancer_strategie(charge, nom)
            essais.append((synthese(charge, lignes, ai), lignes))
        essais.sort(key=lambda e: e[0]["cpu_total_s"])
        s = essais[len(essais) // 2][0]
        s["cpu_total_runs_s"] = [e[0]["cpu_total_s"] for e in essais]
        strat[nom] = s
        print(f"{nom}: CPU {s['cpu_total_s']:.3f} s ; justes {s['justes']}/{s['n']} ; "
              f"halluc. {s['hallucinations']} ; modules {s['modules']}")
    print("planificateur ...")
    plan = banc_planificateur(a.graine, a.mondes)
    print(plan["stats"])
    meta = {"graine": a.graine, "python": platform.python_version(),
            "plateforme": platform.platform(), "repetitions": repetitions_utiles(charge),
            "repetitions_runs": a.repetitions, "runs_b": 0 if a.sans_toujours_tout else 1}
    md = ecrire_rapport(meta, charge, strat, plan)
    with open(os.path.join(ICI, "resultats_objectif_cout.md"), "w", encoding="utf-8") as f:
        f.write(md)
    sortie = {"meta": meta, "strategies": strat,
              "planificateur": {k: v for k, v in plan.items() if k != "instances"},
              "planificateur_instances": plan["instances"],
              "composition": dict(Counter(r["categorie"] for r in charge))}
    with open(os.path.join(ICI, "resultats_objectif_cout.json"), "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=1)
    print("écrit resultats_objectif_cout.md / .json")


if __name__ == "__main__":
    main()
