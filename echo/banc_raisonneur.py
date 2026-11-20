#!/usr/bin/env python3
"""Banc d'essai du raisonneur : tâches synthétiques à réponses connues.

Les tâches sont GÉNÉRÉES ici (graine fixe) et leurs réponses attendues sont
calculées par le générateur, avec son propre code (un petit simulateur du
monde, des entiers Python), jamais par le raisonneur.

Attention : ces tâches sont écrites par nous, dans le français contrôlé que
le système comprend. 100 % ici prouve que le moteur est correct, PAS qu'il
est intelligent. Voir la section « limites » du rapport.

Usage : python3 banc_raisonneur.py [--n 200] [--graine 2024]
Sorties : resultats_raisonneur.md et resultats_raisonneur.json
"""

import argparse
import json
import math
import os
import platform
import random
import time

import raisonneur as rz

ICI = os.path.dirname(os.path.abspath(__file__))
PUISSANCE_W = 10.0   # hypothèse : 10 W par cœur occupé (ESTIMATION, pas une mesure)

# ----------------------------------------------------------------------
# Vocabulaire
# ----------------------------------------------------------------------
FEMMES = ["Marie", "Anne", "Sophie", "Léa", "Julie", "Clara", "Emma", "Alice"]
HOMMES = ["Jean", "Paul", "Luc", "Hugo", "Marc", "Théo", "Pierre", "Louis"]
PERSONNES = [(n, True) for n in FEMMES] + [(n, False) for n in HOMMES]
LIEUX = ["dans la cuisine", "au jardin", "dans le salon", "à la cave",
         "au garage", "dans la chambre", "au bureau", "à la bibliothèque",
         "dans le couloir", "à la gare"]
OBJETS = ["la pomme", "le ballon", "le livre", "la clé", "la lampe",
          "le journal", "la bouteille", "le sac", "l'écharpe", "le stylo"]
INDIVIDUS = ["Socrate", "Médor", "Félix", "Rex", "Titi", "Nemo", "Bambi",
             "Milou", "Garfield", "Idéfix", "Tom", "Jerry", "Pluto", "Simba",
             "Dumbo", "Balou", "Hobbes", "Rantanplan", "Azor", "Jolly"]
CLASSES = [("chien", "chiens"), ("chat", "chats"), ("oiseau", "oiseaux"),
           ("cheval", "chevaux"), ("animal", "animaux"), ("poisson", "poissons"),
           ("mammifère", "mammifères"), ("insecte", "insectes"),
           ("arbre", "arbres"), ("fleur", "fleurs"), ("fruit", "fruits"),
           ("légume", "légumes"), ("outil", "outils"), ("véhicule", "véhicules"),
           ("métal", "métaux"), ("minéral", "minéraux"), ("loup", "loups"),
           ("renard", "renards"), ("vache", "vaches"), ("rongeur", "rongeurs"),
           ("reptile", "reptiles"), ("serpent", "serpents"),
           ("lézard", "lézards"), ("requin", "requins"), ("baleine", "baleines"),
           ("aigle", "aigles"), ("moineau", "moineaux"), ("corbeau", "corbeaux"),
           ("hibou", "hiboux"), ("végétal", "végétaux")]


def cle(texte):
    """Clé attendue (minuscules, sans accents, sans article)."""
    t = rz.normaliser(texte)
    for art in ("le ", "la ", "l'", "les ", "un ", "une "):
        if t.startswith(art):
            return t[len(art):]
    return t


# ----------------------------------------------------------------------
# Simulateur du monde du générateur (indépendant du raisonneur)
# ----------------------------------------------------------------------
class Sim:
    def __init__(self):
        self.lieu, self.porte, self.pose = {}, {}, {}

    def va(self, p, l):
        self.lieu[p] = l

    def prend(self, p, o):
        self.porte[o] = p
        self.pose.pop(o, None)

    def poser(self, p, o):
        del self.porte[o]
        self.pose[o] = self.lieu.get(p)

    def donne(self, p, o, q):
        self.porte[o] = q

    def ou(self, o):
        if o in self.porte:
            return self.lieu.get(self.porte[o])
        return self.pose.get(o)


def phrase_va(rng, p, lieu):
    nom, fem = p
    formes = [f"{nom} va {lieu}.", f"{nom} est allé{'e' if fem else ''} {lieu}."]
    if lieu.startswith("dans"):
        formes.append(f"{nom} entre {lieu}.")
    return rng.choice(formes)


def phrase_prend(rng, p, o):
    return rng.choice([f"{p[0]} prend {o}.", f"{p[0]} a pris {o}.", f"{p[0]} ramasse {o}."])


def phrase_pose(rng, p, o):
    return rng.choice([f"{p[0]} pose {o}.", f"{p[0]} a posé {o}.", f"{p[0]} dépose {o}."])


def phrase_donne(rng, p, o, q):
    return rng.choice([f"{p[0]} donne {o} à {q[0]}.", f"{p[0]} a donné {o} à {q[0]}."])


def phrase_regle(rng, a, b):
    return rng.choice([f"Tout {a[0]} est un {b[0]}.",
                       f"Tous les {a[1]} sont des {b[1]}.",
                       f"Les {a[1]} sont des {b[1]}.",
                       f"Un {a[0]} est un {b[0]}."])


def phrase_fait(x, c):
    return f"{x} est un {c[0]}."


def phrase_non(x, c):
    return f"{x} n'est pas un {c[0]}."


def question_est(rng, x, c):
    return rng.choice([f"{x} est-il un {c[0]} ?", f"Est-ce que {x} est un {c[0]} ?"])


def tirer(rng, pool, n):
    return rng.sample(pool, n)


def distracteurs_classes(rng, classes_libres, individus_libres, k):
    """k phrases sans rapport (faits et règles sur des classes et individus à part)."""
    out = []
    cl = list(classes_libres)
    ind = list(individus_libres)
    rng.shuffle(cl)
    rng.shuffle(ind)
    for _ in range(k):
        t = rng.randrange(3)
        if t == 0 and len(cl) >= 2:
            out.append(phrase_regle(rng, cl.pop(), cl[-1]))
        elif t == 1 and ind and cl:
            out.append(phrase_fait(ind.pop(), cl[-1]))
        elif ind and cl:
            out.append(phrase_non(ind.pop(), cl[-1]))
    return out


def melanger_insere(rng, base, extras):
    """Insère les extras à des positions aléatoires sans réordonner `base`."""
    res = list(base)
    for e in extras:
        res.insert(rng.randint(0, len(res)), e)
    return res


# ----------------------------------------------------------------------
# Générateurs de tâches. Chaque instance : dict(phrases, question, attendu)
# ----------------------------------------------------------------------
def gen_t1(rng):
    n = rng.randint(2, 4)
    pers = rng.sample(PERSONNES, n)
    evts, sim = [], Sim()
    ordre = list(pers) + [rng.choice(pers) for _ in range(rng.randint(3, 8))]
    rng.shuffle(ordre)                   # chacun bouge au moins une fois
    for p in ordre:
        l = rng.choice(LIEUX)
        evts.append(phrase_va(rng, p, l))
        sim.va(p[0], l)
    cible = rng.choice(pers)
    cl = rng.sample(CLASSES, 3)
    ind = rng.sample(INDIVIDUS, 3)
    extras = [phrase_fait(ind[0], cl[0]), phrase_regle(rng, cl[0], cl[1]),
              phrase_non(ind[1], cl[2])][:rng.randint(1, 3)]
    return {"phrases": melanger_insere(rng, evts, extras),
            "question": f"Où est {cible[0]} ?",
            "attendu": lieu_cle(sim.lieu[cible[0]])}


def lieu_cle(l):
    """« dans la cuisine » -> « cuisine »."""
    return cle(l.split(" ", 1)[1])


def gen_t2(rng):
    while True:
        pers = rng.sample(PERSONNES, rng.randint(2, 3))
        objs = rng.sample(OBJETS, rng.randint(2, 3))
        sim, evts = Sim(), []

        def va(p):
            l = rng.choice(LIEUX)
            evts.append(phrase_va(rng, p, l))
            sim.va(p[0], l)
        for p in pers:
            va(p)
        for _ in range(rng.randint(5, 9)):
            a = rng.random()
            tenus = {o: p for o, p in sim.porte.items()}
            if a < 0.30:
                va(rng.choice(pers))
            elif a < 0.60:
                libres = [o for o in objs if cle(o) not in sim.porte]
                if libres:
                    o, p = rng.choice(libres), rng.choice(pers)
                    evts.append(phrase_prend(rng, p, o))
                    sim.prend(p[0], cle(o))
            elif a < 0.78 and tenus:
                o = rng.choice(list(tenus))
                p = next(q for q in pers if q[0] == tenus[o])
                evts.append(phrase_pose(rng, p, next(x for x in objs if cle(x) == o)))
                sim.poser(p[0], o)
            elif tenus:
                o = rng.choice(list(tenus))
                p = next(q for q in pers if q[0] == tenus[o])
                q = rng.choice([x for x in pers if x != p])
                evts.append(phrase_donne(rng, p, next(x for x in objs if cle(x) == o), q))
                sim.donne(p[0], o, q[0])
        veut_pose = rng.random() < 0.25
        cands = [o for o in objs if (cle(o) in sim.pose if veut_pose else cle(o) in sim.porte)
                 and sim.ou(cle(o)) is not None]
        if not cands:
            continue
        o = rng.choice(cands)
        co = cle(o)
        if not veut_pose and rng.random() < 0.25:
            question, attendu = f"Qui a {o} ?", cle(sim.porte[co])
        else:
            question, attendu = f"Où est {o} ?", lieu_cle(sim.ou(co))
        cl = rng.sample(CLASSES, 2)
        ind = rng.sample(INDIVIDUS, 2)
        extras = [phrase_fait(ind[0], cl[0]), phrase_regle(rng, cl[0], cl[1])][:rng.randint(0, 2)]
        return {"phrases": melanger_insere(rng, evts, extras),
                "question": question, "attendu": attendu}


def gen_chaine(rng, longueur, exclus=()):
    """Une chaîne de classes C0 ⊂ C1 ⊂ ... ⊂ C_longueur."""
    pool = [c for c in CLASSES if c not in exclus]
    return rng.sample(pool, longueur + 1)


def gen_t3(rng, longueur=None):
    L = longueur or rng.randint(1, 5)
    ch = gen_chaine(rng, L)
    x, y, z = rng.sample(INDIVIDUS, 3)
    base = [phrase_fait(x, ch[0])]
    base += [phrase_regle(rng, ch[i], ch[i + 1]) for i in range(L)]
    libres = [c for c in CLASSES if c not in ch]
    # chaîne distractrice sur d'autres classes + un autre individu
    d = rng.sample(libres, 3)
    extras = [phrase_fait(y, d[0]), phrase_regle(rng, d[0], d[1]),
              phrase_regle(rng, d[1], d[2]), phrase_non(z, d[2])]
    phrases = base + extras[:rng.randint(2, 4)]
    rng.shuffle(phrases)          # l'ordre n'a pas d'importance
    return {"phrases": phrases, "question": question_est(rng, x, ch[L]),
            "attendu": "oui", "longueur": L}


def gen_t4(rng, genre=None):
    genre = genre if genre is not None else rng.randrange(5)
    x, y = rng.sample(INDIVIDUS, 2)
    if genre == 0:                     # oui par chaîne
        L = rng.randint(1, 3)
        ch = gen_chaine(rng, L)
        ph = [phrase_fait(x, ch[0])] + [phrase_regle(rng, ch[i], ch[i + 1]) for i in range(L)]
        q, att = question_est(rng, x, ch[L]), "oui"
    elif genre == 1:                   # non par fait négatif direct
        c = rng.sample(CLASSES, 2)
        ph = [phrase_fait(x, c[0]), phrase_non(x, c[1])]
        q, att = question_est(rng, x, c[1]), "non"
    elif genre == 2:                   # non par contraposition (modus tollens)
        L = rng.randint(1, 3)
        ch = gen_chaine(rng, L)
        ph = [phrase_regle(rng, ch[i], ch[i + 1]) for i in range(L)] + [phrase_non(x, ch[L])]
        q, att = question_est(rng, x, ch[0]), "non"
    elif genre == 3:                   # non par disjonction (aucun P n'est un Q)
        m = rng.randint(0, 2)
        ch = gen_chaine(rng, m + 1)    # ch[0..m] chaîne de x, P = ch[m]
        P, Q = ch[m], ch[m + 1]
        ph = [phrase_fait(x, ch[0])] + [phrase_regle(rng, ch[i], ch[i + 1]) for i in range(m)]
        ph.append(f"Aucun {P[0]} n'est un {Q[0]}.")
        q, att = question_est(rng, x, Q), "non"
    else:                              # oui, malgré des négations qui traînent
        L = rng.randint(1, 3)
        ch = gen_chaine(rng, L)
        libres = [c for c in CLASSES if c not in ch]
        n1, n2 = rng.sample(libres, 2)
        ph = [phrase_fait(x, ch[0])] + [phrase_regle(rng, ch[i], ch[i + 1]) for i in range(L)]
        ph += [phrase_non(x, n1), phrase_non(y, n2)]
        q, att = question_est(rng, x, ch[L]), "oui"
    utilises = [c for c in CLASSES if any(f" {c[0]}" in p or f" {c[1]}" in p for p in ph)]
    libres = [c for c in CLASSES if c not in utilises]
    ind_libres = [i for i in INDIVIDUS if i not in (x, y)]
    extras = distracteurs_classes(rng, libres, ind_libres, rng.randint(1, 3))
    phrases = ph + extras
    rng.shuffle(phrases)
    return {"phrases": phrases, "question": q, "attendu": att, "genre": genre}


def gen_t5(rng, genre=None):
    genre = genre if genre is not None else rng.randrange(7)
    pers = rng.sample(PERSONNES, 3)
    ph, ind = [], rng.sample(INDIVIDUS, 3)
    cl = rng.sample(CLASSES, 5)
    ph += [phrase_fait(ind[0], cl[0]), phrase_regle(rng, cl[0], cl[1])]
    if genre == 0:                      # individu jamais mentionné
        q = question_est(rng, ind[2], cl[1])
    elif genre == 1:                    # réciproque (non valide)
        ph += [phrase_fait(ind[1], cl[1])]
        q = question_est(rng, ind[1], cl[0])
    elif genre == 2:                    # classe sans lien, ni déductible ni réfutable
        q = question_est(rng, ind[0], cl[3])
    elif genre == 3:                    # personne jamais mentionnée
        ph += [phrase_va(rng, pers[0], rng.choice(LIEUX))]
        q = f"Où est {pers[2][0]} ?"
    elif genre == 4:                    # objet jamais mentionné
        o = rng.sample(OBJETS, 2)
        ph += [phrase_va(rng, pers[0], rng.choice(LIEUX)), phrase_prend(rng, pers[0], o[0])]
        q = f"Où est {o[1]} ?"
    elif genre == 5:                    # qui a un objet jamais mentionné
        o = rng.sample(OBJETS, 2)
        ph += [phrase_va(rng, pers[0], rng.choice(LIEUX)), phrase_prend(rng, pers[0], o[0])]
        q = f"Qui a {o[1]} ?"
    else:                               # le porteur n'a jamais été localisé
        o = rng.choice(OBJETS)
        ph += [phrase_prend(rng, pers[0], o), phrase_va(rng, pers[1], rng.choice(LIEUX))]
        q = f"Où est {o} ?"
    if genre < 3:
        rng.shuffle(ph)
    return {"phrases": ph, "question": q, "attendu": "inconnu", "genre": genre}


def gen_t6(rng):
    def nombre(maxi=10 ** 12):
        d = rng.randint(1, 12)
        return rng.randint(0, min(10 ** d, maxi))
    sym = rng.choice(["plus", "moins", "fois", "×", "x", "divisé par"])
    multi = rng.random() < 0.3
    if sym == "divisé par":
        b = rng.randint(1, 10 ** rng.randint(1, 6))
        a = b * rng.randint(0, 10 ** 12 // b)
    else:
        a, b = nombre(), nombre()
    val = {"plus": a + b, "moins": a - b, "fois": a * b, "×": a * b, "x": a * b,
           "divisé par": a // b if b else 0}[sym]
    texte = f"{a} {sym} {b}"
    if multi and sym in ("plus", "moins"):
        c = nombre(10 ** 6)
        texte += f" fois {c}"
        val = a + b * c if sym == "plus" else a - b * c
    elif multi and sym in ("fois", "×", "x"):
        c = nombre()
        texte = f"{a} {sym} {b} plus {c}"
        val = a * b + c
    q = rng.choice(["Combien font", "Combien fait"]) + f" {texte} ?"
    cl = rng.sample(CLASSES, 2)
    ph = [phrase_fait(rng.choice(INDIVIDUS), cl[0])][:rng.randint(0, 1)]
    return {"phrases": ph, "question": q, "attendu": val}


def gen_t7(rng, i):
    injecte = (i % 2 == 0)
    genre = (i // 2) % 6
    x, y = rng.sample(INDIVIDUS, 2)
    L = rng.randint(1, 5)
    ch = gen_chaine(rng, L)
    libres = [c for c in CLASSES if c not in ch]
    z = rng.choice(libres)
    regles = [phrase_regle(rng, ch[k], ch[k + 1]) for k in range(L)]
    if genre == 0:
        contexte = [phrase_fait(x, ch[0])] + regles
        final = phrase_non(x, ch[0]) if injecte else phrase_non(x, z)
    elif genre == 1:
        contexte = [phrase_fait(x, ch[0])] + regles
        final = phrase_non(x, ch[L]) if injecte else phrase_non(x, z)
    elif genre == 2:
        contexte = [phrase_non(x, ch[L] if injecte else z)] + regles
        final = phrase_fait(x, ch[0])
    elif genre == 3:
        j = rng.randrange(L)
        contexte = [phrase_fait(x, ch[0]), phrase_non(x, ch[L] if injecte else z)]
        contexte += [r for k, r in enumerate(regles) if k != j]
        final = regles[j]
    elif genre == 4:
        contexte = [phrase_fait(x, ch[0])] + regles + [phrase_fait(x, z)]
        autre = rng.choice([c for c in libres if c != z])
        final = f"Aucun {ch[L][0]} n'est un {z[0] if injecte else autre[0]}."
    else:
        pers = rng.sample(PERSONNES, 2)
        o = rng.choice(OBJETS)
        contexte = [phrase_va(rng, pers[0], rng.choice(LIEUX)), phrase_va(rng, pers[1], rng.choice(LIEUX)),
                    phrase_prend(rng, pers[0], o)]
        if injecte:
            final = rng.choice([phrase_pose(rng, pers[1], o), phrase_prend(rng, pers[1], o),
                                phrase_donne(rng, pers[1], o, pers[0])])
        else:
            final = rng.choice([phrase_pose(rng, pers[0], o), phrase_donne(rng, pers[0], o, pers[1])])
    extras = distracteurs_classes(rng, [c for c in libres if c != z], [t for t in INDIVIDUS if t not in (x, y)],
                                  rng.randint(1, 3))
    return {"phrases": melanger_insere(rng, contexte, extras), "final": final,
            "attendu": "contradiction" if injecte else "coherent", "genre": genre}


# ----------------------------------------------------------------------
# Évaluation
# ----------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def construire(inst):
    """Insère les phrases de contexte. Retourne (raisonneur, nb_erreurs, temps_cpu)."""
    r = rz.Raisonneur()
    erreurs, t0 = 0, time.process_time()
    for p in inst["phrases"]:
        if r.ajouter(p).statut not in ("ajoutee", "doublon"):
            erreurs += 1
    return r, erreurs, time.process_time() - t0


def evaluer_question(inst):
    r, erreurs, t_ins = construire(inst)
    t0 = time.process_time()
    rep = r.repondre(inst["question"])
    dt = time.process_time() - t0
    repondu = rep.valeur not in ("inconnu", "incompris")
    # contrôle indépendant : on relance le vérificateur nous-mêmes
    verifie = bool(repondu and rep.preuve and rz.verifier(rep.preuve, r.base)[0])
    attendu = inst["attendu"]
    juste = repondu and rep.valeur == attendu
    return {
        "erreurs_contexte": erreurs, "repondu": repondu,
        "correct": (juste and verifie and not erreurs) or (attendu == "inconnu" and not repondu and not erreurs and rep.valeur == "inconnu"),
        "hallucination": repondu and (rep.valeur != attendu or not verifie),
        "abstention": not repondu,
        "abstention_correcte": (not repondu) and attendu == "inconnu",
        "abstention_a_tort": (not repondu) and attendu != "inconnu",
        "cpu": dt, "cpu_insertion": t_ins, "n_phrases": len(inst["phrases"]),
        "obtenu": rep.valeur if not isinstance(rep.valeur, int) else int(rep.valeur),
        "attendu": attendu,
    }


def evaluer_contradiction(inst):
    r, erreurs, t_ins = construire(inst)
    t0 = time.process_time()
    res = r.ajouter(inst["final"])
    dt = time.process_time() - t0
    detecte = res.statut == "refusee" and bool(res.contradictions)
    # contrôle indépendant des deux preuves
    verifie = detecte and all(rz.verifier_contradiction(c, r.base)[0] for c in res.contradictions)
    injecte = inst["attendu"] == "contradiction"
    ok = (not erreurs) and ((injecte and detecte and verifie) or (not injecte and res.statut == "ajoutee"))
    return {
        "erreurs_contexte": erreurs, "repondu": detecte,
        "correct": ok,
        "hallucination": detecte and (not injecte or not verifie),   # fausse alerte ou preuve non vérifiée
        "abstention": False, "abstention_correcte": False,
        "abstention_a_tort": False,
        "manquee": injecte and not detecte,
        "cpu": dt, "cpu_insertion": t_ins, "n_phrases": len(inst["phrases"]) + 1,
        "obtenu": res.statut, "attendu": inst["attendu"],
    }


TACHES = [
    ("Tâche 1", "un fait suffit : où est X après plusieurs déplacements", gen_t1, evaluer_question),
    ("Tâche 2", "deux faits : où est l'objet porté (ou posé), qui l'a", gen_t2, evaluer_question),
    ("Tâche 3", "déduction : chaînes de règles de longueur 1 à 5", None, evaluer_question),
    ("Tâche 4", "oui/non avec négations (faits, contraposition, disjonction)", gen_t4, evaluer_question),
    ("Tâche 5", "questions sans réponse possible (doit s'abstenir)", gen_t5, evaluer_question),
    ("Tâche 6", "arithmétique exacte, nombres jusqu'à 10^12", gen_t6, evaluer_question),
    ("Tâche 7", "contradictions injectées (détection à l'insertion)", None, evaluer_contradiction),
]


def generer(num, rng, i):
    if num == 3:
        return gen_t3(rng, longueur=(i % 5) + 1)
    if num == 4:
        return gen_t4(rng, genre=i % 5)
    if num == 5:
        return gen_t5(rng, genre=i % 7)
    if num == 7:
        return gen_t7(rng, i)
    return TACHES[num - 1][2](rng)


def lancer(n, graine):
    resultats = []
    for num, (nom, desc, _, ev) in enumerate(TACHES, 1):
        rng = random.Random(graine * 100 + num)
        lignes, echecs = [], []
        for i in range(n):
            inst = generer(num, rng, i)
            m = ev(inst)
            lignes.append(m)
            if not m["correct"] or m["hallucination"]:
                echecs.append({"instance": i, "phrases": inst["phrases"],
                               "question": inst.get("question") or inst.get("final"),
                               "attendu": m["attendu"], "obtenu": m["obtenu"]})
        N = len(lignes)
        k = sum(m["correct"] for m in lignes)
        lo, hi = wilson(k, N)
        cpu = sum(m["cpu"] for m in lignes) / N
        resultats.append({
            "tache": nom, "description": desc, "n": N,
            "correct": k, "exactitude": k / N, "ic95": [lo, hi],
            "hallucinations": sum(m["hallucination"] for m in lignes),
            "taux_hallucination": sum(m["hallucination"] for m in lignes) / N,
            "abstentions": sum(m["abstention"] for m in lignes),
            "abstentions_correctes": sum(m["abstention_correcte"] for m in lignes),
            "abstentions_a_tort": sum(m["abstention_a_tort"] for m in lignes),
            "contradictions_manquees": sum(m.get("manquee", False) for m in lignes),
            "erreurs_contexte": sum(m["erreurs_contexte"] for m in lignes),
            "phrases_moy": sum(m["n_phrases"] for m in lignes) / N,
            "cpu_moyen_us": cpu * 1e6,
            "cpu_insertion_moyen_us": sum(m["cpu_insertion"] / max(1, m["n_phrases"]) for m in lignes) / N * 1e6,
            "energie_uJ": cpu * PUISSANCE_W * 1e6,
            "echecs": echecs[:5],
        })
        if num == 3:   # détail par longueur de chaîne
            par = {}
            for i, m in enumerate(lignes):
                par.setdefault((i % 5) + 1, []).append(m["correct"])
            resultats[-1]["par_longueur"] = {str(L): [sum(v), len(v)] for L, v in sorted(par.items())}
        if num == 4:
            noms = ["oui (chaîne)", "non (fait négatif)", "non (contraposition)",
                    "non (disjonction)", "oui (malgré négations)"]
            par = {}
            for i, m in enumerate(lignes):
                par.setdefault(noms[i % 5], []).append(m["correct"])
            resultats[-1]["par_genre"] = {k2: [sum(v), len(v)] for k2, v in par.items()}
        if num == 7:
            par = {}
            noms = ["fait contre fait", "fait contre chaîne", "fait ajouté après la négation",
                    "règle ajoutée en dernier", "disjonction ajoutée", "événement impossible"]
            for i, m in enumerate(lignes):
                par.setdefault(("injectée " if i % 2 == 0 else "témoin ") + noms[(i // 2) % 6], []).append(m["correct"])
            resultats[-1]["par_genre"] = {k2: [sum(v), len(v)] for k2, v in sorted(par.items())}
    return resultats


# ----------------------------------------------------------------------
# Limites : phrases libres et ce qu'en fait le système
# ----------------------------------------------------------------------
# (phrase, forme attendue d'un lecteur correct — None = doit être refusée)
LIBRES = [
    ("Le chat dort sur le canapé.", None),
    ("Marie et Jean vont au parc.", None),
    ("Marie va au parc, puis elle prend la clé.", None),
    ("Elle prend la pomme.", None),
    ("Il est mortel.", None),
    ("Socrate était un homme.", None),
    ("Socrate est très intelligent.", None),
    ("Si Socrate est un homme alors il est mortel.", None),
    ("Socrate est un homme et Platon aussi.", None),
    ("Jean a une pomme.", None),
    ("Jean mange la pomme.", None),
    ("Jean prend la pomme et part.", None),
    ("Marie a posé la pomme dans la cuisine.", None),
    ("Jean est plus grand que Marie.", None),
    ("Les oiseaux ne volent pas.", None),
    ("Médor aboie.", None),
    ("La plupart des chats sont gris.", None),
    ("Marie est née à Paris.", None),
    ("Quelle est la capitale de la France ?", None),
    ("Pourquoi le ciel est bleu ?", None),
    ("Combien de pommes a Jean ?", None),
    ("Où habite Marie ?", None),
    ("Socrate n'est-il pas mortel ?", None),
    ("Quel âge a Marie ?", None),
    ("Paul est le frère de Marie.", None),
    ("La pomme est rouge.", "discutable"),   # fait particulier lu comme règle générale
    ("Marie est en retard.", None),
    ("Jean est à l'heure.", None),
    ("Marie est allée dans la cuisine hier.", None),
    ("Jean prend le train.", None),
    ("Le livre est sur la table.", None),
    ("Socrate est un homme mortel.", ("est", "socrate", "homme"),),
    ("Marie rentre à la maison.", ("evt", "va", "marie", "maison")),
    ("Jean confie la pomme à Marie.", ("evt", "donne", "jean", "pomme", "marie")),
    ("Marie traverse le jardin.", ("evt", "va", "marie", "jardin")),
    ("Tout humain est mortel.", ("regle", "humain", "mortel")),
    ("Marie se rend dans la cuisine.", ("evt", "va", "marie", "cuisine")),
    ("Le ciel est bleu aujourd'hui.", None),
    ("Tout le monde est mortel.", None),
    ("Socrate est mortel.", ("est", "socrate", "mortel", True)),
    ("Marie va à Paris.", ("evt", "va", "marie", "paris")),
    ("marie va dans la cuisine", ("evt", "va", "marie", "cuisine")),
    ("Les chats sont des animaux", ("regle", "chat", "animal")),
    ("Combien font 7 plus 5 ?", "calcul"),
]


def tester_libres():
    out = []
    for phrase, attendu in LIBRES:
        a = rz.analyser(phrase)
        if a.genre == "incompris":
            obtenu, verdict = "refusée (« Je ne comprends pas »)", \
                ("refus correct" if attendu is None else "refus à tort")
        elif a.genre == "question":
            obtenu = f"question {a.forme[0]}"
            verdict = "lue" if attendu else "lue de travers"
            if attendu == "calcul":
                verdict = "correct"
            elif attendu is None:
                verdict = "MAL INTERPRÉTÉE (aurait dû être refusée)"
        else:
            obtenu = f"acceptée comme {a.forme}"
            if attendu is None:
                verdict = "MAL INTERPRÉTÉE (aurait dû être refusée)"
            elif attendu == "discutable":
                verdict = "DISCUTABLE (fait particulier lu comme règle générale)"
            elif a.forme == attendu:
                verdict = "correct"
            else:
                verdict = "interprétation différente"
        out.append({"phrase": phrase, "obtenu": obtenu, "verdict": verdict})
    return out


# ----------------------------------------------------------------------
# Rapport
# ----------------------------------------------------------------------
def pct(x):
    return f"{100 * x:.1f} %".replace(".", ",")


def fnum(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def rapport(res, libres, n, graine):
    L = []
    L.append("# Résultats du raisonneur d'Écho\n")
    L.append(f"Généré par `banc_raisonneur.py` (graine {graine}, {n} instances par tâche, "
             f"Python {platform.python_version()}). Aucun réseau de neurones, aucun poids, "
             "bibliothèque standard seulement.\n")
    L.append("## Avertissement\n")
    L.append("Les tâches sont **générées par nous**, dans le français contrôlé que le système "
             "comprend, et leurs réponses sont calculées par le générateur (un simulateur "
             "indépendant et des entiers Python). Un score élevé montre que **le moteur est "
             "correct**, pas que le système est intelligent ni qu'il comprend le français.\n")
    L.append("## Résultats par tâche\n")
    L.append("| Tâche | Description | Exactitude | IC 95 % | Hallucination | Abstention correcte | Abstention à tort |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res:
        if r["tache"] == "Tâche 5":
            ab = f"{pct(r['abstentions_correctes'] / r['n'])} ({r['abstentions_correctes']}/{r['n']})"
        else:
            ab = "sans objet"
        L.append(f"| {r['tache']} | {r['description']} | {pct(r['exactitude'])} ({r['correct']}/{r['n']}) "
                 f"| {pct(r['ic95'][0])} à {pct(r['ic95'][1])} "
                 f"| {pct(r['taux_hallucination'])} ({r['hallucinations']}/{r['n']}) "
                 f"| {ab} | {r['abstentions_a_tort']} |")
    L.append("")
    L.append("Définitions (calculées, pas supposées) :\n")
    L.append("- **Exactitude** : l'instance est juste (bonne réponse **avec preuve acceptée par le "
             "vérificateur**, ou bonne abstention pour la tâche 5, ou contradiction détectée "
             "avec ses deux preuves vérifiées / absence de fausse alerte pour la tâche 7). "
             "IC 95 % : intervalle de Wilson.")
    L.append("- **Hallucination** : une réponse a été donnée ET (elle est fausse OU sa preuve "
             "n'est pas acceptée par le vérificateur). Le banc relance lui-même `verifier` sur "
             "chaque preuve renvoyée. Tâche 7 : une alerte est une hallucination si c'est une "
             "fausse alerte (cas témoin) ou si ses preuves ne sont pas vérifiées.")
    L.append("- **Abstention correcte** : « Je ne sais pas » quand aucune réponse n'existe "
             "(tâche 5 uniquement). **Abstention à tort** : « Je ne sais pas » alors qu'une "
             "réponse existait.")
    L.append("- Tâche 7 : 50 % d'instances avec une contradiction injectée (6 types), 50 % de "
             "témoins cohérents. Une contradiction non détectée compte comme erreur d'exactitude "
             "(colonne ci-dessous), pas comme hallucination.\n")
    L.append("## Coût de calcul\n")
    L.append("| Tâche | Phrases de contexte (moy.) | CPU moyen par question (µs) | CPU moyen par phrase insérée (µs) | Énergie estimée par question (µJ) |")
    L.append("|---|---|---|---|---|")
    for r in res:
        L.append(f"| {r['tache']} | {fnum(r['phrases_moy'], 1)} | {fnum(r['cpu_moyen_us'], 1)} "
                 f"| {fnum(r['cpu_insertion_moyen_us'], 1)} | {fnum(r['energie_uJ'], 2)} |")
    L.append("")
    L.append(f"**ESTIMATION, pas une mesure** : l'énergie est calculée comme temps CPU "
             f"(`time.process_time`) × {PUISSANCE_W:.0f} W, en supposant {PUISSANCE_W:.0f} W par cœur "
             "occupé. Aucun wattmètre n'a été utilisé ; la valeur réelle dépend de la machine. "
             "Pour la tâche 7, la « question » est l'insertion de la dernière phrase (avec "
             "recherche de contradiction). Le temps de lecture des phrases de contexte est donné "
             "à part (par phrase insérée).\n")
    L.append("## Détails\n")
    for r in res:
        if "par_longueur" in r:
            L.append("**Tâche 3, par longueur de chaîne** : " + ", ".join(
                f"longueur {k} : {v[0]}/{v[1]}" for k, v in r["par_longueur"].items()) + "\n")
        if "par_genre" in r:
            L.append(f"**{r['tache']}, par type d'instance** :\n")
            L.append("| Type | Justes |\n|---|---|")
            for k, v in r["par_genre"].items():
                L.append(f"| {k} | {v[0]}/{v[1]} |")
            L.append("")
    r7 = res[-1]
    L.append(f"Tâche 7 : contradictions non détectées : {r7['contradictions_manquees']} ; "
             f"erreurs d'insertion du contexte (toutes tâches) : "
             f"{sum(r['erreurs_contexte'] for r in res)}.\n")
    echecs = [(r["tache"], e) for r in res for e in r["echecs"]]
    if echecs:
        L.append("### Échecs observés (jusqu'à 5 par tâche)\n")
        for t, e in echecs:
            L.append(f"- {t}, instance {e['instance']} : question/phrase « {e['question']} », "
                     f"attendu {e['attendu']}, obtenu {e['obtenu']}.")
        L.append("")
    else:
        L.append("Aucun échec sur les instances générées.\n")
    L.append("## Limites (à lire)\n")
    L.append("1. **Le système ne comprend que le français contrôlé** décrit dans `raisonneur.py` "
             "(une affirmation simple par phrase, sujet nominal ou propre, verbes listés). "
             "Tout le reste est refusé (« Je ne comprends pas cette phrase ») — ou, plus grave, "
             "parfois mal interprété (voir le tableau).")
    L.append("2. **Les benchmarks sont écrits par nous** avec les mêmes tournures que le parseur "
             "connaît ; 100 % ici prouve la correction du moteur, **pas une intelligence générale**. "
             "Aucune de ces tâches ne teste la compréhension de textes naturels.")
    L.append("3. **Le monde est modélisé de façon minimale** : une personne a un lieu, un objet est "
             "porté ou posé. Pas de pronoms (« elle »), pas de coordination (« et », « puis »), "
             "pas de négation d'événements, pas de temps verbal, pas de quantités, pas de relations "
             "(« frère de »).")
    L.append("4. **Logique limitée** : classes et individus uniquement (faits, règles « tout A est B », "
             "disjonctions « aucun A n'est B », négations de faits). Pas de relations à plusieurs "
             "arguments, pas de « certains », pas de « si… alors » général, pas de disjonction "
             "(« ou »). Pour une question sur une classe (« tout chien est-il un animal ? ») le "
             "système répond « oui » ou s'abstient, jamais « non ».")
    L.append("5. **Monde ouvert** : ne pas savoir n'est pas savoir que non. « Rex est-il un chien ? » "
             "reçoit « Je ne sais pas » si rien ne le prouve ni ne le réfute. C'est voulu (pas "
             "d'hallucination), mais c'est une abstention fréquente.")
    L.append("6. **Pluriels** : règle simple (retrait de s/x, -aux -> -al). Les pluriels irréguliers "
             "(« œil/yeux ») et l'accord des adjectifs au féminin (« mortelle ») ne sont pas gérés ; "
             "deux formes non reliées sont deux classes différentes.")
    L.append("7. **Vérificateur** : indépendant du démonstrateur (code séparé, rejeu propre des "
             "événements, évaluateur arithmétique propre), mais il partage avec lui l'analyseur de "
             "phrases : si l'analyseur lit mal une phrase, les deux sont d'accord sur une mauvaise "
             "lecture. La vérification garantit « la conclusion découle des phrases telles que lues », "
             "pas « les phrases ont été bien lues ».")
    L.append("8. **Cohérence** : une phrase contradictoire est refusée, la base reste cohérente. "
             "Les contradictions entre règles seules (« tout A est B » et « aucun A n'est B ») ne sont "
             "pas détectées tant qu'aucun individu n'appartient à A (classe vide cohérente).")
    L.append("9. **Estimation d'énergie** : voir plus haut ; un modèle de langage neuronal n'a pas été "
             "mesuré ici, aucune comparaison chiffrée n'est faite ou revendiquée.\n")
    L.append("### Phrases libres testées\n")
    L.append("Transparence : après un premier essai de ces phrases, j'ai durci l'analyseur "
             "(mots de quantité, « très », « à », verbes de relation…) pour que plusieurs "
             "d'entre elles soient refusées au lieu d'être mal lues. Les refus corrects "
             "ci-dessous sont donc en partie le résultat de ce réglage, pas une compréhension. "
             "Les cas « MAL INTERPRÉTÉE » sont ceux qui passent encore entre les mailles ; "
             "les « refus à tort » sont du français contrôlé raisonnable que le système ne "
             "connaît pas (verbes ou tournures absents de la liste).\n")
    L.append("| Phrase | Ce que fait le système | Verdict |\n|---|---|---|")
    for e in libres:
        L.append(f"| {e['phrase']} | {e['obtenu']} | {e['verdict']} |")
    mal = [e for e in libres if e["verdict"].startswith(("MAL", "DISCUTABLE")) or e["verdict"] == "interprétation différente"]
    refus_tort = [e for e in libres if e["verdict"] == "refus à tort"]
    L.append("")
    L.append(f"Sur {len(libres)} phrases libres : "
             f"{sum(1 for e in libres if e['verdict'] in ('refus correct', 'correct'))} traitées comme "
             f"un lecteur correct le ferait, {len(mal)} mal interprétées ou différemment, "
             f"{len(refus_tort)} refusées à tort.\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--graine", type=int, default=2024)
    args = ap.parse_args()
    res = lancer(args.n, args.graine)
    libres = tester_libres()
    for r in res:
        print(f"{r['tache']} exactitude {pct(r['exactitude'])}  hallucination "
              f"{pct(r['taux_hallucination'])}  abst.correcte {r['abstentions_correctes']}  "
              f"abst.à tort {r['abstentions_a_tort']}  CPU {r['cpu_moyen_us']:.1f} µs")
    with open(os.path.join(ICI, "resultats_raisonneur.md"), "w", encoding="utf-8") as f:
        f.write(rapport(res, libres, args.n, args.graine))
    with open(os.path.join(ICI, "resultats_raisonneur.json"), "w", encoding="utf-8") as f:
        json.dump({"graine": args.graine, "n_par_tache": args.n,
                   "hypothese_energie": f"{PUISSANCE_W} W par coeur occupe (estimation)",
                   "taches": res, "phrases_libres": libres}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
