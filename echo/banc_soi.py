#!/usr/bin/env python3
"""Banc d'essai du « soi » d'Écho : calibration, auto-prédiction, description.

Sorties : resultats_soi.md et resultats_soi.json.
Usage   : python3 banc_soi.py [--n-eval 2000] [--graine 2024] [--rapide]

Au plus 2 processus (anglais et français en parallèle). Standard library only.
Rappel : « soi » = métacognition mesurée, pas conscience.
"""
import argparse
import json
import lzma
import math
import multiprocessing
import os
import platform
import random
import sys
import time

import soi
from predicteur import Memoire

ICI = os.path.dirname(os.path.abspath(__file__))
CHEMIN_EN = os.path.join(ICI, "memoire.xz")
CHEMIN_VALID = os.path.join(ICI, "validation.txt")
CHEMIN_FR = os.path.join(ICI, "data", "francais.txt")
FRACTION_MEMOIRE = 0.10      # 10 % de la mémoire (anglais, et français), graine fixe
PUISSANCE_W = soi.PUISSANCE_W


# ----------------------------------------------------------------------
# 1. Calibration
# ----------------------------------------------------------------------
def _lignes(texte):
    return [l for l in texte.split("\n") if l.strip()]


def _sous_ensemble(paras, fraction, graine):
    k = max(1, round(len(paras) * fraction))
    idx = sorted(random.Random(graine).sample(range(len(paras)), k))
    return [paras[i] for i in idx]


def _prendre(paras, n_car):
    """Paragraphes pris dans l'ordre jusqu'à n_car caractères (joints par \\n, tronqués)."""
    morceaux, tot, k = [], 0, 0
    for p in paras:
        morceaux.append(p)
        k += 1
        tot += len(p) + 1
        if tot >= n_car:
            break
    return "\n".join(morceaux)[:n_car], k


def preparer_langue(lang, n_eval, graine):
    """Retourne (texte_memoire, eval, calibration, infos) pour 'en' ou 'fr'."""
    with lzma.open(CHEMIN_EN, "rt", encoding="utf-8") as f:
        en_complet = f.read()
    en_paras = _lignes(en_complet)
    sous_en = _sous_ensemble(en_paras, FRACTION_MEMOIRE, graine + 1)
    infos = {"memoire_en_paragraphes_total": len(en_paras),
             "memoire_en_caracteres_total": len(en_complet)}
    if lang == "en":
        with open(CHEMIN_VALID, encoding="utf-8") as f:
            valid = _lignes(f.read())
        fuites = [p for p in valid if p in en_complet]     # `in`, comme demandé
        sains = [p for p in valid if p not in en_complet]
        random.Random(graine + 2).shuffle(sains)
        ev, n_ev = _prendre(sains, n_eval)
        cal, n_cal = _prendre(sains[n_ev:], n_eval)
        memoire = "\n".join(sous_en) + "\n"
        infos.update({"valid_total": len(valid), "valid_fuites_exclues": len(fuites),
                      "memoire_paragraphes": len(sous_en), "eval_paragraphes": n_ev,
                      "cal_paragraphes": n_cal})
        return memoire, ev, cal, infos
    with open(CHEMIN_FR, encoding="utf-8") as f:
        fr = _lignes(f.read())
    tenues = [l for i, l in enumerate(fr) if i % 100 == 99]
    memo = [l for i, l in enumerate(fr) if i % 100 != 99]
    fr_complet = "\n".join(memo) + "\n"
    fuites = [l for l in tenues if l in fr_complet or l in en_complet]
    sains = [l for l in tenues if l not in fr_complet and l not in en_complet]
    random.Random(graine + 3).shuffle(sains)
    ev, n_ev = _prendre(sains, n_eval)
    cal, n_cal = _prendre(sains[n_ev:], n_eval)
    sous_fr = _sous_ensemble(memo, FRACTION_MEMOIRE, graine + 4)
    memoire = "\n".join(sous_en) + "\n" + "\n".join(sous_fr) + "\n"
    infos.update({"fr_lignes_total": len(fr), "fr_tenues": len(tenues),
                  "fr_tenues_fuites_exclues": len(fuites),
                  "memoire_paragraphes": len(sous_en) + len(sous_fr),
                  "memoire_fr_paragraphes": len(sous_fr),
                  "memoire_fr_caracteres_total": len(fr_complet),
                  "memoire_fr_paragraphes_total": len(memo),
                  "eval_paragraphes": n_ev, "cal_paragraphes": n_cal})
    return memoire, ev, cal, infos


def calibrer_langue(args):
    """Travail d'un processus : collecte sur calibration et évaluation."""
    lang, n_eval, graine = args
    t0 = time.process_time()
    texte, ev, cal, infos = preparer_langue(lang, n_eval, graine)
    m = Memoire(texte)
    infos["memoire_caracteres"] = m.taille()
    r_cal = soi.collecter(m, cal)
    r_ev = soi.collecter(m, ev)
    grp = []
    g = 0
    for ch in ev[:len(r_ev)]:
        grp.append(g)
        if ch == "\n":
            g += 1
    return {"lang": lang, "infos": infos, "recs_cal": r_cal, "recs_ev": r_ev, "grp_ev": grp,
            "n_eval_car": len(ev), "n_cal_car": len(cal),
            "cpu_s": time.process_time() - t0}


def bootstrap_ece(paires, groupes, graine, b=300):
    """IC95 de l'ECE par bootstrap PAR PARAGRAPHE (les caractères d'un paragraphe ne sont pas indépendants)."""
    rng = random.Random(graine)
    par = {}
    for p, g in zip(paires, groupes):
        par.setdefault(g, []).append(p)
    blocs = list(par.values())
    v = []
    for _ in range(b):
        tir = []
        for _ in blocs:
            tir += blocs[rng.randrange(len(blocs))]
        v.append(soi.ece(tir))
    v.sort()
    return v[int(0.025 * b)], v[int(0.975 * b)]


def resume_evaluation(recs_ev, groupes, T, graine):
    p = soi.confiances(recs_ev, T)
    lo, hi = bootstrap_ece(p, groupes, graine)
    return {"ece": soi.ece(p), "ece_ic95": [lo, hi], "brier": soi.brier_top1(p),
            "biais": soi.biais_confiance(p), "conf_moy": sum(c for c, _ in p) / len(p),
            "exactitude": soi.exactitude(p), "bpc": soi.nll_bits(recs_ev, T),
            "table": soi.fiabilite(p)}


def partie_calibration(res_langues, graine):
    """T (Brier) : critère retenu ; T (NLL) : variante donnée à titre de comparaison."""
    out = {}
    pool_cal = []
    for r in res_langues:
        pool_cal += r["recs_cal"]
    T_pool = soi.ajuster_temperature(pool_cal, "brier")
    out["T_pool"] = T_pool
    for r in res_langues:
        lang, g = r["lang"], r["grp_ev"]
        T_l = soi.ajuster_temperature(r["recs_cal"], "brier")
        T_n = soi.ajuster_temperature(r["recs_cal"], "nll")
        out["T_" + lang] = T_l
        out["T_nll_" + lang] = T_n
        av = resume_evaluation(r["recs_ev"], g, 1.0, graine)
        ap_l = resume_evaluation(r["recs_ev"], g, T_l, graine)
        ap_p = resume_evaluation(r["recs_ev"], g, T_pool, graine)
        ap_n = resume_evaluation(r["recs_ev"], g, T_n, graine)
        out[lang] = {
            "n_eval_car": r["n_eval_car"], "n_cal_car": r["n_cal_car"],
            "ece_avant": av["ece"], "ece_avant_ic95": av["ece_ic95"],
            "brier_avant": av["brier"], "biais": av["biais"],
            "conf_moy_avant": av["conf_moy"], "exactitude": av["exactitude"],
            "bpc_avant": av["bpc"], "table_avant": av["table"],
            "ece_apres_T_langue": ap_l["ece"], "ece_apres_T_langue_ic95": ap_l["ece_ic95"],
            "brier_apres_T_langue": ap_l["brier"], "biais_apres_T_langue": ap_l["biais"],
            "bpc_apres_T_langue": ap_l["bpc"], "table_apres_T_langue": ap_l["table"],
            "ece_apres_T_commun": ap_p["ece"], "brier_apres_T_commun": ap_p["brier"],
            "biais_apres_T_commun": ap_p["biais"], "bpc_apres_T_commun": ap_p["bpc"],
            "ece_apres_T_nll": ap_n["ece"], "brier_apres_T_nll": ap_n["brier"],
            "biais_apres_T_nll": ap_n["biais"], "bpc_apres_T_nll": ap_n["bpc"],
            "ece_cal_avant": soi.ece(soi.confiances(r["recs_cal"], 1.0)),
            "ece_cal_apres": soi.ece(soi.confiances(r["recs_cal"], T_l)),
            "infos": r["infos"], "cpu_s": r["cpu_s"],
        }
    return out


# ----------------------------------------------------------------------
# 2. Auto-prédiction
# ----------------------------------------------------------------------
# Phrases libres supplémentaires (écrites ici) ; on attend un REFUS de la part d'un
# lecteur prudent. Si le système les accepte, on compte « mal lu » (convention :
# un fait particulier lu comme règle générale est une mauvaise lecture).
EXTRA_LIBRES = [
    "Le soleil se couche à l'ouest.", "Mon voisin répare sa voiture tous les samedis.",
    "Il pleut depuis ce matin.", "Les enfants jouent dans la cour.",
    "Pierre et Paul sont frères.", "Cette maison appartient à ma tante.",
    "Nous irons au cinéma demain soir.", "Sophie lit un roman policier.",
    "Le train de Lyon part à huit heures.", "J'ai faim.",
    "Qui a gagné le match ?", "Quand est-ce que Jean arrive ?",
    "Comment va Marie ?", "Pourquoi Paul est-il parti ?",
    "Quel temps fait-il à Paris ?", "Combien coûte ce livre ?",
    "Est-ce que tu viens avec nous ?", "Peux-tu fermer la porte ?",
    "Qu'est-ce que Marie a mangé ?", "Qui est le président ?",
    "Écris-moi un poème sur la mer.", "Traduis cette phrase en anglais.",
    "Bonjour, comment allez-vous ?", "Merci beaucoup.",
    "Résume le texte suivant.", "Dis-moi une blague.",
    "Le café est chaud.", "La mer est calme ce matin.",
    "Le facteur passe à midi.", "Anne n'aime pas les épinards.",
    "Tous les matins, Luc boit un café.", "Les étoiles brillent dans le ciel.",
    "Mon frère est plus grand que moi.", "Il faut que tu partes.",
    "Léa a quitté la ville la semaine dernière.", "Paul court très vite.",
    "Hugo ne dit jamais la vérité.", "La Terre tourne autour du Soleil.",
    "Les poissons nagent dans la rivière.", "Aucun homme n'est une île.",
    "Quelqu'un a pris mon stylo.", "Personne n'est venu.",
    "Marie a trois chats et deux chiens.", "Luc travaille chez Renault.",
    "Que fait Marie dans la cuisine ?", "Est-il vrai que Jean est parti ?",
    "Le chat est-il sur le toit ?", "Quelle heure est-il ?",
    "Jean est le père de Paul.", "Julie est médecin.",
    "La Seine traverse Paris.", "Clara écoute de la musique en marchant.",
    "Paul est plus rapide que Luc.", "Donne-moi la clé.",
    "Où se trouve la gare ?", "Combien de personnes sont dans la cuisine ?",
    "Anne chante et Luc danse.", "Les hommes sont mortels parce que tout vit puis meurt.",
    "Emma est arrivée en avance.", "On dit que Léa est partie.",
    "Un jour, Marc ira sur la Lune.", "La clé est dans la poche de Jean.",
    "Tout est possible.", "Rien n'est certain.",
    "Où est passé mon livre ?", "Que penses-tu de Paris ?",
]


def construire_demandes(graine, n_total=300):
    """Mélange fixe : phrases contrôlées, sans réponse, phrases libres, arithmétique.

    Chaque demande : {type, contexte[], requete, attendu}. attendu : valeur
    attendue, "inconnu" (doit s'abstenir), "refus" (doit être refusée), ou
    ("forme", F) (l'analyseur doit la lire comme F).
    """
    import banc_raisonneur as br
    rng = random.Random(graine)
    libres = []
    for phrase, att in br.LIBRES:
        libres.append({"type": "libre", "contexte": [], "requete": phrase,
                       "attendu": "refus" if att is None else ("forme", att)})
    for phrase in EXTRA_LIBRES:
        libres.append({"type": "libre", "contexte": [], "requete": phrase, "attendu": "refus"})
    n_libres = len(libres)
    n_ctrl_par = 22
    gens = [("t1 position", br.gen_t1), ("t2 objet", br.gen_t2), ("t3 déduction", br.gen_t3),
            ("t4 oui/non", br.gen_t4), ("t6 arithmétique", br.gen_t6)]
    demandes = []
    for nom, g in gens:
        for _ in range(n_ctrl_par):
            inst = g(rng)
            demandes.append({"type": "contrôlée (%s)" % nom, "contexte": inst["phrases"],
                             "requete": inst["question"], "attendu": inst["attendu"]})
    n_sans = max(0, n_total - len(demandes) - n_libres)
    for i in range(n_sans):
        inst = br.gen_t5(rng, genre=i % 7)
        demandes.append({"type": "sans réponse", "contexte": inst["phrases"],
                         "requete": inst["question"], "attendu": "inconnu",
                         "genre": i % 7})
    demandes += libres
    return demandes


def classer_resultat(demande, genre, detail):
    """Issue réelle : compris / incompris / abstenu / faux."""
    att = demande["attendu"]
    if genre == "incompris":
        return "incompris"
    if genre == "abstention":
        return "abstenu"
    if genre == "reponse":
        if isinstance(att, tuple) and att[1] == "calcul":
            return "compris"          # question de calcul : la valeur est vérifiée par le banc du raisonneur
        if att in ("inconnu", "refus") or isinstance(att, tuple):
            return "faux"
        return "compris" if detail == att else "faux"
    # genre == "accepte" : une affirmation a été insérée
    if att == "refus":
        return "faux"
    if isinstance(att, tuple):
        return "compris" if detail == att[1] else "faux"
    return "faux"


CLASSES_ISSUE = ["compris", "incompris", "abstenu", "faux"]
CORRESPONDANCE = {soi.PRED_COMPRENDRE: "compris", soi.PRED_NON: "incompris",
                  soi.PRED_ABSTENIR: "abstenu"}


def partie_auto_prediction(graine):
    demandes = construire_demandes(graine)
    t_pred = t_exec = 0.0
    lignes = []
    for d in demandes:
        t0 = time.process_time()
        pred, raison = soi.estimer_competence(d["contexte"], d["requete"])
        t1 = time.process_time()
        genre, detail = soi.executer(d["contexte"], d["requete"])
        t2 = time.process_time()
        t_pred += t1 - t0
        t_exec += t2 - t1
        issue = classer_resultat(d, genre, detail)
        lignes.append({"type": d["type"], "requete": d["requete"], "pred": pred,
                       "raison": raison, "issue": issue, "genre": d.get("genre")})
    n = len(lignes)
    conf = {p: {c: 0 for c in CLASSES_ISSUE} for p in soi.PREDICTIONS}
    for l in lignes:
        conf[l["pred"]][l["issue"]] += 1
    justes = sum(1 for l in lignes if CORRESPONDANCE[l["pred"]] == l["issue"])
    compte_issue = {c: sum(1 for l in lignes if l["issue"] == c) for c in CLASSES_ISSUE}
    maj = max(compte_issue.values())
    nb_comp = sum(conf[soi.PRED_COMPRENDRE].values())
    prec = conf[soi.PRED_COMPRENDRE]["compris"] / nb_comp if nb_comp else None
    faux_conf = nb_comp - conf[soi.PRED_COMPRENDRE]["compris"]
    par_type = {}
    for l in lignes:
        t = l["type"].split(" (")[0]
        e = par_type.setdefault(t, {"n": 0, "justes": 0})
        e["n"] += 1
        e["justes"] += CORRESPONDANCE[l["pred"]] == l["issue"]
    par_sous = {}
    for l in lignes:
        e = par_sous.setdefault(l["type"], {"n": 0, "justes": 0})
        e["n"] += 1
        e["justes"] += CORRESPONDANCE[l["pred"]] == l["issue"]
    erreurs = [l for l in lignes if CORRESPONDANCE[l["pred"]] != l["issue"]]
    return {"n": n, "justes": justes, "exactitude": justes / n,
            "exactitude_majoritaire": maj / n, "compte_issue": compte_issue,
            "matrice": conf, "precision_comprendre": prec, "faux_confiants": faux_conf,
            "par_type": par_type, "par_sous_type": par_sous,
            "erreurs": erreurs[:40], "n_erreurs": len(erreurs),
            "cpu_prediction_moy_us": 1e6 * t_pred / n, "cpu_execution_moy_us": 1e6 * t_exec / n,
            "graine": graine}


# ----------------------------------------------------------------------
# Rapport
# ----------------------------------------------------------------------
def f3(x, d=4):
    return ("%." + str(d) + "f") % x


def table_fiabilite(tab):
    L = ["| Seau de confiance | N | Confiance moy. | Exactitude top-1 | Écart (conf. − exact.) |",
         "|---|---:|---:|---:|---:|"]
    for s in tab:
        if s["n"]:
            L.append("| %s | %d | %s | %s | %+.3f |" % (
                s["seau"], s["n"], f3(s["conf_moy"], 3), f3(s["exactitude"], 3),
                s["conf_moy"] - s["exactitude"]))
        else:
            L.append("| %s | 0 | - | - | - |" % s["seau"])
    return L


def sens(b):
    return ("TROP SÛR DE LUI (surconfiant)" if b > 0.005
            else "TROP MODESTE (sous-confiant)" if b < -0.005 else "à peu près calibré")


def rapport(C, A, M, cpu, duree, params):
    L = ["# Résultats du « soi » d'Écho (auto-modèle mesuré)", ""]
    L.append("Généré par `banc_soi.py` (graine %d, Python %s). Bibliothèque standard seulement, aucun poids appris."
             % (params["graine"], platform.python_version()))
    L += ["", "## Avertissement : ce n'est PAS de la conscience", "",
          "Le « soi » ci-dessous désigne de la **métacognition mesurable** : Écho sait (en chiffres) quand ses"
          " probabilités sont fiables, prévoit s'il va savoir traiter une demande, et se décrit avec des nombres"
          " lus dans ses fichiers de résultats. Ce sont des compteurs et des règles écrites à la main."
          " Aucune expérience subjective, aucune sensibilité, aucune preuve de conscience n'est revendiquée"
          " ni suggérée.", ""]
    L += ["## 1. Calibration d'Écho-P", "",
          "Pour chaque caractère à prédire, on prend le meilleur candidat (top-1) et sa probabilité (la confiance)."
          " Calibré = quand il dit 0,7, il a raison 70 % du temps.", "",
          "**Protocole.** Mémoire = %d %% de la bibliothèque, tirée au hasard par paragraphes (graine fixe %d) ;"
          " pour le français, 10 %% de l'anglais + 10 %% des lignes françaises (graine fixe) — pas les mémoires"
          " complètes, pour tenir le temps de calcul. Évaluation : ~%d caractères par langue ; calibration : un"
          " AUTRE échantillon de ~%d caractères, paragraphes disjoints de l'évaluation."
          % (int(FRACTION_MEMOIRE * 100), params["graine"] + 1, params["n_eval"], params["n_eval"]), ""]
    for lang, nom in (("en", "anglais"), ("fr", "français")):
        if lang not in C:
            continue
        c = C[lang]
        i = c["infos"]
        if lang == "en":
            L.append("- Anglais : mémoire %s paragraphes (%s caractères) sur %s ; `validation.txt` : %d paragraphes,"
                     " **%d exclus car présents mot pour mot dans la mémoire** (test `in`) ; évaluation %d caractères"
                     " (%d paragraphes), calibration %d caractères (%d paragraphes)."
                     % (soi._fmt(i["memoire_paragraphes"]), soi._fmt(i["memoire_caracteres"]),
                        soi._fmt(i["memoire_en_paragraphes_total"]), i["valid_total"], i["valid_fuites_exclues"],
                        c["n_eval_car"], i["eval_paragraphes"], c["n_cal_car"], i["cal_paragraphes"]))
        else:
            L.append("- Français : mémoire %s paragraphes (%s caractères) ; %d lignes tenues à l'écart (1 sur 100),"
                     " %d exclues car présentes dans la mémoire ; évaluation %d caractères (%d lignes), calibration"
                     " %d caractères (%d lignes)."
                     % (soi._fmt(i["memoire_paragraphes"]), soi._fmt(i["memoire_caracteres"]), i["fr_tenues"],
                        i["fr_tenues_fuites_exclues"], c["n_eval_car"], i["eval_paragraphes"],
                        c["n_cal_car"], i["cal_paragraphes"]))
    L.append("")
    L += ["### Résumé (échantillon d'évaluation)", "",
          "| Langue | Exactitude top-1 | Confiance moy. | Biais | Sens | ECE brut [IC95 bootstrap] | Brier top-1 | bpc |",
          "|---|---:|---:|---:|---|---:|---:|---:|"]
    for lang, nom in (("en", "anglais"), ("fr", "français")):
        if lang in C:
            c = C[lang]
            L.append("| %s | %s | %s | %+.4f | %s | %s [%s ; %s] | %s | %s |" % (
                nom, f3(c["exactitude"], 3), f3(c["conf_moy_avant"], 3), c["biais"], sens(c["biais"]),
                f3(c["ece_avant"]), f3(c["ece_avant_ic95"][0]), f3(c["ece_avant_ic95"][1]),
                f3(c["brier_avant"]), f3(c["bpc_avant"], 3)))
    L.append("")
    for lang, nom in (("en", "anglais"), ("fr", "français")):
        if lang in C:
            L += ["### Table de fiabilité brute — %s (évaluation)" % nom, ""] + table_fiabilite(C[lang]["table_avant"]) + [""]
    L += ["## 1bis. Correction paramétrique : une température", "",
          "Les probabilités sont élevées à la puissance 1/T puis renormalisées. T est **1 réel ajusté** (recherche en"
          " section dorée, critère : bits par caractère sur l'échantillon de CALIBRATION, jamais sur l'évaluation)."
          " Le candidat en tête ne change pas, donc l'exactitude top-1 est inchangée.", "",
          "| Réglage | T | ECE éval. avant | ECE éval. après | Brier avant | Brier après | Biais après | bpc avant | bpc après |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for lang, nom in (("en", "anglais"), ("fr", "français")):
        if lang in C:
            c = C[lang]
            L.append("| %s, T propre à la langue | %s | %s | %s | %s | %s | %+.4f | %s | %s |" % (
                nom, f3(C["T_" + lang], 3), f3(c["ece_avant"]), f3(c["ece_apres_T_langue"]),
                f3(c["brier_avant"]), f3(c["brier_apres_T_langue"]), c["biais_apres_T_langue"],
                f3(c["bpc_avant"], 3), f3(c["bpc_apres_T_langue"], 3)))
    for lang, nom in (("en", "anglais"), ("fr", "français")):
        if lang in C:
            c = C[lang]
            L.append("| %s, T commun (calibration EN+FR réunies) | %s | %s | %s | %s | %s | %+.4f | %s | %s |" % (
                nom, f3(C["T_pool"], 3), f3(c["ece_avant"]), f3(c["ece_apres_T_commun"]),
                f3(c["brier_avant"]), f3(c["brier_apres_T_commun"]), c["biais_apres_T_commun"],
                f3(c["bpc_avant"], 3), f3(c["bpc_apres_T_commun"], 3)))
    L.append("")
    for lang, nom in (("en", "anglais"), ("fr", "français")):
        if lang in C:
            c = C[lang]
            L.append("Contrôle sur l'échantillon de calibration (où T a été ajusté) — %s : ECE %s -> %s ; sur l'évaluation, IC95 de l'ECE après T : [%s ; %s]."
                     % (nom, f3(c["ece_cal_avant"]), f3(c["ece_cal_apres"]),
                        f3(c["ece_apres_T_langue_ic95"][0]), f3(c["ece_apres_T_langue_ic95"][1])))
    L.append("")
    for lang, nom in (("en", "anglais"), ("fr", "français")):
        if lang in C:
            L += ["### Table de fiabilité après température (T de la langue) — %s (évaluation)" % nom, ""] \
                 + table_fiabilite(C[lang]["table_apres_T_langue"]) + [""]
    # auto-prédiction
    L += ["## 2. Auto-prédiction de compétence", "",
          "Avant d'inférer, Écho regarde la demande (analyse seule, entités connues dans le contexte) et prédit"
          " « %s », « %s » ou « %s ». Puis on lance le vrai traitement. Issues : **compris** (bonne réponse vérifiée, ou"
          " affirmation lue comme attendu), **incompris** (refus « je ne comprends pas »), **abstenu** (« je ne sais"
          " pas »), **faux** (réponse fausse, ou phrase acceptée mais mal lue).  Une prédiction est juste quand elle"
          " correspond à l'issue ; « faux » n'est jamais prédit (le système ne sait pas qu'il lit de travers)."
          % tuple(soi.PREDICTIONS), "",
          "Mélange fixe (graine %d) de %d demandes : %s." % (
              A["graine"], A["n"], ", ".join("%d %s" % (v["n"], k) for k, v in sorted(A["par_type"].items()))), "",
          "### Matrice de confusion (lignes = prédiction, colonnes = issue réelle)", "",
          "| Prédiction \\ Issue | compris | incompris | abstenu | faux | Total |", "|---|---:|---:|---:|---:|---:|"]
    for p in soi.PREDICTIONS:
        row = A["matrice"][p]
        L.append("| %s | %d | %d | %d | %d | %d |" % (p, row["compris"], row["incompris"], row["abstenu"],
                                                     row["faux"], sum(row.values())))
    ci = A["compte_issue"]
    L.append("| **Total issue** | %d | %d | %d | %d | %d |" % (ci["compris"], ci["incompris"], ci["abstenu"], ci["faux"], A["n"]))
    L += ["", "- **Exactitude de l'auto-prédiction : %s (%d/%d).**" % (soi._pct(A["exactitude"]), A["justes"], A["n"]),
          "- Référence « toujours prédire l'issue la plus fréquente » : %s." % soi._pct(A["exactitude_majoritaire"]),
          "- Précision de « je vais comprendre » : %s ; %d fois j'ai annoncé « je vais comprendre » et c'était faux ou abstenu."
          % (soi._pct(A["precision_comprendre"]) if A["precision_comprendre"] is not None else "n/a", A["faux_confiants"]),
          "- Coût moyen : prédiction %.1f µs, exécution %.1f µs (CPU)." % (A["cpu_prediction_moy_us"], A["cpu_execution_moy_us"]),
          "", "| Type de demande | N | Prédictions justes |", "|---|---:|---:|"]
    for k, v in sorted(A["par_sous_type"].items()):
        L.append("| %s | %d | %d/%d (%s) |" % (k, v["n"], v["justes"], v["n"], soi._pct(v["justes"] / v["n"])))
    L += ["", "Erreurs d'auto-prédiction (jusqu'à 40 sur %d) :" % A["n_erreurs"], "",
          "| Type | Demande | Prédiction | Issue | Raison |", "|---|---|---|---|---|"]
    for e in A["erreurs"]:
        L.append("| %s | %s | %s | %s | %s |" % (e["type"], e["requete"].replace("|", "/"), e["pred"], e["issue"], e["raison"]))
    # mémoire
    L += ["", "## 3. Auto-description", "",
          "`python3 soi.py --qui` relit `resultats_banc.json`, `resultats_raisonneur.json`, `resultats_soi.json`, la taille des"
          " fichiers de mémoire et la liste des modules présents ; aucun nombre n'est écrit en dur dans `soi.py`.", "",
          "Mémoire mesurée : %s paragraphes, %s caractères (anglais %s p., français %s p.)." % (
              soi._fmt(M["paragraphes_total"]), soi._fmt(M["caracteres_total"]),
              soi._fmt(M["en"]["paragraphes"]), soi._fmt(M["fr"]["paragraphes"])), ""]
    # coût
    L += ["## Coût (ESTIMATION)", "",
          "**Estimation, pas une mesure** : énergie = temps CPU (`time.process_time`) × %g W (10 W supposés par cœur"
          " occupé ; aucun wattmètre)." % PUISSANCE_W, "",
          "| Étape | CPU (s) | Énergie estimée (J) |", "|---|---:|---:|"]
    for nom, s in cpu:
        L.append("| %s | %.1f | %.1f |" % (nom, s, s * PUISSANCE_W))
    tot = sum(s for _, s in cpu)
    L.append("| **Total CPU (tous processus)** | **%.1f** | **%.1f** |" % (tot, tot * PUISSANCE_W))
    L += ["", "Durée réelle totale : %.0f s (au plus 2 processus en parallèle)." % duree, ""]
    L += ["## Limites (à lire)", "",
          "1. **Ce n'est pas une conscience.** Calibrer ses probabilités et prédire son propre comportement est de la"
          " métacognition fonctionnelle ; rien ne dit qu'il y ait une expérience derrière, et rien dans le code ne le suggère.",
          "2. **Calibration sur un seul échantillon par langue** (~%d caractères évalués, ~%d pour calibrer) : voir les IC95"
          " bootstrap de l'ECE ; des différences de quelques millièmes ne sont pas significatives. Les caractères voisins"
          " d'un même paragraphe ne sont pas indépendants, l'IC est donc trop étroit." % (params["n_eval"], params["n_eval"]),
          "3. **Mémoires réduites (10 %)** : la calibration de la mémoire complète (15,7 M caractères) n'a pas été mesurée ;"
          " T dépend probablement de la taille de la mémoire et de la langue.",
          "4. **Un seul réel ajusté** par réglage (T). Il corrige un biais global de confiance, pas les erreurs locales ;"
          " le test « T commun » montre à quel point il dépend de la langue.",
          "5. **L'auto-prédiction ne voit pas ses erreurs de lecture** : le système ne prédit jamais « faux » ; les phrases"
          " mal lues passent l'analyseur sans alerte. Elle ne prévoit pas non plus les abstentions qui dépendent d'un"
          " raisonnement (réciproque non valide, porteur jamais localisé…), car on ne lance pas d'inférence avant.",
          "6. **Le mélange de demandes est écrit par nous** : les phrases contrôlées sont celles que l'analyseur connaît (les"
          " tâches viennent de `banc_raisonneur.py`), les phrases libres ont été choisies après avoir vu le comportement du"
          " raisonneur sur une partie d'entre elles (44 sont celles de `resultats_raisonneur.md`). L'exactitude de l'auto-prédiction"
          " dépend directement de ce mélange ; ce n'est pas une garantie sur du français réel.",
          "7. **Prédiction en partie tautologique** : l'analyseur utilisé pour prévoir « je ne vais pas comprendre » est le même que"
          " celui du traitement réel ; les prédictions de non-compréhension sont donc justes par construction. La partie informative"
          " est la distinction « comprendre / m'abstenir » et les cas « faux » non détectés.",
          "8. Énergie : estimation à 10 W/cœur, pas de mesure. Aucune comparaison avec un modèle neuronal n'est faite.", ""]
    return "\n".join(L)


def stats_memoire(infos_en, infos_fr):
    en_p, en_c = infos_en["memoire_en_paragraphes_total"], infos_en["memoire_en_caracteres_total"]
    fr_p, fr_c = infos_fr["memoire_fr_paragraphes_total"], infos_fr["memoire_fr_caracteres_total"]
    return {"en": {"paragraphes": en_p, "caracteres": en_c, "fichier": "memoire.xz"},
            "fr": {"paragraphes": fr_p, "caracteres": fr_c, "fichier": "memoire_fr.xz"},
            "paragraphes_total": en_p + fr_p, "caracteres_total": en_c + fr_c,
            "langues": ["anglais (memoire.xz)", "français (memoire_fr.xz, data/francais.txt)"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--graine", type=int, default=2024)
    args = ap.parse_args()
    t_debut = time.time()
    cpu = []
    print("Calibration : anglais et français en parallèle (2 processus)...", flush=True)
    t0 = time.process_time()
    with multiprocessing.Pool(2) as pool:
        res = pool.map(calibrer_langue, [("en", args.n_eval, args.graine), ("fr", args.n_eval, args.graine)])
    for r in res:
        cpu.append(("Calibration %s (collecte, mémoire 10 %%)" % r["lang"], r["cpu_s"]))
    t1 = time.process_time()
    C = partie_calibration(res, args.graine)
    cpu.append(("Ajustement des températures + tables + bootstrap", time.process_time() - t1))
    print("Auto-prédiction...", flush=True)
    t2 = time.process_time()
    A = partie_auto_prediction(args.graine)
    cpu.append(("Auto-prédiction (300 demandes, prédiction + exécution)", time.process_time() - t2))
    M = stats_memoire(res[0]["infos"], res[1]["infos"])
    duree = time.time() - t_debut
    params = {"graine": args.graine, "n_eval": args.n_eval}
    md = rapport(C, A, M, cpu, duree, params)
    with open(os.path.join(ICI, "resultats_soi.md"), "w", encoding="utf-8") as f:
        f.write(md)
    sortie = {"graine": args.graine, "n_eval_car": args.n_eval, "fraction_memoire": FRACTION_MEMOIRE,
              "hypothese_energie": "%g W par coeur occupe (estimation)" % PUISSANCE_W,
              "memoire": M, "calibration": C, "auto_prediction": A,
              "cpu_s": cpu, "cpu_total_s": sum(s for _, s in cpu),
              "energie_estimee_J": sum(s for _, s in cpu) * PUISSANCE_W, "duree_mur_s": duree}
    with open(os.path.join(ICI, "resultats_soi.json"), "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=1)
    print("Écrit resultats_soi.md et resultats_soi.json (%.0f s)." % duree)


if __name__ == "__main__":
    main()
