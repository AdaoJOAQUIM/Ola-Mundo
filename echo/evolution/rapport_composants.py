"""Sections du rapport pour les cinq composants (soi, objectif, coût, convergence, émergence)."""
import itertools
import json
import os

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))


def _perm(a, b):
    tout = list(a) + list(b)
    n = len(a)
    obs = abs(np.mean(a) - np.mean(b))
    plus = tot = 0
    for idx in itertools.combinations(range(len(tout)), n):
        ga = [tout[i] for i in idx]
        gb = [tout[i] for i in range(len(tout)) if i not in idx]
        tot += 1
        plus += abs(np.mean(ga) - np.mean(gb)) >= obs - 1e-12
    return plus / tot


def _a_cpu(points, niveau, cle):
    """Dernier point dont le CPU cumulé <= niveau (meilleur jusqu'ici à ce CPU)."""
    ok = [p for p in points if p["cpu_s"] <= niveau + 1e-9]
    return ok[-1][cle] if ok else None


def construire(md_tableau):
    chemin = os.path.join(ICI, "archive", "principal_complet_resume.json")
    chemin_abl = os.path.join(ICI, "archive", "ablation_complet.json")
    if not (os.path.exists(chemin) and os.path.exists(chemin_abl)):
        return [], {}
    r = json.load(open(chemin))
    abl = json.load(open(chemin_abl))
    md, out = [], {"principal_complet": {k: r[k] for k in r if k not in ("journal_rejets",)}}
    ch = r["champion"]
    md += ["## 9. Moteur complet : cinq composants ajoutés", "",
           "Les garde-fous et le protocole A/B/C sont inchangés (fichiers épinglés non touchés). "
           "Les runs de cette partie sont jugés sur A et B seulement : C a déjà servi une fois et n'est pas rouvert. "
           "Run complet : %d évaluations, %d générations, %.0f s de CPU cumulé, champion (B) A = %.4f, B = %.4f. "
           "Rappel : départ A = %.4f, B = %.4f." % (r["evals"], r["generations"], r["cpu_total_s"], ch["fit_A"], ch["fit_B"],
                                                    r["seed_A"], r["seed_B"]), ""]
    # --- ablation à CPU égal ---
    runs = abl["runs"]
    niv = [25 * i for i in range(1, int(abl["budget_cpu_s"] // 25) + 1)]
    lignes = []
    for n in niv:
        row = [n]
        for mode in ("simple", "complet"):
            vals = [_a_cpu(x["points"], n, "best_A") for x in runs if x["mode"] == mode]
            valsB = [_a_cpu(x["points"], n, "best_B") for x in runs if x["mode"] == mode]
            if None in vals:
                row += ["-", "-"]
            else:
                row += ["%.4f ± %.4f" % (np.mean(vals), np.std(vals)), "%.4f ± %.4f" % (np.mean(valsB), np.std(valsB))]
        lignes.append(row)
    fin = {m: [x for x in runs if x["mode"] == m] for m in ("simple", "complet")}
    pA = _perm([x["best_A"] for x in fin["simple"]], [x["best_A"] for x in fin["complet"]])
    pB = _perm([x["champion_B"] for x in fin["simple"]], [x["champion_B"] for x in fin["complet"]])
    out["ablation_complet"] = {"niveaux_cpu": niv, "p_A": pA, "p_B": pB,
                               "evals_simple": [x["evals"] for x in fin["simple"]],
                               "evals_complet": [x["evals"] for x in fin["complet"]]}
    # --- Soi ---
    socs = [x["soi"] for x in fin["complet"] if x.get("soi")]
    s = r["soi"]
    sautes = sum(x["enfants_sautes"] for x in socs)
    evalues = sum(x["evals"] for x in fin["complet"])
    md += ["### 9.1 Soi (auto-modèle)", "",
           "Un petit modèle logistique (numpy, Newton + pénalité L2) prédit, avant l'évaluation, si l'enfant bat son parent. "
           "Variables : type des mutations, rang du parent, variation de taille, taux de succès passé des opérateurs, nombre de mutations. "
           "La calibration suit `echo/soi.py` : une température ajustée par section dorée sur les 25 % d'exemples les plus récents, et l'ECE importée de `soi.py`. "
           "On saute un enfant si sa probabilité prédite est inférieure à 0,35 fois le taux de succès observé, sauf exploration (epsilon = 0,1).", "",
           "Run complet principal : %d exemples, taux de succès observé %.3f, **%d enfants sautés** (soit %.1f %% de ceux qui auraient été évalués). "
           "ECE prédictif (prédictions faites avant l'évaluation) = **%.4f** ; Brier = %.4f contre %.4f pour un taux constant ; température T = %.2f." % (
               s["exemples"], s["taux_succes_observe"], s["enfants_sautes"],
               100 * s["enfants_sautes"] / max(1, s["enfants_sautes"] + r["evals"]),
               s.get("ECE_prequentiel", float("nan")), s.get("Brier_prequentiel", float("nan")),
               s.get("Brier_constant_taux_base", float("nan")), s["temperature"]), "",
           "Contrôle de l'exploration : parmi les enfants prédits mauvais mais évalués quand même (epsilon), le taux de succès est %s (n = %d) ; "
           "pour les autres enfants évalués il est %s (n = %d). S'ils étaient proches, le saut ne perdrait pas grand-chose ; un écart net montre que le modèle trie bien." % (
               "%.3f" % s["exploration_bas_p"]["taux_succes"] if s["exploration_bas_p"]["taux_succes"] is not None else "n/a",
               s["exploration_bas_p"]["n"],
               "%.3f" % s["autres_evalues"]["taux_succes"] if s["autres_evalues"]["taux_succes"] is not None else "n/a",
               s["autres_evalues"]["n"]), "",
           "Sur les 4 runs de l'ablation (complet) : %d enfants sautés pour %d évaluations faites ; ECE prédictif moyen %.4f." % (
               sautes, evalues, np.mean([x.get("ECE_prequentiel", np.nan) for x in socs])), "",
           "**Qualité à CPU égal** (budget : %.0f s de CPU cumulé par run ; moteur simple = ancien moteur, moteur complet = les cinq composants ; 4 graines chacun) :" % abl["budget_cpu_s"], "",
           md_tableau(["CPU cumulé (s)", "A simple", "B simple", "A complet", "B complet"], lignes), "",
           "Évaluations faites : simple = %s ; complet = %s. Valeurs finales (meilleur A) : simple = %s ; complet = %s. "
           "Test exact par permutation : p = %.3f (A), %.3f (B sur le champion)." % (
               fin and out["ablation_complet"]["evals_simple"], out["ablation_complet"]["evals_complet"],
               ", ".join("%.4f" % x["best_A"] for x in fin["simple"]), ", ".join("%.4f" % x["best_A"] for x in fin["complet"]), pA, pB), "",
           "Gain par seconde de CPU (bpc gagnés sur A / CPU cumulé) : simple = %.5f, complet = %.5f." % (
               np.mean([x["gain_par_cpu_s"] for x in fin["simple"]]), np.mean([x["gain_par_cpu_s"] for x in fin["complet"]])), ""]
    # --- Objectif ---
    fp = r["front_pareto"]
    md += ["### 9.2 Objectif explicite (bpc + longueur de description) et front de Pareto", "",
           "Deux objectifs à minimiser : le bpc sur A et la longueur de description du génome (MDL : octets compressés par zlib x 8). "
           "Le tri non dominé est celui de NSGA-II (rang et distance d'encombrement ; fonctions testées sur un exemple jouet). "
           "Pour choisir les parents, la fitness est bpc + 0,5 x (bits du génome / %d caractères de A) + pénalité de coût ; les membres du front ont un poids x1,5. "
           "Le facteur 0,5 est un choix arbitraire, pas un réglage optimisé. Le champion reste choisi sur B (bpc seul). "
           "Une cible et un arrêt sont possibles (`--cible`, par exemple un bpc sur B à atteindre)." % 15000, "",
           "Front de Pareto final de l'archive (%d points ; les plus petits en bpc) :" % len(fp), "",
           md_tableau(["Génome", "bpc sur A", "Taille (bits compressés)"],
                      [[g["id"], "%.4f" % g["fit_A"], g["mdl_bits"]] for g in fp[:12]]), ""]
    # --- Coût ---
    c = r["cpu_par_evaluation_s"]
    md += ["### 9.3 Coût (CPU mesuré)", "",
           "Chaque évaluation est chronométrée dans son processus (`time.process_time`, comme le coût mesuré de `aiguilleur.py`). "
           "Moyenne %.3f s, médiane %.3f s, maximum %.3f s par évaluation (4 processus en parallèle sur 4 coeurs). "
           "Gain par seconde de CPU du run complet : %.5f bpc/s (gain de %.3f bpc sur A pour %.0f s de CPU cumulé). "
           "Un génome plus lent que le départ est pénalisé dans la sélection : + 0,01 bpc par multiple de temps de plus, plafonné à 0,1. "
           "Les mesures de CPU sont bruitées (machine chargée) ; c'est un ordre de grandeur." % (
               c["moyenne"], c["mediane"], c["max"], r["gain_par_cpu_s"], r["gain_bpc_A"], r["cpu_total_s"]), ""]
    # --- Convergence ---
    pts = r["points"]
    ech = [p for i, p in enumerate(pts) if i % max(1, len(pts) // 12) == 0 or p is pts[-1]]
    md += ["### 9.4 Convergence", "",
           "Règle d'arrêt précoce standard : arrêt si le meilleur bpc sur B ne baisse pas de plus de delta = %.3f pendant une fenêtre de patience de %s évaluations. "
           "**Raison de l'arrêt du run complet : %s**, après %d évaluations (budget fixé à %d)." % (
               r["options"]["delta"], r["options"]["patience"], r["raison_arret"], r["evals"], r["budget"]), "",
           md_tableau(["Évaluations", "CPU (s)", "meilleur A", "meilleur B", "gain par CPU-s"],
                      [[p["evals"], p["cpu_s"], "%.4f" % p["best_A"], "%.4f" % p["best_B"], "%.5f" % p["gain_par_cpu_s"]] for p in ech]), ""]
    pf = r.get("point_fixe")
    if pf:
        md += ["**Point fixe** (idée de `induction.py` : relancer et vérifier que rien ne change) : on relance l'évolution pendant %d évaluations en partant du champion (A = %.4f). "
               "Elle atteint A = %.4f, soit un gain de %.4f (seuil delta = %.3f). Verdict : **%s**. Sur B : %.4f -> %.4f." % (
                   pf["budget"], pf["A_depart"], pf["A_apres"], pf["gain_A"], pf["delta"],
                   "point fixe atteint (aucun gain notable)" if pf["point_fixe"] else "PAS un point fixe (la recherche trouve encore mieux)",
                   pf["B_depart"], pf["B_apres"]), ""]
    for x in fin["complet"] + fin["simple"]:
        pass
    # --- Émergence ---
    chemin_em = os.path.join(ICI, "archive", "principal_complet_emergence.json")
    e = json.load(open(chemin_em)) if os.path.exists(chemin_em) else r["elites"]
    out["emergence"] = {k: e[k] for k in e if k != "elites_avec_remise"}

    def neutre(v):
        n = v.get("neutralisation")
        return "non testé" if not n else "%d/%d élites inchangées (delta A médian %+.4f)" % (
            round(n["part_neutres"] * n["n_testes"]), n["n_testes"], n["delta_A_median"])
    lig = [[k, v["nb_elites"], "%.4f" % v["meilleur_A"], neutre(v), v["technique_connue"]]
           for k, v in sorted(e["caracteristiques_absentes_du_depart"].items(), key=lambda kv: kv[1]["meilleur_A"])]
    md += ["### 9.5 Émergence (archive MAP-Elites)", "",
           "Grille simple (pyribs n'est pas utilisé). Descripteurs : taille de l'AST (5 classes) x utilise u1 ou u2 x contient une remise (soustraction d'un terme directement au compte brut n[k], dans la boucle) x constantes propres à chaque ordre : %d cases possibles. "
           "Chaque case garde son meilleur génome. **Couverture : %d cases sur %d (%.0f %%)**, soit %d élites." % (
               e["cases_total"], e["cases_remplies"], e["cases_total"], 100 * e["couverture"], e["nb_elites"]), "",
           "Structures présentes dans des élites et absentes du génome de départ :", "",
           md_tableau(["Structure", "Élites concernées", "Meilleur A", "Retirer la structure change-t-il A ?", "Technique connue ?"], lig), ""]
    if e["genomes_avec_remise_dans_archive"]:
        md += ["**Remise (type Kneser-Ney) :** %d génomes de l'archive soustraient un terme directement à n[k]. Le meilleur d'entre eux (A = %.4f) est :" % (
            e["genomes_avec_remise_dans_archive"], e["meilleur_avec_remise_fit_A"]), "", "```", e["meilleur_avec_remise"].rstrip(), "```", ""]
    else:
        md += ["**Remise (type Kneser-Ney) :** aucun génome de l'archive ne soustrait un terme à n[k].", ""]
    md += [open(os.path.join(ICI, "texte_emergence.md"), encoding="utf-8").read().strip() if os.path.exists(os.path.join(ICI, "texte_emergence.md")) else "", ""]
    return md, out
