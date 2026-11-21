"""Assemble resultats_evolution.json et resultats_evolution.md à partir des mesures."""
import itertools
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))


def charger(chemin):
    with open(os.path.join(ICI, chemin)) as f:
        return json.load(f)


def test_permutation(a, b):
    """Test exact par permutation (deux côtés) sur la différence des moyennes."""
    tout = list(a) + list(b)
    n = len(a)
    obs = abs(np.mean(a) - np.mean(b))
    plus = total = 0
    for idx in itertools.combinations(range(len(tout)), n):
        ga = [tout[i] for i in idx]
        gb = [tout[i] for i in range(len(tout)) if i not in idx]
        total += 1
        if abs(np.mean(ga) - np.mean(gb)) >= obs - 1e-12:
            plus += 1
    return plus / total


def ablation(abl):
    evals = [p["evals"] for p in abl["runs"][0]["points"]]
    sortie = {"evals": evals, "modes": {}}
    for mode in ("auto", "fixe"):
        runs = [r for r in abl["runs"] if r["meta"] == mode]
        # on prend le dernier point <= chaque palier de 250 évaluations
        courbes = {"best_A": [], "best_B": []}
        for r in runs:
            for cle in courbes:
                courbes[cle].append([p[cle] for p in r["points"]])
        L = min(len(c) for c in courbes["best_A"])
        sortie["modes"][mode] = {
            "graines": [r["graine"] for r in runs],
            "finale_A": [r["points"][-1]["best_A"] for r in runs],
            "finale_B_champion": [r["champion_B"] for r in runs],
            "moy_best_A": [float(np.mean([c[i] for c in courbes["best_A"]])) for i in range(L)],
            "moy_best_B": [float(np.mean([c[i] for c in courbes["best_B"]])) for i in range(L)],
            "std_best_A": [float(np.std([c[i] for c in courbes["best_A"]])) for i in range(L)],
            "std_best_B": [float(np.std([c[i] for c in courbes["best_B"]])) for i in range(L)],
            "palier_evals": evals[:L],
        }
    a, f = sortie["modes"]["auto"], sortie["modes"]["fixe"]
    sortie["p_permutation_A"] = test_permutation(a["finale_A"], f["finale_A"])
    sortie["p_permutation_B"] = test_permutation(a["finale_B_champion"], f["finale_B_champion"])
    return sortie


def md_tableau(entetes, lignes):
    s = "| " + " | ".join(entetes) + " |\n|" + "|".join("---" for _ in entetes) + "|\n"
    for l in lignes:
        s += "| " + " | ".join(str(x) for x in l) + " |\n"
    return s


def main():
    principal = charger("archive/principal_resume.json")
    abl = charger("archive/ablation.json")
    test = charger("resultats_test.json")
    tests_txt = open(os.path.join(ICI, "resultats_tests_unitaires.txt"), encoding="utf-8").read().strip()
    ab = ablation(abl)
    rejets = Counter()
    for r in [principal] + abl["runs"]:
        for k, v in r["rejets"].items():
            rejets[k] += v
    rejets_malv = {}
    sys.path.insert(0, ICI)
    import garde_fous
    import test_evolution as te
    for nom, corps in te.MALVEILLANTS.items():
        rejets_malv[nom] = garde_fous.verifier(te._gen(corps))[1]
    temps_total = principal["temps_mur_s"] + abl.get("temps_total_s", 0) + test["duree_totale_s"]
    out = {"principal": {k: principal[k] for k in ("evals", "generations", "taille_archive", "seed_A", "seed_B",
                                                   "champion", "temps_mur_s", "points", "meta_moyen_final",
                                                   "ecartes_trop_mauvais", "rejets")},
           "ablation": ab, "test_C": test, "rejets_total_par_raison": dict(rejets),
           "rejets_malveillants_de_test": rejets_malv, "tests_unitaires": tests_txt,
           "temps_mur_total_s": round(temps_total, 1)}
    with open(os.path.join(ICI, "resultats_evolution.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    C, se = test["C"], test["sous_ensemble_echo_p"]
    md = ["# Résultats de l'évolution d'estimateurs", "",
          "Ce document est généré par `rapport.py` à partir des mesures (`archive/*.json`, `resultats_test.json`).", ""]
    md += [open(os.path.join(ICI, "texte_intro.md"), encoding="utf-8").read().strip(), ""]
    md += ["## 1. Test final sur C (utilisé une seule fois)", "",
           "C = %d caractères de `validation.txt`, jamais utilisés pour choisir quoi que ce soit." % test["taille_C"], "",
           md_tableau(["Modèle", "bpc sur C"],
                      [["Départ (PPM-C interpolé, ordre 10, comptages figés)", "%.4f" % C["depart_bpc"]],
                       ["**Champion évolué**", "**%.4f**" % C["champion_bpc"]],
                       ["Référence à la main (e = %s·d, puissance %s ; réglages choisis sur A)" % (test["reference_a_la_main"]["c"], test["reference_a_la_main"]["gamma"]), "%.4f" % test["reference_a_la_main"]["bpc_C"]],
                       ["LZMA conditionnel (biblio + C moins biblio)", "%.4f" % test["lzma_conditionnel"]["bpc"]]]),
           "Gain relatif du champion sur le départ : **%.1f %%** (intervalle à 95 %% par blocs de 500 caractères : %.1f %% à %.1f %%)." % (
               100 * C["gain_relatif"], 100 * C["gain_relatif_IC95_blocs500"][0], 100 * C["gain_relatif_IC95_blocs500"][1]), "",
           "Écho-P d'origine (`predicteur.py`, ordre 16, exclusion) est lent. Il a été évalué sur un **sous-ensemble de C** : "
           "%d blocs de %d caractères répartis régulièrement sur C (%d caractères). "
           "Départ et champion sont mesurés sur exactement les mêmes positions." % (se["blocs"], se["longueur_bloc"], se["positions"]), "",
           md_tableau(["Modèle (sous-ensemble de C)", "bpc"],
                      [["Écho-P d'origine", "%.4f" % se["echo_p_bpc"]], ["Départ", "%.4f" % se["depart_bpc"]],
                       ["Champion", "%.4f" % se["champion_bpc"]],
                       ["Référence à la main", "%.4f" % se["reference_main_bpc"]]]), "",
           "Scores sur A et B (champion choisi sur B) : départ A = %.4f, B = %.4f ; champion A = %.4f, B = %.4f." % (
               test["A_B"]["depart_A"], test["A_B"]["depart_B"], test["A_B"]["champion_A"], test["A_B"]["champion_B"]), "",
           "### Diagnostic par catégorie de caractère (sur C)", "",
           md_tableau(["Catégorie", "Part des caractères", "bpc départ", "bpc champion", "Part du gain total"],
                      [[k, "%.1f %%" % (100 * v["part_des_caracteres"]), "%.3f" % v["bpc_depart"],
                        "%.3f" % v["bpc_champion"], "%.1f %%" % (100 * v["part_du_gain_total"])]
                       for k, v in sorted(test["categories"].items(), key=lambda kv: -kv[1]["n"])]), ""]
    md += ["## 2. Run principal (méta-génome automatique)", "",
           "%d évaluations, %d générations, archive de %d génomes, %.0f s de temps mur. Départ : A = %.4f, B = %.4f. "
           "Champion (génome n°%d) : A = %.4f, B = %.4f." % (
               principal["evals"], principal["generations"], principal["taille_archive"], principal["temps_mur_s"],
               principal["seed_A"], principal["seed_B"], principal["champion"]["id"],
               principal["champion"]["fit_A"], principal["champion"]["fit_B"]), "",
           md_tableau(["Évaluations", "meilleur A", "meilleur B (champion)"],
                      [[p["evals"], "%.4f" % p["best_A"], "%.4f" % p["best_B"]] for p in principal["points"]]), ""]
    ma, mf = ab["modes"]["auto"], ab["modes"]["fixe"]
    lignes = []
    for i, e in enumerate(ma["palier_evals"]):
        lignes.append([e, "%.4f ± %.4f" % (ma["moy_best_A"][i], ma["std_best_A"][i]),
                       "%.4f ± %.4f" % (mf["moy_best_A"][i], mf["std_best_A"][i]),
                       "%.4f ± %.4f" % (ma["moy_best_B"][i], ma["std_best_B"][i]),
                       "%.4f ± %.4f" % (mf["moy_best_B"][i], mf["std_best_B"][i])])
    md += ["## 3. Ablation : méta-génome automatique contre fixe", "",
           "%d graines par mode, %d évaluations par run, même génome de départ. Moyenne ± écart-type entre graines." % (
               len(ma["graines"]), abl["budget"]), "",
           md_tableau(["Évaluations", "A auto", "A fixe", "B auto", "B fixe"], lignes), "",
           "Valeurs finales par graine, meilleur A : auto = %s ; fixe = %s." % (
               ", ".join("%.4f" % x for x in ma["finale_A"]), ", ".join("%.4f" % x for x in mf["finale_A"])),
           "Valeurs finales par graine, B du champion : auto = %s ; fixe = %s." % (
               ", ".join("%.4f" % x for x in ma["finale_B_champion"]), ", ".join("%.4f" % x for x in mf["finale_B_champion"])),
           "Test exact par permutation sur la différence des moyennes : p = %.3f (A), p = %.3f (B)." % (
               ab["p_permutation_A"], ab["p_permutation_B"]), "",
           open(os.path.join(ICI, "texte_ablation.md"), encoding="utf-8").read().strip(), ""]
    chemin_long = os.path.join(ICI, "archive", "long_resume.json")
    if os.path.exists(chemin_long):
        lg = charger("archive/long_resume.json")
        ana = charger("analyse_champion.json")
        out["run_long"] = {k: lg[k] for k in ("evals", "generations", "taille_archive", "champion", "temps_mur_s", "points", "rejets")}
        with open(os.path.join(ICI, "resultats_evolution.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        pts = [p for p in lg["points"] if p["evals"] % 1000 < 250 or p is lg["points"][-1]]
        md += ["## 3 bis. Run long supplémentaire (A et B seulement, jamais testé sur C)", "",
               "Pour voir si un budget plus grand dépasse la référence à la main : %d évaluations, graine 2, méta automatique, %.0f s. "
               "Son champion n'a **pas** été passé sur C (C ne sert qu'une fois, pour le champion du run principal)." % (lg["evals"], lg["temps_mur_s"]), "",
               md_tableau(["Évaluations", "meilleur A", "meilleur B"], [[p["evals"], "%.4f" % p["best_A"], "%.4f" % p["best_B"]] for p in pts]), "",
               "Champion du run long : A = %.4f, B = %.4f. Référence à la main : A = %.4f, B = %.4f. Champion du run principal : A = %.4f, B = %.4f." % (
                   lg["champion"]["fit_A"], lg["champion"]["fit_B"], ana["reference_a_la_main"]["A"], ana["reference_a_la_main"]["B"],
                   principal["champion"]["fit_A"], principal["champion"]["fit_B"]),
               "Sur A, le run long reste %s que la référence à la main ; sur B, il est %s (B a servi à choisir son champion, donc cette mesure est un peu optimiste)." % (
                   "derrière" if lg["champion"]["fit_A"] > ana["reference_a_la_main"]["A"] else "devant",
                   "devant" if lg["champion"]["fit_B"] < ana["reference_a_la_main"]["B"] else "derrière"), ""]
    md += ["## 4. Génomes rejetés, par raison", "",
           "Total sur le run principal et les 10 runs d'ablation (le run long n'est pas compté). Aucune mutation n'a été refusée par la liste blanche : les rejets viennent tous du juge (sortie de mauvaise forme, non finie, négative, ou erreur à l'exécution). Aucun délai dépassé, aucune mémoire dépassée.", "",
           md_tableau(["Raison", "Nombre"], [[k, v] for k, v in sorted(rejets.items(), key=lambda kv: -kv[1])]), "",
           "Génomes malveillants écrits à la main pour les tests, tous refusés par la liste blanche :", "",
           md_tableau(["Essai", "Raison du rejet"], [[k, v] for k, v in rejets_malv.items()]), ""]
    md += ["## 5. Tests unitaires", "", "```", tests_txt, "```", ""]
    md += [open(os.path.join(ICI, "texte_formule.md"), encoding="utf-8").read().strip(), ""]
    import rapport_composants
    md_c, out_c = rapport_composants.construire(md_tableau)
    md += md_c
    if out_c:
        out["composants"] = out_c
        with open(os.path.join(ICI, "resultats_evolution.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    md += ["## 10. Temps de calcul", "",
           "Temps mur total des runs et du test : environ %.0f minutes, sur 4 coeurs (4 processus d'évaluation). "
           "Le temps CPU cumulé est de l'ordre de 3 fois le temps mur." % (temps_total / 60), ""]
    md += [open(os.path.join(ICI, "texte_limites.md"), encoding="utf-8").read().strip(), ""]
    with open(os.path.join(ICI, "resultats_evolution.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
