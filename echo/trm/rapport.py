"""Génère resultats_trm.md à partir de resultats_trm.json et runs/*_train.json."""
import json
import os

import donnees

ICI = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ICI, "resultats_trm.json")))
T = {k: json.load(open(os.path.join(ICI, "runs", f"{k}_train.json"))) for k in "ABC"}
jeux = donnees.charger(os.path.join(ICI, "donnees_sudoku.pt"))
N = R["n_test"]
NOM = {"A": "A  mini-TRM récursif (2 blocs, n=3, T=2, N_sup=8)",
       "B": "B  même réseau, sans récursion (n=1, T=1, N_sup=1)",
       "C": "C  réseau 8 blocs, sans récursion (n=1, T=1, N_sup=1)"}
pc = lambda x: f"{100 * x:.1f} %"
ci = lambda e: f"[{100 * e[0]:.1f} ; {100 * e[1]:.1f}]"
L = []
w = L.append
w("# Mini-TRM sur Sudoku 9x9 (CPU) : la récursion remplace-t-elle la taille ?\n")
w("Tous les chiffres ci-dessous sont mesurés ici (échelle réduite, CPU, 2 threads). "
  "Le papier arXiv 2510.04871 n'a pas pu être consulté (réseau) : la spécification de la tâche a été suivie "
  "et aucun chiffre du papier n'est cité.\n")
w("## Données\n")
for k, nom in (("train", "entraînement"), ("test", "test")):
    st = donnees.stats_indices([p.tolist() for p in jeux[k][0]])
    prop = sum(donnees.resolu_par_propagation(p.tolist()) for p in jeux[k][0])
    w(f"- {nom} : 1000 puzzles, indices min/moy/max = {st['min']}/{st['moyenne']:.2f}/{st['max']}, "
      f"distribution {st['distribution']}, résolus par propagation seule (sans hypothèse) : {prop}/1000.")
w("- Solution unique garantie par le solveur ; train/test disjoints (puzzles et solutions, modulo renommage des "
  "chiffres et transposition ; la disjonction sous toutes les symétries n'est pas vérifiée).")
w("- Référence symbolique (contexte, pas de l'apprentissage) : le solveur exact résout 100 % des puzzles.\n")
w("## Modèles et budget\n")
w("| Modèle | Params | int8 (octets) | fp16 (octets) | Profondeur effective (applications de bloc/pas) | Horloge entraînement | CPU process | Énergie estimée* | Pas optimiseur | Échantillons-pas vus |")
w("|---|---|---|---|---|---|---|---|---|---|")
prof = {"A": 2 * 2 * 4 * 3 // 3 * 1, "B": 4, "C": 16}
prof = {"A": "2 blocs x 8 appels = 16 par pas (x8 pas)", "B": "2 x 2 = 4", "C": "8 x 2 = 16"}
for k in "ABC":
    t, e = T[k], R[k]["export"]
    w(f"| {NOM[k]} | {R[k]['params']:,} | {e['int8_octets']:,} | {e['fp16_octets']:,} | {prof[k]} | "
      f"{t['horloge_entrainement_s']:.0f} s | {t['cpu_process_s']:.0f} s | {t['cpu_process_s'] * 10 / 3600:.2f} Wh | "
      f"{t['pas_optimiseur']} | {t['pas_supervision_echantillons']:,} |")
w("\n*Estimation : 10 W par coeur occupé x temps CPU mesuré (somme des threads) ; ce n'est PAS une mesure.\n")
w("Budget identique : 1200 s d'horloge d'entraînement par modèle (temps des évaluations de courbe exclu), "
  "mêmes données, mêmes augmentations (chiffres, lignes/bandes, colonnes/piles, transposition), AdamW "
  "(lr 1e-3, wd 0.05, warmup 50 pas puis cosinus sur l'horloge, batch 32, gradient clip 1), EMA (0.999 avec "
  "montée en température). Perte : CE sur les cases vides (indices recopiés) + 0.5 BCE d'arrêt ; ACT activé pour A "
  "(arrêt si la tête dit « fini » ; 10 % des emplacements imposent un minimum aléatoire de pas). "
  "Mêmes réglages pour B/C (N_sup=1). Les machines étaient partagées : l'horloge n'est pas un coût CPU strictement égal.\n")
w("## Résultats test (1000 puzzles, poids EMA)\n")
w("| Modèle | Pas test S | Exact | IC95 Wilson | Cases vides OK | Grilles violant une contrainte |")
w("|---|---|---|---|---|---|")
for k in "ABC":
    for s, v in R[k]["ema"]["par_pas"].items():
        w(f"| {k} | {s} | {v['exact']}/{N} ({pc(v['exact'] / N)}) | {ci(v['exact_ic95'])} | {pc(v['cases_vides'])} | {pc(v['viol_contrainte'])} |")
w("\nPour B et C, S>1 est hors distribution d'entraînement (N_sup=1) ; donné à titre indicatif.\n")
w("### Poids bruts (sans EMA) et int8 (S maximal)\n")
w("| Modèle | Variante | Exact | Cases vides OK |")
w("|---|---|---|---|")
for k in "ABC":
    for cle in ("raw", "ema_int8"):
        pp = R[k][cle]["par_pas"]
        s = max(map(int, pp))
        v = pp[str(s)] if str(s) in pp else pp[s]
        w(f"| {k} | {cle} (S={s}) | {v['exact']}/{N} | {pc(v['cases_vides'])} |")
w("\n## Arrêt adaptatif (A, max 16 pas) et calibration\n")
ad = R["A"]["ema"]["adaptatif"]
w(f"- Exact avec arrêt adaptatif : {ad['exact']}/{N} {ci(ad['exact_ic95'])}, pas moyens utilisés : {ad['pas_moyens']:.2f}, "
  f"puzzles où la tête a déclaré « fini » : {ad['n_arret_declare']}, précision quand elle le dit : {ad['precision_quand_dit_fini']}.")
for cle, titre in (("calibration_pas1", "pas 1"), ("calibration_pasN", "pas 8")):
    w(f"\nCalibration de la tête d'arrêt au {titre} :\n")
    w("| Tranche de proba | n | proba moyenne | exactitude réelle |")
    w("|---|---|---|---|")
    for l in R["A"]["ema"][cle]:
        w(f"| {l['tranche']} | {l['n']} | {l['proba_moy']} | {l['exact_reel']} |")
w("\n## Courbes d'entraînement (EMA, 200 premiers puzzles)\n")
for k in "ABC":
    w(f"\n**{k}**\n")
    w("| t (s) | pas opt | perte CE train | cases OK (batch) | test200 exact | test200 cases | train200 exact | train200 cases |")
    w("|---|---|---|---|---|---|---|---|")
    for c in T[k]["courbe"]:
        w(f"| {c['t_s']:.0f} | {c['pas_opt']} | {c['perte_ce_train']:.3f} | {pc(c['cases_ok_train_pas'])} | "
          f"{pc(c['ema_test200_exact'])} | {pc(c['ema_test200_cases'])} | {pc(c['ema_train200_exact'])} | {pc(c['ema_train200_cases'])} |")
w("\n(Les courbes sont évaluées avec S=N_sup du modèle : 8 pour A, 1 pour B/C. Aucun réglage n'a été fait sur le test.)\n")
w("## Limites\n")
w("- Échelle : ~0,3 M de paramètres pour A/B, ~1,3 M pour C, 20 minutes CPU par modèle, un seul tirage (graine 0). "
  "Le papier original entraîne bien plus longtemps, avec bien plus de calcul ; ses hyperparamètres n'ont pas pu être lus ici et ne sont pas cités. "
  "Nos modèles voient très peu d'époques, donc l'absence de résolution exacte peut refléter un sous-entraînement et non une limite de l'approche.")
w(f"- Test de {N} puzzles : IC de Wilson 95 % larges ; à 0 succès la borne haute reste ~0,4 %. Écarts de quelques puzzles non significatifs.")
w("- Un seul tirage, pas de variance entre graines ; horloge partagée avec d'autres tâches ; 'profondeur effective' et calcul par échantillon diffèrent entre A, B, C (A dépense bien plus de calcul par puzzle) : l'égalité porte sur le temps d'horloge, pas sur les FLOP par puzzle.")
w("- Le réseau est un MLP-mixer à jetons fixes (81) ; pas d'attention.\n")
w("## Conclusion\n")
open(os.path.join(ICI, "resultats_trm.md"), "w").write("\n".join(L) + "\n__CONCLUSION__\n")
