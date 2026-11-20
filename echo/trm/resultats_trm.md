# Mini-TRM sur Sudoku 9x9 (CPU) : la récursion remplace-t-elle la taille ?

Tous les chiffres ci-dessous sont mesurés ici (échelle réduite, CPU, 2 threads). Le papier arXiv 2510.04871 n'a pas pu être consulté (réseau) : la spécification de la tâche a été suivie et aucun chiffre du papier n'est cité.

## Données

- entraînement : 1000 puzzles, indices min/moy/max = 22/25.68/29, distribution {22: 6, 23: 41, 24: 201, 25: 244, 26: 194, 27: 160, 28: 153, 29: 1}, résolus par propagation seule (sans hypothèse) : 525/1000.
- test : 1000 puzzles, indices min/moy/max = 22/25.76/29, distribution {22: 4, 23: 57, 24: 160, 25: 235, 26: 204, 27: 179, 28: 159, 29: 2}, résolus par propagation seule (sans hypothèse) : 565/1000.
- Solution unique garantie par le solveur ; train/test disjoints (puzzles et solutions, modulo renommage des chiffres et transposition ; la disjonction sous toutes les symétries n'est pas vérifiée).
- Référence symbolique (contexte, pas de l'apprentissage) : le solveur exact résout 100 % des puzzles.

## Modèles et budget

| Modèle | Params | int8 (octets) | fp16 (octets) | Profondeur effective (applications de bloc/pas) | Horloge entraînement | CPU process | Énergie estimée* | Pas optimiseur | Échantillons-pas vus |
|---|---|---|---|---|---|---|---|---|---|
| A  mini-TRM récursif (2 blocs, n=3, T=2, N_sup=8) | 331,128 | 331,236 | 662,256 | 2 blocs x 8 appels = 16 par pas (x8 pas) | 1200 s | 2374 s | 6.60 Wh | 2771 | 88,672 |
| B  même réseau, sans récursion (n=1, T=1, N_sup=1) | 331,128 | 331,236 | 662,256 | 2 x 2 = 4 | 1200 s | 2234 s | 6.20 Wh | 9784 | 313,088 |
| C  réseau 8 blocs, sans récursion (n=1, T=1, N_sup=1) | 1,284,162 | 1,284,486 | 2,568,324 | 8 x 2 = 16 | 1200 s | 2374 s | 6.60 Wh | 3295 | 105,440 |

*Estimation : 10 W par coeur occupé x temps CPU mesuré (somme des threads) ; ce n'est PAS une mesure.

Budget identique : 1200 s d'horloge d'entraînement par modèle (temps des évaluations de courbe exclu), mêmes données, mêmes augmentations (chiffres, lignes/bandes, colonnes/piles, transposition), AdamW (lr 1e-3, wd 0.05, warmup 50 pas puis cosinus sur l'horloge, batch 32, gradient clip 1), EMA (0.999 avec montée en température). Perte : CE sur les cases vides (indices recopiés) + 0.5 BCE d'arrêt ; ACT activé pour A (arrêt si la tête dit « fini » ; 10 % des emplacements imposent un minimum aléatoire de pas). Mêmes réglages pour B/C (N_sup=1). Les machines étaient partagées : l'horloge n'est pas un coût CPU strictement égal.

## Résultats test (1000 puzzles, poids EMA)

| Modèle | Pas test S | Exact | IC95 Wilson | Cases vides OK | Grilles violant une contrainte |
|---|---|---|---|---|---|
| A | 1 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 55.8 % | 100.0 % |
| A | 2 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 59.2 % | 100.0 % |
| A | 4 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 61.5 % | 100.0 % |
| A | 8 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 61.3 % | 100.0 % |
| A | 16 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 60.1 % | 100.0 % |
| A | 32 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 58.9 % | 100.0 % |
| B | 1 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 66.3 % | 100.0 % |
| B | 2 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 55.9 % | 100.0 % |
| B | 4 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 39.2 % | 100.0 % |
| B | 8 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 24.9 % | 100.0 % |
| C | 1 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 60.7 % | 100.0 % |
| C | 2 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 47.9 % | 100.0 % |
| C | 4 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 33.9 % | 100.0 % |
| C | 8 | 0/1000 (0.0 %) | [0.0 ; 0.4] | 24.0 % | 100.0 % |

Pour B et C, S>1 est hors distribution d'entraînement (N_sup=1) ; donné à titre indicatif.

### Poids bruts (sans EMA) et int8 (S maximal)

| Modèle | Variante | Exact | Cases vides OK |
|---|---|---|---|
| A | raw (S=32) | 0/1000 | 58.5 % |
| A | ema_int8 (S=32) | 0/1000 | 59.2 % |
| B | raw (S=8) | 0/1000 | 24.7 % |
| B | ema_int8 (S=8) | 0/1000 | 24.8 % |
| C | raw (S=8) | 0/1000 | 23.5 % |
| C | ema_int8 (S=8) | 0/1000 | 24.0 % |

## Arrêt adaptatif (A, max 16 pas) et calibration

- Exact avec arrêt adaptatif : 0/1000 [0.0 ; 0.4], pas moyens utilisés : 16.00, puzzles où la tête a déclaré « fini » : 0, précision quand elle le dit : None.

Calibration de la tête d'arrêt au pas 1 :

| Tranche de proba | n | proba moyenne | exactitude réelle |
|---|---|---|---|
| [0.0,0.1) | 1000 | 2.2704110726884386e-12 | 0.0 |
| [0.1,0.3) | 0 | None | None |
| [0.3,0.5) | 0 | None | None |
| [0.5,0.7) | 0 | None | None |
| [0.7,0.9) | 0 | None | None |
| [0.9,1.0) | 0 | None | None |

Calibration de la tête d'arrêt au pas 8 :

| Tranche de proba | n | proba moyenne | exactitude réelle |
|---|---|---|---|
| [0.0,0.1) | 1000 | 3.932603132028589e-09 | 0.0 |
| [0.1,0.3) | 0 | None | None |
| [0.3,0.5) | 0 | None | None |
| [0.5,0.7) | 0 | None | None |
| [0.7,0.9) | 0 | None | None |
| [0.9,1.0) | 0 | None | None |

## Courbes d'entraînement (EMA, 200 premiers puzzles)


**A**

| t (s) | pas opt | perte CE train | cases OK (batch) | test200 exact | test200 cases | train200 exact | train200 cases |
|---|---|---|---|---|---|---|---|
| 121 | 249 | 2.113 | 17.1 % | 0.0 % | 32.8 % | 0.0 % | 32.1 % |
| 240 | 492 | 1.435 | 42.1 % | 0.0 % | 47.8 % | 0.0 % | 47.5 % |
| 360 | 755 | 1.243 | 48.6 % | 0.0 % | 51.0 % | 0.0 % | 51.7 % |
| 480 | 1009 | 1.156 | 51.6 % | 0.0 % | 54.1 % | 0.0 % | 54.7 % |
| 600 | 1298 | 1.103 | 53.4 % | 0.0 % | 55.7 % | 0.0 % | 55.4 % |
| 720 | 1554 | 1.062 | 54.9 % | 0.0 % | 57.1 % | 0.0 % | 57.1 % |
| 840 | 1849 | 1.014 | 56.8 % | 0.0 % | 58.9 % | 0.0 % | 58.2 % |
| 960 | 2167 | 0.975 | 58.1 % | 0.0 % | 60.4 % | 0.0 % | 59.3 % |
| 1080 | 2482 | 0.945 | 59.4 % | 0.0 % | 61.1 % | 0.0 % | 60.1 % |
| 1200 | 2771 | 0.923 | 60.2 % | 0.0 % | 61.9 % | 0.0 % | 60.3 % |

**B**

| t (s) | pas opt | perte CE train | cases OK (batch) | test200 exact | test200 cases | train200 exact | train200 cases |
|---|---|---|---|---|---|---|---|
| 120 | 876 | 1.345 | 43.1 % | 0.0 % | 52.9 % | 0.0 % | 52.4 % |
| 240 | 2020 | 1.073 | 53.1 % | 0.0 % | 57.3 % | 0.0 % | 56.6 % |
| 401 | 2762 | 0.985 | 57.0 % | 0.0 % | 59.6 % | 0.0 % | 58.8 % |
| 480 | 3210 | 0.943 | 58.6 % | 0.0 % | 60.2 % | 0.0 % | 59.7 % |
| 600 | 4039 | 0.898 | 60.5 % | 0.0 % | 62.3 % | 0.0 % | 61.5 % |
| 720 | 4874 | 0.843 | 62.8 % | 0.0 % | 64.3 % | 0.0 % | 63.6 % |
| 840 | 6067 | 0.812 | 64.1 % | 0.0 % | 65.6 % | 0.0 % | 64.9 % |
| 960 | 7313 | 0.793 | 64.8 % | 0.0 % | 66.0 % | 0.0 % | 65.7 % |
| 1080 | 8551 | 0.777 | 65.4 % | 0.0 % | 66.0 % | 0.0 % | 65.5 % |
| 1200 | 9784 | 0.769 | 65.7 % | 0.0 % | 66.5 % | 0.0 % | 65.7 % |

**C**

| t (s) | pas opt | perte CE train | cases OK (batch) | test200 exact | test200 cases | train200 exact | train200 cases |
|---|---|---|---|---|---|---|---|
| 120 | 318 | 1.556 | 36.0 % | 0.0 % | 51.5 % | 0.0 % | 51.6 % |
| 240 | 633 | 1.156 | 49.9 % | 0.0 % | 52.7 % | 0.0 % | 52.5 % |
| 360 | 959 | 1.116 | 51.3 % | 0.0 % | 53.6 % | 0.0 % | 53.9 % |
| 480 | 1303 | 1.059 | 53.9 % | 0.0 % | 56.2 % | 0.0 % | 56.0 % |
| 600 | 1646 | 1.012 | 55.8 % | 0.0 % | 57.6 % | 0.0 % | 57.7 % |
| 720 | 1984 | 0.982 | 57.0 % | 0.0 % | 58.3 % | 0.0 % | 58.4 % |
| 840 | 2309 | 0.957 | 58.0 % | 0.0 % | 59.1 % | 0.0 % | 58.8 % |
| 960 | 2631 | 0.930 | 59.0 % | 0.0 % | 59.6 % | 0.0 % | 59.2 % |
| 1080 | 2953 | 0.912 | 59.9 % | 0.0 % | 60.0 % | 0.0 % | 59.9 % |
| 1200 | 3295 | 0.900 | 60.4 % | 0.0 % | 60.4 % | 0.0 % | 60.2 % |

(Les courbes sont évaluées avec S=N_sup du modèle : 8 pour A, 1 pour B/C. Aucun réglage n'a été fait sur le test.)

## Limites

- Échelle : ~0,3 M de paramètres pour A/B, ~1,3 M pour C, 20 minutes CPU par modèle, un seul tirage (graine 0). Le papier original entraîne bien plus longtemps, avec bien plus de calcul ; ses hyperparamètres n'ont pas pu être lus ici et ne sont pas cités. Nos modèles voient très peu d'époques, donc l'absence de résolution exacte peut refléter un sous-entraînement et non une limite de l'approche.
- Test de 1000 puzzles : IC de Wilson 95 % larges ; à 0 succès la borne haute reste ~0,4 %. Écarts de quelques puzzles non significatifs.
- Un seul tirage, pas de variance entre graines ; horloge partagée avec d'autres tâches ; 'profondeur effective' et calcul par échantillon diffèrent entre A, B, C (A dépense bien plus de calcul par puzzle) : l'égalité porte sur le temps d'horloge, pas sur les FLOP par puzzle.
- Le réseau est un MLP-mixer à jetons fixes (81) ; pas d'attention.

## Conclusion

**Verdict honnête : cette expérience ne permet pas de conclure que la récursion remplace la taille, et ne la contredit pas non plus.** Les trois modèles obtiennent 0/1000 puzzles exacts (IC95 Wilson [0 ; 0,38] %) et 100 % de leurs grilles violent au moins une contrainte Sudoku : aucun n'a appris à résoudre en 20 minutes CPU. Le niveau atteint (~55-66 % de cases vides correctes) correspond à un apprentissage de régularités partielles, pas à du raisonnement.

Observations mesurées (et leurs limites) :
- A (récursif, 0,33 M params) : cases vides OK 55,8 % (S=1), 59,2 % (S=2), 61,5 % (S=4), 61,3 % (S=8), 60,1 % (S=16), 58,9 % (S=32) : « réfléchir plus longtemps » aide un peu jusqu'à ~4 pas puis ne sert plus ; la dérive reste modérée au-delà de N_sup=8.
- B (même réseau, sans récursion) : 66,3 % à S=1 (mieux que A à budget d'horloge égal, car 3x plus de pas d'optimisation : 9784 contre 2771) ; C (1,28 M params) : 60,7 % à S=1 (3295 pas). À S>1 B et C se dégradent fortement (hors distribution), alors que A reste stable : seule la robustesse aux pas supplémentaires distingue A, pas la précision.
- Aucune différence entre A, B, C n'est interprétable comme « récursion contre taille » : B > C > A à S=1 est dominé par le nombre de pas d'optimisation dans un budget fixe (la récursion coûte ~3,5x plus par pas).
- La tête d'arrêt n'a jamais dit « fini » (proba ~4e-9 partout) : correct car aucune réponse n'est exacte, mais cela rend l'arrêt adaptatif (16 pas moyens, 0 exact) et la calibration non informatifs.
- Quantification int8 : variations de <1 point de cases vides, 0 exact. EMA vs poids bruts : écarts de quelques dixièmes de point.
- Pour tester l'hypothèse il faudrait un entraînement bien plus long (probablement des heures/jours CPU ou GPU), plusieurs graines, et un budget FLOP égal par puzzle.
