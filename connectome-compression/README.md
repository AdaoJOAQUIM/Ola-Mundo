# Compression du connectome de la mouche : test de l'étape 1

Question : peut-on remplacer les 15 091 983 connexions du connectome (FlyWire v783) par une
description beaucoup plus courte, sans perdre le comportement sucre → MN9 ?

Réponse mesurée : non. Voir `compression_connectome.xlsx` (feuille Résultats).

| Méthode | Rapport | MN9 (référence : 63 Hz) |
|---|---|---|
| Élagage, 5 synapses et plus | 5,6 : 1 | 61 Hz |
| Élagage, 10 synapses et plus | 14 : 1 | 47 Hz |
| Élagage, 20 synapses et plus | 41 : 1 | 5 Hz |
| Règles entre groupes (30 à 10 000 groupes) | 6 : 1 à 324 : 1 | 0 Hz |
| Règles + graine (câblage régénéré) | 5 : 1 à 149 : 1 | 0 Hz |
| Rang faible (1 à 64) | 3 : 1 à 163 : 1 | 0 Hz |

Moyennes sur 3 graines, 20 neurones sucre à 100 Hz, 1 s, cerveau entier.

## Reproduire

Les données viennent de la feuille `Cerveau` du classeur `cerveau_auto_construit.xlsx` (non versionné, 92 Mo).

```
python3 decode.py cerveau_auto_construit.xlsx conn.npz   # 15 091 983 connexions, 54 492 922 synapses
python3 groups.py                                        # groupes de neurones (SVD + k-moyennes)
python3 exper.py                                         # simulations (écrit dans le journal)
python3 summary.py                                       # results.json
python3 build_xlsx.py compression_connectome.xlsx
```

Dépendances : numpy, scipy, scikit-learn, openpyxl. `simlib.py` contient le simulateur
(modèle de Shiu et al., Nature 2024) ; il reproduit la référence de la notice : 356 neurones
actifs, 9 054 spikes, MN9 = 61 (graine 1).

## Limites

- Un seul comportement mesuré (sucre → MN9).
- Les groupes sont tirés du câblage, pas des types cellulaires annotés.
- Power Query et DAX (feuille LISEZ-MOI du classeur) n'ont pas été exécutés : pas d'Excel ici.

## Mini-circuit à 19 000 : 1 (une seule tâche)

`mini.py` classe chaque connexion par ce qu'elle transporte réellement vers MN9 (trafic mesuré dans la
simulation complète, importance remontée depuis MN9), garde les k meilleures et rejoue le cerveau entier.

| Connexions gardées | Rapport | MN9 (3 graines ; référence 61, 71, 58) |
|---|---|---|
| 1 200 | 12 577 : 1 | 66, 69, 66 |
| 794 | 19 008 : 1 | 38, 47, 49 |
| 794, gain 1,1 | 19 008 : 1 | 58, 60, 59 |
| 400 | 37 730 : 1 | 7, 12, 12 |

`circuit_794.csv` contient ces 794 connexions (151 neurones). Réponse à d'autres intensités de sucre,
jamais utilisées pour construire le circuit (complet / mini, moyenne de 3 graines) :
25 Hz : 0 / 0 ; 50 Hz : 14 / 9 ; 75 Hz : 52 / 34 ; 150 Hz : 79 / 85 ; 200 Hz : 87 / 99.

Limites : ce circuit ne reproduit que la sortie MN9 pour la stimulation sucre ; le gain a été réglé sur
la mesure qu'il sert à évaluer ; ce n'est pas une compression du cerveau entier.

## Cerveau entier dans Power Query (moteur événementiel)

- `evenementiel.py` : moteur qui ne calcule que les neurones recevant un signal (forme fermée entre deux
  événements, prédiction du pas exact du prochain spike). Sur 3 graines : 0 neurone en écart avec la
  simulation complète des 138 639 neurones, environ 10 000 neurones réveillés, 1,3 million d'événements.
- `Simuler.pq`, `Spikes.pq` : le même moteur en langage M. Syntaxe validée par l'analyseur de Microsoft ;
  jamais exécuté (pas d'Excel ici).
- `pq_miroir.py` : miroir Python ligne à ligne du code M, qui relit le classeur produit. 0 écart avec la
  simulation complète (valeurs dans `attendu.json`).
- `build_pq_xlsx.py` : construit `cerveau_entier_PQ.xlsx` (49 Mo, non versionné) : câblage complet en
  138 639 cellules, réglages, valeurs de contrôle, code M et mesures DAX.

## Chercher les lois plutôt que compresser le câblage

### Étape 1 : résoudre le régime permanent au lieu de simuler (échec)

`regime.py` / `regime2.py` résolvent les fréquences de tous les neurones (entrées de Poisson, noyau exact du
modèle, formule de Siegert pour le LIF à bruit, signaux perdus en période réfractaire). Critère fixé avant le
test : MN9 à moins de 20 % du moteur exact sur la courbe dose-réponse du sucre.

| Sucre | MN9 exact | MN9 régime permanent | Neurones actifs (exact / régime) |
|---|---|---|---|
| 25 Hz | 0 | 0 | 61 / 43 |
| 50 Hz | 11 | 224 | 231 / 462 |
| 100 Hz | 63 | 0,3 à 14 | 352 / 12 806 |
| 200 Hz | 87 | 243 | 401 / 12 933 |

Critère non rempli : le solveur ne converge pas dès 50 Hz (une première version avait un défaut de seuil ;
corrigé dans `regime2.py`, même résultat). Cause non établie : pas de point fixe stable, ou point fixe de
haute activité que la simulation d'une seconde n'atteint pas ; l'hypothèse de bruit de Poisson indépendant est
probablement en cause.

### Plan de repli : campagne sur le moteur exact

`campagne.py` : 400 motifs de stimulation (fréquence tirée au hasard entre 0 et 200 Hz pour chacun des 20
neurones sucre, un tiers muets), cerveau entier exact, 1 s, plus 40 répétitions avec une autre graine pour
mesurer le bruit. `lois_fit.py` : 300 motifs pour ajuster, 100 pour tester.

| Loi | R² test | Erreur moyenne |
|---|---|---|
| Constante | -0,06 | 14,5 Hz |
| Courbe dose-réponse de la fréquence moyenne | 0,49 | 9,4 Hz |
| **20 poids + courbe monotone (62 points)** | **0,86** | **4,9 Hz** |
| max(0, b + Σ aᵢ rᵢ), 21 nombres | 0,81 | 5,7 Hz |
| Réseau 20-32-32-1 (référence non compacte) | 0,79 | 6,5 Hz |

Plancher de bruit des graines : 5,7 Hz (écart-type) ; R² maximal atteignable sur ce test : 0,88.
`loi_MN9.xlsx` (36 Ko, 503 formules, 0 erreur) contient la loi et les 100 tests, recalculés par formules.

Limites : une seule sortie (MN9), une seule famille d'entrées (20 neurones sucre), 1 s de stimulation
constante. La loi ne contient pas le câblage et ne permet pas de le retrouver.
