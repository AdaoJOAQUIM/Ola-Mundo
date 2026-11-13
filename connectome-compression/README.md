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
