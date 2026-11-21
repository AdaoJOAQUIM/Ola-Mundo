# CompressARC officiel (CPU, 1 thread) - resultats

Code officiel (iliao2345/CompressARC, ~76K parametres, entraine from scratch par tache), copie patchee dans `compressarc/code/` : seul changement = `torch.set_default_device('cuda')` -> `'cpu'` (arc_compressor.py). Hyperparametres par defaut conserves (Adam lr 0.01, betas 0.5/0.9, perte KL + 10*reconstruction) ; 2 tentatives = les 2 solutions les plus frequentes/scorees (logique `solution_selection.py`). Adaptateur de donnees : `adapter.py` (fichiers fchollet par tache -> JSON combine style Kaggle `dataset/arc-agi_evaluation_*.json`). Scoring avec les sorties officielles uniquement a la fin.

## Reduction par rapport aux defauts du papier (2000 pas/tache, 1500 dans analyze_example)
Mesure CPU (1 thread, machine partagee) : 0.66 s/pas (grille 6x6, 3 exemples), 2.0 s/pas (19x25), 4.1 s/pas (28x29, 5 exemples) ; donc 1500 pas = 17 a 100 min/tache, impossible pour 20 taches en ~3 h. Budget par tache ~500 s : pas = clamp(500/s_pas, 150, 1500), de 150 a 783 pas (soit 8 a 52 % de 1500), voir `pas_par_tache`. Les grandes taches (150 pas) sont tres probablement sous-entrainees : resultats = borne basse de CompressARC.

## Resultats (20 taches)
pass@2 : 1/20 ; pass@1 : 1/20 ; temps total 2.36 h

| tache | pas | s | pass@1 | pass@2 |
|---|---|---|---|---|
| 00576224 | 783 | 523 | non | non |
| 11e1fe23 | 436 | 460 | non | non |
| 15696249 | 490 | 450 | non | non |
| 1a2e2828 | 312 | 561 | oui | oui |
| 1da012fc | 215 | 430 | non | non |
| 25094a63 | 150 | 426 | non | non |
| 358ba94e | 194 | 397 | non | non |
| 48131b3c | 604 | 439 | non | non |
| 4f537728 | 245 | 318 | non | non |
| 7d1f7ee8 | 150 | 576 | non | non |
| 7e02026e | 404 | 322 | non | non |
| 81c0276b | 196 | 463 | non | non |
| 891232d6 | 150 | 624 | non | non |
| 8cb8642d | 176 | 391 | non | non |
| 9def23fe | 156 | 345 | non | non |
| b15fca0b | 394 | 363 | non | non |
| bc4146bd | 494 | 485 | non | non |
| bf89d739 | 194 | 256 | non | non |
| d94c3b52 | 186 | 267 | non | non |
| fafd9572 | 340 | 390 | non | non |
