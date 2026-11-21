# TRM officiel (checkpoint ARC Prize, ARC-AGI-1) sur 20 tâches d'évaluation

**Modèle** : `arcprize/trm_arc_prize_verification/arc_v1_public/step_518071` (TRM 7M, config `all_config.yaml`), code officiel copié dans `code/` (aucun patch du modèle ; inférence CPU float32 écrite dans `infer.py`, 16 pas ACT comme l'évaluateur officiel, pas de torch.compile / flash-attn).

## Résultats (K = 8 augmentations par tâche : identité + 7 augmentations de la pipeline officielle)
| Métrique | Score |
|---|---|
| pass@1 | **10 / 20 (50 %)** |
| pass@2 | **12 / 20 (60 %)** |
| CPU par tâche | ~346 s CPU (~175 s mur, 2 threads), soit ~22 s mur / échantillon ; 8 échantillons par tâche (toutes les tâches ont 1 entrée de test) |

Résolues en pass@2 : 00576224, 15696249, 1a2e2828, 25094a63, 358ba94e, 48131b3c, 4f537728, 7d1f7ee8, 7e02026e, 9def23fe, b15fca0b, bc4146bd. Échecs : 11e1fe23, 1da012fc, 81c0276b, 891232d6, 8cb8642d, bf89d739, d94c3b52, fafd9572. (15696249 et 7e02026e : bonne réponse seulement en 2e tentative.)

## Reconstruction des identifiants (point critique)
- Le modèle utilise un embedding par identifiant d'augmentation (876 406 lignes). `build_subset.py` rejoue `dataset/build_arc_dataset.py` à l'identique (seed 42, 1000 aug, sous-ensembles training/evaluation/concept, `kaggle/combined`), sans écrire le split train.
- Vérifications : (1) nombre d'identifiants = **876 406**, identique au checkpoint ; (2) sur 3 tâches du sous-ensemble, le modèle reproduit **7/7** paires de démonstration avec le bon identifiant contre **0/7** avec un identifiant d'une autre tâche ; (3) premières sondes sur des tâches train/eval concordantes. Une première version (groupes tronqués) donnait 857 877 identifiants : détectée et corrigée.
- Les K augmentations sont les K premiers éléments du groupe officiel (identité puis augmentations dans l'ordre de génération), et non un tirage parmi les ~1000.

## Vote
Comme `evaluators/arc.py` : prédictions ramenées au repère d'origine (inverse de l'augmentation), tri par (nombre de votes, q_halt moyen). `scores` = nombre de votes (sur 8). Les solutions officielles ne servent qu'au scoring (`finalize.py`).

## Réserves
- K=8 au lieu de ~1000 : le chiffre officiel (~40 % ARC-1 public eval complet) n'est pas comparable ; 20 tâches = échantillon très bruité (±~11 pts d'écart-type).
- Les tâches sont des tâches d'évaluation, mais **le checkpoint a été entraîné sur les paires de démonstration de ces tâches d'évaluation** (protocole TRM officiel, test-time training implicite) : ce n'est pas du zéro-shot ; seules les sorties de test sont inédites.
- `cpu_s` = temps CPU process (2 threads), pas le temps mur.

Fichiers : `infer.py`, `build_subset.py`, `finalize.py`, `res_K8.json` (top-5 candidats), `eval_K8.log`, sortie fusion `../fusion/trm.json`.
