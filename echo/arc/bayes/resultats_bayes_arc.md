# Fusion bayésienne TRM + DSL sur ARC, et valeur du calcul

## Ce qui a été fait
Remplacer la règle de fusion fixée à l'avance (« programme DSL cohérent d'abord », 43 % de justesse) par de la décision bayésienne,
avec des méthodes standard uniquement : calibration bêta–binomiale (a priori Beta(1,1) ; Dirichlet(1,…,1) pour les variables à
plusieurs modalités), combinaison naïve de Bayes, choix à utilité espérée, puis un routeur « valeur du calcul » (Russell & Wefald 1991).

**Nouvelles données de calibration.** 20 nouvelles tâches d'évaluation, tirées avec la graine 2026 parmi les tâches à une seule entrée de test,
disjointes des 20 anciennes (`sous_ensemble_bayes.json`), passées dans le TRM officiel exactement comme avant (K = 8, même inférence et même
vote ; `build_subset.py` et `infer.py` étendus par des paramètres optionnels, comportement par défaut inchangé ; 2 threads au plus). Les sorties
attendues n'ont été lues qu'après la fin de l'inférence. Résultats bruts : `res_K8_bayes.json`, `eval_K8_bayes.log`. TRM seul sur ces 20 : pass@1
9, pass@2 9 / 20 ; ~363 s CPU par tâche.

## Modèle
Candidats d'une tâche : top-5 de TRM (votes sur 8, q) ∪ tentatives DSL cohérentes avec les exemples. P(juste | évidence) par naïf de Bayes sur :
(1) classe de votes TRM {absent, 1, 2, 3-4, 5+} ; (2) rang TRM {absent, 1, 2, 3+} ; (3) produit par le DSL (oui/non) ; (4) accord TRM–DSL (le candidat est dans les deux).
- Vraisemblances TRM (1, 2, 4) et probabilité a priori : estimées sur l'**autre pli** (comptes + 1)/(N + K).
- Vraisemblances DSL (3) : estimées sur **toutes les tâches d'évaluation hors pli de test** (380) : P(DSL | juste) = (tâches dont la solution est produite par le DSL + 1)/(380 + 2) ;
  P(DSL | faux) = (candidats DSL faux + 1)/(taille estimée du lot de faux candidats + 2), où ce lot = faux DSL + 380 × (faux TRM par tâche, mesuré sur l'autre pli).
- Les paramètres sont écrits dans `params_pli1_complet.json` / `params_pli2_complet.json` (empreinte SHA-256) **avant** la notation du pli de test, relus puis vérifiés ; `Verite` interdit en code la lecture des solutions du pli de test pendant l'ajustement (testé).
- Choix : les 2 candidats de plus forte probabilité a posteriori.
- Variante « sobre » (secondaire, non pré-déclarée comme principale) : seulement classe de votes + DSL, pour mesurer l'effet de la dépendance entre variables que le naïf de Bayes ignore.

Paramètres appris, pli 1 (test = anciennes 20) : a priori 0.146 ; log-RV votes (absent,1,2,3-4,5+) = +0.62, -1.71, -0.26, +1.24, +2.78 ; rang = +0.67, +1.40, -1.31, -1.36 ; accord = -0.25, +2.80 ; DSL (non, oui) = -0.08, +0.51.
Pli 2 (test = nouvelles 20) : a priori 0.170 ; votes = -1.09, -1.37, +0.44, +0.99, +3.48 ; rang = -1.04, +1.44, -0.17, -1.10 ; accord = -0.18, +1.94 ; DSL = -0.06, +0.39.

## Résultats de fusion (pass@2 ; pli 1 = anciennes 20, pli 2 = nouvelles 20)
| Méthode | Pli 1 | Pli 2 | Total 40 |
|---|---|---|---|
| TRM seul (2 premières tentatives) | 12 / 20 | 9 / 20 | 21 / 40 |
| DSL seul | 2 / 20 | 3 / 20 | 5 / 40 |
| Ancienne règle (DSL cohérent d'abord) | 8 / 20 | 10 / 20 | 18 / 40 |
| **Fusion bayésienne (4 variables, principale)** | 11 / 20 | 9 / 20 | 20 / 40 |
| Fusion bayésienne « sobre » (votes + DSL) | 11 / 20 | 10 / 20 | 21 / 40 |
| Plafond (une bonne réponse dans TRM top-5 ∪ DSL) | 14 / 20 | 11 / 20 | 25 / 40 |

Contrôle : la règle ancienne recalculée (sans CompressARC) donne 8 / 20 sur le pli 1, comme le chiffre publié (8).
Pass@1 : TRM 19 / 40, fusion bayésienne 18 / 40.

## Calibration des posteriors sur les plis de test (Brier / ECE, `ece()` de `soi.py`, 10 seaux)
Variante principale :

| Population | n (pli 1) | Brier / ECE | n (pli 2) | Brier / ECE | n (40) | Brier / ECE |
|---|---|---|---|---|---|---|
| toutes les candidates | 86 | 0.100 / 0.122 | 80 | 0.063 / 0.069 | 166 | 0.082 / 0.090 |
| (référence : a priori seul) | | 0.137 / – | | 0.120 / – | | 0.128 / – |
| meilleur candidat (top-1) | 20 | 0.111 / 0.183 | 20 | 0.138 / 0.139 | 40 | 0.124 / 0.158 |
| (référence : a priori seul) | | 0.340 / – | | 0.326 / – | | 0.333 / – |

Variante sobre : Brier / ECE sur toutes les candidates, 40 tâches = 0.077 / 0.057 ; top-1 = 0.142 / 0.079.

## Valeur du calcul (DSL d'abord, TRM seulement si gain espéré > λ × coût)
Gain espéré de TRM pour une tâche = max(0, p_après − p_maintenant(g)), où g = nombre de candidats DSL distincts (0, 1 ou 2) ;
p_maintenant(g) = taux de réussite du DSL (Beta(1,1)) sur les tâches d'évaluation hors pli de test de même groupe (0 si g = 0) ; p_après = taux pass@2 de TRM (Beta(1,1)) sur le pli de calibration.
Règle de choix de λ fixée avant de voir le pli de test : le plus grand λ dont le pass@2 sur le pli de calibration reste à moins d'une tâche de « toujours TRM ».
Coûts : DSL = CPU mesuré (≈ 7,5 s en moyenne, ≤ 12 s) ; TRM = CPU mesuré par tâche. Pli de test 1 : calibration sur les nouvelles 20 ; pli de test 2 : sur les anciennes 20.
Dans la branche « TRM lancé », la réponse est celle de la fusion bayésienne figée.

**Pli de test 1 (anciennes 20)** (λ choisi sur l'autre pli : 1.20e-03 ; coût TRM moyen mesuré sur le pli de calibration : 363 s CPU ; gain espéré par groupe de candidats DSL 0 / 1 / 2 : 0.455 / 0.106 / 0.000 ; p_après = 0.455)

| Politique | pass@2 | CPU total (s) | Tâches où TRM est lancé |
|---|---|---|---|
| DSL seul (jamais TRM) | 2 / 20 | 139.5 | 0 |
| **VoC, λ choisi** | **6 / 20** | **3939.7** | 11 |
| Toujours TRM + fusion bayésienne | 11 / 20 | 7053.3 | 20 |

Courbe (tous les λ de la grille, politiques distinctes) :

| λ | pass@2 | CPU (s) | TRM lancé |
|---|---|---|---|
| 0.00e+00 | 8 | 5328.8 | 15 |
| 2.97e-04 | 6 | 3939.7 | 11 |
| 1.26e-03 | 2 | 139.5 | 0 |

**Pli de test 2 (nouvelles 20)** (λ choisi sur l'autre pli : 1.41e-04 ; coût TRM moyen mesuré sur le pli de calibration : 346 s CPU ; gain espéré par groupe de candidats DSL 0 / 1 / 2 : 0.591 / 0.246 / 0.050 ; p_après = 0.591)

| Politique | pass@2 | CPU total (s) | Tâches où TRM est lancé |
|---|---|---|---|
| DSL seul (jamais TRM) | 3 / 20 | 164.4 | 0 |
| **VoC, λ choisi** | **9 / 20** | **7423.8** | 20 |
| Toujours TRM + fusion bayésienne | 9 / 20 | 7423.8 | 20 |

Courbe (tous les λ de la grille, politiques distinctes) :

| λ | pass@2 | CPU (s) | TRM lancé |
|---|---|---|---|
| 0.00e+00 | 9 | 7423.8 | 20 |
| 1.48e-04 | 9 | 6697.2 | 18 |
| 7.30e-04 | 10 | 5614.4 | 15 |
| 1.79e-03 | 3 | 164.4 | 0 |

## Lecture honnête
- **La fusion bayésienne répare la règle ancienne mais ne bat pas TRM seul.** Sur 40 tâches : ancienne règle 18, bayésienne 20 (variante sobre : 21), TRM seul 21, plafond 25. Pli 1 : 8 → 11 (TRM : 12) ; pli 2 : 10 → 9 (TRM : 9). Sur le pli 2, l'ancienne règle fait mieux (10) car le DSL y résout c3202e5a, que TRM rate : le modèle, calibré sur le pli 1 où le DSL ne résout rien que TRM rate, lui donne 0,15 de probabilité et l'écarte. Ce sont des écarts d'une ou deux tâches, dans le bruit.
- **Pourquoi pas mieux** : le plafond (25) n'est que 4 tâches au-dessus de TRM, et ces 4 réponses sont soit en 3e à 5e position de TRM, soit des sorties DSL isolées de probabilité faible (0,15 à 0,64). Avec 20 tâches de calibration par pli (11 à 14 candidates justes), les vraisemblances sont très incertaines (a priori Beta(1,1) pesant lourd).
- **Calibration** : les posteriors battent nettement l'a priori seul (Brier toutes candidates 0,082 contre 0,128 sur 40 tâches), mais sont surconfiants pour le meilleur candidat (ECE top-1 = 0,158 pour la variante principale ; la variante sobre est mieux calibrée : 0,079). Cohérent avec le double comptage de variables dépendantes. Avec 20 points par pli, l'ECE est elle-même très bruitée.
- **Valeur du calcul : résultat négatif avec la règle de choix de λ fixée d'avance.** TRM représente plus de 98 % du CPU (DSL : ~140–160 s par pli de 20 tâches ; TRM : ~7 000 s). Pli 1 : λ choisi sur le pli 2 (où le DSL résout 3 tâches contre 9 pour TRM) coupe TRM sur 9 tâches : CPU −44 % mais pass@2 6 contre 11, perte inacceptable. Pli 2 : λ choisi sur le pli 1 donne exactement « toujours TRM » (aucune économie). Le point 10 / 20 à −24 % de CPU de la courbe du pli 2 n'est pas retenu par la règle (choix a posteriori, à ne pas revendiquer). Raison de fond : le DSL ne produit une réponse que sur 149 tâches sur 400, et ses réponses ne sont justes que dans 43 % des cas ; aucune configuration ne le rend assez fiable pour se passer de TRM, et la décision « lancer TRM » est presque toujours rentable (TRM résout ~50 % des tâches contre 15 % pour le DSL).
- **Conclusion** : la décision bayésienne est un cadre correct et mieux calibré que la règle d'ordre fixe, qui avait un défaut de principe (sur-confiance dans « vérifié »), mais sur cet échantillon elle égale TRM à une tâche près et n'économise du CPU que si on accepte de perdre des tâches. Ce n'est pas une amélioration démontrée.

## Réserves
- 20 + 20 tâches : écart-type d'environ ±8 points sur chaque taux ; une tâche = 5 points. Aucune différence entre méthodes ici n'est significative.
- TRM a été entraîné sur les paires de démonstration de ces tâches d'évaluation (protocole officiel) : seules les sorties de test sont inédites. Ses votes sont donc plus fiables ici que sur des tâches jamais vues.
- Le naïf de Bayes double-compte des variables dépendantes (votes/rang/accord/DSL) ; la variante « sobre » le montre. Le côté DSL est estimé sur toutes les solutions, alors que le lot de candidats ne contient la solution que si une méthode l'a trouvée : approximation notée dans le code.
- La courbe VoC du pli de test est descriptive ; seul le point « λ choisi » respecte la règle fixée d'avance.

## Fichiers
`bayes_arc.py` (modèle), `lancer.py` (protocole : ajustement → gel → notation), `rapport_bayes.py`, `test_bayes_arc.py` (8 tests), `sous_ensemble_bayes.json`, `res_K8_bayes.json`, `eval_K8_bayes.log`, `params_pli*_{complet,sobre}.json`, `resultats_bayes_arc.json`.
Modifs de code existant (comportement par défaut inchangé) : `../trm_officiel/build_subset.py` (arguments optionnels), `../trm_officiel/infer.py` (variables TRM_PKL, TRM_OUT). Aucun fichier de résultats existant n'a été modifié.
