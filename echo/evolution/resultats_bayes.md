# Bayes, la dernière pierre

Ce document est généré par `rapport_bayes.py` à partir de `archive_bayes/*.json`. Code : `bayes.py` (méthodes), `lancer_bayes.py` (expériences), `test_bayes.py` (tests). Rien d'inventé : Pitman-Yor hiérarchique (Teh 2006 ; Wood et al. 2009), moyenne bayésienne de modèles, fixed-share (Herbster et Warmuth 1998 ; Volf et Willems 1998), échantillonnage de Thompson, bootstrap bayésien (Rubin 1981).

## 1. Protocole et donnée fraîche

- Tout ce qui se règle (δ_k, θ_k de Pitman-Yor ; α de fixed-share) l'est sur **A** seulement. **B** sert à regarder. Le test final se fait **une seule fois** sur D.
- **Du texte frais existait** : `validation.txt` compte 141271 caractères ; A+B+C en prennent 90000. La tranche **D** = les 51271 caractères restants (positions 90000 à 141271). Elle est construite par le même pipeline de comptage que A, B et C (`donnees.table_ordre`, `Empreintes`, mêmes sels), avec des comptages issus de la seule bibliothèque `memoire.xz`, ordres 0 à 10.
- **C n'a pas été rouvert** : ni lu ni utilisé. Un témoin (`cache/d_utilise.json`) empêche de relancer le test sur D par accident.
- D ne ressemble pas forcément à A, B, C : les bpc de D (autour de 1,9) sont plus hauts que ceux de B (1,46 à 1,48) parce que le texte est plus difficile, pas parce que les modèles sont moins bons. **Ne comparer que des lignes d'une même colonne.**

## 2. Résultats (bpc, plus petit = mieux)

| Modèle | bpc A (réglage) | bpc B (regard) | **bpc D (test unique, frais)** | P(meilleur que le champion) sur D |
|---|---|---|---|---|
| Départ (PPM-C) | 2.0480 | 1.7144 | **2.4715** | 0.000 |
| Champion évolué | 1.6276 | 1.4772 | **1.8968** | - |
| Référence à la main (e = 16·d) | 1.6083 | 1.4608 | **1.8973** | 0.411 |
| Pitman-Yor (A) | 1.6414 | 1.4934 | **1.9744** | 0.000 |
| BMA en ligne | 1.6084 | 1.4609 | **1.8969** | 0.450 |
| Fixed-share (α = 0.0003) | 1.6081 | 1.4229 | **1.8892** | 1.000 |
| LZMA conditionnel | - | - | 2.3137 | - |

Sur B, le champion a servi à la sélection (l'écart est donc un peu optimiste pour lui) ; la référence à la main a eu ses deux réglages choisis sur A ; Pitman-Yor est ajusté sur A et donc légèrement avantagé sur A. LZMA n'est calculé que sur D (bibliothèque + D moins bibliothèque, comme `tester.py`).

### Probabilité postérieure de battre le champion, sur D

P(meilleur) = P(bpc du modèle < bpc du champion), par **bootstrap bayésien** (poids de Dirichlet sur les 102 blocs de 500 caractères), avec en contrôle le bootstrap par blocs. Différence de bpc moyenne (modèle moins champion, négatif = le modèle gagne) et intervalle crédible à 95 %.

| Modèle | P(meilleur), bayésien | P(meilleur), blocs | différence moyenne | intervalle à 95 % |
|---|---|---|---|---|
| Départ (PPM-C) | 0.000 | 0.000 | +0.5760 | [+0.5461 ; +0.6065] |
| Référence à la main (e = 16·d) | 0.411 | 0.418 | +0.0005 | [-0.0034 ; +0.0045] |
| Pitman-Yor (A) | 0.000 | 0.000 | +0.0773 | [+0.0668 ; +0.0881] |
| BMA en ligne | 0.450 | 0.471 | +0.0000 | [-0.0018 ; +0.0017] |
| Fixed-share (α = 0.0003) | 1.000 | 1.000 | -0.0076 | [-0.0107 ; -0.0050] |

Contre la **référence à la main** (D) : champion P = 0.589, Pitman-Yor P = 0.000, BMA P = 0.591, fixed-share P = 1.000. Le champion et la référence à la main sont statistiquement indiscernables sur D (différence +0.0005 bpc).

## 3. Estimateur de Pitman-Yor : hyperparamètres ajustés sur A

Formule : `p_k = (max(n_k − δ_k, 0) + (θ_k + δ_k·d_k)·p_{k−1}) / (N_k + θ_k)`, base uniforme, `p_k = p_{k−1}` si `N_k = 0`. Les n_k sont les comptages bruts de la bibliothèque (pas de comptes de continuation). Recherche par coordonnées (section dorée sur δ_k, puis sur log(θ_k+δ_k), 6 balayages, 4 points de départ ; 8 s en tout). Les 4 départs finissent à 1.6414, 1.6415, 1.6422, 1.6414 bpc sur A : la vraisemblance est plate dans plusieurs directions.

| ordre k | δ_k | erreur std. δ (approx.) | θ_k | remise de Ney n1/(n1+2·n2) de la bibliothèque | D1 / D2 (Chen-Goodman) |
|---|---|---|---|---|---|
| 0 | 0.000 | 103.059 | 1e+04 | 0.385 | - |
| 1 | 0.000 | 1.398 | 0.001 | 0.507 | 0.51 / 1.01 |
| 2 | 0.000 | 0.044 | 0.001 | 0.528 | 0.53 / 0.95 |
| 3 | 0.000 | 0.046 | 55.1 | 0.557 | 0.56 / 1.02 |
| 4 | 0.440 | 0.143 | 14.1 | 0.588 | 0.59 / 1.03 |
| 5 | 0.379 | 0.238 | 136 | 0.620 | 0.62 / 1.04 |
| 6 | 0.525 | 0.075 | 17.3 | 0.654 | 0.65 / 1.06 |
| 7 | 0.000 | 0.021 | 310 | 0.687 | 0.69 / 1.07 |
| 8 | 0.000 | 0.060 | 1e+04 | 0.719 | 0.72 / 1.10 |
| 9 | 0.000 | 0.118 | 1e+04 | 0.750 | 0.75 / 1.13 |
| 10 | 0.698 | 0.032 | 5.64 | 0.780 | 0.78 / 1.16 |

Lecture :

- **Ça ne reproduit pas partout des remises de Kneser-Ney de 0,7 à 0,9.** Seuls les ordres 4, 5, 6, 10 ont un δ strictement positif et un θ fini : δ = 0.44, 0.38, 0.53, 0.70. L'ordre le plus long (10) a δ = 0.70, proche du bas de la fourchette 0,7 à 0,9 ; les ordres 4 à 6 sont plus bas (0,4 à 0,5).
- Aux ordres 0 à 3 (et 7 à 9), δ = 0 et θ n'est pas identifiable : les contextes courts ont des comptes énormes (N de l'ordre de 10⁵ à 10⁶) et δ ou θ y changent à peine la probabilité (erreurs standard très grandes ou θ collé à une borne). Ce sont des valeurs sans signification, pas des mesures.
- Aux ordres 8 et 9, θ est collé à la borne haute (10⁴) : le modèle choisit d'**ignorer** ces ordres (`p_k ≈ p_{k−1}`). Les comptages bruts de longs contextes sont trop peu fiables dans cette bibliothèque.
- Les remises de Ney calculées sur les comptes de comptes de la bibliothèque donnent la valeur « attendue » (colonne 5). Elles sont élevées (0,5 à 0,8, croissantes avec l'ordre ; 0,78 à l'ordre 10) aux ordres moyens et longs ; le δ ajusté est en dessous. Cela peut venir de ce que A est un texte d'évaluation (pas la distribution de la bibliothèque), et surtout de ce que nous utilisons des comptes bruts à tous les ordres : le vrai Kneser-Ney remplace les comptes des ordres inférieurs par des comptes de continuation, ce que le jeu de données ne fournit pas. La comparaison est donc indicative, pas une mesure de Kneser-Ney.
- Pitman-Yor ainsi réglé est **moins bon que le champion et que la référence à la main** : 1.6414 bpc sur A, 1.4934 sur B, 1.9744 sur D. L'écart sur D est de +0.0773 bpc contre le champion (P(meilleur) = 0.000). La raison probable : le champion et la référence gagnent beaucoup en aplatissant les scores (puissance 0,65) ou en envoyant beaucoup de masse vers l'ordre inférieur (e = 16·d), deux réglages que la forme Pitman-Yor à θ constant ne peut pas imiter ; et les comptages bruts ne sont pas des comptes de continuation.

## 4. Moyenne bayésienne de modèles et fixed-share

Membres : départ, champion, référence à la main, Pitman-Yor (4 membres, a priori uniforme 1/4). En ligne : poids a posteriori ∝ a priori × vraisemblance des caractères déjà vus, remis à l'a priori au début de chaque tranche. Fixed-share : après chaque caractère, w ← (1−α)·w + α/M, avec α choisi sur A dans {1e-5, ..., 0,1} : **α = 0.0003**.

**Borne de regret connue.** Pour le BMA simple, la perte cumulée en bits vérifie, pour tout membre i : `perte_BMA ≤ perte_i + log2(1/π_i)` (c'est une identité : la perte du mélange est −log2 Σ π_i Π p_i). Avec 4 membres et a priori uniforme, le regret ne dépasse jamais **2 bits** au total sur toute la tranche, quelle que soit sa longueur. Pour fixed-share, contre la meilleure suite de membres avec m changements : `≤ log2 M + m·log2(M/α) + (T−1−m)·log2(1/(1−α))` (forme Herbster-Warmuth / Volf-Willems, le dernier terme est le prix de l'agilité). Les deux bornes sont vérifiées par les tests.

Regret observé sur D du BMA contre le meilleur membre (champion) : **2.0000 bits** au total (borne : 2 bits), soit environ 3.9e-05 bit par caractère. Sur A et B : 2.0000 et 2.0000 bits.

Poids moyens sur D (départ / champion / référence / Pitman-Yor) : BMA = 0.000 / 0.807 / 0.192 / 0.001 ; fixed-share = 0.017 / 0.446 / 0.509 / 0.028.

Lecture : le BMA simple **ne gagne rien** (D : 1.8969 bpc contre 1.8968 pour le meilleur membre) : il se verrouille très vite sur un seul membre, et son regret est borné. Fixed-share fait mieux que tous les membres : 1.8892 bpc sur D, soit -0.0076 contre le champion (P(meilleur) = 1.000, intervalle à 95 % [-0.0107 ; -0.0050]). Il peut changer de membre quand le texte change (sur B le gain est bien plus grand : 1.4229 contre 1.4608), mais le gain vient d'un mélange entre quatre prédicteurs très semblables (mêmes comptages) : c'est un petit gain sûr, pas un nouveau modèle.

Attention à un biais : α et les hyperparamètres de Pitman-Yor sont réglés sur A, où Pitman-Yor est ajusté (légèrement avantagé). Les poids d'A ne sont pas réutilisés (les poids repartent de l'a priori à chaque tranche).

## 5. Thompson sampling pour choisir les opérateurs de mutation

Un bandit de Bernoulli par opérateur (9 opérateurs de `mutations.py`) : succès = l'enfant bat son parent sur A. Postérieur Beta(a, b) avec a priori Beta(1, 1). Pour chaque mutation on tire θ_op ~ Beta et on prend l'argmax. **Il remplace les poids du méta-génome** ; le nombre de mutations (1,5) et le pas des constantes (0,3) sont fixes. Même boucle, même sélection des parents, même fitness A et même choix du champion sur B que `moteur.py` ; seule la façon de choisir l'opérateur change. Comparaison à budget égal : 1500 évaluations d'enfants par run, 3 graines (les mêmes pour les deux modes), moteur existant = méta-génome automatique.

| Graine | meilleur A, existant | meilleur A, Thompson | B du champion, existant | B du champion, Thompson | CPU (s) existant / Thompson |
|---|---|---|---|---|---|
| 1 | 1.6157 | 1.6548 | 1.4664 | 1.4903 | 199 / 265 |
| 2 | 1.6090 | 1.6657 | 1.4485 | 1.4763 | 270 / 145 |
| 3 | 1.7010 | 1.6236 | 1.5083 | 1.4729 | 290 / 223 |
| **moyenne ± écart-type** | 1.6419 ± 0.0513 | 1.6480 ± 0.0218 | 1.4744 ± 0.0307 | 1.4798 ± 0.0092 | |

Probabilité postérieure que Thompson soit meilleur (bpc plus bas), modèle de Student sur les moyennes, a priori plat : **sur A : 0.44** (différence +0.0062, intervalle à 95 % [-0.1372 ; +0.1508]) ; **sur B : 0.41** (différence +0.0054, intervalle [-0.0761 ; +0.0871]).

Opérateurs préférés par Thompson (graine 1), moyenne postérieure du taux de succès / nombre de choix :

| Opérateur | taux de succès postérieur | nombre de choix |
|---|---|---|
| perturbe | 0.197 | 1058 |
| supprime | 0.144 | 233 |
| post | 0.143 | 427 |
| croise | 0.134 | 179 |
| enveloppe | 0.118 | 187 |
| insere | 0.090 | 78 |
| instruction | 0.075 | 39 |
| feuille | 0.073 | 42 |
| par_ordre | 0.056 | 43 |

Avec 3 graines, l'incertitude est grande : lire ce résultat comme une indication, pas une preuve. Le budget (1500 évaluations) est plus court que celui des runs précédents (2000 à 15000).

## 6. Limites

- D vient du même fichier `validation.txt` que A, B, C : même corpus, mêmes sources. Elle n'est « fraîche » que pour les décisions prises ici (aucun modèle ni réglage n'y a été ajusté), pas pour la distribution.
- Le champion et la référence à la main ont été sélectionnés avec A et B (et le champion avec C pour un test déjà publié) ; leur avance éventuelle sur D est donc honnête, mais leur comparaison à Pitman-Yor porte sur des modèles qui ont eu des réglages « puissance » ou « masse d'échappement » que Pitman-Yor n'a pas.
- Les comptes sont bruts à tous les ordres, ce qui n'est pas Kneser-Ney (comptes de continuation) : l'interprétation des δ_k est indicative. Le critère d'ajustement est la vraisemblance prédictive de A (Bayes empirique de type II), pas une évidence marginale complète ; les erreurs standard (hessienne diagonale, texte non indépendant) sont optimistes.
- Le BMA et fixed-share mélangent des modèles presque identiques (mêmes comptages). Le gain de fixed-share vient de la variation locale du meilleur membre ; il se limite à quelques millièmes de bit par caractère sur D.
- Thompson : 3 graines, 1500 évaluations, un seul réglage (pas d'oubli, a priori Beta(1,1)). Les opérateurs ne sont pas indépendants (une mutation réussit ou échoue avec les autres du même enfant : le crédit est attribué à tous les opérateurs utilisés).
- Incident d'exécution : le run Thompson a été lancé deux fois par erreur ; le doublon a été tué après environ 12 minutes, l'autre a terminé seul. Les résultats en bpc ne dépendent pas du temps (graines fixes, évaluations déterministes), mais les temps CPU du tableau sont gonflés par cette contention et ne sont pas comparables ; pendant ces minutes il y a eu plus de 4 processus de calcul. Les journaux `archive_bayes/*.jsonl` des deux copies se sont écrasés mutuellement : seul `thompson.json` fait foi.
- LZMA conditionnel est une référence grossière (pas un modèle de même nature).

## 7. Fichiers

`bayes.py`, `lancer_bayes.py`, `rapport_bayes.py`, `test_bayes.py`, `archive_bayes/*.json`, `cache/d.npz` (tranche D), `cache/bayes_comptes_de_comptes.json`, `cache/d_utilise.json` (témoin). Aucun fichier épinglé n'a été modifié.
