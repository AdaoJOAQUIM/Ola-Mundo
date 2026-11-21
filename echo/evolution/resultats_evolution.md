# Résultats de l'évolution d'estimateurs

Ce document est généré par `rapport.py` à partir des mesures (`archive/*.json`, `resultats_test.json`).

## Ce qu'on a fait

Un moteur réécrit le code Python d'un estimateur de probabilités (la fonction `estimer`). Il garde un génome seulement s'il est meilleur sur des données qu'il n'a pas servi à fabriquer. Il conserve une archive ouverte de toutes les variantes valides (comme la Darwin Gödel Machine). Il fait aussi évoluer sa propre façon de muter (le « méta-génome »).

- Les comptages viennent uniquement de la bibliothèque (`memoire.xz`), figés, sans adaptation. Ordres 0 à 10.
- A (15 000 caractères de `validation.txt`) donne la fitness. B (15 000 suivants) sert à choisir le champion. C (60 000 suivants) n'est lu qu'une fois, par `tester.py`.
- Le juge, la liste blanche et les fichiers de données sont épinglés par SHA-256 (`empreintes.json`). Le moteur vérifie ces empreintes au départ puis toutes les 25 générations.
- Départ : PPM-C interpolé, sans exclusion, avec des comptages figés.

## 1. Test final sur C (utilisé une seule fois)

C = 60000 caractères de `validation.txt`, jamais utilisés pour choisir quoi que ce soit.

| Modèle | bpc sur C |
|---|---|
| Départ (PPM-C interpolé, ordre 10, comptages figés) | 1.8632 |
| **Champion évolué** | **1.5377** |
| Référence à la main (e = 16·d, puissance 1.0 ; réglages choisis sur A) | 1.5241 |
| LZMA conditionnel (biblio + C moins biblio) | 1.9253 |

Gain relatif du champion sur le départ : **17.5 %** (intervalle à 95 % par blocs de 500 caractères : 16.4 % à 18.5 %).

Écho-P d'origine (`predicteur.py`, ordre 16, exclusion) est lent. Il a été évalué sur un **sous-ensemble de C** : 24 blocs de 400 caractères répartis régulièrement sur C (9600 caractères). Départ et champion sont mesurés sur exactement les mêmes positions.

| Modèle (sous-ensemble de C) | bpc |
|---|---|
| Écho-P d'origine | 1.9182 |
| Départ | 1.8991 |
| Champion | 1.5444 |
| Référence à la main | 1.5328 |


Scores sur A et B (champion choisi sur B) : départ A = 2.0480, B = 1.7144 ; champion A = 1.6276, B = 1.4772.

### Diagnostic par catégorie de caractère (sur C)

| Catégorie | Part des caractères | bpc départ | bpc champion | Part du gain total |
|---|---|---|---|---|
| minuscule | 74.8 % | 1.940 | 1.636 | 69.8 % |
| espace | 18.6 % | 0.441 | 0.396 | 2.6 % |
| ponctuation | 3.4 % | 4.738 | 3.553 | 12.3 % |
| majuscule | 2.5 % | 5.801 | 4.115 | 13.1 % |
| retour_ligne | 0.5 % | 1.244 | 0.971 | 0.4 % |
| chiffre | 0.3 % | 6.233 | 4.154 | 1.8 % |


## 2. Run principal (méta-génome automatique)

6000 évaluations, 750 générations, archive de 3926 génomes, 482 s de temps mur. Départ : A = 2.0480, B = 1.7144. Champion (génome n°3321) : A = 1.6276, B = 1.4772.

| Évaluations | meilleur A | meilleur B (champion) |
|---|---|---|
| 256 | 1.6682 | 1.4923 |
| 504 | 1.6682 | 1.4923 |
| 752 | 1.6682 | 1.4923 |
| 1000 | 1.6682 | 1.4923 |
| 1256 | 1.6682 | 1.4923 |
| 1504 | 1.6682 | 1.4876 |
| 1752 | 1.6506 | 1.4876 |
| 2000 | 1.6414 | 1.4876 |
| 2256 | 1.6414 | 1.4876 |
| 2504 | 1.6414 | 1.4876 |
| 2752 | 1.6414 | 1.4876 |
| 3000 | 1.6414 | 1.4857 |
| 3256 | 1.6414 | 1.4857 |
| 3504 | 1.6414 | 1.4857 |
| 3752 | 1.6414 | 1.4857 |
| 4000 | 1.6414 | 1.4857 |
| 4256 | 1.6414 | 1.4857 |
| 4504 | 1.6414 | 1.4857 |
| 4752 | 1.6414 | 1.4857 |
| 5000 | 1.6414 | 1.4857 |
| 5256 | 1.6276 | 1.4772 |
| 5504 | 1.6276 | 1.4772 |
| 5752 | 1.6276 | 1.4772 |
| 6000 | 1.6276 | 1.4772 |


## 3. Ablation : méta-génome automatique contre fixe

5 graines par mode, 2000 évaluations par run, même génome de départ. Moyenne ± écart-type entre graines.

| Évaluations | A auto | A fixe | B auto | B fixe |
|---|---|---|---|---|
| 256 | 1.6509 ± 0.0164 | 1.6866 ± 0.0451 | 1.5068 ± 0.0239 | 1.4902 ± 0.0257 |
| 504 | 1.6509 ± 0.0164 | 1.6580 ± 0.0277 | 1.5068 ± 0.0239 | 1.4800 ± 0.0238 |
| 752 | 1.6479 ± 0.0168 | 1.6580 ± 0.0277 | 1.5068 ± 0.0239 | 1.4793 ± 0.0238 |
| 1000 | 1.6464 ± 0.0157 | 1.6536 ± 0.0291 | 1.5060 ± 0.0230 | 1.4793 ± 0.0238 |
| 1256 | 1.6426 ± 0.0179 | 1.6402 ± 0.0289 | 1.5019 ± 0.0250 | 1.4770 ± 0.0245 |
| 1504 | 1.6425 ± 0.0178 | 1.6402 ± 0.0289 | 1.5014 ± 0.0245 | 1.4770 ± 0.0245 |
| 1752 | 1.6425 ± 0.0178 | 1.6402 ± 0.0289 | 1.5014 ± 0.0245 | 1.4770 ± 0.0245 |
| 2000 | 1.6425 ± 0.0179 | 1.6402 ± 0.0289 | 1.5014 ± 0.0245 | 1.4770 ± 0.0245 |


Valeurs finales par graine, meilleur A : auto = 1.6551, 1.6232, 1.6183, 1.6557, 1.6601 ; fixe = 1.6355, 1.6971, 1.6215, 1.6255, 1.6212.
Valeurs finales par graine, B du champion : auto = 1.4918, 1.4814, 1.4729, 1.5317, 1.5292 ; fixe = 1.4782, 1.5222, 1.4674, 1.4680, 1.4491.
Test exact par permutation sur la différence des moyennes : p = 0.984 (A), p = 0.159 (B).

Lecture, en phrases simples :

- **Pas de différence démontrée.** Avec 5 graines, la différence entre méta-génome automatique et fixe n'est pas significative (p = 0,98 sur A, 0,16 sur B). L'écart entre graines est plus grand que l'écart entre modes.
- Au début (250 évaluations), le mode automatique est un peu meilleur en moyenne sur A. À 2000 évaluations, le mode fixe est un peu meilleur, surtout sur B. On ne peut pas conclure que l'auto-adaptation aide dans ce budget.
- **Ce que le méta-génome a fait quand même :** dans les 5 runs automatiques, le nombre moyen de mutations par enfant est descendu de 2,0 à environ 1,3 – 1,7. Les poids des opérateurs et le pas des constantes ont dérivé (voir `meta_moyen_final` dans `resultats_evolution.json`). Le moteur « préfère » donc des changements plus petits. C'est un signal réel, mais il ne se traduit pas en meilleur score mesuré.
- Les courbes sont des « meilleur jusqu'ici » : elles ne descendent jamais. Les longs paliers montrent que la recherche stagne vite après le premier gain.

## 3 bis. Run long supplémentaire (A et B seulement, jamais testé sur C)

Pour voir si un budget plus grand dépasse la référence à la main : 15000 évaluations, graine 2, méta automatique, 1217 s. Son champion n'a **pas** été passé sur C (C ne sert qu'une fois, pour le champion du run principal).

| Évaluations | meilleur A | meilleur B |
|---|---|---|
| 1000 | 1.6331 | 1.4557 |
| 2000 | 1.6122 | 1.4557 |
| 3000 | 1.6122 | 1.4557 |
| 4000 | 1.6121 | 1.4557 |
| 5000 | 1.6121 | 1.4557 |
| 6000 | 1.6119 | 1.4537 |
| 7000 | 1.6119 | 1.4537 |
| 8000 | 1.6119 | 1.4537 |
| 9000 | 1.6097 | 1.4537 |
| 10000 | 1.6097 | 1.4537 |
| 11000 | 1.6097 | 1.4537 |
| 12000 | 1.6097 | 1.4537 |
| 13000 | 1.6077 | 1.4537 |
| 14000 | 1.6077 | 1.4537 |
| 15000 | 1.6077 | 1.4537 |


Champion du run long : A = 1.6299, B = 1.4537. Référence à la main : A = 1.6083, B = 1.4608. Champion du run principal : A = 1.6276, B = 1.4772.
Sur A, le run long reste derrière que la référence à la main ; sur B, il est devant (B a servi à choisir son champion, donc cette mesure est un peu optimiste).

## 4. Génomes rejetés, par raison

Total sur le run principal et les 10 runs d'ablation (le run long n'est pas compté). Aucune mutation n'a été refusée par la liste blanche : les rejets viennent tous du juge (sortie de mauvaise forme, non finie, négative, ou erreur à l'exécution). Aucun délai dépassé, aucune mémoire dépassée.

| Raison | Nombre |
|---|---|
| forme_incorrecte | 1136 |
| non_fini | 1094 |
| negatif | 789 |
| erreur_execution | 3 |


Génomes malveillants écrits à la main pour les tests, tous refusés par la liste blanche :

| Essai | Raison du rejet |
|---|---|
| open | appel_interdit |
| __import__ | dunder |
| import os | noeud_interdit |
| np.load | attribut_interdit |
| n.__class__ | dunder |
| exec | appel_interdit |
| lambda | noeud_interdit |
| while | noeud_interdit |
| getattr | appel_interdit |
| eval | appel_interdit |
| compréhension | noeud_interdit |
| np.save | attribut_interdit |
| attribut .sum | attribut_interdit |
| puissance géante | puissance_non_bornee |
| chaîne | constante_interdite |
| dunder nom | dunder |
| boucle non range | boucle_interdite |


## 5. Tests unitaires

```
OK     test_200_mutations_parsent
OK     test_arret_precoce
OK     test_comptages_contre_force_brute
OK     test_cout_mesure
OK     test_empreintes_detectent_modification
OK     test_empreintes_du_depot_intactes
OK     test_evaluateur_en_processus
OK     test_juge_constant_donne_log2_A
OK     test_juge_rejette_nan_negatif_forme
OK     test_liste_blanche_graine
OK     test_liste_blanche_rejette_malveillants
OK     test_mdl_et_descripteurs_map_elites
OK     test_meta_heritee_borne
OK     test_moteur_complet_court_et_arrets
OK     test_moteur_ne_touche_pas_c
OK     test_pareto_jouet
OK     test_soi_apprend_et_se_calibre
OK     test_taille_et_syntaxe
18 tests, 0 échec(s)
```

## 6. Ce que le moteur a découvert (lecture de la formule du champion)

Le champion (voir `champion.py`, et `lignee.md` pour son histoire) fait, à chaque ordre k :

```
m = n[k] + d[k]
e = d[k] * (une constante proche de 2,2, propre à chaque ordre) + u1[k]
p = si N[k] > 0.25 :  (n[k] + (e + m) * p) / N[k]     sinon p
...
p = p ** 0.649
```

En clair :

1. **Il a donné beaucoup plus de poids aux ordres courts.** Dans le départ (PPM-C), la masse envoyée vers l'ordre inférieur vaut `d` (le nombre de symboles distincts vus). Dans le champion elle vaut environ `3,2·d + u1 + n`. Le départ faisait trop confiance aux longs contextes vus peu de fois. Retirer ce terme fait remonter le bpc de A de 1,628 à 1,749 (`analyse_champion.json`).
2. **Il a ajouté une puissance finale (0,65).** Comme le juge renormalise les scores, élever à une puissance < 1 aplatit la distribution : c'est un réglage de « température ». Retirer cette ligne coûte environ 0,10 bpc sur A (1,628 à 1,729). C'est la plus grosse pièce isolée.
3. **Il a changé le dénominateur** : `N[k]` seul, au lieu de `N[k] + e`. Comme les scores ne sont pas des probabilités exactes, le juge les renormalise ; ce n'est donc pas une erreur, mais ce n'est plus du PPM au sens strict.
4. **Les autres pièces sont neutres.** Le terme `u1` (symboles vus une seule fois, un clin d'oeil à Good-Turing) et les constantes différentes par ordre n'ont aucun effet mesurable : A passe de 1,6276 à 1,6289 sans `u1` et à 1,6272 sans constantes par ordre. Ce sont des restes de la dérive aléatoire.
5. **Il n'a PAS trouvé de décompte à la Kneser-Ney.** Il n'y a ni soustraction d'une remise D sur les comptes, ni comptes de continuation. Le `np.maximum(n[k], 0.0)` est inutile (n est déjà positif). L'idée de la « remise absolue » demande plusieurs mutations coordonnées, que la recherche n'a pas assemblées avec ce budget.

**Point d'honnêteté important.** Une référence écrite à la main, avec seulement deux réglages choisis sur A (`e = 16·d` et aucune puissance), fait mieux que le champion : voir le tableau de la section 1 (et `analyse_champion.json`). Le moteur a donc surtout redécouvert que « le génome de départ envoie trop peu de masse vers l'ordre inférieur et qu'il faut aplatir ». Il ne l'a pas fait de façon plus efficace qu'un réglage manuel à une constante. Les mutations de constantes de pas ~30 % montent lentement de 1 vers 16 ; c'est une limite de ce moteur à budget de quelques milliers d'évaluations.

## 9. Moteur complet : cinq composants ajoutés

Les garde-fous et le protocole A/B/C sont inchangés (fichiers épinglés non touchés). Les runs de cette partie sont jugés sur A et B seulement : C a déjà servi une fois et n'est pas rouvert. Run complet : 4240 évaluations, 530 générations, 749 s de CPU cumulé, champion (B) A = 1.6270, B = 1.4768. Rappel : départ A = 2.0480, B = 1.7144.

### 9.1 Soi (auto-modèle)

Un petit modèle logistique (numpy, Newton + pénalité L2) prédit, avant l'évaluation, si l'enfant bat son parent. Variables : type des mutations, rang du parent, variation de taille, taux de succès passé des opérateurs, nombre de mutations. La calibration suit `echo/soi.py` : une température ajustée par section dorée sur les 25 % d'exemples les plus récents, et l'ECE importée de `soi.py`. On saute un enfant si sa probabilité prédite est inférieure à 0,35 fois le taux de succès observé, sauf exploration (epsilon = 0,1).

Run complet principal : 4240 exemples, taux de succès observé 0.117, **428 enfants sautés** (soit 9.2 % de ceux qui auraient été évalués). ECE prédictif (prédictions faites avant l'évaluation) = **0.0130** ; Brier = 0.1008 contre 0.1028 pour un taux constant ; température T = 0.90.

Contrôle de l'exploration : parmi les enfants prédits mauvais mais évalués quand même (epsilon), le taux de succès est 0.055 (n = 55) ; pour les autres enfants évalués il est 0.117 (n = 3985). S'ils étaient proches, le saut ne perdrait pas grand-chose ; un écart net montre que le modèle trie bien.

Sur les 4 runs de l'ablation (complet) : 3180 enfants sautés pour 9604 évaluations faites ; ECE prédictif moyen 0.0124.

**Qualité à CPU égal** (budget : 300 s de CPU cumulé par run ; moteur simple = ancien moteur, moteur complet = les cinq composants ; 4 graines chacun) :

| CPU cumulé (s) | A simple | B simple | A complet | B complet |
|---|---|---|---|---|
| 25 | - | - | - | - |
| 50 | 1.6752 ± 0.0346 | 1.4902 ± 0.0138 | 1.6614 ± 0.0386 | 1.4872 ± 0.0156 |
| 75 | 1.6718 ± 0.0325 | 1.4854 ± 0.0122 | 1.6337 ± 0.0210 | 1.4711 ± 0.0165 |
| 100 | 1.6665 ± 0.0341 | 1.4840 ± 0.0147 | 1.6206 ± 0.0056 | 1.4697 ± 0.0159 |
| 125 | 1.6641 ± 0.0325 | 1.4833 ± 0.0144 | 1.6182 ± 0.0064 | 1.4641 ± 0.0089 |
| 150 | 1.6637 ± 0.0323 | 1.4830 ± 0.0143 | 1.6172 ± 0.0072 | 1.4641 ± 0.0089 |
| 175 | 1.6637 ± 0.0323 | 1.4830 ± 0.0143 | 1.6146 ± 0.0063 | 1.4641 ± 0.0089 |
| 200 | 1.6591 ± 0.0296 | 1.4814 ± 0.0129 | 1.6113 ± 0.0063 | 1.4602 ± 0.0079 |
| 225 | 1.6421 ± 0.0288 | 1.4723 ± 0.0172 | 1.6113 ± 0.0063 | 1.4602 ± 0.0079 |
| 250 | 1.6421 ± 0.0288 | 1.4723 ± 0.0172 | 1.6099 ± 0.0043 | 1.4592 ± 0.0082 |
| 275 | 1.6421 ± 0.0288 | 1.4723 ± 0.0172 | 1.6097 ± 0.0044 | 1.4592 ± 0.0082 |
| 300 | 1.6421 ± 0.0288 | 1.4723 ± 0.0172 | 1.6093 ± 0.0046 | 1.4582 ± 0.0084 |


Évaluations faites : simple = [888, 1392, 2720, 1752] ; complet = [2048, 2240, 2356, 2960]. Valeurs finales (meilleur A) : simple = 1.6073, 1.6214, 1.6614, 1.6725 ; complet = 1.6073, 1.6161, 1.6061, 1.6067. Test exact par permutation : p = 0.057 (A), 0.229 (B sur le champion).

Gain par seconde de CPU (bpc gagnés sur A / CPU cumulé) : simple = 0.00135, complet = 0.00146.

### 9.2 Objectif explicite (bpc + longueur de description) et front de Pareto

Deux objectifs à minimiser : le bpc sur A et la longueur de description du génome (MDL : octets compressés par zlib x 8). Le tri non dominé est celui de NSGA-II (rang et distance d'encombrement ; fonctions testées sur un exemple jouet). Pour choisir les parents, la fitness est bpc + 0,5 x (bits du génome / 15000 caractères de A) + pénalité de coût ; les membres du front ont un poids x1,5. Le facteur 0,5 est un choix arbitraire, pas un réglage optimisé. Le champion reste choisi sur B (bpc seul). Une cible et un arrêt sont possibles (`--cible`, par exemple un bpc sur B à atteindre).

Front de Pareto final de l'archive (13 points ; les plus petits en bpc) :

| Génome | bpc sur A | Taille (bits compressés) |
|---|---|---|
| 2281 | 1.6209 | 2264 |
| 2659 | 1.6220 | 1832 |
| 2359 | 1.6222 | 1768 |
| 2654 | 1.6227 | 1752 |
| 1422 | 1.6239 | 1680 |
| 1814 | 1.6239 | 1672 |
| 1475 | 1.6239 | 1640 |
| 2138 | 1.6249 | 1288 |
| 1980 | 1.7364 | 1200 |
| 2549 | 1.7364 | 1192 |
| 1901 | 1.7364 | 1152 |
| 2451 | 1.7766 | 1104 |


### 9.3 Coût (CPU mesuré)

Chaque évaluation est chronométrée dans son processus (`time.process_time`, comme le coût mesuré de `aiguilleur.py`). Moyenne 0.165 s, médiane 0.157 s, maximum 0.706 s par évaluation (4 processus en parallèle sur 4 coeurs). Gain par seconde de CPU du run complet : 0.00057 bpc/s (gain de 0.427 bpc sur A pour 749 s de CPU cumulé). Un génome plus lent que le départ est pénalisé dans la sélection : + 0,01 bpc par multiple de temps de plus, plafonné à 0,1. Les mesures de CPU sont bruitées (machine chargée) ; c'est un ordre de grandeur.

### 9.4 Convergence

Règle d'arrêt précoce standard : arrêt si le meilleur bpc sur B ne baisse pas de plus de delta = 0.002 pendant une fenêtre de patience de 2000 évaluations. **Raison de l'arrêt du run complet : patience (aucun gain > 0.0020 sur B depuis 2000 évaluations)**, après 4240 évaluations (budget fixé à 6000).

| Évaluations | CPU (s) | meilleur A | meilleur B | gain par CPU-s |
|---|---|---|---|---|
| 192 | 26.7 | 1.7360 | 1.5308 | 0.01169 |
| 432 | 77.0 | 1.6388 | 1.4896 | 0.00531 |
| 664 | 126.3 | 1.6378 | 1.4822 | 0.00325 |
| 912 | 175.1 | 1.6316 | 1.4822 | 0.00238 |
| 1192 | 225.6 | 1.6293 | 1.4822 | 0.00186 |
| 1464 | 275.9 | 1.6269 | 1.4822 | 0.00153 |
| 1752 | 326.1 | 1.6263 | 1.4822 | 0.00129 |
| 2056 | 376.2 | 1.6263 | 1.4822 | 0.00112 |
| 2352 | 425.7 | 1.6239 | 1.4768 | 0.00100 |
| 2752 | 494.6 | 1.6239 | 1.4768 | 0.00086 |
| 3000 | 537.9 | 1.6233 | 1.4768 | 0.00079 |
| 3256 | 579.9 | 1.6233 | 1.4768 | 0.00073 |
| 3528 | 625.6 | 1.6233 | 1.4768 | 0.00068 |
| 3824 | 675.7 | 1.6209 | 1.4768 | 0.00063 |
| 4112 | 726.1 | 1.6209 | 1.4768 | 0.00059 |
| 4240 | 748.5 | 1.6209 | 1.4768 | 0.00057 |


**Point fixe** (idée de `induction.py` : relancer et vérifier que rien ne change) : on relance l'évolution pendant 600 évaluations en partant du champion (A = 1.6270). Elle atteint A = 1.6212, soit un gain de 0.0059 (seuil delta = 0.002). Verdict : **PAS un point fixe (la recherche trouve encore mieux)**. Sur B : 1.4768 -> 1.4722.

### 9.5 Émergence (archive MAP-Elites)

Grille simple (pyribs n'est pas utilisé). Descripteurs : taille de l'AST (5 classes) x utilise u1 ou u2 x contient une remise (soustraction d'un terme directement au compte brut n[k], dans la boucle) x constantes propres à chaque ordre : 40 cases possibles. Chaque case garde son meilleur génome. **Couverture : 14 cases sur 40 (35 %)**, soit 14 élites.

Structures présentes dans des élites et absentes du génome de départ :

| Structure | Élites concernées | Meilleur A | Retirer la structure change-t-il A ? | Technique connue ? |
|---|---|---|---|---|
| np.maximum | 7 | 1.6220 | non testé | fonction utilitaire (clip, abs, ...) : pas une technique nommée |
| variable_intermediaire | 13 | 1.6220 | non testé | pas une technique : simple réécriture |
| constantes_par_ordre | 7 | 1.6220 | 6/7 élites inchangées (delta A médian +0.0000) | poids différents par ordre : technique connue (interpolation à poids par ordre) |
| utilise_u2 | 3 | 1.6220 | 2/3 élites inchangées (delta A médian +0.0000) | nombre de symboles vus deux fois : idée de Good-Turing / Kneser-Ney modifié |
| utilise_u1 | 4 | 1.6236 | 4/4 élites inchangées (delta A médian +0.0000) | nombre de symboles vus une fois : idée de Good-Turing / Witten-Bell / PPM-D |
| np.log1p | 4 | 1.6249 | non testé | fonction utilitaire (clip, abs, ...) : pas une technique nommée |
| remise | 5 | 1.6287 | non testé | remise absolue / Kneser-Ney (soustraire une remise D du compte) : à vérifier à la main |
| np.abs | 1 | 1.6308 | non testé | fonction utilitaire (clip, abs, ...) : pas une technique nommée |
| np.tanh | 1 | 1.7437 | non testé | fonction utilitaire (clip, abs, ...) : pas une technique nommée |


**Remise (type Kneser-Ney) :** 168 génomes de l'archive soustraient un terme directement à n[k]. Le meilleur d'entre eux (A = 1.6287) est :

```
def estimer(n, N, d, u1, u2, K):
    p = u2[0] + (1.0 / n.shape[-1] - u1[K])
    for k in range(K + 1):
        e = d[k] * 4.93917
        m = -p + p
        p = np.where((N[k] > p * [1.07275, 1.13818, 1.07297, 1.02518, 1.01381, 0.98328, 1.01632, 0.98032, 0.98368, 1.03458, 0.97649][k]) - e, div(n[k] - p + e * p, N[k] + 0.2087), p)
    p = np.maximum(p, 6.36e-05)
    return p
```

**Lecture honnête de l'émergence.**

- Les structures « nouvelles » des élites sont presque toutes des habillages ou des restes : variables intermédiaires, `np.maximum`, `np.log1p`, termes `u1` et `u2` dans la boucle. Quand on les retire (colonne « Retirer la structure change-t-il A ? »), le bpc sur A ne bouge pas dans la grande majorité des cas. Elles ne servent pas : ce sont des restes de la dérive aléatoire, pas des découvertes.
- Les constantes propres à chaque ordre sont aussi, en général, neutres : la même valeur partout donne le même score.
- **Pas de Kneser-Ney.** Une « remise » au sens du détecteur (soustraire un terme directement au compte n[k]) existe dans de nombreux génomes de l'archive, mais le terme soustrait est la probabilité `p` de l'ordre inférieur (de l'ordre de 0,01), et non une remise constante D proportionnelle aux comptes ; il n'y a pas non plus de comptes de continuation. Ce n'est pas du Kneser-Ney, et ces génomes ne sont pas meilleurs que les autres. Nous ne revendiquons donc aucune technique connue redécouverte, sauf l'aplatissement des scores (puissance) et l'augmentation de la masse d'échappement, déjà signalés plus haut et qui sont du réglage, pas de la structure nouvelle.
- La couverture de la grille reste partielle : peu de cases sont remplies, surtout les structures simples.

## 10. Temps de calcul

Temps mur total des runs et du test : environ 32 minutes, sur 4 coeurs (4 processus d'évaluation). Le temps CPU cumulé est de l'ordre de 3 fois le temps mur.

## 11. Limites et précautions

- **Un seul domaine de texte.** Bibliothèque et validation viennent des mêmes sources (romans anglais, textes bibliques, etc.). Rien ne dit que le champion vaut autant sur un autre type de texte.
- **A sert à la fitness, B à la sélection, C a servi une seule fois.** Mais A, B et C sont trois morceaux consécutifs d'un même fichier : les trois peuvent se ressembler plus que des textes réellement différents. Les bpc de A, B et C ne sont pas comparables entre eux (textes différents).
- **Le champion est choisi sur B parmi le top 10 de A.** Le biais de sélection est faible, mais il existe. C seul donne le chiffre honnête.
- **Les comptages sont figés** (aucune adaptation au texte en cours) et l'ordre maximal est 10. Le modèle de départ n'est donc pas Écho-P : il est plus pauvre (pas d'exclusion, ordre 10 au lieu de 16).
- **Le juge normalise les scores.** Le génome n'a pas à produire de vraies probabilités. Une grande partie du gain vient d'ailleurs de cette liberté (la puissance finale, la division par N seul).
- **Écho-P et le sous-ensemble.** Écho-P a été évalué sur un sous-ensemble de C (blocs répartis), pas sur C entier : la comparaison porte sur les mêmes positions pour tous les modèles, mais l'intervalle d'erreur est plus large.
- **LZMA conditionnel** compresse biblio + C d'un bloc. C'est une référence grossière, pas un modèle de même nature.
- **Peu de graines.** L'ablation compare 5 graines par mode ; la différence éventuelle est à lire avec le test de permutation indiqué.
- **Mutations aléatoires, pas de gradient.** Le moteur ne descend pas les constantes de façon fine : voir la comparaison avec la référence à la main.
- **Moteur complet : A/B seulement.** Les composants ajoutés (soi, MDL/Pareto, coût, MAP-Elites, arrêt précoce) n'ont pas été jugés sur C : C avait déjà servi pour le champion du run principal. Les comparaisons « complet contre simple » reposent sur A (fitness) et B (sélection).
- **Coût mesuré, bruité.** Le CPU par évaluation vient de `time.process_time` sur une machine à 4 processus en parallèle ; le budget « CPU égal » inclut le CPU du processus parent. Les facteurs de pénalité (MDL 0,5 ; coût 0,01) sont des choix, pas des optimums.
- **Garde-fous inchangés.** Les fichiers épinglés (`juge.py`, `garde_fous.py`, `donnees.py`, caches) n'ont pas été modifiés : la mesure du CPU est faite dans `composants.py`, autour de la tâche d'évaluation d'origine.
- **Détection de structures.** Les « caractéristiques » des élites sont repérées par des règles syntaxiques simples ; une règle peut se tromper (une première version voyait de fausses remises, corrigée).
- **Attribution.** L'ablation « complet contre simple » compare le paquet de cinq composants, pas chacun isolé. Le gain à CPU égal peut venir autant de la pénalité de coût et du MDL (génomes plus petits et plus rapides : plus d'évaluations pour le même CPU) que du soi. Le soi seul a sauté 9 pour cent (run principal) à 25 pour cent (ablation) des enfants ; la séparation entre enfants prédits mauvais et les autres est réelle mais modeste (taux de succès 0,055 contre 0,117).
- **Perturbation du budget CPU du moteur simple.** Pendant les premiers runs « simple » de l'ablation, d'autres calculs tournaient sur la machine (tests, analyses) : le CPU mesuré y est gonflé et le nombre d'évaluations faites y est plus faible (888, 1392, 2720, 1752 contre 2048 à 2960 pour le complet). Quatre graines et p = 0,057 : tendance favorable au moteur complet, pas une preuve.
- **Temps de calcul supplémentaire** : run complet 6 min (749 s de CPU cumulé) plus 1 min de vérification du point fixe ; ablation complet/simple : 15 min de temps mur.
