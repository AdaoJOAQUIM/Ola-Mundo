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
OK     test_comptages_contre_force_brute
OK     test_empreintes_detectent_modification
OK     test_empreintes_du_depot_intactes
OK     test_evaluateur_en_processus
OK     test_juge_constant_donne_log2_A
OK     test_juge_rejette_nan_negatif_forme
OK     test_liste_blanche_graine
OK     test_liste_blanche_rejette_malveillants
OK     test_meta_heritee_borne
OK     test_moteur_ne_touche_pas_c
OK     test_taille_et_syntaxe
12 tests, 0 échec(s)
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

## 7. Temps de calcul

Temps mur total des runs et du test : environ 32 minutes, sur 4 coeurs (4 processus d'évaluation). Le temps CPU cumulé est de l'ordre de 3 fois le temps mur.

## 8. Limites et précautions

- **Un seul domaine de texte.** Bibliothèque et validation viennent des mêmes sources (romans anglais, textes bibliques, etc.). Rien ne dit que le champion vaut autant sur un autre type de texte.
- **A sert à la fitness, B à la sélection, C a servi une seule fois.** Mais A, B et C sont trois morceaux consécutifs d'un même fichier : les trois peuvent se ressembler plus que des textes réellement différents. Les bpc de A, B et C ne sont pas comparables entre eux (textes différents).
- **Le champion est choisi sur B parmi le top 10 de A.** Le biais de sélection est faible, mais il existe. C seul donne le chiffre honnête.
- **Les comptages sont figés** (aucune adaptation au texte en cours) et l'ordre maximal est 10. Le modèle de départ n'est donc pas Écho-P : il est plus pauvre (pas d'exclusion, ordre 10 au lieu de 16).
- **Le juge normalise les scores.** Le génome n'a pas à produire de vraies probabilités. Une grande partie du gain vient d'ailleurs de cette liberté (la puissance finale, la division par N seul).
- **Écho-P et le sous-ensemble.** Écho-P a été évalué sur un sous-ensemble de C (blocs répartis), pas sur C entier : la comparaison porte sur les mêmes positions pour tous les modèles, mais l'intervalle d'erreur est plus large.
- **LZMA conditionnel** compresse biblio + C d'un bloc. C'est une référence grossière, pas un modèle de même nature.
- **Peu de graines.** L'ablation compare 5 graines par mode ; la différence éventuelle est à lire avec le test de permutation indiqué.
- **Mutations aléatoires, pas de gradient.** Le moteur ne descend pas les constantes de façon fine : voir la comparaison avec la référence à la main.
