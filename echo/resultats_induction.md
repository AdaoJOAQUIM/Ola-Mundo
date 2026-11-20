# Résultats de l'induction d'Écho

Généré par `banc_induction.py` (graine 2024, 30 mondes synthétiques par configuration, Python 3.13.18). Bibliothèque standard seulement : aucun réseau de neurones, aucun poids appris, aucun gradient. L'induction ne fait que **compter** des exemples et des contre-exemples.

## Avertissement

Les mondes sont **générés par nous** : une taxonomie cachée (arbre de 4 à 8 classes) plus 2 à 4 « traits » dont la probabilité dépend du nœud (1 = règle universelle, 0 = jamais, 0,7 = tendance/exception, 0,2 = rare), soit 6 à 12 classes au total. La vérité terrain (règles universelles vraies, paires disjointes, réponses attendues) est calculée par le banc à partir du modèle générateur. Le système ne voit qu'une fraction des faits vrais : chaque appartenance vraie est révélée avec la probabilité r = 0,7, chaque non-appartenance vraie avec r_neg = 0,5. Aucune règle n'est donnée (sauf dans la variante signalée). Les résultats disent ce que fait le mécanisme **sur ce type de monde** ; ils ne prouvent rien sur le monde réel.

Définitions : s = exemples, c = contre-exemples ; hypothèse « tout A est B » proposée si s >= s_min et c = 0 ; confiance = règle de succession de Laplace (s+1)/(s+2) (moyenne a posteriori de Beta(s+1, c+1), prior uniforme). k = nombre d'individus par classe de l'arbre (échantillon d'exemples). Les questions portent sur des (individu, classe) **non révélés**, donc inconnus du raisonneur seul ; une réponse « probable » est un savoir que personne n'a écrit.

## 1. Émergence : règles induites et nouvelles questions

Précision = hypothèses qui sont de vraies règles universelles / hypothèses proposées. Rappel = vraies règles universelles retrouvées / vraies règles universelles du monde (donc limité par l'échantillon : une règle sur une classe sans individu observé est invisible).

### Règles « Tout A est B » (moyenne sur les mondes)

| k | s_min | hypothèses / monde | précision | rappel |
|---|---|---|---|---|
| 1 | 2 | 6,0 | 64,8 % | 21,4 % |
| 1 | 3 | 2,5 | 66,7 % | 9,2 % |
| 1 | 5 | 0,6 | 66,7 % | 2,2 % |
| 2 | 2 | 10,7 | 79,1 % | 46,6 % |
| 2 | 3 | 5,4 | 79,6 % | 23,8 % |
| 2 | 5 | 1,7 | 80,0 % | 7,4 % |
| 3 | 2 | 13,7 | 82,9 % | 62,6 % |
| 3 | 3 | 7,1 | 84,6 % | 33,3 % |
| 3 | 5 | 3,5 | 83,8 % | 16,2 % |
| 5 | 2 | 17,1 | 88,5 % | 83,6 % |
| 5 | 3 | 13,2 | 91,9 % | 66,9 % |
| 5 | 5 | 6,4 | 92,7 % | 32,8 % |
| 10 | 2 | 18,6 | 94,4 % | 96,9 % |
| 10 | 3 | 17,4 | 95,0 % | 91,2 % |
| 10 | 5 | 13,3 | 97,0 % | 71,3 % |
| 20 | 2 | 17,9 | 97,8 % | 96,9 % |
| 20 | 3 | 17,8 | 97,9 % | 96,5 % |
| 20 | 5 | 17,5 | 98,3 % | 95,2 % |

### Règles « Aucun A n'est B »

| k | s_min | hypothèses / monde | précision | rappel |
|---|---|---|---|---|
| 1 | 2 | 6,2 | 30,6 % | 8,2 % |
| 1 | 3 | 2,1 | 19,0 % | 1,7 % |
| 1 | 5 | 0,1 | 0,0 % | 0,0 % |
| 2 | 2 | 11,5 | 60,7 % | 30,2 % |
| 2 | 3 | 5,3 | 50,0 % | 11,5 % |
| 2 | 5 | 1,1 | 42,4 % | 2,0 % |
| 3 | 2 | 13,1 | 73,2 % | 41,3 % |
| 3 | 3 | 6,0 | 64,8 % | 16,7 % |
| 3 | 5 | 2,1 | 54,8 % | 4,9 % |
| 5 | 2 | 19,5 | 81,9 % | 69,1 % |
| 5 | 3 | 12,4 | 80,3 % | 42,9 % |
| 5 | 5 | 5,1 | 77,8 % | 17,1 % |
| 10 | 2 | 23,0 | 93,5 % | 92,8 % |
| 10 | 3 | 20,0 | 93,5 % | 80,7 % |
| 10 | 5 | 12,1 | 94,8 % | 49,5 % |
| 20 | 2 | 23,4 | 96,6 % | 97,6 % |
| 20 | 3 | 23,0 | 97,0 % | 96,1 % |
| 20 | 5 | 21,6 | 97,2 % | 90,8 % |

### Nouvelles questions rendues répondables par l'induction

300 questions par monde et par configuration, toutes sur des faits non révélés (aucune règle donnée : le raisonneur seul répond « Je ne sais pas » à toutes). Colonne « toujours non » : exactitude d'un témoin trivial qui répondrait toujours non (la plupart des appartenances sont fausses), à comparer à l'exactitude des réponses probables.

| k | s_min | répondues « probablement » | part des questions | exactitude | dont oui | exact. oui | dont non | exact. non | témoin « toujours non » |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 333 | 46,2 % | 73,9 % | 50 | 64,0 % | 283 | 75,6 % | 71,8 % |
| 1 | 3 | 140 | 19,4 % | 76,4 % | 32 | 84,4 % | 108 | 74,1 % | 71,8 % |
| 1 | 5 | 16 | 2,2 % | 81,2 % | 7 | 85,7 % | 9 | 77,8 % | 71,8 % |
| 2 | 2 | 966 | 67,8 % | 87,3 % | 178 | 88,8 % | 788 | 86,9 % | 69,9 % |
| 2 | 3 | 663 | 46,5 % | 85,2 % | 166 | 89,8 % | 497 | 83,7 % | 69,9 % |
| 2 | 5 | 206 | 14,5 % | 83,0 % | 86 | 91,9 % | 120 | 76,7 % | 69,9 % |
| 3 | 2 | 1467 | 66,8 % | 89,1 % | 327 | 82,3 % | 1140 | 91,1 % | 71,4 % |
| 3 | 3 | 1063 | 48,4 % | 90,3 % | 259 | 88,8 % | 804 | 90,8 % | 71,4 % |
| 3 | 5 | 572 | 26,0 % | 90,7 % | 178 | 91,6 % | 394 | 90,4 % | 71,4 % |
| 5 | 2 | 2483 | 69,7 % | 94,5 % | 585 | 92,8 % | 1898 | 95,0 % | 71,0 % |
| 5 | 3 | 2188 | 61,4 % | 94,7 % | 506 | 94,9 % | 1682 | 94,7 % | 71,0 % |
| 5 | 5 | 1487 | 41,7 % | 95,4 % | 421 | 95,7 % | 1066 | 95,3 % | 71,0 % |
| 10 | 2 | 4329 | 65,4 % | 98,5 % | 1157 | 98,0 % | 3172 | 98,7 % | 70,9 % |
| 10 | 3 | 4196 | 63,4 % | 98,6 % | 1114 | 98,3 % | 3082 | 98,7 % | 70,9 % |
| 10 | 5 | 3701 | 55,9 % | 99,2 % | 1001 | 99,3 % | 2700 | 99,2 % | 70,9 % |
| 20 | 2 | 5500 | 64,2 % | 99,0 % | 1531 | 98,7 % | 3969 | 99,1 % | 70,8 % |
| 20 | 3 | 5474 | 63,9 % | 99,0 % | 1525 | 98,7 % | 3949 | 99,2 % | 70,8 % |
| 20 | 5 | 5433 | 63,5 % | 99,1 % | 1514 | 98,8 % | 3919 | 99,2 % | 70,8 % |

Réponses certaines (preuves vérifiées) dans cette expérience : 0 (attendu 0 sans règle donnée).

### Composition des hypothèses « Tout A est B » fausses (s_min = 3)

« tendance » : B vrai pour au moins 50 % des A mais pas tous (exceptions) ; « rare » : B vrai pour moins de la moitié des A ; « disjointe » : A et B ne coexistent jamais. Totaux sur tous les mondes.

| k | universelles vraies | tendance | rare | disjointe |
|---|---|---|---|---|
| 1 | 50 | 25 | 0 | 0 |
| 2 | 129 | 33 | 0 | 0 |
| 3 | 181 | 32 | 1 | 0 |
| 5 | 363 | 32 | 0 | 0 |
| 10 | 495 | 25 | 1 | 0 |
| 20 | 524 | 11 | 0 | 0 |

### Variante : la moitié des règles vraies est donnée comme phrases (s_min = 3)

Les règles données sont certaines (le raisonneur répond avec preuve vérifiée) ; le rappel est calculé sur les règles universelles non déjà certaines.

| k | certaines (justes) | probables (justes) | inconnues | rappel des règles restantes | précision |
|---|---|---|---|---|---|
| 3 | 549 (549) | 789 (729) | 859 | 50,3 % | 74,4 % |
| 5 | 724 (724) | 1549 (1466) | 1290 | 74,2 % | 82,1 % |
| 10 | 1343 (1343) | 2920 (2856) | 2357 | 94,4 % | 89,5 % |

## 2. Calibration de la confiance

Réponses « probablement » regroupées par confiance annoncée (k >= 2, tous mondes). ECE = somme pondérée des |confiance moyenne − exactitude observée| par seau.

**s_min = 2** : 14745 réponses ; confiance moyenne 89,1 % ; exactitude globale 96,4 % ; **ECE = 0,072** (globalement sous-confiant (prudent)). Réponses « probablement oui » seules : 3778 réponses, confiance moyenne 91,3 %, exactitude 95,7 %, ECE = 0,045.

| confiance annoncée | n | confiance moyenne | exactitude observée |
|---|---|---|---|
| [0,00 ; 0,70[ | 237 | 60,5 % | 80,6 % |
| [0,70 ; 0,80[ | 1320 | 75,0 % | 88,3 % |
| [0,80 ; 0,85[ | 2100 | 81,6 % | 92,6 % |
| [0,85 ; 0,90[ | 2391 | 87,2 % | 96,6 % |
| [0,90 ; 0,95[ | 4543 | 92,4 % | 98,3 % |
| [0,95 ; 1,00[ | 4154 | 96,7 % | 99,5 % |

**s_min = 3** : 13584 réponses ; confiance moyenne 90,6 % ; exactitude globale 96,9 % ; **ECE = 0,063** (globalement sous-confiant (prudent)). Réponses « probablement oui » seules : 3570 réponses, confiance moyenne 92,5 %, exactitude 96,9 %, ECE = 0,046.

| confiance annoncée | n | confiance moyenne | exactitude observée |
|---|---|---|---|
| [0,00 ; 0,70[ | 107 | 65,1 % | 80,4 % |
| [0,70 ; 0,80[ | 207 | 74,5 % | 80,7 % |
| [0,80 ; 0,85[ | 2149 | 81,6 % | 91,9 % |
| [0,85 ; 0,90[ | 2418 | 87,2 % | 96,2 % |
| [0,90 ; 0,95[ | 4549 | 92,4 % | 98,2 % |
| [0,95 ; 1,00[ | 4154 | 96,7 % | 99,5 % |

**s_min = 5** : 11399 réponses ; confiance moyenne 92,5 % ; exactitude globale 97,9 % ; **ECE = 0,054** (globalement sous-confiant (prudent)). Réponses « probablement oui » seules : 3200 réponses, confiance moyenne 93,8 %, exactitude 98,0 %, ECE = 0,042.

| confiance annoncée | n | confiance moyenne | exactitude observée |
|---|---|---|---|
| [0,70 ; 0,80[ | 176 | 76,9 % | 89,8 % |
| [0,80 ; 0,85[ | 59 | 81,7 % | 89,8 % |
| [0,85 ; 0,90[ | 2448 | 87,2 % | 95,7 % |
| [0,90 ; 0,95[ | 4562 | 92,4 % | 98,1 % |
| [0,95 ; 1,00[ | 4154 | 96,7 % | 99,5 % |

ECE par k (s_min = 3) :

| k | n réponses | confiance moyenne | exactitude | ECE |
|---|---|---|---|---|
| 1 | 140 | 79,8 % | 76,4 % | 0,058 |
| 2 | 663 | 81,5 % | 85,2 % | 0,040 |
| 3 | 1063 | 83,4 % | 90,3 % | 0,069 |
| 5 | 2188 | 86,6 % | 94,7 % | 0,082 |
| 10 | 4196 | 90,7 % | 98,6 % | 0,079 |
| 20 | 5474 | 94,5 % | 99,0 % | 0,045 |

## 3. Convergence

### 3.1 Une règle vraie : confiance en fonction du nombre d'exemples

Mesuré dans le vrai système (s_min = 1, n individus « chien » et « aboyeur »). `Beta 5 %` : borne inférieure à 5 % de la loi Beta(n+1, 1).

| n exemples | confiance | erreur \|confiance − 1\| | Beta moyenne | Beta 5 % |
|---|---|---|---|---|
| 1 | 66,7 % | 0,3333 | 0,6667 | 0,224 |
| 2 | 75,0 % | 0,2500 | 0,7500 | 0,368 |
| 3 | 80,0 % | 0,2000 | 0,8000 | 0,473 |
| 4 | 83,3 % | 0,1667 | 0,8333 | 0,549 |
| 5 | 85,7 % | 0,1429 | 0,8571 | 0,607 |
| 8 | 90,0 % | 0,1000 | 0,9000 | 0,717 |
| 10 | 91,7 % | 0,0833 | 0,9167 | 0,762 |
| 15 | 94,1 % | 0,0588 | 0,9412 | 0,829 |
| 20 | 95,5 % | 0,0455 | 0,9545 | 0,867 |
| 50 | 98,1 % | 0,0192 | 0,9808 | 0,943 |
| 100 | 99,0 % | 0,0098 | 0,9902 | 0,971 |

L'erreur vaut 1/(n+2) : elle décroît, mais lentement et sans jamais atteindre 0 (une induction n'est jamais une preuve).

### 3.2 Une relation « exception » : proposée, puis retirée

Flux d'individus « chien » dont la propriété « aboyeur » est vraie avec la probabilité p (chaque exemple = 2 phrases). On arrête au premier contre-exemple. Le retrait est fait par le système lui-même (vérifié à chaque essai). Théorie : P(proposée) = p^s_min ; nombre moyen d'exemples entre la proposition et le retrait = 1/(1−p).

| p | s_min | essais | proposée au moins une fois | théorie p^s_min | exemples moy. après proposition jusqu'au retrait | théorie 1/(1−p) | exemples moy. jusqu'au 1er contre-exemple | retraits vérifiés |
|---|---|---|---|---|---|---|---|---|
| 0,5 | 3 | 2000 | 12,6 % | 12,5 % | 1,88 | 2,00 | 1,98 | 252/252 |
| 0,7 | 3 | 2000 | 34,6 % | 34,3 % | 3,26 | 3,33 | 3,33 | 693/693 |
| 0,9 | 3 | 2000 | 72,4 % | 72,9 % | 10,51 | 10,00 | 10,31 | 1447/1447 |
| 0,7 | 5 | 2000 | 16,0 % | 16,8 % | 3,32 | 3,33 | 3,27 | 320/320 |

Exactitude de la réponse « probablement oui » tant que l'hypothèse est active, par soutien s (exemples où l'hypothèse était active juste avant l'exemple suivant) :

| p | s | confiance annoncée | observations | exactitude observée |
|---|---|---|---|---|
| 0,5 | 3 | 80,0 % | 252 | 50,0 % |
| 0,5 | 4 | 83,3 % | 126 | 42,1 % |
| 0,5 | 5 | 85,7 % | 53 | 45,3 % |
| 0,5 | 6 | 87,5 % | 24 | 41,7 % |
| 0,7 | 3 | 80,0 % | 693 | 67,7 % |
| 0,7 | 4 | 83,3 % | 469 | 70,4 % |
| 0,7 | 5 | 85,7 % | 330 | 66,1 % |
| 0,7 | 6 | 87,5 % | 218 | 68,8 % |
| 0,9 | 3 | 80,0 % | 1447 | 89,5 % |
| 0,9 | 4 | 83,3 % | 1295 | 90,0 % |
| 0,9 | 5 | 85,7 % | 1166 | 89,6 % |
| 0,9 | 6 | 87,5 % | 1045 | 89,0 % |

Lecture : comparer la colonne « confiance annoncée » à « exactitude observée ». Pour p = 1 la confiance de Laplace est trop basse (80 % pour s = 3 alors que la règle est vraie) ; pour p < 1 elle est trop haute, et d'autant plus que p est faible (la règle de Laplace suppose un taux inconnu à prior uniforme, pas le mélange de relations à 100 %, 70 % ou 50 % que contient un vrai monde). Après un contre-exemple, la confiance en la règle tombe à zéro : l'hypothèse n'existe plus.

### 3.3 Boucle de point fixe (déduire → induire → appliquer → ré-induire)

Une boucle par monde, par k et par s_min. Les faits provisoires ne comptent pas comme exemples. Le plafond est de 50 itérations ; la preuve de terminaison est dans le docstring de `induction.py`.

| s_min | boucles | itérations moyennes | max | histogramme (itérations : nombre) | plafond atteint | non convergées | faits provisoires moy. | conflits moy. |
|---|---|---|---|---|---|---|---|---|
| 2 | 180 | 2,81 | 5 | 2 : 79, 3 : 64, 4 : 30, 5 : 7 | 0 | 0 | 114,6 | 6,4 |
| 3 | 180 | 2,58 | 4 | 2 : 99, 3 : 57, 4 : 24 | 0 | 0 | 103,2 | 3,6 |
| 5 | 180 | 2,45 | 5 | 2 : 112, 3 : 56, 4 : 11, 5 : 1 | 0 | 0 | 87,7 | 1,8 |

« Conflits » : faits provisoires contradictoires entre eux ou avec un fait déduit (par exemple deux hypothèses incompatibles) ; ils sont comptés, jamais cachés, et la requête s'abstient dans ce cas.

### 3.4 Pourquoi les faits provisoires ne comptent pas : mesure de l'auto-renforcement (s_min = 3)

Mode sain = faits provisoires exclus du soutien. Mode naïf = ils comptent comme exemples (cercle vicieux, uniquement pour la mesure). « Hypothèses » compte « tout » et « aucun » ; la précision ne porte que sur « tout ». Soutien moyen = nombre moyen d'exemples par hypothèse ; la confiance en découle.

| k | mode | hypothèses / monde | précision « tout » | soutien moyen | confiance moyenne annoncée | itérations moy. | plafond atteint |
|---|---|---|---|---|---|---|---|
| 1 | sain | 4,6 | 66,7 % | 3,6 | 81,8 % | 2,40 | 0 |
| 1 | naif | 4,9 | 68,2 % | 5,1 | 85,1 % | 2,50 | 0 |
| 2 | sain | 10,7 | 79,6 % | 4,0 | 82,6 % | 2,83 | 0 |
| 2 | naif | 13,6 | 80,3 % | 6,5 | 86,6 % | 3,33 | 0 |
| 3 | sain | 13,1 | 84,6 % | 4,9 | 84,2 % | 2,93 | 0 |
| 3 | naif | 18,3 | 94,1 % | 6,6 | 85,9 % | 3,67 | 0 |
| 5 | sain | 25,5 | 91,9 % | 5,6 | 84,9 % | 2,73 | 0 |
| 5 | naif | 35,9 | 93,2 % | 7,8 | 86,9 % | 3,60 | 0 |
| 10 | sain | 37,4 | 95,0 % | 8,2 | 87,6 % | 2,47 | 0 |
| 10 | naif | 40,9 | 95,8 % | 14,5 | 91,4 % | 2,97 | 0 |
| 20 | sain | 40,8 | 97,9 % | 15,1 | 91,9 % | 2,13 | 0 |
| 20 | naif | 40,8 | 97,9 % | 29,3 | 95,2 % | 2,27 | 0 |

Lecture : le mode naïf gonfle le soutien (donc la confiance) sans aucune observation nouvelle ; la colonne précision montre si cette confiance supplémentaire est méritée. Si la précision n'augmente pas autant que la confiance, c'est de la confiance fabriquée.


### 3.5 Option : garde de couverture (s / |A| >= seuil), s_min = 3

L'absence d'information n'est pas un contre-exemple : sans garde, « tout animal est un chat » peut être proposée si seuls des chats ont été vus avec leur espèce. Cette garde est une option (désactivée par défaut, conformément à la spécification) ; on mesure son effet.

| k | seuil | précision « tout » | rappel « tout » | réponses probables / monde | exactitude |
|---|---|---|---|---|---|
| 1 | 0,0 | 66,7 % | 9,2 % | 4,7 | 76,4 % |
| 1 | 0,5 | 69,4 % | 9,2 % | 4,4 | 73,7 % |
| 2 | 0,0 | 79,6 % | 23,8 % | 22,1 | 85,2 % |
| 2 | 0,5 | 85,1 % | 23,2 % | 15,9 | 86,0 % |
| 3 | 0,0 | 84,6 % | 33,3 % | 35,4 | 90,3 % |
| 3 | 0,5 | 86,5 % | 31,9 % | 24,6 | 90,9 % |
| 5 | 0,0 | 91,9 % | 66,9 % | 72,9 | 94,7 % |
| 5 | 0,5 | 93,0 % | 65,7 % | 61,9 | 96,5 % |
| 10 | 0,0 | 95,0 % | 91,2 % | 139,9 | 98,6 % |
| 10 | 0,5 | 95,6 % | 87,7 % | 125,6 | 98,8 % |
| 20 | 0,0 | 97,9 % | 96,5 % | 182,5 | 99,0 % |
| 20 | 0,5 | 98,6 % | 91,3 % | 160,4 | 99,2 % |

## 4. Coût de calcul

Temps CPU mesuré avec `time.process_time()` (moyenne sur les mondes). Une « passe d'induction » inclut la fermeture déductive de la base. **ESTIMATION, pas une mesure** : énergie = temps CPU × 10 W (hypothèse de 10 W par cœur occupé) ; aucun wattmètre n'a été utilisé.

| k | s_min | CPU par passe d'induction (ms) | CPU par question (µs) | énergie estimée par passe (mJ) | énergie estimée par question (µJ) |
|---|---|---|---|---|---|
| 1 | 2 | 0,14 | 79 | 1,44 | 794 |
| 1 | 3 | 0,11 | 70 | 1,08 | 703 |
| 1 | 5 | 0,08 | 63 | 0,83 | 631 |
| 2 | 2 | 0,25 | 89 | 2,53 | 888 |
| 2 | 3 | 0,20 | 79 | 2,04 | 793 |
| 2 | 5 | 0,16 | 68 | 1,62 | 680 |
| 3 | 2 | 0,30 | 82 | 3,02 | 819 |
| 3 | 3 | 0,26 | 78 | 2,56 | 777 |
| 3 | 5 | 0,23 | 70 | 2,26 | 701 |
| 5 | 2 | 0,54 | 90 | 5,39 | 895 |
| 5 | 3 | 0,49 | 86 | 4,87 | 860 |
| 5 | 5 | 0,41 | 79 | 4,08 | 785 |
| 10 | 2 | 1,00 | 88 | 9,99 | 879 |
| 10 | 3 | 1,00 | 88 | 10,01 | 883 |
| 10 | 5 | 0,87 | 83 | 8,73 | 829 |
| 20 | 2 | 2,35 | 98 | 23,49 | 978 |
| 20 | 3 | 2,27 | 96 | 22,68 | 961 |
| 20 | 5 | 2,26 | 96 | 22,56 | 955 |

Durée totale du banc : 23 s.

## 5. Limites (à lire avant de citer un chiffre)

- **L'induction peut avoir tort.** Une hypothèse soutenue par 3, 10 ou 100 exemples sans contre-exemple
  reste fausse si le contre-exemple n'est pas encore observé (cygne noir). La confiance (Laplace) est une
  probabilité sous une hypothèse d'échangeabilité, pas une garantie. Elle n'est pas bien calibrée sur un
  monde qui mélange des règles universelles et des relations « exception » (voir sections 2 et 3.2).
- **Compromis s_min.** Un s_min bas propose beaucoup plus de règles (rappel, nouvelles réponses) mais
  davantage de fausses règles ; un s_min haut est plus sûr mais répond à moins de questions et exige des
  exemples qui n'existent pas toujours. Les tableaux de la section 1 le montrent chiffres à l'appui.
- **Monde ouvert, pas de contre-exemple par défaut.** Ne pas savoir qu'un chien n'est pas un chat n'est
  pas un contre-exemple de « tout animal est un chat » ; l'inducteur peut donc proposer des règles
  contradictoires entre elles (comptées comme conflits). La garde de couverture est une atténuation
  partielle, pas une solution.
- **Seules des relations d'appartenance de classes** sont induites (« tout A est B », « aucun A n'est B »),
  dans le français contrôlé du raisonneur. Pas de relations à plusieurs arguments, pas de lieux, pas de
  nombres, pas de règles conditionnelles, pas de causalité.
- **Biais d'échantillonnage.** Les exemples sont supposés tirés sans biais. Un biais de sélection (on ne
  parle que des chiens qui aboient) produit des règles fausses et très confiantes.
- **Le banc est écrit par nous.** Mondes en arbre + traits, vérité terrain issue du même générateur,
  révélation aléatoire uniforme des faits : c'est un cas favorable et structuré. Les performances sur
  des données réelles (langage naturel, taxonomies incohérentes, négations rares) ne sont pas mesurées ici.
- **Chaînage et produit de confiances** : multiplier les confiances suppose l'indépendance des
  hypothèses, ce qui est faux quand elles reposent sur les mêmes individus. Le chiffre d'un chemin de
  plusieurs hypothèses est une approximation.
- **Coût énergétique estimé**, non mesuré (10 W par cœur occupé supposés).
- **Les hypothèses ne sont pas des faits** : elles ne sont jamais insérées dans la base du raisonneur ;
  elles disparaissent si on recrée l'inducteur sans recalculer, et se recalculent entièrement (pas
  d'apprentissage incrémental fin) à chaque phrase acceptée.
