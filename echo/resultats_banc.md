# Résultats du banc d'essai Écho-P

Écho-P = `predicteur.py` : PPM (échappement de méthode C, avec exclusion), ordres 16 à 0, au plus 64 occurrences par ordre, uniquement `str.find` sur le texte brut, 0 paramètre appris. bpc = bits par caractère (plus bas = mieux).

Échantillon d'évaluation anglais : 3000 caractères (paragraphes de `validation.txt` tirés au hasard avec une graine fixe, joints par un saut de ligne).

## A. Courbe d'apprentissage (efficacité en données)

Mémoire = échantillon aléatoire de paragraphes de la bibliothèque anglaise (graine fixe). Lignes de base : bpc de compression *conditionnelle* = 8 x (taille(mémoire + éval) - taille(mémoire)) / nb de caractères de l'éval. lzma : préset 9|EXTREME ; bz2 et zlib : niveau 9.

| Mémoire | Caractères | Écho-P (bpc) | lzma (bpc) | bz2 (bpc) | zlib (bpc) |
|---|---:|---:|---:|---:|---:|
| 0,1 % | 14 164 | 3.238 | 3.264 | 2.979 | 3.501 |
| 1 % | 133 076 | 2.562 | 2.720 | 2.512 | 3.291 |
| 10 % | 1 344 461 | 2.281 | 2.411 | 2.672 | 3.357 |
| 100 % | 15 679 400 | 2.042 | 2.176 | 2.461 | 3.371 |

Précautions de lecture : la fenêtre de zlib n'est que de 32 Ko et les blocs de bz2 de 900 Ko ; ces deux compresseurs « oublient » donc la quasi-totalité d'une mémoire de plusieurs mégaoctets et n'en profitent presque pas. Seul lzma (dictionnaire de 64 Mo au préset 9) voit toute la mémoire : c'est la seule vraie comparaison. Ces trois chiffres ne sont pas un classement de « méthodes » mais une mesure de ce que chaque outil retient d'une longue mémoire.

Écho-P contre lzma, écart de bpc (négatif = Écho-P meilleur) :

| Mémoire | Écho-P - lzma (bpc) |
|---|---:|
| 0,1 % | -0.026 |
| 1 % | -0.158 |
| 10 % | -0.129 |
| 100 % | -0.134 |

## B. Apprentissage continu (anglais, puis ajout du français)

Mémoire anglaise = 100 % de la bibliothèque. Français : 56402 paragraphes en mémoire, 569 paragraphes tenus à l'écart (1 ligne sur 100) ; échantillon d'évaluation français : 3000 caractères. L'ajout se fait par `Memoire.ajouter`, sans aucun recalcul.

| Condition | Mémoire | Évaluation | bpc |
|---|---|---|---:|
| (a) | anglais seul | anglais | 2.042 |
| (b) | anglais seul | français | 5.471 |
| (c) | anglais + français | anglais | 2.091 |
| (c) | anglais + français | français | 1.759 |

- **Oubli** = c_en - a_en = **+0.050 bpc** (positif = l'anglais s'est dégradé ; 0 = aucun oubli).
- **Apprentissage immédiat** = b_fr - c_fr = **+3.711 bpc** (positif = le français est mieux prédit).
- Durée de `ajouter` (7 114 138 caractères) : 0.0716 s de temps réel, 0.0703 s de CPU.
- Pour comparaison, la mémoire anglaise compressée se charge en une dizaine de secondes au plus ; aucune phase d'entraînement n'existe.

## C. Énergie (ESTIMATION)

**Estimation, pas une mesure** : énergie = temps CPU (`time.process_time`) x 10 W, en supposant 10 W par cœur de CPU occupé. Aucune mesure de puissance réelle n'a été faite. Les temps CPU ont été relevés avec plusieurs processus en parallèle sur 4 cœurs.

| Mesure | CPU (s) | CPU (s / caractère) | Énergie estimée (mJ / caractère) |
|---|---:|---:|---:|
| A, mémoire 0,1 % | 0.35 | 0.000116 | 1.16 |
| A, mémoire 1 % | 2.64 | 0.000882 | 8.82 |
| A, mémoire 10 % | 24.65 | 0.008216 | 82.16 |
| A, mémoire 100 % | 280.85 | 0.093617 | 936.17 |
| B (a) anglais / mém. anglaise | 281.84 | 0.093946 | 939.46 |
| B (b) français / mém. anglaise | 140.67 | 0.046889 | 468.89 |
| B (c) anglais / mém. EN+FR | 408.45 | 0.136149 | 1361.49 |
| B (c) français / mém. EN+FR | 448.00 | 0.149334 | 1493.34 |

Le coût par caractère croît avec la taille de la mémoire, car chaque `str.find` balaie davantage de texte.

## D. Taille

| Fichier | Octets |
|---|---:|
| memoire.xz (anglais) | 4 048 144 |
| memoire_fr.xz (français, lzma 9\|EXTREME) | 1 638 508 |
| **Somme** | **5 686 652** |

Limite 10 000 000 octets : **RESPECTÉE**.

## Durée

Durée totale du banc : 764 s (12.7 min).

## Limites et honnêteté

- Un seul échantillon d'évaluation de ~3000 caractères par langue : l'incertitude est de l'ordre de quelques centièmes de bpc.
- L'évaluation prédit chaque caractère à partir du texte d'évaluation déjà lu (contexte <= 16 caractères).
- Fuite écartée : 106 paragraphes de `validation.txt` figurent mot pour mot dans la mémoire anglaise (versets bibliques en double). Ils sont exclus des échantillons d'évaluation (anglais et français), sinon Écho-P et lzma les recopieraient.
- Écho-P n'est pas un compresseur complet : on calcule -log2 P sans coder réellement.
