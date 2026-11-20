# Résultats du « soi » d'Écho (auto-modèle mesuré)

Généré par `banc_soi.py` (graine 2024, Python 3.13.18). Bibliothèque standard seulement, aucun poids appris.

## Avertissement : ce n'est PAS de la conscience

Le « soi » ci-dessous désigne de la **métacognition mesurable** : Écho sait (en chiffres) quand ses probabilités sont fiables, prévoit s'il va savoir traiter une demande, et se décrit avec des nombres lus dans ses fichiers de résultats. Ce sont des compteurs et des règles écrites à la main. Aucune expérience subjective, aucune sensibilité, aucune preuve de conscience n'est revendiquée ni suggérée.

## 1. Calibration d'Écho-P

Pour chaque caractère à prédire, on prend le meilleur candidat (top-1) et sa probabilité (la confiance). Calibré = quand il dit 0,7, il a raison 70 % du temps.

**Protocole.** Mémoire = 10 % de la bibliothèque, tirée au hasard par paragraphes (graine fixe 2025) ; pour le français, 10 % de l'anglais + 10 % des lignes françaises (graine fixe) — pas les mémoires complètes, pour tenir le temps de calcul. Évaluation : ~2000 caractères par langue ; calibration : un AUTRE échantillon de ~2000 caractères, paragraphes disjoints de l'évaluation.

- Anglais : mémoire 8 211 paragraphes (1 416 976 caractères) sur 82 109 ; `validation.txt` : 829 paragraphes, **106 exclus car présents mot pour mot dans la mémoire** (test `in`) ; évaluation 2000 caractères (7 paragraphes), calibration 2000 caractères (22 paragraphes).
- Français : mémoire 13 851 paragraphes (2 126 299 caractères) ; 569 lignes tenues à l'écart (1 sur 100), 19 exclues car présentes dans la mémoire ; évaluation 2000 caractères (16 lignes), calibration 2000 caractères (16 lignes).

### Résumé (échantillon d'évaluation)

| Langue | Exactitude top-1 | Confiance moy. | Biais | Sens | ECE brut [IC95 bootstrap] | Brier top-1 | bpc |
|---|---:|---:|---:|---|---:|---:|---:|
| anglais | 0.593 | 0.530 | -0.0628 | TROP MODESTE (sous-confiant) | 0.0855 [0.0703 ; 0.1050] | 0.1943 | 2.224 |
| français | 0.648 | 0.578 | -0.0701 | TROP MODESTE (sous-confiant) | 0.0766 [0.0617 ; 0.0965] | 0.1838 | 1.939 |

### Table de fiabilité brute — anglais (évaluation)

| Seau de confiance | N | Confiance moy. | Exactitude top-1 | Écart (conf. − exact.) |
|---|---:|---:|---:|---:|
| [0.0 ; 0.1[ | 3 | 0.084 | 0.000 | +0.084 |
| [0.1 ; 0.2[ | 89 | 0.154 | 0.112 | +0.041 |
| [0.2 ; 0.3[ | 202 | 0.251 | 0.178 | +0.072 |
| [0.3 ; 0.4[ | 134 | 0.367 | 0.373 | -0.006 |
| [0.4 ; 0.5[ | 66 | 0.456 | 0.394 | +0.062 |
| [0.5 ; 0.6[ | 936 | 0.504 | 0.605 | -0.101 |
| [0.6 ; 0.7[ | 178 | 0.661 | 0.803 | -0.143 |
| [0.7 ; 0.8[ | 122 | 0.747 | 0.869 | -0.122 |
| [0.8 ; 0.9[ | 163 | 0.830 | 0.902 | -0.072 |
| [0.9 ; 1.0] | 107 | 0.935 | 0.944 | -0.009 |

### Table de fiabilité brute — français (évaluation)

| Seau de confiance | N | Confiance moy. | Exactitude top-1 | Écart (conf. − exact.) |
|---|---:|---:|---:|---:|
| [0.0 ; 0.1[ | 3 | 0.085 | 0.333 | -0.248 |
| [0.1 ; 0.2[ | 62 | 0.150 | 0.113 | +0.037 |
| [0.2 ; 0.3[ | 179 | 0.247 | 0.223 | +0.023 |
| [0.3 ; 0.4[ | 132 | 0.371 | 0.417 | -0.045 |
| [0.4 ; 0.5[ | 85 | 0.456 | 0.518 | -0.062 |
| [0.5 ; 0.6[ | 707 | 0.503 | 0.605 | -0.103 |
| [0.6 ; 0.7[ | 284 | 0.665 | 0.799 | -0.134 |
| [0.7 ; 0.8[ | 180 | 0.752 | 0.850 | -0.098 |
| [0.8 ; 0.9[ | 161 | 0.840 | 0.870 | -0.029 |
| [0.9 ; 1.0] | 207 | 0.963 | 0.971 | -0.008 |

## 1bis. Correction paramétrique : une température

Les probabilités sont élevées à la puissance 1/T puis renormalisées. T est **1 réel ajusté** (recherche en section dorée, critère : bits par caractère sur l'échantillon de CALIBRATION, jamais sur l'évaluation). Le candidat en tête ne change pas, donc l'exactitude top-1 est inchangée.

| Réglage | T | ECE éval. avant | ECE éval. après | Brier avant | Brier après | Biais après | bpc avant | bpc après |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anglais, T propre à la langue | 1.115 | 0.0855 | 0.1165 | 0.1943 | 0.2016 | -0.1059 | 2.224 | 2.230 |
| français, T propre à la langue | 1.058 | 0.0766 | 0.0964 | 0.1838 | 0.1875 | -0.0914 | 1.939 | 1.943 |
| anglais, T commun (calibration EN+FR réunies) | 1.087 | 0.0855 | 0.1077 | 0.1943 | 0.1994 | -0.0956 | 2.224 | 2.226 |
| français, T commun (calibration EN+FR réunies) | 1.087 | 0.0766 | 0.1062 | 0.1838 | 0.1897 | -0.1019 | 1.939 | 1.947 |

Contrôle sur l'échantillon de calibration (où T a été ajusté) — anglais : ECE 0.0404 -> 0.0765 ; sur l'évaluation, IC95 de l'ECE après T : [0.1011 ; 0.1373].
Contrôle sur l'échantillon de calibration (où T a été ajusté) — français : ECE 0.0759 -> 0.0960 ; sur l'évaluation, IC95 de l'ECE après T : [0.0811 ; 0.1176].

### Table de fiabilité après température (T de la langue) — anglais (évaluation)

| Seau de confiance | N | Confiance moy. | Exactitude top-1 | Écart (conf. − exact.) |
|---|---:|---:|---:|---:|
| [0.0 ; 0.1[ | 6 | 0.086 | 0.000 | +0.086 |
| [0.1 ; 0.2[ | 109 | 0.151 | 0.128 | +0.022 |
| [0.2 ; 0.3[ | 202 | 0.239 | 0.213 | +0.027 |
| [0.3 ; 0.4[ | 137 | 0.352 | 0.336 | +0.017 |
| [0.4 ; 0.5[ | 940 | 0.453 | 0.601 | -0.148 |
| [0.5 ; 0.6[ | 92 | 0.555 | 0.663 | -0.108 |
| [0.6 ; 0.7[ | 210 | 0.643 | 0.843 | -0.200 |
| [0.7 ; 0.8[ | 156 | 0.749 | 0.891 | -0.142 |
| [0.8 ; 0.9[ | 95 | 0.852 | 0.937 | -0.085 |
| [0.9 ; 1.0] | 53 | 0.938 | 0.962 | -0.025 |

### Table de fiabilité après température (T de la langue) — français (évaluation)

| Seau de confiance | N | Confiance moy. | Exactitude top-1 | Écart (conf. − exact.) |
|---|---:|---:|---:|---:|
| [0.0 ; 0.1[ | 7 | 0.090 | 0.143 | -0.052 |
| [0.1 ; 0.2[ | 73 | 0.157 | 0.151 | +0.006 |
| [0.2 ; 0.3[ | 171 | 0.242 | 0.222 | +0.020 |
| [0.3 ; 0.4[ | 135 | 0.360 | 0.415 | -0.055 |
| [0.4 ; 0.5[ | 754 | 0.473 | 0.603 | -0.130 |
| [0.5 ; 0.6[ | 42 | 0.553 | 0.524 | +0.029 |
| [0.6 ; 0.7[ | 280 | 0.642 | 0.804 | -0.162 |
| [0.7 ; 0.8[ | 236 | 0.741 | 0.864 | -0.124 |
| [0.8 ; 0.9[ | 122 | 0.851 | 0.885 | -0.034 |
| [0.9 ; 1.0] | 180 | 0.961 | 0.978 | -0.017 |

## 2. Auto-prédiction de compétence

Avant d'inférer, Écho regarde la demande (analyse seule, entités connues dans le contexte) et prédit « je vais comprendre », « je ne vais pas comprendre » ou « je vais devoir m'abstenir ». Puis on lance le vrai traitement. Issues : **compris** (bonne réponse vérifiée, ou affirmation lue comme attendu), **incompris** (refus « je ne comprends pas »), **abstenu** (« je ne sais pas »), **faux** (réponse fausse, ou phrase acceptée mais mal lue).  Une prédiction est juste quand elle correspond à l'issue ; « faux » n'est jamais prédit (le système ne sait pas qu'il lit de travers).

Mélange fixe (graine 2024) de 300 demandes : 110 contrôlée, 110 libre, 80 sans réponse.

### Matrice de confusion (lignes = prédiction, colonnes = issue réelle)

| Prédiction \ Issue | compris | incompris | abstenu | faux | Total |
|---|---:|---:|---:|---:|---:|
| je vais comprendre | 116 | 0 | 23 | 15 | 154 |
| je ne vais pas comprendre | 0 | 87 | 0 | 0 | 87 |
| je vais devoir m'abstenir | 0 | 0 | 59 | 0 | 59 |
| **Total issue** | 116 | 87 | 82 | 15 | 300 |

- **Exactitude de l'auto-prédiction : 87.3 % (262/300).**
- Référence « toujours prédire l'issue la plus fréquente » : 38.7 %.
- Précision de « je vais comprendre » : 75.3 % ; 38 fois j'ai annoncé « je vais comprendre » et c'était faux ou abstenu.
- Coût moyen : prédiction 86.0 µs, exécution 168.2 µs (CPU).

| Type de demande | N | Prédictions justes |
|---|---:|---:|
| contrôlée (t1 position) | 22 | 22/22 (100.0 %) |
| contrôlée (t2 objet) | 22 | 22/22 (100.0 %) |
| contrôlée (t3 déduction) | 22 | 22/22 (100.0 %) |
| contrôlée (t4 oui/non) | 22 | 22/22 (100.0 %) |
| contrôlée (t6 arithmétique) | 22 | 22/22 (100.0 %) |
| libre | 110 | 95/110 (86.4 %) |
| sans réponse | 80 | 57/80 (71.2 %) |

Erreurs d'auto-prédiction (jusqu'à 40 sur 38) :

| Type | Demande | Prédiction | Issue | Raison |
|---|---|---|---|---|
| sans réponse | Médor est-il un cheval ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Où est le stylo ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Est-ce que Milou est un fleur ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Où est le sac ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Tom est-il un métal ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Où est le ballon ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Rantanplan est-il un cheval ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Où est le journal ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Titi est-il un loup ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Où est le journal ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Bambi est-il un insecte ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Où est le ballon ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Est-ce que Rantanplan est un aigle ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Où est la bouteille ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Félix est-il un poisson ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Où est le livre ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Socrate est-il un mammifère ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Où est la lampe ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Azor est-il un reptile ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Où est le journal ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Milou est-il un fleur ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Où est le ballon ? | je vais comprendre | abstenu | question analysable, entités connues |
| sans réponse | Est-ce que Rex est un chien ? | je vais comprendre | abstenu | question analysable, entités connues |
| libre | Paul est le frère de Marie. | je vais comprendre | faux | affirmation analysable |
| libre | La pomme est rouge. | je vais comprendre | faux | affirmation analysable |
| libre | Marie est en retard. | je vais comprendre | faux | affirmation analysable |
| libre | Jean est à l'heure. | je vais comprendre | faux | affirmation analysable |
| libre | Marie est allée dans la cuisine hier. | je vais comprendre | faux | affirmation analysable |
| libre | Jean prend le train. | je vais comprendre | faux | affirmation analysable |
| libre | Socrate est un homme mortel. | je vais comprendre | faux | affirmation analysable |
| libre | Le ciel est bleu aujourd'hui. | je vais comprendre | faux | affirmation analysable |
| libre | Tout le monde est mortel. | je vais comprendre | faux | affirmation analysable |
| libre | Combien font 7 plus 5 ? | je vais comprendre | faux | calcul analysable |
| libre | Le café est chaud. | je vais comprendre | faux | affirmation analysable |
| libre | Aucun homme n'est une île. | je vais comprendre | faux | affirmation analysable |
| libre | Jean est le père de Paul. | je vais comprendre | faux | affirmation analysable |
| libre | Julie est médecin. | je vais comprendre | faux | affirmation analysable |
| libre | Tout est possible. | je vais comprendre | faux | affirmation analysable |

## 3. Auto-description

`python3 soi.py --qui` relit `resultats_banc.json`, `resultats_raisonneur.json`, `resultats_soi.json`, la taille des fichiers de mémoire et la liste des modules présents ; aucun nombre n'est écrit en dur dans `soi.py`.

Mémoire mesurée : 138 511 paragraphes, 22 793 538 caractères (anglais 82 109 p., français 56 402 p.).

## Coût (ESTIMATION)

**Estimation, pas une mesure** : énergie = temps CPU (`time.process_time`) × 10 W (10 W supposés par cœur occupé ; aucun wattmètre).

| Étape | CPU (s) | Énergie estimée (J) |
|---|---:|---:|
| Calibration en (collecte, mémoire 10 %) | 43.5 | 434.7 |
| Calibration fr (collecte, mémoire 10 %) | 64.2 | 641.9 |
| Ajustement des températures + tables + bootstrap | 2.1 | 20.8 |
| Auto-prédiction (300 demandes, prédiction + exécution) | 0.1 | 1.0 |
| **Total CPU (tous processus)** | **109.8** | **1098.4** |

Durée réelle totale : 68 s (au plus 2 processus en parallèle).

## Limites (à lire)

1. **Ce n'est pas une conscience.** Calibrer ses probabilités et prédire son propre comportement est de la métacognition fonctionnelle ; rien ne dit qu'il y ait une expérience derrière, et rien dans le code ne le suggère.
2. **Calibration sur un seul échantillon par langue** (~2000 caractères évalués, ~2000 pour calibrer) : voir les IC95 bootstrap de l'ECE ; des différences de quelques millièmes ne sont pas significatives. Les caractères voisins d'un même paragraphe ne sont pas indépendants, l'IC est donc trop étroit.
3. **Mémoires réduites (10 %)** : la calibration de la mémoire complète (15,7 M caractères) n'a pas été mesurée ; T dépend probablement de la taille de la mémoire et de la langue.
4. **Un seul réel ajusté** par réglage (T). Il corrige un biais global de confiance, pas les erreurs locales ; le test « T commun » montre à quel point il dépend de la langue.
5. **L'auto-prédiction ne voit pas ses erreurs de lecture** : le système ne prédit jamais « faux » ; les phrases mal lues passent l'analyseur sans alerte. Elle ne prévoit pas non plus les abstentions qui dépendent d'un raisonnement (réciproque non valide, porteur jamais localisé…), car on ne lance pas d'inférence avant.
6. **Le mélange de demandes est écrit par nous** : les phrases contrôlées sont celles que l'analyseur connaît (les tâches viennent de `banc_raisonneur.py`), les phrases libres ont été choisies après avoir vu le comportement du raisonneur sur une partie d'entre elles (44 sont celles de `resultats_raisonneur.md`). L'exactitude de l'auto-prédiction dépend directement de ce mélange ; ce n'est pas une garantie sur du français réel.
7. **Prédiction en partie tautologique** : l'analyseur utilisé pour prévoir « je ne vais pas comprendre » est le même que celui du traitement réel ; les prédictions de non-compréhension sont donc justes par construction. La partie informative est la distinction « comprendre / m'abstenir » et les cas « faux » non détectés.
8. Énergie : estimation à 10 W/cœur, pas de mesure. Aucune comparaison avec un modèle neuronal n'est faite.
