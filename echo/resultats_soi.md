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

| Langue | Exactitude top-1 | Confiance moy. | Biais | Sens | ECE brut [IC95 par paragraphes] | Brier top-1 | bpc |
|---|---:|---:|---:|---|---:|---:|---:|
| anglais | 0.593 | 0.530 | -0.0628 | TROP MODESTE (sous-confiant) | 0.0855 [0.0743 ; 0.1259] | 0.1943 | 2.224 |
| français | 0.648 | 0.578 | -0.0701 | TROP MODESTE (sous-confiant) | 0.0766 [0.0603 ; 0.1022] | 0.1838 | 1.939 |

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

Les probabilités sont élevées à la puissance 1/T puis renormalisées (T < 1 : plus tranché ; T > 1 : plus modeste). T est **1 réel ajusté** (recherche en section dorée sur l'échantillon de CALIBRATION, jamais sur l'évaluation). Critère retenu : Brier du top-1. À titre de comparaison, T ajusté sur les bits par caractère (NLL) : critère standard du « temperature scaling », mais il vise toute la distribution, pas la confiance du top-1. Le candidat en tête ne change pas, donc l'exactitude top-1 est inchangée.

| Réglage | T | ECE éval. avant | ECE éval. après [IC95 par paragraphes] | Brier avant | Brier après | Biais après | bpc avant | bpc après |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anglais, T Brier propre à la langue (retenu) | 0.989 | 0.0855 | 0.0809 [0.0695 ; 0.1208] | 0.1943 | 0.1938 | -0.0584 | 2.224 | 2.224 |
| français, T Brier propre à la langue (retenu) | 0.882 | 0.0766 | 0.0434 [0.0294 ; 0.0635] | 0.1838 | 0.1797 | -0.0249 | 1.939 | 1.958 |
| anglais, T Brier commun (calibration EN+FR réunies) | 0.936 | 0.0855 | 0.0683 | 0.1943 | 0.1922 | -0.0375 | 2.224 | 2.233 |
| français, T Brier commun (calibration EN+FR réunies) | 0.936 | 0.0766 | 0.0600 | 0.1838 | 0.1809 | -0.0460 | 1.939 | 1.945 |
| anglais, T NLL (comparaison) | 1.115 | 0.0855 | 0.1165 | 0.1943 | 0.2016 | -0.1059 | 2.224 | 2.230 |
| français, T NLL (comparaison) | 1.058 | 0.0766 | 0.0964 | 0.1838 | 0.1875 | -0.0914 | 1.939 | 1.943 |

Contrôle sur l'échantillon de calibration (où T a été ajusté) — anglais : ECE 0.0404 -> 0.0390 ; IC95 de l'ECE brut sur l'évaluation : [0.0743 ; 0.1259].
Contrôle sur l'échantillon de calibration (où T a été ajusté) — français : ECE 0.0759 -> 0.0395 ; IC95 de l'ECE brut sur l'évaluation : [0.0603 ; 0.1022].

### Table de fiabilité après température (T de la langue) — anglais (évaluation)

| Seau de confiance | N | Confiance moy. | Exactitude top-1 | Écart (conf. − exact.) |
|---|---:|---:|---:|---:|
| [0.0 ; 0.1[ | 3 | 0.084 | 0.000 | +0.084 |
| [0.1 ; 0.2[ | 88 | 0.155 | 0.114 | +0.041 |
| [0.2 ; 0.3[ | 203 | 0.253 | 0.177 | +0.076 |
| [0.3 ; 0.4[ | 77 | 0.346 | 0.351 | -0.005 |
| [0.4 ; 0.5[ | 107 | 0.424 | 0.393 | +0.031 |
| [0.5 ; 0.6[ | 952 | 0.508 | 0.602 | -0.093 |
| [0.6 ; 0.7[ | 177 | 0.666 | 0.802 | -0.136 |
| [0.7 ; 0.8[ | 123 | 0.752 | 0.870 | -0.117 |
| [0.8 ; 0.9[ | 161 | 0.834 | 0.901 | -0.066 |
| [0.9 ; 1.0] | 109 | 0.938 | 0.945 | -0.007 |

### Table de fiabilité après température (T de la langue) — français (évaluation)

| Seau de confiance | N | Confiance moy. | Exactitude top-1 | Écart (conf. − exact.) |
|---|---:|---:|---:|---:|
| [0.0 ; 0.1[ | 2 | 0.095 | 0.000 | +0.095 |
| [0.1 ; 0.2[ | 59 | 0.162 | 0.119 | +0.043 |
| [0.2 ; 0.3[ | 161 | 0.263 | 0.224 | +0.040 |
| [0.3 ; 0.4[ | 69 | 0.349 | 0.275 | +0.074 |
| [0.4 ; 0.5[ | 133 | 0.450 | 0.489 | -0.039 |
| [0.5 ; 0.6[ | 690 | 0.550 | 0.593 | -0.043 |
| [0.6 ; 0.7[ | 88 | 0.647 | 0.750 | -0.103 |
| [0.7 ; 0.8[ | 317 | 0.742 | 0.808 | -0.065 |
| [0.8 ; 0.9[ | 213 | 0.841 | 0.859 | -0.018 |
| [0.9 ; 1.0] | 268 | 0.967 | 0.951 | +0.016 |

## 2. Auto-prédiction de compétence

Avant d'inférer, Écho regarde la demande (analyse seule, entités connues dans le contexte) et prédit « je vais comprendre », « je ne vais pas comprendre » ou « je vais devoir m'abstenir ». Puis on lance le vrai traitement. Issues : **compris** (bonne réponse vérifiée, ou affirmation lue comme attendu), **incompris** (refus « je ne comprends pas »), **abstenu** (« je ne sais pas »), **faux** (réponse fausse, ou phrase acceptée mais mal lue).  Une prédiction est juste quand elle correspond à l'issue ; « faux » n'est jamais prédit (le système ne sait pas qu'il lit de travers).

Mélange fixe (graine 2024) de 300 demandes : 110 contrôlée, 110 libre, 80 sans réponse.

### Matrice de confusion (lignes = prédiction, colonnes = issue réelle)

| Prédiction \ Issue | compris | incompris | abstenu | faux | Total |
|---|---:|---:|---:|---:|---:|
| je vais comprendre | 117 | 0 | 23 | 14 | 154 |
| je ne vais pas comprendre | 0 | 87 | 0 | 0 | 87 |
| je vais devoir m'abstenir | 0 | 0 | 59 | 0 | 59 |
| **Total issue** | 117 | 87 | 82 | 14 | 300 |

- **Exactitude de l'auto-prédiction : 87.7 % (263/300).**
- Référence « toujours prédire l'issue la plus fréquente » : 39.0 %.
- Précision de « je vais comprendre » : 76.0 % ; 37 fois j'ai annoncé « je vais comprendre » et c'était faux ou abstenu.
- Coût moyen : prédiction 74.3 µs, exécution 147.6 µs (CPU).

| Type de demande | N | Prédictions justes |
|---|---:|---:|
| contrôlée (t1 position) | 22 | 22/22 (100.0 %) |
| contrôlée (t2 objet) | 22 | 22/22 (100.0 %) |
| contrôlée (t3 déduction) | 22 | 22/22 (100.0 %) |
| contrôlée (t4 oui/non) | 22 | 22/22 (100.0 %) |
| contrôlée (t6 arithmétique) | 22 | 22/22 (100.0 %) |
| libre | 110 | 96/110 (87.3 %) |
| sans réponse | 80 | 57/80 (71.2 %) |

Erreurs d'auto-prédiction (jusqu'à 40 sur 37) :

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
| Calibration en (collecte, mémoire 10 %) | 43.6 | 435.7 |
| Calibration fr (collecte, mémoire 10 %) | 65.3 | 652.9 |
| Ajustement des températures + tables + bootstrap | 2.0 | 19.5 |
| Auto-prédiction (300 demandes, prédiction + exécution) | 0.1 | 0.9 |
| **Total CPU (tous processus)** | **110.9** | **1109.0** |

Durée réelle totale : 69 s (au plus 2 processus en parallèle).

## Limites (à lire)

1. **Ce n'est pas une conscience.** Calibrer ses probabilités et prédire son propre comportement est de la métacognition fonctionnelle ; rien ne dit qu'il y ait une expérience derrière, et rien dans le code ne le suggère.
2. **Calibration sur un seul échantillon par langue** (~2000 caractères évalués, ~2000 pour calibrer) : voir les IC95 bootstrap par paragraphes de l'ECE ; des différences de quelques millièmes ne sont pas significatives. Peu de paragraphes distincts (voir protocole) : l'IC par paragraphes est large et l'ECE d'une langue peut varier du simple au double d'un échantillon à l'autre.
3. **Mémoires réduites (10 %)** : la calibration de la mémoire complète (15,7 M caractères) n'a pas été mesurée ; T dépend probablement de la taille de la mémoire et de la langue.
4. **Un seul réel ajusté** par réglage (T). Il corrige un biais global de confiance, pas les erreurs locales ; le test « T commun » montre à quel point il dépend de la langue.
5. **L'auto-prédiction ne voit pas ses erreurs de lecture** : le système ne prédit jamais « faux » ; les phrases mal lues passent l'analyseur sans alerte. Elle ne prévoit pas non plus les abstentions qui dépendent d'un raisonnement (réciproque non valide, porteur jamais localisé…), car on ne lance pas d'inférence avant.
6. **Le mélange de demandes est écrit par nous** : les phrases contrôlées sont celles que l'analyseur connaît (les tâches viennent de `banc_raisonneur.py`), les phrases libres ont été choisies après avoir vu le comportement du raisonneur sur une partie d'entre elles (44 sont celles de `resultats_raisonneur.md`). L'exactitude de l'auto-prédiction dépend directement de ce mélange ; ce n'est pas une garantie sur du français réel.
7. **Prédiction en partie tautologique** : l'analyseur utilisé pour prévoir « je ne vais pas comprendre » est le même que celui du traitement réel ; les prédictions de non-compréhension sont donc justes par construction. La partie informative est la distinction « comprendre / m'abstenir » et les cas « faux » non détectés.
8. Énergie : estimation à 10 W/cœur, pas de mesure. Aucune comparaison avec un modèle neuronal n'est faite.
