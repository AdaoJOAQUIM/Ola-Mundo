# Bilan : tout ce que nous avons construit et mesuré

Ce bilan fusionne toute la conversation. Il ne contient que des chiffres **mesurés**
(les fichiers de résultats sont indiqués). Ce qui a échoué est dit ; ce qui tourne
encore est marqué « en cours ». L'énergie est toujours une **estimation** (temps CPU ×
10 W par cœur), jamais une mesure.

Machine : 4 cœurs CPU, 15 Go de RAM, aucun GPU. Répartition du travail demandée :
modèles économiques (Haiku) pour les tâches simples, Sonnet pour programmer, Opus
pour la planification, la vérification et la fusion.

## 1. Point de départ : « un vrai modèle de langage de 10 Mo »

| Essai | Résultat |
|---|---|
| Transformer classique de 4,76 M paramètres (9,54 Mo en fp16), `mini-llm/` | Arrêté après 100 itérations à votre demande (« pas de transformer ») |
| **Écho** : aucun paramètre, aucun entraînement, Python standard (`echo/echo.py`) | Bibliothèque compressée de **4,05 Mo** ; 79 à 84 % des suites de 5 mots générées sont inédites ; plus long passage recopié : 35 à 42 caractères |

## 2. Les pièces manquantes de l'IA actuelle, à petite échelle

| Pièce | Module | Mesure principale |
|---|---|---|
| Apprendre avec peu d'exemples | `predicteur.py`, `banc.py` | Écho-P bat LZMA à toutes les tailles de mémoire (2,04 contre 2,18 bits/caractère à 100 %) ; bz2 le bat quand la mémoire est très petite |
| Apprendre en continu | `banc.py` | Français ajouté en 0,07 s : 5,47 → 1,76 bit/car. ; anglais 2,04 → 2,09 (légère dilution, rien d'effacé) |
| Modèle du monde + raisonnement vérifiable | `raisonneur.py` | 7 tâches, 1 400 questions : 100 %, **0 hallucination** ; un vérificateur indépendant rejette les preuves falsifiées |
| Le soi (métacognition, pas conscience) | `soi.py` | Prévoit sa propre réussite 263 fois sur 300 (87,7 %, contre 39 % au hasard) ; calibration française corrigée (erreur 0,077 → 0,043) |
| L'objectif | `planificateur.py` | 152 plans sur 152 trouvés, vérifiés et optimaux ; 48 objectifs impossibles détectés sur 48 |
| Le coût | `aiguilleur.py` | 400 demandes justes sur 400 ; 9,5 s de calcul contre 271 s en lançant tout à chaque fois |
| Émergence et convergence | `induction.py` | Règles découvertes justes à 67 % (1 exemple par classe) → 98 % (20 exemples) ; point fixe atteint 540 fois sur 540 |

Limite commune : français contrôlé seulement, bancs d'essai écrits par nous. Les 100 %
prouvent que les moteurs sont corrects, pas qu'ils sont intelligents.

## 3. Intelligence fluide : ARC

| Système | Entraînement | **Évaluation** (un seul passage) |
|---|---|---|
| Solveur par recherche de programmes, 376 opérations, sans paramètre (`arc/solveur.py`) | 129 / 400 (32 %) | **63 / 400 (15,8 %)** |

Contrôle : 6 énigmes sur 6 restent résolues quand on retire leurs réponses du fichier.

## 4. Petits modèles récursifs (TRM)

| Sur 1 000 Sudokus de test | Paramètres | Entraînement | Résolus |
|---|---|---|---|
| Mini-TRM entraîné (et deux témoins sans récursion) | 331 128 | ≈ 20 min CPU | **0** (résultat négatif) |
| **Réseau écrit à la main**, poids 0/1, récursif (`trm/poids_ecrits.py`) | 8 829 (10,9 Ko) | **aucun** | 565 |
| **Réseau écrit + recherche parallèle** (`trm/recherche_ecrite.py`) | 8 829 | aucun | **1 000** (0 faux, 1,2 s pour 1 000) |

## 5. Programmation dans l'espace des poids appliquée à la langue

| Expérience | Résultat mesuré |
|---|---|
| Transformer à 2 couches **écrit à la main** (tête d'induction) | 2,29 bits/car., contre Écho-P 2,63 et LZMA 2,74 (même bibliothèque, même texte) |
| Chirurgie de SmolLM2-135M contre les boucles (0,055 % des poids) | Boucles 50 % → 35 % ; témoin aléatoire 47,5 % ; perplexité inchangée |
| **Circuit de tâche écrit** dans SmolLM2-135M (1 153 poids : un détecteur + une injection) | Sans exemple : capitales **0 → 57 %**, passé irrégulier **0 → 33 %** ; témoin 0 % ; perplexité inchangée ; échecs : contraires 7 %, traduction 0 % |

## 6. Reproduction de l'étude du MIT (« Sparks of Weight-Space Programming »)

| Vérification | Résultat |
|---|---|
| Audit des poids : Qwen3.5-0.8B d'origine contre corrigé | Seuls **2 tenseurs sur 488** diffèrent (couches 15 et 19, `mlp.down_proj`, 64 colonnes chacun) : **130 188 poids**, exactement le manifeste des auteurs |
| Comportement, 20 problèmes MATH-500 faciles, 2 048 jetons au plus | **Non concluant** : les deux modèles sont coupés à la limite dans 95 % des cas (19 sur 20) ; chacun ne termine qu'un problème (juste). Les « réussites » brutes (45 % et 35 %) viennent surtout de réponses repérées au milieu d'un raisonnement coupé : elles ne mesurent rien. Il faudrait au moins 16 000 jetons par réponse, soit environ une journée de CPU ici (3,4 h pour le modèle corrigé à 2 048 jetons) |

## 7. Fusion des gagnants des Paper Awards de l'ARC Prize 2025

Même sous-ensemble de 20 énigmes d'évaluation pour tous (notre solveur : 2 sur 20).

| Méthode | Ce qui est utilisé | Résultat (2 essais par énigme) |
|---|---|---|
| TRM (1er prix) | point de contrôle officiel `arcprize/trm_arc_prize_verification`, 8 votes par énigme au lieu d'environ 1 000 | **12 sur 20** (10 au premier essai), environ 6 min de CPU par énigme ; recompté par nous sur les réponses officielles. Réserve : ce modèle a appris les exemples de démonstration de ces énigmes pendant son entraînement (protocole officiel de TRM) ; seules les réponses de test sont inédites. 20 énigmes, c'est peu : environ ±11 points d'incertitude |
| SOAR (2e prix) | son idée, sans grand modèle de langage : notre solveur apprend de ses propres solutions (3 rondes sur l'entraînement seulement), puis une seule évaluation | **2 sur 20** ; sur les 400 énigmes d'évaluation, 63 → 64 : gain dans le bruit (`arc/soar_lite/`) |
| CompressARC (3e prix) | code officiel, entraînement réduit à environ 500 s par énigme sur 1 cœur (150 à 783 pas au lieu de 2 000) | **1 sur 20** (1a2e2828, déjà résolue par TRM), 2,4 h au total. Borne basse : les grandes énigmes n'ont eu que 150 pas ; l'article annonce environ 20 % sur l'évaluation avec l'entraînement complet |
| **Fusion** (`arc/fusion/fusionner.py`, règle fixée avant les résultats) | réponses « vérifiées » d'abord, puis accord entre méthodes, puis TRM, puis CompressARC | **8 sur 20 : pire que TRM seul** |

Pourquoi la fusion perd : un programme « vérifié » reproduit tous les exemples, mais cela
ne prouve pas qu'il a trouvé la règle. Sur les 400 énigmes d'évaluation, 149 programmes
reproduisent les exemples et 64 seulement donnent la bonne réponse au test (43 %) ; sur nos
20 énigmes, 9 contre 2. Placés en tête, ils ont chassé 4 bonnes réponses de TRM. Et aucune
méthode ne résout ici une énigme que TRM rate (au mieux 12 sur 20) : sur cet échantillon,
aucune règle de fusion ne pouvait battre TRM seul.

## 7 bis. Moteur d'auto-amélioration (`evolution/`)

Le moteur réécrit lui-même le code de l'estimateur d'Écho, sous garde-fous : juge protégé
par empreinte, liste blanche du code produit, exécution isolée. Il garde une archive de
variantes (idée de la Darwin Gödel Machine). Le texte de validation est coupé en trois :
A pour évoluer, B pour choisir, C pour le test, lu une seule fois.

| Mesure | Résultat |
|---|---|
| Test C (60 000 car.) | départ 1,863 → **variante évoluée 1,538 bit/car. (−17,5 %)** ; LZMA 1,925 ; une retouche faite à la main (une constante) : 1,524 |
| « S'améliorer à s'améliorer » (méta-génome qui évolue) | pas de gain significatif (5 graines) |
| Soi, objectif (longueur minimale + front de Pareto), coût, convergence, émergence (MAP-Elites) | moteur complet meilleur à calcul égal, mais pas significativement (p = 0,057 sur A) ; le soi saute 9 à 25 % des essais ; aucune technique connue redécouverte ; les « structures émergentes » sont surtout du code inutile |
| Garde-fous | 17 codes malveillants sur 17 refusés ; 18 tests sur 18 ; une tentative de modifier le fichier de garde-fous protégé a été bloquée par les permissions |

### Bayes, la dernière pierre (`evolution/bayes.py`)

Test sur un texte neuf D (51 271 car., lu une seule fois ; réglages choisis sur A seulement) :

| Modèle | bit/car. sur D | P(meilleur que le champion) |
|---|---|---|
| Départ (PPM-C) | 2,472 | 0 |
| Champion évolué | 1,897 | — |
| Pitman–Yor hiérarchique (bayésien, réglages par l'évidence) | 1,974 | 0 |
| Mélange bayésien des modèles | 1,897 | 0,45 |
| **Mélange bayésien qui peut changer de favori (« fixed share »)** | **1,889** | **1,00** (gain faible : 0,008) |
| LZMA | 2,314 | — |

Le mélange simple se fixe sur un modèle en quelques caractères (perte exacte : 2 bits, la
borne théorique). L'échantillonnage de Thompson pour choisir les mutations : non concluant
(P = 0,44). Pitman–Yor ne retrouve pas les décomptes de Kneser–Ney, car il part des comptes
bruts ; Kneser–Ney utilise des comptes de continuation.

### La fusion ARC refaite avec Bayes (`arc/bayes/`)

Fiabilité de chaque méthode mesurée sur d'autres énigmes (20 nouvelles énigmes pour TRM),
paramètres figés avant la notation, validation croisée en 2 plis (40 énigmes en tout).

| Sur 40 énigmes | TRM seul | Notre solveur seul | Ancienne règle | **Fusion bayésienne** | Au mieux |
|---|---|---|---|---|---|
| Résolues (2 essais) | 21 | 5 | 18 | **20 à 21** | 25 |

Bayes répare la règle (18 → 21) mais ne bat pas TRM seul : les écarts tiennent en une ou
deux énigmes. Routeur « valeur du calcul » : aucune économie sans perte (TRM fait 98 % du
calcul ; couper TRM sur 9 énigmes économise 44 % mais fait tomber le pli 1 de 11 à 6).

### Bayes comme principe unique (`bayes/`)

Lire = mettre à jour ses croyances (la bibliothèque sert d'a priori) ; prédire = garder
tous les modèles (429), pondérés par leur probabilité a posteriori.

| Mesure | Résultat |
|---|---|
| Texte neuf E (56 195 car., retiré de la bibliothèque, lu une fois) | mélange « fixed share » **1,634** contre champion 1,657 et LZMA 2,057 |
| Changement de langue (bibliothèque anglaise, lecture de français) | 5,14 → **2,45** bit/car. par simple mise à jour, sans entraînement ; meilleur que LZMA dans 24 fenêtres sur 24. Une partie du gain vient des caractères absents de l'anglais |
| Optimisation bayésienne (Optuna) contre évolution | même niveau en **7 à 16 fois moins d'essais** ; sur E, égale la retouche faite à la main |

## 7 ter. ARC-AGI-3 : le harnais de Prime Agent sans grand modèle (`arc3/`)

Protocole et code d'accès au jeu repris de Prime Agent (licence MIT) ; les jeux sont des
boîtes noires, leur code n'est jamais lu. Réglage sur 5 jeux, test une seule fois sur 20.

| 20 jeux de test | Niveaux | Jeux gagnés |
|---|---|---|
| Hasard (3 graines) | 1,7 / 150 | 0 |
| **Cerveau bayésien sur CPU** (2 à 5 s par jeu) | **7 / 150** (recompté par nous : identique) | 0 |
| Prime Agent + Claude Opus 5 (environ 1 000 $ par passage) | 145 / 150 | 19 |

Ce qui manque : des connaissances de base (objet, personnage, but), deviner le but du jeu,
et un modèle du monde sous forme de programme testé par l'expérience. Prime Agent réussit
déjà tous les niveaux dans un passage sur trois ; ses derniers points perdus viennent de
l'efficacité en actions.

## 8. Ce qui reste à faire

Moteur pour un modèle de ≈ 100 milliards de paramètres sur votre machine (4 Go de RAM,
128 Go de disque) ; compression du modèle ; monde infini procédural ; agent « phone
use » avec garde-fous. Possibles ensuite : la notation stricte de Qwen (environ une
journée de CPU) ; vérifier sur plus d'énigmes si notre solveur résout ce que TRM rate
(sur nos 20 : aucune).

## Ce qu'il faut retenir

1. **Écrire la connaissance dans les poids bat l'apprentissage quand les règles sont
   connues et exactes** : Sudoku 1 000 sur 1 000 sans entraînement, contre 0 pour un réseau entraîné.
2. **Sur un vrai modèle, on débloque ce qu'il sait déjà, on ne crée pas de savoir** :
   capitales 0 → 57 %, mais jamais au-dessus de ce qu'il fait avec des exemples.
3. **La vérification change tout, si elle vérifie la bonne chose** : 0 hallucination quand
   le vérificateur contrôle la réponse elle-même (raisonneur, Sudoku). Sur ARC, « reproduit
   les exemples » n'est juste que 43 fois sur 100 : s'y fier aveuglément a fait perdre la fusion.
4. **Les comportements diffus résistent** (boucles d'un petit modèle de base), comme l'a aussi constaté l'étude du MIT.

## Notes pratiques

- Accès réseau ajouté à l'environnement « Par défaut », avec votre accord : `huggingface.co`, `cas-bridge.xethub.hf.co`, `cdn-lfs.hf.co`, `us.aws.cdn.hf.co`.
- Le corpus français (Europarl, UD) n'est pas republiable : il reste hors de Git.
- Code tiers (TRM, CompressARC, SOAR) cloné sous `arc/externes/`, hors de Git, sous leurs licences.
