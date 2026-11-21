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
| Comportement, 20 problèmes MATH-500 faciles, 2 048 jetons au plus | Modèle d'origine : **95 % des réponses coupées** (les boucles) ; modèle corrigé : **en cours**. La réussite du modèle d'origine (45 %) est gonflée par une notation trop souple, à refaire sur la seule réponse encadrée |

## 7. Fusion des gagnants des Paper Awards de l'ARC Prize 2025 — **en cours**

Même sous-ensemble de 20 énigmes d'évaluation pour tous (notre solveur : 2 sur 20).

| Méthode | Ce qui est utilisé | État |
|---|---|---|
| TRM (1er prix) | point de contrôle officiel `arcprize/trm_arc_prize_verification`, moins de votes | en cours |
| SOAR (2e prix) | son idée (s'améliorer à partir de ses propres solutions), appliquée à notre solveur | en cours |
| CompressARC (3e prix) | code officiel, entraînement réduit à ≈ 500 s par énigme sur CPU | en attente du CPU |
| **Fusion** (`arc/fusion/fusionner.py`) | réponses vérifiées d'abord, puis accord entre méthodes, puis TRM, puis CompressARC | prête |

## 8. Ce qui reste à faire

Moteur pour un modèle de ≈ 100 milliards de paramètres sur votre machine (4 Go de RAM,
128 Go de disque) ; compression du modèle ; monde infini procédural ; agent « phone
use » avec garde-fous. Une fois le CPU libre : la notation stricte de Qwen et le
résultat de la fusion ARC.

## Ce qu'il faut retenir

1. **Écrire la connaissance dans les poids bat l'apprentissage quand les règles sont
   connues et exactes** : Sudoku 1 000 sur 1 000 sans entraînement, contre 0 pour un réseau entraîné.
2. **Sur un vrai modèle, on débloque ce qu'il sait déjà, on ne crée pas de savoir** :
   capitales 0 → 57 %, mais jamais au-dessus de ce qu'il fait avec des exemples.
3. **La vérification change tout** : 0 hallucination partout où un vérificateur existe.
4. **Les comportements diffus résistent** (boucles d'un petit modèle de base), comme l'a aussi constaté l'étude du MIT.

## Notes pratiques

- Accès réseau ajouté à l'environnement « Par défaut », avec votre accord : `huggingface.co`, `cas-bridge.xethub.hf.co`, `cdn-lfs.hf.co`, `us.aws.cdn.hf.co`.
- Le corpus français (Europarl, UD) n'est pas republiable : il reste hors de Git.
- Code tiers (TRM, CompressARC, SOAR) cloné sous `arc/externes/`, hors de Git, sous leurs licences.
