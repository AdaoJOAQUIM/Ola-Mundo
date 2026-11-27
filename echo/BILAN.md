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

## 7 quater. Graines : Kimi K3.1 complet dans 128 Go ? (`graines/`)

**Question.** K3.1 a 2,78 T de paramètres. 128 Go = 1,024·10¹² bits, soit environ 0,375 bit par poids.
Des générateurs de poids (SeedLM, PRANC, hyper-réseau) peuvent-ils atteindre ce débit sans casser le modèle ?

**Critère fixé à l'avance.** « Plausible » seulement si une méthode à ≤ 0,375 bit/poids reste à +10 % de
perplexité sur les deux modèles testés. Sinon, dire où la qualité casse.

**Résultats** (MLP comprimés, tous les bits comptés ; perplexité / original ; SmolLM2 = 23,12, Qwen = 27,20).

| Débit (bit/poids) | Meilleure méthode | SmolLM2-135M | Qwen3.5-0.8B |
|---|---|---|---|
| 4,25 | RTN 4 bits | ×1,20 | ×1,09 |
| 4,0 | SeedLM (recherche complète) | ×1,44 | ×1,28 (recherche partielle) |
| 3,25 | RTN 3 bits | ×3,0 | non mesuré |
| 2,25 | RTN 2 bits | ×4,5·10⁴ | ×168 |
| 1,0 | SeedLM | ×2,4·10⁶ | non mesuré |
| 0,375 | SeedLM (PRANC, hyper-réseau : pires) | ×2,7·10⁷ | ×2,3·10⁴ |

**Verdict : critère non atteint.** À 0,375 bit/poids le modèle est détruit sur les deux modèles. L'erreur relative de
SeedLM est de 0,86 : 16 bits de graine choisissent parmi 65 536 motifs aléatoires, et le meilleur ne conserve
qu'environ un quart d'un bloc de 64 poids. PRANC (0,99) et l'hyper-réseau (0,999) n'apprennent rien de mieux que la moyenne.
À 4 bits, SeedLM (×1,44) est moins bon que l'arrondi simple (RTN ×1,20) : la graine ne fait pas mieux que de bonnes échelles.

**Vérification indépendante.** Tests : 4 réussis. Comptabilité des bits : exacte pour RTN, SeedLM, PRANC et hyper-réseau.
Perplexité de base 23,12 et RTN 4 bits ×1,20 retrouvées. Erreur d'un bloc à 0,375 bit : 0,862 en moyenne
(théorie ≈ 0,86 ; des blocs de bruit pur donnent aussi 0,862). Parts de paramètres comprimés : 59,2 % et 35,1 %. Scripts et chiffres dans `graines/verif/`.

**Réserves.**
- Petits modèles (135 M et 0,8 B). Le plus grand est plus robuste (jusqu'à trois ordres de grandeur de moins
  de dégradation à débit égal, en partie parce qu'on y comprime moins de paramètres), mais reste ruiné à 0,375 bit.
- Écarts documentés : recherche de graines partielle sur certaines lignes, exposant partagé élargi, PRANC par blocs,
  un seul hyper-réseau essayé (25 min d'entraînement) ; PRANC et l'hyper-réseau n'ont pas été essayés sur Qwen.
- Le test comprime 59 % (SmolLM2) et 35 % (Qwen) des paramètres. Les experts de K3.1 en font environ 98 %,
  donc il n'y a presque rien d'intact pour compenser : le cas réel est plus dur.

## 7 quinquies. Moteur Monde Infini (phase 1) (`monde/`)

**Idée.** Le modèle est un monde ouvert : on ne charge que ce dont le mot courant a besoin. Le noyau (attention, normes,
routeur, embeddings) reste en RAM ; chaque expert est un bloc contigu lu sur disque d'un seul `pread`. Test sur OLMoE-1B-7B (CPU, 15 Go).

**Critères fixés à l'avance, revérifiés indépendamment.**
- (a) Sans perte vs transformers : RÉUSSI. 16 tokens identiques, écart max de logit 6,7·10⁻⁶ (revendiqué 1,7·10⁻⁵ sur 3×32 tokens ; limite 10⁻²).
- (b) RAM ≤ 4 Go : RÉUSSI. Pic mesuré 2,83 Go (16 tokens, tous les mécanismes). Le 3,70 Go revendiqué vient du run à cache 25 %.
- (c) Prefetch sans effet sur les sorties : RÉUSSI (tests, 13 sur 13).
- (d) Moitié des octets du 4 bits naïf à ≤ +2 % de perplexité : **ÉCHEC**. Tous mécanismes : 103 % des octets (101,7 % sur mes 16 tokens), +1,79 %.
- Octets : 16 couches × 8 experts × 3 538 944 o = 453 Mo/token. Le bloc est bien 4 bits, groupes de 32, échelle fp16, arrondi à 4096 o.

**Ce que chaque mécanisme donne vraiment.**
- Cache LRU : 0 % de succès à 0,25 %, 1 % et 5 % (confirmé), 50,8 % à 25 %. Cause : un token touche 128 blocs distincts (12,5 % du volume),
  la distance de réutilisation médiane est 226 blocs, donc un LRU plus petit qu'un token entier ne sert à rien.
- Cache épinglé par fréquence (mes mesures, appris sur 64 tokens, testé sur 64 autres) : 1,1 % à 0,25 %, 4,9 % à 1 %, 16,4 % à 5 %, 49,5 % à 25 %.
- Prefetch : devine 86 % des experts, cache la latence disque (attente 70 ms/token au lieu de 319), mais lit 13 % d'octets en plus.
- LOD et culling : environ −12 % d'octets pour +1,8 % de perplexité ; au-delà la qualité s'effondre.
- Vitesse : le modèle additionne lecture et calcul (le recouvrement parfait n'est donné qu'entre parenthèses). OLMoE sur NVMe 3,5 Go/s :
  1,36 s/token en somme, 1,22 s avec recouvrement parfait. Le calcul (CPU, torch) domine, pas le disque.

**Extrapolations corrigées (calcul, rien n'a été exécuté).** Configs lues sur le Hub. Les paramètres retrouvent 2 780 Md (total) et 104 Md (actifs) pour K3.1,
et la taille des fichiers (1 561 Go). 3 Go de poids non routés chauds gardés en RAM. Secondes par token, lecture seule, NVMe 3,5 / NVMe 7 / SATA 0,5 / USB 1 Go/s.
- Kimi-Linear-48B-A3B (49,1 Md, 3,1 Md actifs) : publié en bf16 : 3,2 Go/token, 0,92 / 0,46 / 6,4 / 3,2 s. À 4,25 bits : 0,78 Go, 0,22 / 0,11 / 1,6 / 0,78 s.
- K3.1 (experts latents 3584 × 3072, 33,0 M de paramètres) : publié (MXFP4 + bf16) : 134 Go/token, 38 / 19 / 269 / 134 s.
  Tout à 4,25 bits : 52 Go/token, 15 / 7,5 / 105 / 52 s.
- Écart avec le premier rapport (26 Go, 7,7 s sur NVMe) : il ne lisait que les experts. Les poids non routés (56 Md, bf16 : 111 Go)
  se lisent à chaque token ; seuls 25,8 Go sont des experts. Son expert de 33,7 M était rétro-calculé depuis le total ; la valeur lue est 33,0 M.

**Réserves.**
- Un seul modèle petit, un seul run par configuration, 128 tokens ; perplexité non refaite (chiffres de l'agent).
- Extrapolations sans noyau natif ni mesure : les formes des couches KDA/MLA sont déduites des configs (le total concorde, les détails fins non).
- Latences SATA/USB supposées dans le premier rapport ; ici seulement le débit. Les 3 Go résidents sont une hypothèse.
- Le 4 bits simple coûte déjà +6 % de perplexité contre bf16 ; l'objectif (d) était hors de portée avec ces mécanismes.

## 7 sexies. ARC-AGI-3 : du cerveau au Rétro-Créateur

Chronologie des jalons mesurés (jeux ARC-AGI-3 locaux, sans grand modèle à la partie) :

- **Cerveau bayésien (base)** : 7/150 niveaux sur les 20 jeux HELD, 500 actions par jeu.
- **Skill-Pro** (compétences apprises avec un grand modèle comme professeur) : 9/150 sur HELD.
- **v1 Scientifique** : 6/99 sur DEV.
- **v2** : 5/99 sur DEV, et 181/1119 niveaux sur les mondes procéduraux.
- **Rétro-Créateur J1**, 300 mondes procéduraux, vérifié de façon indépendante : 82 % des niveaux résolus, 76 % avec le « programme exact », 1,64 fois l'optimum d'actions. Deux réserves du vérificateur : « exact » est mesuré sur des états perçus, pas sur l'état réel ; l'a priori de plateau carré est propre au générateur de mondes.
- **J3 sur DEV** : 9,1 % des niveaux, seuil visé non atteint. Sur ls20, les niveaux 1 et 2 sont résolus en moins d'actions que les humains.
- **J3-bis (schémas appris)** : 12,1 %, seuil non atteint. Premier niveau 1 de cn04 résolu.
- **J4 (examen final, agent gelé au commit d57daf4, jeux HELD joués une seule fois, sans réglage)** : **6/150 niveaux (4,0 %)**, aucun jeu terminé, aucun plantage. Niveau 1 résolu sur ar25, bp35, lp85, m0r0, r11l et vc33 ; jamais le niveau 2. C'est moins que la base (7) et que Skill-Pro (9). Détail : `echo/arc3_retrocreateur/examen/resultats_j4.md`.

Ce qui bloque, sans enjolivure :

- **La découverte du but.** L'agent ne sait pas, sur un jeu inconnu, ce qu'il faut atteindre. Les mondes procéduraux donnent le but dans un cadre connu ; les vrais jeux non.
- **Des mécaniques absentes du langage.** Ce qui n'est pas exprimable dans le langage de règles ne peut ni être appris ni planifié. Les schémas appris sur DEV ne se transfèrent presque pas aux 20 jeux HELD.
- **Les niveaux résolus coûtent cher** : de 1,6 à 46 fois les actions de référence, et le budget de 500 actions s'épuise sur le niveau suivant.

Distance au but : 4,0 % contre 100 %, soit 96 points. Pour situer, le meilleur agent publié sans modèle de langage, StochasticGoose, a obtenu 12,58 % dans la compétition de préversion d'ARC-AGI-3. Ce n'est pas directement comparable : autres jeux, autre protocole. Notre 4,0 % sur HELD n'est pas non plus à comparer à notre 12,1 % sur DEV, qui avait servi au réglage.

## 7 septies. Le géant démonté (`geant_demonte/`)

Question : un petit modèle qui lit une bibliothèque locale bat-il un modèle plus gros qui répond de mémoire, sous 4 Go de RAM ? Le test porte sur 300 questions TEST à choix multiple (hasard 25 %), générées à partir des pages de man locales. Le critère était déposé d'avance.

| Système | Précision | IC 95 % | s/question |
|---|---|---|---|
| A : OLMoE 7B en 4 bits, de mémoire | 68,0 % | [62,7 ; 73,0] | 8,5 |
| B : Qwen 0,8B, de mémoire | 67,0 % | [61,7 ; 72,3] | 0,76 |
| **C : Qwen 0,8B + BM25** | **99,3 %** | [98,3 ; 100] | 1,9 |
| Contrôle : C avec passages hors sujet | 49,7 % | [44,0 ; 55,3] | 2,0 |
| D : OLMoE + BM25 (30 questions) | 80 % | [66,7 ; 93,3] | 13,1 |

**Critère réussi.** C dépasse A de +31,3 points, IC [+26,0 ; +36,7], pour un seuil fixé à +10. La RAM de pic de C est de 3 637 Mo, index compris. Les chiffres ont été recontrôlés depuis `resultats.json`.

Le contrôle confirme que le gain vient des bons passages :
- avec des passages hors sujet, C tombe de 99,3 % à 49,7 % ;
- la récupération trouve un passage contenant la réponse pour 98,7 % des questions ;
- la ligne de base lexicale vaut 25 %, et Qwen sans l'énoncé fait 25,3 % : pas de fuite.

Limites :
- les énoncés citent la page mot pour mot, ce qui avantage la récupération ;
- le test mesure la lecture d'un passage fourni, pas le raisonnement sur une documentation paraphrasée ;
- C et A n'utilisent pas le même format d'invite. Chaque format a été choisi sur TUNE puis gelé avant TEST.

Le premier générateur de questions fuyait lexicalement. Il a été refait, et les critères amendés, avant tout calcul sur TEST.

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
