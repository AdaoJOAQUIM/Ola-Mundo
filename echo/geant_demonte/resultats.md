# Le géant démonté : résultats

Tout est mesuré sur ce CPU (4 cœurs, 15 Go de RAM, pas de GPU), sur les **300 questions TEST** (4 choix, hasard = 25 %, 71 commandes). Les intervalles sont des IC à 95 % par bootstrap (10 000 tirages, graine 0). Aucun texte de page de man n'est commité : seulement les questions et des nombres.

## Verdict

**Critère RÉUSSI** (les deux conditions du fichier `criteres.md`).

- C (Qwen3.5-0.8B + BM25) bat A (OLMoE-1B-7B de mémoire) de **+31,3 points** (seuil : +10). Différence appariée : IC 95 % [+26,0 ; +36,7], donc la borne basse dépasse même 10.
- Pic de RAM de C : **3 637 Mo** (3,55 Gio, soit 3,81 Go décimaux), index BM25 compris (seuil : 4 Go). La marge est de 0,2 à 0,45 Go selon l'unité : serrée.

Réserve importante, à lire avant de conclure quoi que ce soit : les questions citent mot pour mot la description qui se trouve dans la page de man. La récupération n'a donc qu'à retrouver la bonne phrase, ce que BM25 fait presque toujours (rappel 98,7 %). Ce test mesure « le petit modèle sait lire un passage fourni », pas « le petit modèle sait raisonner à partir d'une documentation qui paraphrase la question ». Voir « Limites ».

## Précision sur TEST

| Système | Description | Précision | IC 95 % | Secondes / question | RAM max |
|---|---|---|---|---|---|
| A | OLMoE-1B-7B, 4 bits en streaming disque, sans retrieval | 68,0 % | [62,7 ; 73,0] | 8,54 | 2,79 Go |
| B | Qwen3.5-0.8B base, sans retrieval | 67,0 % | [61,7 ; 72,3] | 0,76 | 2,71 Go |
| **C** | **Qwen3.5-0.8B + BM25, 3 passages** | **99,3 %** | **[98,3 ; 100,0]** | **1,90** | **3,55 Go** |
| Contrôle | Qwen3.5-0.8B + 3 passages au hasard d'autres commandes | 49,7 % | [44,0 ; 55,3] | 1,98 | 3,54 Go |
| B sans énoncé | Qwen3.5-0.8B, invite sans la description | 25,3 % | [20,7 ; 30,3] | 0,74 | 2,53 Go |
| D (30 premières questions TEST seulement) | OLMoE + BM25, 3 passages | 80,0 % | [66,7 ; 93,3] | 13,13 | 2,86 Go |

(Les « Go » de la colonne RAM sont des Gio : Mo mesurés divisés par 1024. Le hasard vaut 25 %.)

Différences appariées (mêmes 300 questions) :

| Comparaison | Écart | IC 95 % |
|---|---|---|
| C moins A | +31,3 points | [+26,0 ; +36,7] |
| C moins B | +32,3 points | [+27,0 ; +37,7] |
| C moins contrôle | +49,7 points | [+44,0 ; +55,3] |
| A moins B | +1,0 point | [-5,3 ; +7,3] |
| B moins B sans énoncé | +41,7 points | [+33,7 ; +49,3] |

Lecture : le modèle 8 fois plus gros ne fait pas mieux que le petit (A moins B est indistinguable de 0) ; ce qui change tout, c'est de lui donner les bonnes pages.

Par famille (A / B / C) : JDK, 157 questions : 67,5 / 61,8 / 98,7 % ; PostgreSQL, 91 questions : 62,6 / 70,3 / 100 % ; cargo, 52 questions : 78,8 / 76,9 / 100 %.

## Système D (optionnel, sous-ensemble)

Faute de temps, D n'a tourné que sur les 30 premières questions TEST (13,1 s par question, 2,86 Go de RAM pour le processus). Sur ces mêmes 30 questions : A 63,3 % [46,7 ; 80,0], B 70,0 % [53,3 ; 86,7], C 100 %, **D 80,0 % [66,7 ; 93,3]**. Les passages aident OLMoE (+16,7 points sur A, mais l'IC est très large avec n = 30), sans l'amener au niveau de C, alors qu'il est 8 fois plus gros : le goulot semble être la lecture de l'invite longue par un modèle quantifié en 4 bits avec le format « liste A-D » (choisi pour A sans passages), pas l'absence d'information. Ce n'est qu'un indice : un seul format, 30 questions.

## Tailles

- Bibliothèque : 99 pages de man, 0,79 Mo sur disque (compressées ou brutes selon la page), 1,34 Mo en texte brut extrait.
- Index BM25 (fenêtres de 120 mots, pas de 60 : 3 323 fenêtres, texte des fenêtres inclus) : 6,1 Mo (pickle, dans `cache/`, ignoré par git).
- Poids de Qwen3.5-0.8B en bf16 : environ 1,6 Go. Le pic de 3,55 Gio de C vient surtout du chargement du modèle, de PyTorch et des invites d'environ 700 jetons.

## Honnêteté : contrôles

**Rappel de la récupération.** Sur TEST, les 3 passages renvoyés par BM25 contiennent la chaîne du vrai drapeau dans 296 questions sur 300 (**98,7 %**) ; avec les alias du drapeau (forme courte ou longue) : 99,0 %. La bonne page est dans les 3 passages dans 99,7 % des cas. Les passages contiennent en moyenne 0,82 des 3 drapeaux distracteurs par question (les pages parlent des autres options). Quand la réponse est dans les passages, C a juste dans 99,7 % des cas ; quand elle n'y est pas (4 questions), 75 % (3 sur 4, donc quasi sans valeur statistique).

**Contrôle « passages mélangés, non pertinents ».** Avec 3 passages tirés au hasard dans des pages d'autres commandes (graine fixée par question), la réponse figure dans les passages dans 1,7 % des cas et la précision tombe à **49,7 %** : le gain de C ne vient donc pas du seul fait d'avoir un long contexte, il vient de la lecture des bons passages. On note que ce contrôle est aussi sous B (67 %). Ce n'est pas tout à fait comparable : B a utilisé le format « liste A-D » choisi sur TUNE, le contrôle et C le format « option seule » ; sur TUNE, B en format « option seule » faisait 54 %, soit à peu près le niveau du contrôle. Des passages hors sujet ne semblent donc pas faire beaucoup de mal, mais on ne l'a pas mesuré proprement.

**Fuite de l'énoncé.**
1. Aucun énoncé ne contient le drapeau de la réponse ni celui d'un distracteur (vérifié par test unitaire sur les 400 questions).
2. Ligne de base lexicale sans modèle (l'option qui partage le plus de mots avec l'énoncé) : **25,0 %** sur TEST comme sur TUNE, par construction (voir Écarts, point 1). La première version du générateur donnait 68 % sur TUNE : c'est ce qui a provoqué l'amendement.
3. Qwen sans l'énoncé (les options seules) : 25,3 %, donc rien ne fuit par les options ou leur position.
4. Reste un signal que le filtre ne retire pas et que l'on ne cherche pas à retirer : B de mémoire fait 67 %, bien au-dessus du hasard, parce que beaucoup de noms d'options sont sémantiquement guessables (`--help`, `--version`, `-q` pour « quiet »…) sans recouvrement de mots exact. C'est une connaissance réelle des conventions, pas une fuite de l'énoncé.

## Écarts par rapport à la consigne (liste complète)

1. **Générateur v2 et amendement 1 des critères.** Les premiers essais sur TUNE ont montré que les questions se résolvaient par recouvrement de mots (68 % sans modèle). Avant tout calcul sur TEST, j'ai refait le générateur (la bonne option et les 3 distracteurs ont le même recouvrement lexical avec l'énoncé, quasi toujours nul) et amendé `criteres.md` (commité avant TEST ; historique git conservé). Les résultats de la v1 sur TUNE ne sont pas conservés. Seuil de longueur d'énoncé passé de 6 à 4 mots minimum pour atteindre 400 questions.
2. **Bibliothèque élargie hors `/usr/share/man`.** `/usr/share/man/man1` ne contient que 60 pages (PostgreSQL et JDK) ; sous le filtre lexical elles donnaient environ 390 questions. J'ai ajouté les 39 pages de cargo, rustc et rustdoc de la chaîne Rust (`~/.rustup`). Les pages de rustc et rustdoc n'ont pas de table d'options lisible par mon analyseur : elles sont dans l'index mais ne fournissent pas de questions. Pas de repli sur les docstrings Python. Total : 400 questions, 76 commandes (71 dans TEST).
3. **Réglage de A sur 40 questions TUNE seulement** (au lieu de 100), pour tenir le temps (8,5 s par question et par format). Format retenu pour A : liste A-D, somme des log-probabilités (72,5 % sur ces 40, contre 67,5 % pour l'autre format). B et C réglés sur les 100.
4. **A est quantifié en 4 bits** (moteur de `echo/monde/`, +6 % de perplexité par rapport au bf16 selon `resultats_monde.md`), pas en bf16. Sa précision est donc peut-être un peu sous-estimée. Les experts sont lus via le cache de pages du système (sans O_DIRECT), pour que 300 questions tiennent dans le temps ; ce cache n'est pas compté dans la RAM du processus (2,79 Go), mais il occupe ~3,6 Go de la mémoire de la machine. Le critère de RAM ne porte que sur C.
5. **Une seule passe par question pour A et D** : invite et 4 options dans une même séquence avec un masque d'attention (chaque option ne voit que l'invite et elle-même). C'est mathématiquement équivalent à 4 passes séparées ; je ne l'ai pas vérifié numériquement contre 4 passes.
6. **Qwen en bf16** sur CPU, avec les noyaux de repli de transformers pour l'attention linéaire de Qwen3.5. Le premier appel de chaque processus (~47 s de mise en route) est exclu des chronomètres. Le temps d'une question de C comprend la recherche BM25.
7. **D : seulement 30 questions TEST** (les 30 premières), faute de temps, avec le format gelé pour A (non réglé pour D).
8. Le contrôle et « B sans énoncé » reprennent les formats gelés de C et de B.
9. La RAM est le `ru_maxrss` de chaque processus. Il inclut le pic de chargement du modèle.

## Limites

- Les énoncés sont des citations exactes de la bibliothèque : le rappel de 98,7 % est favorable. Avec des questions paraphrasées ou des réponses à composer à partir de plusieurs passages, il serait plus bas, et l'écart C moins A aussi.
- Un seul type de question (« quelle option fait ceci ? »), un seul domaine (outils PostgreSQL, JDK, cargo), une seule graine : pas de barre d'erreur liée au tirage des questions, seulement celle du bootstrap sur 300 questions.
- La bibliothèque est petite (1,3 Mo de texte). Rien ne dit que BM25 tiendrait sur une bibliothèque de plusieurs Go.
- Qwen3.5-0.8B pourrait avoir vu ces pages à l'entraînement ; le résultat B de 67 % ne permet pas de le distinguer.

## Fichiers

`bibliotheque.py` (analyse roff), `questions.py` (générateur) et `test_questions.py` (tests), `fuite.py` (ligne de base lexicale), `bm25.py`, `invites.py`, `qwen_score.py`, `olmoe_score.py`, `run_systeme.py`, `geler.py`, `analyse.py`, `rapport.py`, `questions.json` (400 questions), `format_gele.json`, `results/runs/*.jsonl` (log-vraisemblances par question), `resultats.json`.
