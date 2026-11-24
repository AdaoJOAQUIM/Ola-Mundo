# Mondes infinis : résultats des baselines

Généré par `calibrer.py`. Mondes procéduraux de type ARC-AGI-3 ; chaque niveau est prouvé soluble par un oracle A* sur l'état interne, qui donne aussi l'optimum d'actions. Protocole du harnais : courtier + garde, 500 actions par monde, RESET automatique et compté après un game-over.

## Le jeu de mondes

- 200 mondes, 1049 niveaux (5.25 par monde, de 3 à 8).
- Optimum de l'oracle par niveau : moyenne 16.2 actions (min 2, max 51).
- 6.61 mécaniques actives par monde en moyenne.

## Résultat global

| agent | niveaux résolus | taux | mondes gagnés | efficacité vs optimum (niveaux résolus) |
|---|---|---|---|---|
| Cerveau bayésien (sans LLM) | 99/1049 | 9 % | 1/200 | 4 % |
| Aléatoire | 35/1049 | 3 % | 0/200 | 3 % |
| *Cerveau sur les 5 jeux DEV réels* | 2/33 | 6 % | 0/5 | — |
| *Aléatoire sur les 5 jeux DEV réels* | 0/33 | 0 % | 0/5 | — |

L'efficacité est le rapport optimum/actions réellement dépensées, sur les niveaux résolus (RESET et game-over inclus). 100 % = jouer comme l'oracle.

## Par difficulté (paramètre du monde : 1 facile, 5 le plus long)

| groupe | mondes | cerveau | aléatoire |
|---|---|---|---|
| 1 | 40 | 42/197 (21 %) | 16/197 (8 %) |
| 2 | 40 | 32/207 (16 %) | 12/207 (6 %) |
| 3 | 40 | 11/215 (5 %) | 7/215 (3 %) |
| 4 | 40 | 12/228 (5 %) | 0/228 (0 %) |
| 5 | 40 | 2/202 (1 %) | 0/202 (0 %) |

## Par rang de niveau dans le monde (1 = premier)

| groupe | mondes | cerveau | aléatoire |
|---|---|---|---|
| 1 | — | 60/200 (30 %) | 27/200 (14 %) |
| 2 | — | 25/200 (12 %) | 7/200 (4 %) |
| 3 | — | 10/200 (5 %) | 1/200 (0 %) |
| 4 | — | 4/159 (2 %) | 0/159 (0 %) |
| 5 | — | 0/117 (0 %) | 0/117 (0 %) |
| 6 | — | 0/92 (0 %) | 0/92 (0 %) |
| 7 | — | 0/56 (0 %) | 0/56 (0 %) |
| 8 | — | 0/25 (0 %) | 0/25 (0 %) |

## Par type de but

| groupe | mondes | cerveau | aléatoire |
|---|---|---|---|
| cibles | 50 | 14/259 (5 %) | 12/259 (5 %) |
| panel | 43 | 34/224 (15 %) | 4/224 (2 %) |
| sortie | 107 | 51/566 (9 %) | 19/566 (3 %) |

## Par mécanique (niveaux des mondes qui contiennent la mécanique ; un monde compte pour chacune de ses mécaniques)

| groupe | mondes | cerveau | aléatoire |
|---|---|---|---|
| annuler | 99 | 34/516 (7 %) | 8/516 (2 %) |
| attente | 69 | 11/366 (3 %) | 6/366 (2 %) |
| blocs_poussables | 88 | 31/441 (7 %) | 20/441 (4 %) |
| cibles | 50 | 14/259 (5 %) | 12/259 (5 %) |
| cles_portes | 64 | 23/346 (7 %) | 11/346 (3 %) |
| collectibles | 71 | 21/371 (6 %) | 9/371 (2 %) |
| couleurs | 41 | 10/231 (4 %) | 2/231 (1 %) |
| dangers | 74 | 25/393 (6 %) | 14/393 (4 %) |
| deplacement | 157 | 65/825 (8 %) | 31/825 (4 %) |
| energie | 106 | 36/537 (7 %) | 11/537 (2 %) |
| ennemis | 69 | 11/366 (3 %) | 6/366 (2 %) |
| gravite | 50 | 23/251 (9 %) | 15/251 (6 %) |
| interrupteurs | 73 | 25/394 (6 %) | 8/394 (2 %) |
| lumieres | 56 | 27/293 (9 %) | 3/293 (1 %) |
| motif | 51 | 7/274 (3 %) | 1/274 (0 %) |
| murs | 157 | 65/825 (8 %) | 31/825 (4 %) |
| selection | 10 | 0/55 (0 %) | 0/55 (0 %) |
| teleporteurs | 36 | 25/186 (13 %) | 11/186 (6 %) |

## Lecture : ces mondes sont-ils aussi difficiles que les jeux DEV réels ?

- Sur les 5 jeux DEV réels, le cerveau résout 2/33 niveaux (6 %) et l'aléatoire 0/33. Sur l'ensemble des mondes générés (difficultés 1 à 5 en proportions égales), le cerveau résout 9 % et l'aléatoire 3 %.
- Par palier : difficulté 1 : cerveau 21 %, aléatoire 8 % ; difficulté 2 : cerveau 16 %, aléatoire 6 % ; difficulté 3 : cerveau 5 %, aléatoire 3 % ; difficulté 4 : cerveau 5 %, aléatoire 0 % ; difficulté 5 : cerveau 1 %, aléatoire 0 %.
- Les paliers dont le taux du cerveau est à moins de 4 points du taux DEV réel : 3, 4. Les paliers plus faciles servent à mesurer un progrès que les vrais jeux ne laissent pas voir (le cerveau y est proche de zéro) ; les plus durs rapprochent l'évaluation de la difficulté réelle.
- Le premier niveau d'un monde est nettement plus facile que les suivants (cerveau : niveau 1 : 30 %, niveau 2 : 12 %, niveau 3 : 5 %, niveau 4 : 2 %). Les niveaux suivants réutilisent les mêmes règles avec plus d'éléments.
- Mécaniques les plus difficiles pour le cerveau : selection (0 %), motif (3 %), attente (3 %), ennemis (3 %). Les plus faciles : murs (8 %), gravite (9 %), lumieres (9 %), teleporteurs (13 %).
- Efficacité : sur les niveaux qu'il résout, le cerveau dépense en moyenne 25 fois l'optimum de l'oracle (aléatoire : 30 fois), loin d'un joueur humain.
- Limite : 5 jeux DEV et 33 niveaux seulement côté réel ; l'écart entre 2/33 et un taux mesuré sur plus de 1000 niveaux générés n'a pas de barre d'erreur serrée côté réel. Les mondes sont plus petits et plus réguliers que les vrais jeux (grilles de 6 à 10 cases, règles toutes déterministes à l'état complet) : un agent qui progresse ici peut ne pas progresser sur les vrais jeux. Utiliser ces mondes pour développer, pas pour prédire le score final.

## Contenu des mondes

18 mécaniques composables (3 à 8 niveaux par monde, mêmes règles dans tout le monde) : déplacement, murs, clés et portes,
blocs poussables, collectibles avec compteur (sortie verrouillée), dangers mortels, téléporteurs, interrupteurs qui
basculent des murs, couleurs (cases de peinture, sortie de la bonne couleur), lumières (lights-out), sélection puis
déplacement (clic sur une pièce, puis flèches), motif à recopier, gravité (4 directions possibles, sur les blocs),
énergie limitée (barre en bas), ennemis à patrouille déterministe (+ action d'attente), cibles (blocs sur cibles),
annuler (ACTION7). Règles cachées, tirées de la graine : but, mécaniques actives, correspondance ACTION1..4 vers
directions, couleurs et formes de chaque rôle, sens de la gravité. Aucune consigne n'est affichée ; seuls indices : la
sortie est creuse tant qu'un verrou (gemmes, panneau) n'est pas levé, la barre d'énergie, les pastilles de niveaux.
Les mécaniques sont corrélées à la difficulté et entre elles : le tableau par mécanique n'est pas une expérience contrôlée.

Garanties : chaque niveau est résolu par un A* sur l'état interne (optimum = proxy d'efficacité humaine), la solution
optimale est rejouée par l'interface publique dans les tests, tout est déterministe selon (graine, difficulté).

## Utilisation

- `MondeEnv(seed, difficulte)` : même surface que l'env du SDK vu par le harnais (`reset()`, `step(GameAction, data)`,
  `action1..7`), observations 64x64 en 16 couleurs, états NOT_FINISHED / WIN / GAME_OVER.
- `CourtierMonde(seed, difficulte)` : même protocole que `arc3_local_broker.Broker` (observe/act, lots de 20, 500 actions,
  RESET automatique compté après game-over). `jouer(seed, difficulte, fabrique)` reproduit `harnais.jouer`.
- `python evaluer.py --agent cerveau --mondes 20 --graine 0 [--difficulte 1..5|mix]` : niveaux résolus et actions contre
  l'optimum. L'agent est un module (ou `module:Classe`) avec une classe `Classe(graine).choisir(obs)`.
- `python calibrer.py` : régénère ce rapport (200 mondes, cerveau et aléatoire).
- Tests : `python -m pytest tests -q`.
