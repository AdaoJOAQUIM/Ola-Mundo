# ARC-AGI-3 sans LLM : cerveau bayésien sur CPU

Généré par `rapport.py` à partir de `res_*.json`. Protocole Prime Agent (courtier + garde réutilisés), 500 actions/jeu, graine d'environnement 0.

## Held-out (20 jeux, agent gelé, un seul passage)

| agent | niveaux | jeux gagnés | actions totales |
|---|---|---|---|
| Cerveau bayésien (nous) | 7/150 | 0 | 10000 |
| Aléatoire (moy. 3 graines) | 1.67/150 | 0.0 | 10000.0 |
| Prime Agent run-1 (Claude Opus 5) | 145/150 | 19 | 9886 |

Score SDK local moyen par jeu : nous 0.587, aléatoire 0.12. CPU total nous : 63.9 s (aléatoire : 44.87 s).

## Dev (5 jeux : utilisés pour le débogage et le réglage, NON indépendants)

| agent | niveaux | jeux gagnés | actions totales |
|---|---|---|---|
| Cerveau bayésien (nous) | 2/33 | 0 | 2500 |
| Aléatoire (moy. 3 graines) | 0.0/33 | 0.0 | 2500.0 |
| Prime Agent run-1 (Claude Opus 5) | 33/33 | 5 | 1483 |

Score SDK local moyen par jeu : nous 0.642, aléatoire 0.0. CPU total nous : 12.0 s (aléatoire : 6.57 s).

## Les 25 jeux (mélange dev + held-out)

| agent | niveaux | jeux gagnés | actions totales |
|---|---|---|---|
| Cerveau bayésien (nous) | 9/183 | 0 | 12500 |
| Aléatoire (moy. 3 graines) | 1.67/183 | 0.0 | 12500.0 |
| Prime Agent run-1 (Claude Opus 5) | 178/183 | 24 | 11369 |

Score SDK local moyen par jeu : nous 0.598, aléatoire 0.1. CPU total nous : 75.9 s (aléatoire : 51.4 s).

## Détail par jeu

| jeu | split | niveaux | nous | état | score SDK | aléatoire (3 graines) | Prime | CPU s |
|---|---|---|---|---|---|---|---|---|
| ar25 | held-out | 8 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 8 (WIN) | 2.51 |
| bp35 | held-out | 9 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 9 (WIN) | 5.91 |
| cd82 | held-out | 6 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 6 (WIN) | 2.09 |
| cn04 | dev | 6 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 6 (WIN) | 2.06 |
| dc22 | held-out | 6 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 6 (WIN) | 2.31 |
| ft09 | held-out | 6 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 6 (WIN) | 2.66 |
| g50t | held-out | 7 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 7 (WIN) | 4.83 |
| ka59 | dev | 7 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 7 (WIN) | 1.95 |
| lf52 | held-out | 10 | 1 | ACTION_CAP | 0.06079406307977736 | [0, 0, 0] | 5 (ACTIVE) | 4.76 |
| lp85 | held-out | 8 | 1 | ACTION_CAP | 1.6586317722681359 | [0, 0, 0] | 8 (WIN) | 2.08 |
| ls20 | dev | 7 | 1 | ACTION_CAP | 0.04409171075837742 | [0, 0, 0] | 7 (WIN) | 2.31 |
| m0r0 | held-out | 6 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 6 (WIN) | 2.42 |
| r11l | held-out | 6 | 1 | ACTION_CAP | 4.761904761904762 | [1, 1, 0] | 6 (WIN) | 4.13 |
| re86 | held-out | 8 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 8 (WIN) | 1.01 |
| s5i5 | held-out | 8 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 8 (WIN) | 0.97 |
| sb26 | held-out | 8 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 8 (WIN) | 12.47 |
| sc25 | dev | 6 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 6 (WIN) | 2.89 |
| sk48 | held-out | 8 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 8 (WIN) | 2.12 |
| sp80 | held-out | 6 | 1 | ACTION_CAP | 0.19047619047619052 | [1, 1, 1] | 6 (WIN) | 2.77 |
| su15 | held-out | 9 | 1 | ACTION_CAP | 0.009359086290195487 | [0, 0, 0] | 9 (WIN) | 3.28 |
| tn36 | dev | 7 | 1 | ACTION_CAP | 3.1636183885318836 | [0, 0, 0] | 7 (WIN) | 2.83 |
| tr87 | held-out | 6 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 6 (WIN) | 1.06 |
| tu93 | held-out | 9 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 9 (WIN) | 4.27 |
| vc33 | held-out | 7 | 2 | ACTION_CAP | 5.0584249679314075 | [0, 0, 0] | 7 (WIN) | 1.39 |
| wa30 | held-out | 9 | 0 | ACTION_CAP | 0.0 | [0, 0, 0] | 9 (WIN) | 0.83 |

## Harnais continu (journal des versions)

Version finale gelée : **v2** (transfert de groupes : ['C']).

| version | description | test (jeux postérieurs) | décision |
|---|---|---|---|
| v0 | prieurs Beta non informatifs, aucun transfert | - | base |
| v1 | transfert des prieurs d'actions simples (A1-A5, A7) | 0 victoires / 1 défaites par niveau, P(meilleur)=0.245 | RETOUR ARRIERE |
| v2 | transfert des prieurs de clics par couleur/taille d'objet | 3 victoires / 0 défaites par niveau, P(meilleur)=0.9418 | GARDE |

## Valeur du calcul : économie si l'agent bon marché traite d'abord les niveaux qu'il sait résoudre

| run Prime Agent | coût total estimé | économie (niveaux de coût égal) | économie (coût ~ actions de référence) |
|---|---|---|---|
| run-1 | 1059.35 $ | 63.8 $ (6.02 %) | 20.5 $ (1.93 %) |
| run-2 | 1288.07 $ | 97.9 $ (7.6 %) | 29.8 $ (2.32 %) |
| run-3 | 944.0 $ | 39.4 $ (4.17 %) | 14.2 $ (1.51 %) |

Méthode : `results.json` ne donne pas le coût par jeu ; on répartit le coût total du run au prorata des appels modèle de chaque jeu.
Pour un jeu, l'agent bon marché joue d'abord ; les k premiers niveaux qu'il résout (et que Prime résolvait) sont retirés du LLM.
Variante 1 : tous les niveaux coûtent autant. Variante 2 : le coût d'un niveau est proportionnel à son nombre d'actions de référence (scorecard du SDK).

Réserves : (i) coût par niveau inconnu, il est modélisé ; (ii) le LLM doit reprendre un jeu déjà avancé : il lui faut rejouer/observer l'état, ce que ce calcul ignore ; (iii) les actions de l'agent bon marché consomment une partie du plafond de 500 actions ; (iv) les niveaux faciles sont probablement moins chers que la moyenne, d'où la variante 2 ; (v) 20 jeux held-out seulement + 5 jeux de dev tunés : estimation bruitée ; (vi) coût CPU de notre agent négligeable (quelques secondes par jeu).

## Notes et limites

- Le courtier de Prime Agent appelle `env.action1()...` qui n'existe plus dans le SDK installé : `harnais.CourtierCompat` ne remplace que `_apply` (via `env.step(GameAction, data)`). Le reste (dispatch, validation, garde de phase) est réutilisé tel quel, sans copie.
- Découpe dev/held-out : `random.Random(2026).sample(sorted(25 préfixes), 5)`. Réglages faits uniquement sur les jeux de dev (masque HUD, marche aléatoire de repli).
- Le harnais continu est testé sur les jeux de dev (jeux postérieurs, 3 graines du cerveau, appariées) ; le résultat v2 repose sur très peu de comparaisons (3 victoires/0 défaite) et est fragile. Les 20 jeux held-out utilisent les prieurs gelés, sans mise à jour entre jeux.
- Le « score SDK » est le score local du scorecard hors ligne (échelle en %, par jeu) ; il n'est pas comparable au RHAE de Prime Agent.
- Limites du cerveau : l'espace d'états des jeux est vaste, la nouveauté ne suffit pas comme signal de progrès ; pas de modèle d'objets/règles. Le masque HUD ne gère que des barres monotones en ligne/colonne.
