# J4 : examen final sur les 20 jeux HELD

Agent gele au commit d57daf4, joue une seule fois : 500 actions par jeu, graine d'environnement 0, graine de l'agent 0, aucun reseau, aucun modele de langage. Les jeux HELD n'ont pas ete lus par l'agent ; le garde de decoupe du runner J3 n'a pas ete utilise (pilote d'examen separe, meme adaptateur).

**Resultat : 6/150 niveaux (4.0 %)** contre 7/150 (4,67 %) pour le cerveau de base et 9/150 (6,0 %) pour Skill-Pro. Aucun jeu termine, aucun plantage.

## Controles avant examen

| controle | resultat |
|---|---|
| tests | 21 passes (tests/) |
| no_game_ids | pass |
| garde_socket | pass |
| decoupe_held | pass |
| import_mondes_arc3 | pass (test d'import isole) |
| sha256_retro | identiques avant/apres (17 fichiers) |

## Detail par jeu

| jeu | niveaux | actions | actions par niveau resolu (reference) | base | Skill-Pro |
|---|---|---|---|---|---|
| ar25 | 1/8 | 500 | 134 (32) | 0 | 0 |
| bp35 | 1/9 | 500 | 486 (21) | 0 | 0 |
| cd82 | 0/6 | 500 | - | 0 | 0 |
| dc22 | 0/6 | 500 | - | 0 | 0 |
| ft09 | 0/6 | 500 | - | 0 | 0 |
| g50t | 0/7 | 500 | - | 0 | 1 |
| lf52 | 0/10 | 500 | - | 1 | 1 |
| lp85 | 1/8 | 500 | 27 (17) | 1 | 3 |
| m0r0 | 1/6 | 500 | 68 (30) | 0 | 0 |
| r11l | 1/6 | 500 | 138 (22) | 1 | 1 |
| re86 | 0/8 | 500 | - | 0 | 0 |
| s5i5 | 0/8 | 500 | - | 0 | 0 |
| sb26 | 0/8 | 500 | - | 0 | 0 |
| sk48 | 0/8 | 500 | - | 0 | 0 |
| sp80 | 0/6 | 500 | - | 1 | 1 |
| su15 | 0/9 | 500 | - | 1 | 0 |
| tr87 | 0/6 | 500 | - | 0 | 0 |
| tu93 | 0/9 | 500 | - | 0 | 0 |
| vc33 | 1/7 | 500 | 321 (7) | 2 | 2 |
| wa30 | 0/9 | 500 | - | 0 | 0 |
| **total** | **6/150** | 10000 | | **7** | **9** |

## Lecture

- Les niveaux resolus l'ont ete avec beaucoup plus d'actions que la reference (de 1,6 a 46 fois) ; les niveaux 1 de ar25, bp35, lp85, m0r0, r11l, vc33 sont resolus, le niveau 2 jamais.
- La mesure est faite une fois, sans reglage ni nouvel essai. Le resultat (4,0 %) est inferieur a la baseline (7) et a Skill-Pro (9) sur ces 20 jeux ; la distance a 100 % est de 96 points.
- Remarque de protocole : le dossier echo/arc/externes (non versionne) a ete relie par un lien symbolique dans le worktree ; le SDK vient de /tmp/claude-0/arcsdk. Les 20 premiers essais du pilote ont plante a l'import (module absent) avant toute action de jeu et ont ete relances : aucun jeu n'a ete joue deux fois.
