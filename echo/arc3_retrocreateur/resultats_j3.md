# Résultats J3 — Le Rétro-Créateur sur les vrais jeux de développement (non vérifié)

Critères pré-enregistrés dans `criteres.md` (commit 750ec06), inchangés. Données : `resultats_j3.json`, `runs/j3_*.json`.

## Verdict

**J3 NON atteint.** Seuil : >= 50 % des niveaux des 5 jeux DEV à 500 actions.
Mesure : **9,1 %** (9 niveaux sur 99 = 3 graines x 33 niveaux). Le 5000 actions ne change rien (9,1 %).

## DEV-train (cn04, ls20, sc25 : ont servi au développement, non équitable)

| jeu | niveaux, graines 0 / 1 / 2 (500 actions) | 5000 actions |
|---|---|---|
| cn04 | 0 / 0 / 0 sur 6 | 0 / 0 / 0 |
| ls20 | 2 / 2 / 2 sur 7 (niveaux 1 et 2 en 16 et 102 coups) | 2 / 2 / 2 |
| sc25 | 0 / 0 / 0 sur 6 | 0 / 0 / 0 |
Total : 6 / 57 = **10,5 %**. Référence de départ (v1 / v2) : 1 niveau par graine sur ls20, 0 ailleurs.

## DEV-val (tn36, ka59 : un seul passage, agent gelé au commit 3a72a74 ; seul signal équitable)

| jeu | niveaux, graines 0 / 1 / 2 (500 actions) | 5000 actions |
|---|---|---|
| tn36 | 1 / 1 / 1 sur 7 (niveau 1 en 368 coups) | 1 / 1 / 1 |
| ka59 | 0 / 0 / 0 sur 7 | 0 / 0 / 0 |
Total : 3 / 42 = **7,1 %**. Généralisation : très faible. Sur tn36 le niveau 1 coûte 368 coups (référence humaine 32, soit 11,5 fois), contre 48 pour la v1 seule : c'est un recul dû au filet (voir plus bas).

## Actions par niveau contre la référence humaine (niveaux résolus)
- DEV-train : 6 niveaux, rapport 0,78 (16/22 et 102/123 sur ls20 : mieux que la référence humaine sur ces deux niveaux).
- DEV-val : 3 niveaux (un par graine, le même), rapport 11,5.

## Régression sur les mondes
100 graines J1 tirées par `random.Random(11)` (budget 500, avec doublons du tirage avec remise) : niveaux 438 -> 439 sur 562 (77,9 % -> 78,1 %, +0,18 point). Seuil (pas plus de 5 points de baisse) : tenu. `runs/regression_mondes.json`.

## Nouvelles primitives (détails et preuves dans ARCHITECTURE.md, phase 3)
1. Perception : réseau par arêtes de couleur (sans plateau centré), sol appris, avatar = partie la plus rare.
2. Registre à motif (D4), déclencheur de transformation `mod`, but « registre = cible » `enterreg`, porte à motif `solidreg`.
3. Déclencheur d'énergie `refill` et planification avec énergie `chercher_u`.
4. Filet de sécurité : v2 (`bibliotheque`), pris à l'échec de perception ou après 250 actions sans plan sur un niveau.

## Causes d'échec restantes
- **Perception sans réseau** (sc25, tn36) : la période n'est pas trouvée ; seul le filet joue, sans programme. Jeux à clics sur des objets non alignés.
- **cn04** : corps rigide de plusieurs cases translaté / tourné par les actions, sans avatar d'une case ; le langage n'a pas de primitive « objet articulé » ni de but d'assemblage. Aucune mécanique comprise.
- **ls20 niveau 3** : le registre y est un motif colorié dont la couleur change aussi (case « changeur de couleur ») ; le registre ne gère que D4, pas encore une permutation de palette. Le niveau 3 est vu mais non résolu (382 coups).
- **ka59** : perçu en réseau, mais aucun plan ni but trouvé (196 explorations).
- **Filet** : n'ajoute aucun niveau ; sur tn36 il remplace la v1 et coûte 320 coups de plus pour le niveau 1.
- Le budget n'est pas la cause : 5000 actions ne donnent aucun niveau de plus.

## Écarts et honnêteté
- Les priors du générateur sont relâchés mais pas « appris » : réseau par arêtes en repli du réseau par composantes, sol appris à la première translation.
- Le filet est l'assemblage complet de la v2 (pas seulement ses briques) ; un RESET de la v2 est bloqué en cours de partie (il avait ramené un jeu au niveau 0).
- Budget large : plafond du courtier relevé en mémoire (comme en v2), aucun fichier modifié ; garde CPU 400 s.
- Un accident de disque plein (volume partagé) a interrompu une première série ; refaite en entier avec des dossiers d'enregistrement par processus, effacés après chaque partie. Les enregistrements des jeux ne sont jamais commités.
- Le graine d'agent ne change presque rien (jeu et agent déterministes) : les 3 graines donnent les mêmes niveaux.
- Les 20 jeux HELD n'ont jamais été ouverts ; le garde du lanceur lève une erreur (test). Aucun fichier de jeu lu.
- Non vérifié de façon indépendante.
