# Compétences exécutables pour ARC-AGI-3 (Skill-Pro adapté), élève sans modèle de langage

Un enseignant (Claude) lit les trajectoires de jeux de dev et écrit de petites compétences Python déterministes ; l'élève (cerveau bayésien de `echo/arc3`, sans LLM) les exécute à pleine vitesse. Un juge épinglé (SHA-256) décide de leur acceptation sur des jeux de dev réservés. Test final : une seule exécution sur les 20 jeux held-out, protocole identique à celui de la baseline des 7/150 sur 150.

## Critères pré-enregistrés (`criteres.md`)

- **B1 (>= 14/150) : NON ATTEINT.** 9/150 niveaux held-out contre 7/150 pour la baseline (gain net de 2).
- **B2 (actions par niveau terminé) : PLUS MAUVAIS.** 236.6 actions par niveau contre 111.4 pour la baseline ; sur les 6 niveaux résolus par les deux : 261.3 contre 73.5. Les compétences allongent le premier niveau des jeux que la baseline résolvait vite (vc33 : 16 -> 142 actions, r11l : 10 -> 169, lp85 : 22 -> 68).
- **B3 (aucun réseau ni LLM à l'exécution) : CONFIRMÉ.** Analyse statique des imports (`eleve.py`, `outils.py`, `liste_blanche.py`, `cerveau.py`) et test dynamique : un jeu entier joué avec `socket.socket` remplacé par une fonction qui lève une erreur (`test_competences.py`).
- **B4 (transfert) :** 3/3 compétences acceptées s'activent sur des jeux held-out ; sonde_clics : 15/20 jeux, 8 niveaux gagnés pendant ou juste après; couloir : 12/20 jeux, 1 niveaux gagnés pendant ou juste après; paires_clics : 12/20 jeux, 0 niveaux gagnés pendant ou juste après.

## Compétences acceptées

1. `sonde_clics` : au début d'un niveau, clique une fois chaque objet de l'image initiale (couleurs rares d'abord). Acceptée par efficacité (actions dans les niveaux résolus de DEV-gate : 206 -> 50).
2. `couloir` : répète une flèche tant qu'elle déplace quelque chose vers un état nouveau (8 actions au plus). Acceptée en groupe à la ronde 3 (borne inférieure 0,016).
3. `paires_clics` : après RESET, essaie les paires ordonnées de clics parmi les 6 objets qui ont répondu. Acceptée en groupe à la ronde 3 ; aucun succès attribué sur DEV (0/18 activations), 1 en held-out (lp85).

Rejetées : tour_objets (acceptée en groupe puis élaguée : 0 succès sur 140 activations, aucun niveau perdu au retrait), repetition_clic, clic_puis_fleches, couverture, visite_objets (jamais activées sur DEV-gate), clic_nouveaux (fait perdre sc25 niveau 1), révisions paires_clics r5 et sonde_clics r5.

## Niveaux held-out par jeu (nous / baseline)

| jeu | niveaux nous | niveaux baseline | actions par niveau (nous) | actions par niveau (baseline) | activations (sonde / couloir / paires) |
|---|---|---|---|---|---|
| ar25 | 0/8 | 0 | [] | [] | 1 / 58 / 0 |
| bp35 | 0/9 | 0 | [] | [] | 1 / 56 / 1 |
| cd82 | 0/6 | 0 | [] | [] | 1 / 53 / 1 |
| dc22 | 0/6 | 0 | [] | [] | 1 / 33 / 1 |
| ft09 | 0/6 | 0 | [] | [] | 1 / 0 / 0 |
| g50t | 1/7 | 0 | [302] | [] | 0 / 66 / 0 |
| lf52 | 1/10 | 1 | [458] | [175] | 2 / 0 / 2 |
| lp85 | 3/8 | 1 | [68, 170, 89] | [22] | 4 / 0 / 3 |
| m0r0 | 0/6 | 0 | [] | [] | 1 / 98 / 1 |
| r11l | 1/6 | 1 | [169] | [10] | 2 / 0 / 2 |
| re86 | 0/8 | 0 | [] | [] | 0 / 53 / 0 |
| s5i5 | 0/8 | 0 | [] | [] | 1 / 0 / 1 |
| sb26 | 0/8 | 0 | [] | [] | 1 / 0 / 1 |
| sk48 | 0/8 | 0 | [] | [] | 1 / 68 / 0 |
| sp80 | 1/6 | 1 | [461] | [195] | 2 / 26 / 2 |
| su15 | 0/9 | 1 | [] | [339] | 1 / 0 / 1 |
| tr87 | 0/6 | 0 | [] | [] | 0 / 73 / 0 |
| tu93 | 0/9 | 0 | [] | [] | 0 / 6 / 0 |
| vc33 | 2/7 | 2 | [142, 270] | [16, 23] | 3 / 0 / 3 |
| wa30 | 0/9 | 0 | [] | [] | 0 / 69 / 0 |

Total : 9/150 contre 7/150. Niveaux gagnés par rapport à la baseline : [('g50t', 1), ('lp85', 2), ('lp85', 3)]. Niveaux perdus : [('su15', 1)].

## Lignée (jugements sur DEV-gate : tn36, sc25 ; graines 0 à 4 ; 10 paires)

| ronde | candidat | décision | niveaux base -> candidat | borne inférieure |
|---|---|---|---|---|
| r1 | couloir__r1 | rejeté | 5 -> 6 | -0.038 |
| r1 | sonde_clics__r1 | accepté | 5 -> 5 | 0.0 |
| r1 | tour_objets__r1 | rejeté | 5 -> 7 | -0.077 |
| r2 | paires_clics__r2 | rejeté | 5 -> 6 | -0.038 |
| r2 | repetition_clic__r2 | rejeté | 5 -> 5 | 0.0 |
| r2 | clic_puis_fleches__r2 | rejeté | 5 -> 5 | 0.0 |
| r3 | couloir__r3+tour_objets__r3+paires_clics__r3 | accepté | 5 -> 7 | 0.016 |
| r4 | couverture__r4+visite_objets__r4+clic_nouveaux__r4 | rejeté | 7 -> 5 | -0.384 |
| r4s | couverture__r4 | rejeté | 7 -> 7 | 0.0 |
| r4s | visite_objets__r4 | rejeté | 7 -> 7 | 0.0 |
| r4s | clic_nouveaux__r4 | rejeté | 7 -> 5 | -0.384 |
| r4 | retrait couloir | rejeté | 7 -> 5 | -0.384 |
| r4 | retrait tour_objets | accepté | 7 -> 7 | 0.0 |
| r5 | paires_clics__r5 | rejeté | 7 -> 8 | -0.283 |
| r5 | sonde_clics__r5 | rejeté | 7 -> 6 | -0.238 |

Postérieurs Beta (DEV-train graines 0 à 2 et DEV-gate graines 0 à 4, fin de ronde 4) : sonde_clics 5/23 (moyenne 0.24); couloir 7/443 (moyenne 0.018); tour_objets 0/140 (moyenne 0.007); paires_clics 0/18 (moyenne 0.05).

## Ce que ces résultats disent, et ne disent pas

- Le gain brut (+2 niveaux) est dans le bruit : un seul passage, une graine, 20 jeux. Il vient de g50t (couloir, +1) et lp85 (+2, après sonde_clics), contre une perte (su15). Aucun test statistique ne le distingue de zéro.
- Le juge n'a que 2 jeux (tn36, sc25), tous deux à clics : il ne peut pas valider des compétences de navigation (couverture, visite_objets n'ont jamais été activées par lui). Les acceptations reposent sur 10 paires et un 3e jugement sur les mêmes jeux (comparaisons multiples) : la borne de 0,016 de la ronde 3 est fragile.
- L'efficacité du gate (sonde_clics : 206 -> 50 actions) ne se retrouve pas en held-out : l'efficacité mesurée sur deux jeux ne transfère pas, elle s'inverse sur les jeux que la baseline résolvait vite. Je n'ai pas fait l'ablation sur held-out (ce serait un second passage).
- Les trajectoires DEV-train (cn04, ka59) demandent un raisonnement sur les objets (pièce à emboîter, bloc à pousser) que des compétences génériques ne capturent pas ; aucune compétence ne gagne de niveau sur ces deux jeux.
- Sur DEV-train, le bassin final gagne le niveau 1 de ls20 en graines 0 et 2 (305 et 427 actions) contre une seule graine pour la baseline (198 actions).

## Écarts par rapport au plan

- La liste blanche est un nouveau module (`liste_blanche.py`) qui reprend le schéma de `echo/evolution/garde_fous.py` (Rejet, vérification AST, bornes) sans l'importer ni le modifier : celui-ci ne convient qu'aux expressions numériques `estimer`.
- Le substrat de perception `h` (`outils.py` : objets, différences, mouvements, BFS) et le cerveau de repli sont écrits par moi ; seules les politiques sont des compétences. Sans compétence active, l'élève est identique à la baseline (vérifié par test).
- Découpe : DEV-train = ls20, cn04, ka59 ; DEV-gate = tn36, sc25 (2 jeux seulement : 5 jeux de dev au total). Les prieurs gelés de la baseline ont été calculés sur les 5 jeux de dev, donc aussi sur le gate.
- Jugement de groupe aux rondes 3 et 4 (plusieurs compétences révisées ensemble, 3 au plus), séquentiel aux rondes 1, 2 et 5. 5 rondes sur 6 utilisées : aucune idée nouvelle justifiée par les trajectoires.
- `couloir` est gardée malgré une moyenne a posteriori de 0,018 (< 0,05) parce que le juge (mode retrait) constate que son retrait fait perdre sc25 niveau 1.
- Succès d'une activation = niveau gagné pendant l'exécution ou dans les 25 actions suivantes ; l'attribution est donc approximative.
- Honnêteté : aucun fichier source de jeu n'a été ouvert ni inspecté ; seules les images, actions disponibles, `levels_completed` et `state` ont été utilisés. Les images des jeux held-out et de DEV-gate n'ont pas été examinées avant le test final.

## Fichiers

`criteres.md`, `liste_blanche.py`, `outils.py`, `eleve.py`, `jouer.py`, `juge.py` + `empreintes.json` (épinglés), `ronde.py`, `enseignant.py`, `final.py`, `skills/` (toutes les versions), `bassin.json`, `lignee.json`, `gradients/`, `test_competences.py`, `res_final.json`.
