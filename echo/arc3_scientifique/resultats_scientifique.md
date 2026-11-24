# Le Scientifique : résultats (DEV seulement, HELD non ouvert)

HELD est reporté à l'évaluation finale (amendement de criteres.md). **M1 (>= 30/150 sur HELD) n'est donc pas encore jugé.** Tout ce qui suit vient des 5 jeux DEV (cn04, ka59, ls20, sc25, tn36), qui ont servi au développement : les scores ne sont pas indépendants.

## Ce qui est construit
- `percep.py` : HUD (bandes monotones), mouvement par translation (groupement par co-mouvement), gabarit de sprite et suivi.
- `theorie.py` : langage de règles par couleur (traversable, bloquant, mortel, ramassable, objectif), postérieur par propagation de croyance sur des motifs d'observation (explication par élimination), vraisemblance bruitée, gain d'information = information mutuelle.
- `agent.py` : expériences au gain d'information + bonus de nouveauté (archive des positions), BFS en imagination dans la théorie, replanification à chaque pas ; repli générique (graphe d'états, clics par classes d'objets, gain BALD) quand aucun avatar n'est trouvé ; rejeu de la solution du niveau précédent.
- `adapter.py` / `run.py` : point d'entrée unique et adaptateur d'environnement. `sommeil.py` + `juge.py` (épinglé, `empreintes.json`).
- Tests : `test_scientifique.py` (8 tests passent : pas d'id de jeu, garde-socket, juge épinglé, déterminisme, inférence sur monde jouet, élimination, translation).

## DEV (3 graines d'agent ; l'environnement est déterministe, donc les graines changent peu)
| jeu | niveaux (3 graines, sur 3x) | actions niveau 1 (graine 0) | référence humaine niv. 1 | cerveau echo/arc3 niv. 1 |
|---|---|---|---|---|
| ls20 | 3 (1 par graine) | 41 | 22 | 198 |
| tn36 | 3 | 48 | 32 | 34 |
| cn04, ka59, sc25 | 0 | - | - | 0 |
Total : 6/99 (soit 2/33 par graine), identique au cerveau de echo/arc3 (2/33) : **pas d'amélioration du nombre de niveaux sur DEV**, seulement moins d'actions sur ls20 niveau 1.

## Ablations DEV (niveaux sur 99)
| variante | niveaux |
|---|---|
| complète (gain d'information + plan + prior) | 6 |
| exploration aléatoire | 1 |
| jeu réactif (sans plan) | 3 |
| sans prior appris | 6 |
Le gain d'information et la planification aident (petits nombres, bruit possible). Le prior appris ne change rien : voir plus bas.

## Courbe d'apprentissage intra-jeu
Un seul niveau par jeu est résolu : la courbe (actions du niveau k en fonction de k) ne se mesure pas. Le niveau 2 de ls20 n'est pas résolu en 459 actions (l'agent n'infère que « toucher la couleur X »).

## Sommeil
51 étiquettes de règles (jeux DEV, variantes d'exploration poolées). Fréquences apprises : traversable 0.63, bloquant 0.25, ramassable 0.09, mortel 0.02, objectif 0.02. L'énumération a testé des conjonctions de primitives de forme (1 à 2) ; le juge épinglé n'en a gardé **aucune** : le meilleur candidat (une seule composante + rectangulaire, gain 0.17 nat) n'a de support que dans 1 pli de validation croisée (il en faut 2). Le prior brut rendait « objectif » trop improbable et faisait perdre ls20 niveau 1 ; il est tempéré à moitié vers l'uniforme (choix fait sur DEV). Au final, le prior appris ≈ uniforme : l'apprentissage de l'apprentissage n'a rien apporté de mesurable ici.

## Limites et déviations
- Seul le mode « avatar qui se déplace par translation » est modélisé finement ; cn04 (grands changements), ka59 (avatar instable), sc25 et les jeux à clics retombent sur le repli générique, sans gain.
- Les objectifs « tout collecter / tout effacer / motif de référence » ne sont pas implémentés (seul « atteindre X » + un bonus de ramassage).
- Le HUD n'est appris qu'après 6 transitions.
- Honnêteté : en début de session, une commande Bash a inclus un listage du dossier des jeux ; elle a été refusée par le système avant exécution et n'a rien renvoyé. Aucun fichier de jeu n'a été lu. Le SDK charge les jeux lui-même via le courtier réutilisé.
- Budget : 500 actions, graine d'environnement 0 (comme lancer.py). HELD intact.
