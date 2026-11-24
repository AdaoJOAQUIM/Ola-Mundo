# Gradient sémantique, ronde 1 (DEV-train : ls20, cn04, ka59 ; graines 0..2)

Données : `resume_r0.json` (baseline), `resume_r1.json` (3 candidats). Seul gain de la baseline : ls20 graine 0, niveau 1 en 198 actions.

Attribution a posteriori
- Baseline : 495 actions de cerveau sur ls20, 33 positions d'avatar distinctes en 197 actions (27 passages sur la même case) : le tirage de Thompson revisite beaucoup. Des « pertes de vie » (changement de 138 cellules : barre de pas remplie, avatar ramené au départ) tombent toutes les ~43 actions : le niveau doit être trouvé en peu de pas.
- couloir : 45 activations sur ls20, 3 succès attribués (fenêtre de 25 actions) sur la graine 0, mais le niveau gagné par la baseline en 198 n'est pas retrouvé avec les trois compétences (0/3 graines) : tour_objets prend la main 75 % du temps et casse la couverture du cerveau.
- tour_objets : activé 14 à 19 fois sur ls20 sans succès ; l'avatar oscille gauche/droite (estimation de position fausse de ±5 px : le centroïde des cellules « libérées » est biaisé par le sprite) puis insiste sur la même cible.
- sonde_clics : ne s'active qu'une fois par niveau (par construction), 0 succès sur cn04 et ka59.

Ce qu'il faut changer
1. tour_objets : position = milieu de la boîte des cellules changées + moitié du déplacement ; abandon d'une cible après 8 pas sans progrès (révision ronde 3).
2. couloir : budget plus court (8) et activation seulement après 10 actions dans le niveau (révision ronde 3).
3. Juger les trois ; garder ce que le juge accepte.

Verdict du juge (DEV-gate, 10 paires) : couloir +1 niveau, borne -0.04 : rejeté ; sonde_clics : niveaux égaux, actions dans les niveaux résolus 206 -> 50 : ACCEPTÉ ; tour_objets +2 niveaux, borne -0.08 : rejeté.
