# Gradient sémantique, ronde 3

Révisions : couloir r3, tour_objets r3, paires_clics r3 (jugement de groupe, 3 compétences révisées).
Données DEV-train (`resume_r3.json`) : 0 niveau (la perte du niveau gagné par la baseline sur ls20 graine 0 persiste avec ces trois compétences : 293 actions de compétences contre 204 de cerveau).

Verdict du juge sur DEV-gate : niveaux 5 -> 7 (+2), borne inférieure 0.016 > 0, aucune perte : ACCEPTÉ, mais de justesse (2 paires sur 10, coût des actions dans les niveaux résolus 50 -> 686). Mise en garde : c'est le 3e jugement sur les mêmes 2 jeux de gate (comparaisons multiples), donc l'acceptation est fragile.

Attribution a posteriori (ronde 4, `posterieurs_r4.json`) : le gain DEV-gate est attribué à couloir (7 succès sur 443 activations, moyenne 0.018) et pas à paires_clics (0 succès / 18 activations) ni tour_objets (0 / 140). Le retrait de couloir par le juge perd sc25 niveau 0 : couloir est la seule compétence qui compte réellement sur le gate.
