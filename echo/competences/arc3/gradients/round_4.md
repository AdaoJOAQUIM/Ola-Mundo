# Gradient sémantique, ronde 4

Constat sur ls20 (DEV-train) : la couverture déterministe du graphe d'états (couverture) et la planification sur grille avec murs appris (visite_objets) gagnent toutes deux le niveau 1 vers l'action 330-336 en graine 0 (baseline : 198 en graine 0, jamais en graine 1 et 2) ; l'espace d'états (avatar x états du monde) est trop grand pour moins. Pool + couverture : ls20 niveau 1 en 348.

Candidats : couverture, visite_objets (navigation), clic_nouveaux (cliquer ce qui vient d'apparaître).
Verdict : couverture et visite_objets ne s'activent jamais sur DEV-gate (les jeux de gate ont l'action 6 : la couverture exclut les jeux à clic) : niveaux et coût identiques, rejetés car aucun gain mesurable. clic_nouveaux perd sc25 niveau 0 (7 -> 5 niveaux) : il vole la main à couloir ; rejeté.
Limite de la procédure : le gate ne contient que des jeux à clic, il ne peut pas valider des compétences de navigation (non testables, donc non acceptées).

Maintenance : posterieurs_r4.json ; tour_objets (0 succès / 140 activations, moyenne 0.007) élagué après confirmation du juge (aucun niveau perdu) ; couloir (moyenne 0.018) NON élagué car le juge constate la perte de sc25 niveau 0.
