# Gradient sémantique, ronde 2

Hindsight sur ronde 1 : les clics isolés ne suffisent pas pour les niveaux 2 et plus (baseline : lp85, vc33, r11l bloquent après 10-23 actions sur le niveau 1). Hypothèses : (a) il faut des séquences de deux clics ordonnés après retour à l'état initial, (b) un clic efficace est un basculement qu'il faut répéter, (c) clic de sélection puis flèches (cn04 : un clic recolore une pièce, une flèche la déplace de ~10 px).

Candidats : paires_clics (RESET puis clic a, clic b sur les répondants), repetition_clic, clic_puis_fleches.

Attribution : sur DEV-train, aucun des trois ne gagne de niveau (cn04 : 292 images distinctes contre 406 pour la baseline, c'est-à-dire moins de diversité visitée). repetition_clic et clic_puis_fleches ne changent pas un seul niveau sur DEV-gate. paires_clics gagne 1 niveau sur 1 paire de 10 mais coûte 414 actions de plus (464 contre 50) : cher car 8 répondants donnent 56 paires à 3 actions.

Changements demandés : réduire à 6 répondants (révision ronde 3, 30 paires), reprendre couloir et tour_objets corrigés dans le même jugement.

Verdict : paires_clics rejeté (borne -0.04), repetition_clic rejeté (aucun effet), clic_puis_fleches rejeté (aucun effet).
