## Protocole suivi

- 300 mondes de difficulté 1 à 3, 100 par difficulté : d=1 graines 9000001 à 9000184, d=2 9000002 à 9000211, d=3 9000000 à 9000213 (les mondes ayant une mécanique scellée — cibles, ennemis, téléporteurs — sont sautés). Disjointes des graines de développement (1 à 2999). Critères commités avant toute expérience (commit 9e17283).
- Budgets : 500 actions (protocole) et 5000 (large). Mesures : programme exact (40 séquences de validation par monde, vrai env comme vérité), niveaux résolus, actions par niveau résolu contre l'optimum de l'oracle, le tout par mécanique.
- Seuils J1 : exact >= 70 %, niveaux >= 70 %, actions <= 2 x optimum (agrégé sur les niveaux résolus). Par mécanique, le rapport dépasse 2 pour les couleurs, la gravité, le motif et les blocs poussables : le seuil n'est tenu qu'en agrégé.
- L'efficacité à 5000 actions (3,45) monte parce que les niveaux gagnés tard coûtent beaucoup de coups ; le seuil J1 ne porte que sur le budget de 500.

## Lecture des chiffres (à lire avant de conclure)

- **Mesure unique.** Les 300 mondes J1 (graines 9 000 000 et suivantes, mécaniques scellées exclues) ont été joués une seule fois par la version figée de l'agent (commit précédant l'essai). Aucun réglage après. Les graines de développement (1 à 2999) donnent des chiffres plus bas sur 180 mondes : 75,9 % de niveaux et 70,6 % de programmes exacts (non indépendants, mesurés pendant la mise au point).
- **Effet de composition.** 143 des 300 mondes J1 ont pour but un panneau (lumières ou motif) : l'agent les résout tous (100 % des niveaux) et retrouve un programme exact dans 98 % des cas. Les 157 mondes « sortie » sont nettement plus durs : 64 % des niveaux et 55 % de programmes exacts. Le score global dépend donc de la part de mondes « panneau » (48 % ici, 41 % en développement). La marge sur le programme exact est mince.
- **Sélection de pièces** (30 mondes, 10 %) : prise en charge partielle ajoutée en fin de développement ; 25 % des niveaux de ces mondes, 0 % de programme exact (la validation ne sait pas distinguer pièce et sortie de même apparence quand une pièce est posée sur sa sortie).

## Causes d'échec, classées (mondes J1, 500 actions)

1. **Mondes à sélection** : 118 des 300 niveaux perdus (39 %). Plusieurs pièces de même forme que les sorties, état caché (pièce sur sa sortie), planification conjointe coûteuse ; l'identification pièce/sortie est fragile quand un danger a la même forme que les pièces.
2. **But « sortie » avec panneau verrouillant la sortie** (lumières ou motif) combiné à couleurs, dangers ou énergie : l'agent doit deviner quel motif ouvre la sortie, puis le résoudre avec peu de coups ; en monde à énergie, une mauvaise hypothèse coûte une vie entière.
3. **Couleurs** (peinture qui change la couleur de l'avatar, sortie de la bonne couleur) : fusion de l'avatar avec une case de même couleur, hypothèse de victoire « couleur égale » apprise tard.
4. **Énergie et dangers** : l'exploration du premier niveau consomme des vies ; la capacité d'énergie est estimée d'après la jauge et peut être fausse de 1 à 3 coups.
5. **Exactitude** : interrupteurs (22 des 43 mondes inexacts hors sélection en contiennent ; cause probable : bascule ou recoloriage, effet vu une seule fois), énergie, dangers et couleurs (types jamais touchés), annulation, gravité (non résolue : chute des blocs mal prédite).

## Écarts par rapport au plan (déclarés)

- **Perception.** Réutilise `composantes` de la v1 pour retrouver le réseau de cases ; ajoute deux a priori de rendu : le plateau est un carré centré dont l'origine est celle du réseau, et les barres fines et longues sont des jauges du bandeau (jamais du plateau). Ces a priori décrivent le rendu de ces mondes, pas des jeux réels.
- **Énergie** : lue sur la jauge (barre fine), estimée par la formule de remplissage ; la mort par épuisement est prédite quand le compteur d'actions atteint cette capacité.
- **Annulation** : une action 7 d'essai par monde (après un coup effectif) pour apprendre qu'elle revient en arrière ; l'agent ne l'utilise pas pour jouer.
- **Filet de sécurité.** La v2 (`arc3_scientifique/v2`) n'est pas utilisée (non terminée, non modifiée). Le repli est la v1 : il ne se déclenche que si le réseau de cases est introuvable ou après 25 exceptions ; il ne s'est jamais déclenché sur ces 300 mondes. Une garde de temps (60 s de CPU par monde à 500 actions, 150 s à 5000) passe aux actions aléatoires sûres : elle s'est déclenchée dans 6 mondes à 500 actions.
- **Budget large (5000 actions)** : l'agent est déterministe et ignore son budget ; les mondes déjà terminés à 500 actions sont repris tels quels, les autres sont rejoués avec 5000 actions.
- **Cerveau et v1** : mêmes 300 mondes, 500 actions seulement.
- **Validation.** 40 séquences aléatoires de 1 à 4 actions par monde, depuis des états visités ; clics aux pixels tirés au hasard sur 64 x 56 ; une seule erreur rend le monde inexact. Elle compare l'état perçu (types, position, couleur de l'avatar, drapeaux victoire/mort) ; elle ne voit pas le contenu caché sous l'avatar.
- **Phases 2 à 5** de l'architecture (intuition amortie, rêve dirigé, sommeil, vrais jeux) : non touchées.
