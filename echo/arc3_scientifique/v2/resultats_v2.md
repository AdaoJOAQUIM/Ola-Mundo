# Le Scientifique v2 : résultats (DEV seulement, HELD jamais lancé)

Critères pré-enregistrés : `criteres.md` (commité avant toute expérience). Les jeux DEV (cn04, ka59, ls20, sc25, tn36) ont servi au développement : leurs scores ne sont pas indépendants. Chiffres détaillés : `resultats_v2.json` (agrégé par `agrege.py` à partir de `runs/campagne/`, ignoré par git).

## Ce qui est construit
- `effet.py` : CNN petit (3 convolutions 16 canaux), entraîné en ligne sur (image, action ou pixel de clic) -> « l'image a changé ? », dédoublonné par (état, action) ; échantillonnage proportionnel à la probabilité prédite. Mélangé à des statistiques de classes d'actions (changé / nouvel état / victoire), transférées d'un niveau au suivant.
- `graphe.py` : graphe d'états (hachage de l'image stable, HUD masqué par le détecteur de v1), frontière étendue au meilleur score (probabilité d'effet + valeur, escompté par la distance), retour par le plus court chemin connu ou RESET + rejeu si c'est moins cher, arcs mortels évités.
- `buts.py` : hypothèses de but génériques en signaux de progrès (tout effacer, tout collecter, motif de référence, transformation/couleurs disparues) + régression logistique de valeur, apprise à chaque niveau gagné, conservée au niveau suivant.
- `agent2.py` : assemble le tout et délègue à la théorie MAP de v1 (avatar par translation) pendant 120 actions par niveau, puis par blocs alternés de 80 ; rejeu de la dernière solution quand aucun avatar n'est connu.
- `bibliotheque.py` : façade pour réutiliser les composants (signatures dans le docstring). Tests : `test_v2.py` (12 passent : pas d'id de jeu, garde-socket, déterminisme, chemins de retour sur graphes jouets dont sens uniques et RESET, modèle d'effet, valeur, motif de référence).

## D1 : 500 actions, 3 graines, DEV
**v2 : 5/99. Critère (>= 18/99) NON atteint.** v1 refait dans le même protocole : 6/99.
| jeu | v2 (3 graines) | v1 |
|---|---|---|
| ls20 | 1, 1, 1 | 1, 1, 1 |
| tn36 | 1, 0, 1 | 1, 1, 1 |
| cn04, ka59, sc25 | 0 | 0 |
v2 n'améliore pas v1 sur DEV : il retrouve ls20 niveau 1 (41 actions, grâce à la théorie) mais tn36 niveau 1 est moins fiable (123 à 153 actions contre 48 pour v1, un échec sur 3).

## D2 : 5000 actions (graine 0 seule : le CPU partagé ne permettait pas 3 graines)
Le courtier plafonne à 500 actions ; j'ai relevé ce plafond en mémoire dans le processus (`adaptateur_large.py`, aucun fichier modifié). Résultat : v2 2/33, v1 2/33 : ls20 niveau 1 (41 actions, référence humaine 22) et tn36 niveau 1 (153 actions pour v2, 48 pour v1 ; référence 32). Dix fois plus d'actions n'ajoutent aucun niveau : le blocage n'est pas le budget.

## D3 : ablations (500 actions, 3 graines, niveaux sur 99)
| variante | niveaux |
|---|---|
| v2 complète | 5 |
| sans modèle d'effet (probabilités uniformes) | 4 |
| sans graphe (échantillonnage local + RESET) | 3 |
| sans théorie (v1) | 3 |
| sans modèle de valeur | 4 |
| v2 + modèle d'effet pré-entraîné sur mondes | 4 |
Écarts de 1 à 2 niveaux sur des totaux de 3 à 6 : à la limite du bruit, aucune conclusion ferme sauf que la théorie v1 porte ls20 et que le graphe porte tn36.

## D4 : 200 mondes jamais vus (graines 100000..100199, difficulté 1/2/3 en rotation, 500 actions)
Le générateur `echo/mondes_arc3` (MondeEnv) était utilisable (un instantané « en cours » : le runner `evaluer.py` n'existait pas, j'ai écrit mon propre adaptateur `adaptateur_monde.py`, lecture seule).
| agent | niveaux sur 1119 | mondes avec >= 1 niveau |
|---|---|---|
| v2 | 181 | 71 |
| v1 | 161 | 60 |
| aléatoire | 48 | 36 |
v2 fait 12 % de niveaux de plus que v1 et près de quatre fois l'aléatoire. Pas d'intervalle de confiance calculé (une graine d'agent par monde). Avec le modèle d'effet pré-entraîné sur des mondes (graines 1000..1044) : 187 niveaux (écart de 6 niveaux avec 181, dans le bruit).

## Ce qui bloque encore le niveau 2 et au-delà
- Sur DEV, cn04, ka59 et sc25 ne rendent aucun niveau : puzzles à plusieurs étapes (clics en séquence, objets poussés) qu'un explorateur sans but explicite ne trouve pas en 500, ni en 5000 actions. Les hypothèses de but (effacer, collecter, motif) ne se sont jamais déclenchées de façon utile : le modèle de valeur n'apprend qu'après une première victoire, et il n'y en a qu'une par jeu.
- ls20 niveau 2 : la théorie de v1 ne sait modéliser que « toucher la couleur X » ; un but du type « changer d'abord de forme puis aller à la porte » exige une planification séquentielle absente ; le graphe sur positions devient trop grand.
- Le graphe d'états explose sur les jeux d'avatar (chaque position est un nœud) ; il faudrait des macro-actions « aller toucher cet objet ».

## Déviations et honnêteté
- D1 non atteint ; v2 est au niveau de v1 sur DEV, légèrement meilleur sur mondes. Je ne le présente pas comme une amélioration sur les jeux réels.
- Réglages sur DEV : deux choix faits en regardant DEV (le terme d'information BALD sur les classes, repris de v1 ; la fenêtre théorie 120/80). Un essai d'échantillonnage plus net (puissance 4) n'a rien apporté et n'est pas utilisé. Les mondes de test D4 n'ont servi à aucun réglage (la mise au point sur mondes a utilisé les graines 2000..2029).
- D2 : une seule graine ; plafond du courtier relevé en mémoire.
- Pré-entraînement : graines 1000..1044, distinctes de D4. Les mondes sont d'un style visuel différent des vrais jeux : le transfert n'est pas démontré.
- Durée écrite dans les journaux perturbée par une coupure de session (usage), sans effet sur les résultats.
- Aucun accès au dossier des jeux ; HELD intact ; ni LLM ni réseau (test de garde-socket).
