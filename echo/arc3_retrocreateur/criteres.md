# Le Rétro-Créateur, phase 1 : critères pré-enregistrés (J1)

Ce fichier est commité AVANT toute expérience. Les écarts éventuels seront signalés dans resultats_j1.md, pas ici.

## Mondes
- Mondes procéduraux de `echo/mondes_arc3` (MondeEnv). L'agent (paquet `retro/`) ne les importe jamais (test) ; il ne voit que l'interface d'environnement : observations `obs` du courtier, actions `{"name","data"}`.
- Mécaniques scellées exclues de TOUS les mondes utilisés (développement compris) : `cibles`, `ennemis`, `teleporteurs` (scelle.json). Un monde est écarté si sa liste de mécaniques en contient une (donc aussi les mondes de but « cibles » et ceux qui ont `attente`, qui vient avec `ennemis`).
- Le vrai programme du générateur ne sert qu'à l'évaluateur (dossier `evaluation/`).

## Graines
- Développement (seules graines autorisées pendant la mise au point) : 1 à 2999.
- J1 : 300 mondes jamais vus, 100 par difficulté 1, 2, 3. Pour chaque difficulté d, les 100 premières graines >= 9 000 000 dont le monde (graine, d) n'a aucune mécanique scellée (`evaluation/choisir_seeds.py`, déterministe) :
  - d=1 : 9000001 à 9000184 ; d=2 : 9000002 à 9000211 ; d=3 : 9000000 à 9000213 (plages parcourues, avec les mondes scellés sautés).
- Disjointes de toute graine de développement (1 à 2999).

## Budgets
- Protocole : 500 actions par monde (RESET et game-over compris), courtier + garde de `mondes_arc3.env.jouer`.
- Budget large : 5000 actions, rapporté à part.
- CPU seulement, aucun réseau (test de garde de socket), aucun identifiant de jeu dans le code (test).

## Mesures (sur les 300 mondes)
1. **Taux de programme exact** : le programme MAP final de l'agent prédit 100 % des transitions de validation d'un monde. Validation générée par l'évaluateur : à partir d'états visités par l'agent (rejeu de ses actions sur le vrai env), séquences aléatoires de 1 à 4 actions, vérité = vrai env ; 40 séquences par monde ; la prédiction de l'agent porte sur l'état perçu (grille de types canonique, position et type de l'avatar, drapeaux niveau gagné / mort). Une seule erreur = monde non exact. Les mondes où l'agent n'a joué aucun coup utile comptent comme non exacts.
2. **Niveaux résolus** : somme des niveaux terminés / somme des niveaux des 300 mondes (budgets 500 puis 5000).
3. **Actions par niveau résolu contre l'optimum de l'oracle** : somme des actions sur les niveaux résolus / somme des optimums sur ces niveaux (aussi la médiane par niveau).
4. Mêmes mesures par mécanique : un monde compte pour chaque mécanique qu'il contient.
5. Cerveau bayésien (`echo/arc3/cerveau.py`) et Scientifique v1 sur les mêmes 300 mondes (budget 500) : niveaux résolus et efficacité.

## Seuils J1
J1 atteint si : taux de programme exact >= 70 %, niveaux résolus (500 actions) >= 70 %, et actions par niveau résolu <= 2 x l'optimum.

## Engagements
- Résultats rapportés tels quels. Aucun réglage sur les 300 mondes J1 : un seul passage final (les résultats de développement ne sont pas indépendants).
- Le dossier des vrais jeux (`echo/arc/externes/arc3-jeux`) n'est jamais ouvert. `echo/arc3_scientifique/v2` n'est pas modifié.
