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

---

# Phase 3 : critères pré-enregistrés (J3, vrais jeux de développement)

Cette section est commitée AVANT toute mesure J3. Les écarts éventuels seront signalés dans resultats_j3.md, pas ici.

## Jeux et découpe
- La découpe vient de `echo/arc3/lancer.py` (DEV = 5 jeux tirés par `random.Random(2026)`, HELD = les 20 autres), importée, jamais recopiée. Un garde du lanceur lève une erreur si un jeu HELD est demandé (test).
- **DEV-train** = cn04, ls20, sc25 : seuls jeux autorisés pour le développement (observations et rejeux uniquement ; jamais le code ni les fichiers des jeux).
- **DEV-val** = tn36, ka59 : jamais joués pendant le développement. Un seul passage final, agent gelé (commit marqué). Seul signal équitable, car DEV-train a servi à développer.
- Les 20 jeux HELD ne sont pas touchés.

## Protocole
- Accès par `AdaptateurARC3(jeu_id, budget)` (v1). Aucun réseau ni modèle de langue à la partie (test de garde de socket), aucun identifiant de jeu dans le code de l'agent (test).
- Budget de protocole : 500 actions par jeu ; graines d'agent 0, 1 et 2 (le jeu lui-même est déterministe : le courtier le crée avec la graine 0). Budget large : 5 000 actions, rapporté à part.
- CPU seulement.

## Mesures
1. **Niveaux résolus** = niveaux terminés / niveaux totaux, par jeu et par graine, puis agrégé (somme sur les graines / somme des niveaux).
2. **Actions par niveau contre la référence humaine** (`niveaux_reference` du SDK) : pour chaque niveau résolu, actions utilisées sur ce niveau / actions de référence ; rapport agrégé et médiane.
3. Programme prédit : taux de prédiction exacte des transitions de l'agent sur ses propres observations (information, pas un seuil).
4. Régression sur les mondes : 100 graines J1 tirées par `random.Random(11)` parmi les 300 graines J1 (budget 500) ; la part de niveaux ne doit pas baisser de plus de 5 points par rapport au commit f9bd261 (mêmes 100 mondes).

## Seuil J3
- **J3 atteint** si les niveaux résolus à 500 actions sont >= 50 % sur l'ensemble des 5 jeux DEV (moyenne sur les 3 graines, agrégée sur les niveaux).
- Lecture équitable : DEV-val seul >= 50 % est la seule preuve de généralisation ; si DEV-val < 50 % alors que le total >= 50 %, le résultat est rapporté comme « atteint mais non généralisé ».
- Rapporté tel quel, quel que soit le résultat. Aucun réglage sur tn36 et ka59.

---

# Phase 4 : critères pré-enregistrés (J3-bis, couche de schémas causaux locaux)

Cette section est commitée AVANT toute mesure J3-bis. Les écarts éventuels seront signalés dans resultats_j3bis.md, pas ici.

## Jeux et protocole
- Mêmes jeux et même découpe que J3 (DEV-train = cn04, ls20, sc25 ; DEV-val = tn36, ka59). HELD jamais touché, garde du lanceur inchangé.
- DEV-val a déjà été mesuré en J3 et ses causes d'échec ont été publiées (resultats_j3.md) : DEV-val est donc une **validation affaiblie** ; DEV-train a servi au développement. Aucun des deux n'est une preuve de généralisation. Le rapport le dit.
- Budget de protocole : 500 actions par jeu. Les parties sont déterministes (le jeu et l'agent le sont ; J3 a montré que les graines 0, 1, 2 donnent les mêmes niveaux) : **la graine d'agent 0 suffit**. Budget large (5 000) rapporté à part, si le temps le permet.
- DEV-val : un seul passage, à la fin, agent gelé (commit marqué), sans réglage ensuite.
- Aucun réseau ni modèle de langue à la partie (test de garde de socket), aucun identifiant de jeu dans le code de l'agent (test), jamais de lecture du dossier des jeux.
- Les mécaniques ajoutées à une copie privée du générateur (corps rigide, rotation, recoloration) ne servent qu'au développement ; `echo/mondes_arc3` n'est pas modifié.

## Mesures
1. **Niveaux résolus** à 500 actions sur l'ensemble des 5 jeux DEV (somme des niveaux terminés / somme des niveaux), puis DEV-train et DEV-val séparément, et par jeu.
2. Actions par niveau résolu contre la référence humaine (information).
3. Schémas appris par jeu (information).
4. **Régression sur les mondes** : mêmes 100 graines que J3 (`random.Random(11)` parmi les 300 graines J1, budget 500), comparées à la mesure du commit 8ff1aea (439 niveaux sur 562, 78,1 %).

## Seuils J3-bis
- **J3-bis atteint** si les niveaux résolus à 500 actions sur les 5 jeux DEV sont >= 25 %.
- Régression : la part de niveaux sur les 100 mondes ne doit pas baisser de plus de 5 points par rapport à 78,1 % (donc >= 73,1 %).
- Rapporté tel quel, DEV-train et DEV-val séparés.
