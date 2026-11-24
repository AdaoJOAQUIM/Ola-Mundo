# Le Rétro-Créateur — architecture pour ARC-AGI-3, sans grand modèle

Objectif : 100 % des niveaux d'ARC-AGI-3, sans modèle de langue, sans réseau pendant le jeu,
sans jamais lire le code d'un jeu, sans code propre à un jeu. Les 20 jeux de test ne sont joués qu'une fois, à la fin.

## L'idée en une phrase

**Ne pas apprendre à jouer : reconstruire le programme du jeu et l'intention de son créateur,
en rêvant d'abord des milliers de mondes, puis jouer parfaitement dans le programme retrouvé.**

## Trois changements de paradigme

1. **Du réflexe au programme.** L'agent ne cherche pas une politique : il reconstruit, à partir de ce qu'il voit,
   le *programme causal* du jeu (objets, variables cachées, règles « déclencheur → effet », condition de victoire).
   Une fois ce programme trouvé, chaque niveau se résout par planification exacte, au nombre de coups minimal.
   La limite n'est plus la « puissance » de l'agent, mais la couverture de son langage et la vitesse de son inférence.

2. **Du jeu au créateur.** Chaque niveau est un message d'un concepteur coopératif qui veut être compris.
   L'agent raisonne sur ce concepteur (inférence pragmatique) :
   le niveau 1 isole la mécanique centrale ; chaque niveau suivant ajoute peu de nouveautés, et ce qui est nouveau compte ;
   un objet unique et saillant a un rôle ; la solution prévue utilise ce qui a été placé ; les formes qui se ressemblent vont ensemble.
   Le générateur de mondes infinis sert de **modèle du créateur** : P(disposition du niveau | règles) dit
   quelles règles rendent ce niveau « logique à construire ».

3. **De l'essai réel au rêve.** Dans ARC-AGI-3, chaque coup réel compte ; le calcul, lui, est gratuit.
   L'agent échange des coups contre du calcul :
   - *intuition amortie* : un réseau entraîné sur des millions de mondes générés, dont on connaît le vrai programme,
     propose en une passe les règles et le but probables ; Bayes vérifie sur le vrai jeu ;
   - *rêve dirigé* : pendant la partie, l'agent génère des milliers de variantes cohérentes avec ses hypothèses,
     s'y entraîne et y teste ses plans, puis revient au vrai jeu avec le moins de coups possible.

## Les sept organes

| Organe | Rôle | Base existante | Travaux réutilisés |
|---|---|---|---|
| 1. Perception | objets, relations (contact, alignement, inclusion, symétrie, formes semblables), compteurs | v1 `arc3_scientifique` | — |
| 2. Langage de programmes | état = objets + attributs + variables cachées ; règles déclencheur → effet ; condition de victoire | nouveau | AutumnSynth (POPL 2023), EMPA (2021) |
| 3. Inférence bayésienne | propositions (intuition + énumération par taille) ; score = prédiction exacte des transitions × a priori (longueur, pragmatique, fréquences apprises) | `bayes/`, `evolution/bayes.py` | Inference compilation (Le et al. 2017), Pu et al. (NeurIPS 2020) |
| 4. Expériences cruciales | coup qui départage le mieux les meilleurs programmes, pondéré par son coût | v1 | EMPA |
| 5. Planification exacte | BFS/A* symbolique dans le programme le plus probable, macro-actions ; ré-inférence à la première surprise | v1 | — |
| 6. Mémoire | mécaniques confirmées, solutions de niveaux → a priori pour la suite ; porte de contrôle = juge verrouillé | `competences/`, `evolution/` | Skill-Pro (ICML 2026), sans grand modèle |
| 7. Filet de sécurité | si aucun programme ne prédit bien : exploration par graphe d'états + modèle effet-action + valeur, jusqu'à avoir assez de données | v2 `arc3_scientifique/v2` | StochasticGoose, Blind Squirrel |

## Trois boucles

- **Boucle courte (pendant la partie)** : percevoir → inférer le programme → expérience cruciale ou plan exact → surprise ? ré-inférer.
  En parallèle, rêve dirigé : variantes cohérentes avec les hypothèses, entraînement et test des plans, zéro coup réel.
- **Boucle moyenne (entre niveaux et entre jeux)** : mémoire des mécaniques ; sommeil à la DreamCoder (PLDI 2021) :
  les surprises inexpliquées deviennent de nouvelles primitives du langage, gardées seulement si le juge verrouillé
  mesure un gain de prédiction sur des transitions mises de côté.
- **Boucle longue (ouverte)** : un générateur-professeur fait évoluer des mondes juste au-delà de ce que l'agent sait résoudre
  (conception non supervisée d'environnements : POET 2019, PAIRED 2020), avec les primitives inventées par le sommeil.
  Le système s'invente ses propres jeux pour progresser, sans fin et sans aide extérieure.

## Jalons fixés d'avance

Les mondes générés servent au développement ; leur vrai programme ne sert qu'à **évaluer**, jamais d'entrée à l'agent.
Trois des 18 mécaniques des mondes infinis sont **tenues secrètes** pendant tout le développement
(tirées au hasard, graine 2026, liste scellée dans `scelle.json`) pour mesurer la vraie généralisation.

- **J1 — mondes de difficulté 1 à 3, 300 mondes jamais vus** : programme exact retrouvé (100 % des transitions de validation prédites)
  dans ≥ 70 % des mondes ; niveaux résolus ≥ 70 % ; coups ≤ 2 × l'optimum de l'oracle.
- **J2 — mondes de difficulté 4 et 5, et mondes avec les mécaniques secrètes** : niveaux résolus ≥ 50 %.
- **J3 — les 5 vrais jeux d'entraînement** : ≥ 50 % des niveaux (budget du protocole : 500 coups par jeu ; budget large 5 000 rapporté à part).
- **J4 — examen final, les 20 jeux de test, une seule fois, agent gelé** : objectif 150/150.
  Le chiffre réel est publié tel quel, vérifié de façon indépendante, avec le détail jeu par jeu.

## Pourquoi cette architecture peut aller jusqu'à 100 %, et ce qui peut l'en empêcher

Si le programme d'un jeu s'écrit dans le langage et que l'inférence le trouve, la planification est exacte :
le niveau est résolu, presque au nombre de coups minimal. Chaque échec a donc une cause identifiable :
une mécanique hors langage (le sommeil et la boucle ouverte l'attaquent), un but ambigu (l'a priori pragmatique l'attaque),
un écart entre mondes rêvés et vrais jeux (J3 le mesure), ou le budget de coups (le rêve dirigé l'attaque).

## Phases d'exécution

0. Finir la v2 : elle devient le filet de sécurité.
1. Langage + inférence + expériences cruciales + planification exacte, sur les mondes infinis → **J1**.
2. Intuition amortie et modèle du créateur, entraînés sur des millions de mondes générés.
3. Rêve dirigé pendant la partie.
4. Sommeil (croissance du langage) et boucle ouverte → **J2**.
5. Vrais jeux d'entraînement → **J3** ; puis examen final, une seule fois → **J4**.

## Règles inchangées

Aucun grand modèle, aucun réseau pendant le jeu (test par garde de socket) ; jamais de lecture du code des jeux ;
aucun identifiant de jeu dans le code (test) ; juges verrouillés par empreinte SHA-256 ; jeux de test touchés une seule fois.
