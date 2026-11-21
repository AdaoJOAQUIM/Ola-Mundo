# Programmation dans l'espace des poids : un réseau écrit, pas entraîné

Inspiré de « Sparks of Weight-Space Programming » (MIT CSAIL, novembre 2026).
Les poids de `ReseauEcrit` (`poids_ecrits.py`) ont été écrits à la main : deux règles
logiques du Sudoku traduites en matrices 0/1 et en ReLU, appliquées en boucle
(récursion façon TRM) jusqu'au point fixe. Aucune donnée, aucun gradient.

| Modèle (1 000 Sudokus de test) | Paramètres | Taille int8 | Entraînement | Résolus | Grilles violant une règle |
|---|---|---|---|---|---|
| Mini-TRM entraîné (A) | 331 128 | 331 Ko | environ 20 min de CPU | 0 | 100 % |
| **Réseau écrit à la main** | **8 829** | **10,9 Ko** | **aucun** | **565** | **0** |

- Les 565 grilles résolues sont exactement celles que résout la propagation symbolique
  classique de `donnees.py` (accord sur 1 000 grilles sur 1 000) : le réseau EST cet
  algorithme, écrit dans l'espace des poids.
- 8,2 étapes de réflexion en moyenne (21 au maximum), 4,6 s de CPU pour 1 000 grilles.
- Jeu d'entraînement (jamais utilisé pour écrire les poids) : 525 sur 1 000, accord 1 000 sur 1 000.
- Aucune contradiction, aucune grille fausse : quand le réseau ne sait pas, il laisse
  les candidats ouverts au lieu d'inventer.

## Étape 2 : recherche complète pilotée par le réseau écrit (`recherche_ecrite.py`)

Quand le réseau bloque, un contrôleur de quelques lignes choisit la case la plus
contrainte et ouvre une branche par chiffre possible, en parallèle dans le même lot ;
le réseau propage, les branches contradictoires meurent. Chaque grille trouvée passe
un contrôle indépendant (règles du Sudoku, indices conservés).

| Jeu | Résolus | Réponses fausses | Contrôle indépendant | Sans recherche | Nœuds moyen / max | CPU |
|---|---|---|---|---|---|---|
| Test | **1 000 / 1 000** | 0 | 1 000 / 1 000 | 565 | 3,4 / 49 | 1,2 s |
| Entraînement | **1 000 / 1 000** | 0 | 1 000 / 1 000 | 525 | 3,5 / 37 | 1,1 s |

## Limites

- La capacité vient de celui qui a écrit les poids : elle est transférée, pas créée de rien.
- 435 grilles demandent de deviner puis revenir en arrière (recherche) : les deux règles
  écrites ne suffisent pas. D'autres règles peuvent s'écrire de la même façon.
- Le Sudoku a des règles connues et exactes ; un comportement flou (style, flagornerie)
  ne se laisse pas écrire ainsi — c'est aussi un échec rapporté par l'étude du MIT.
