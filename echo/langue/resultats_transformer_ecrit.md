# Rétro-ingénierie appliquée à la langue : un transformer écrit à la main

`transformer_ecrit.py` écrit les poids d'un transformer à 2 couches, sans gradient :
10 têtes « caractère précédent » (couche 1) et une tête d'induction (couche 2), le
circuit de copie en contexte trouvé par l'interprétabilité dans les vrais modèles.
Son score d'attention privilégie le plus long suffixe identique : c'est la règle d'Écho.
La version rapide (tests d'égalité) est vérifiée identique à la version matricielle
littérale (vecteurs one-hot et matrices de requête/clé écrites).

Même bibliothèque (1 % de la mémoire anglaise, 133 429 caractères), même texte
d'évaluation (3 000 caractères de validation, paragraphes présents dans la mémoire exclus).
Deux réels (beta = 16, epsilon = 0,05) choisis sur une calibration disjointe de 1 000 caractères.

| Modèle (moins = mieux) | bits / caractère |
|---|---|
| Unigramme (fréquence des lettres) | 4,392 |
| Transformer écrit, contexte seul (sans bibliothèque) | 4,354 |
| LZMA avec la bibliothèque | 2,741 |
| Écho-P avec la bibliothèque | 2,632 |
| **Transformer écrit avec la bibliothèque** | **2,316** |
| **Transformer écrit, bibliothèque + texte déjà lu** | **2,285** |

Copie de séquences aléatoires jamais vues (20 × 100 symboles parmi 500, répétées) :
**100 %** de réussite, contre 1,2 % si l'on mélange les valeurs et 0,2 % au hasard.

## Ce que ça montre

- Le circuit écrit à la main fait mieux qu'Écho-P (−0,32 bit/car.) et LZMA (−0,43) avec
  les mêmes données : son attention pèse TOUTES les occurrences de chaque contexte,
  alors qu'Écho-P en échantillonne 64 au plus.
- Sans bibliothèque, il ne gagne presque rien (4,354 contre 4,392) : la capacité vient
  des données mises dans son contexte, le mécanisme seul ne sait rien de l'anglais.

## Limites

- Niveau caractère, 1 % de la bibliothèque : le texte généré n'a pas de sens.
- Ce circuit est la partie « copie en contexte » d'un modèle de langage. La partie
  « connaissances et raisonnement » des vrais modèles est apprise ; personne ne sait
  l'écrire à la main.
- Une seule évaluation de 3 000 caractères : l'écart avec Écho-P est net, mais sans test
  de significativité.
