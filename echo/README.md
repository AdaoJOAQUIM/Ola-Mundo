# Écho — un modèle de langage sans paramètres

Pas de transformer, pas de réseau de neurones, pas de paramètres, pas d'entraînement,
pas de PyTorch, pas même numpy : **Python standard uniquement**.

## L'idée

Un modèle de langage classique *compresse* ses données d'entraînement dans des millions
de poids. Écho fait l'inverse : il garde **les textes eux-mêmes**, compressés, et les relit
au moment d'écrire.

1. Le « modèle » est une bibliothèque de 15,7 millions de caractères compressée avec LZMA :
   `memoire.xz`, **4,05 Mo**.
2. Pour écrire le mot suivant, Écho cherche dans la bibliothèque le plus long passage
   (jusqu'à 32 caractères) qui finit comme le texte en cours.
3. Il ne s'appuie sur ce passage que s'il est suivi d'**au moins 2 mots différents**
   dans la bibliothèque (réglable avec `--diversite`). Sinon, il raccourcit le contexte.
   Cette règle l'oblige à choisir et à recombiner au lieu de recopier un livre.
4. Il prend un de ces mots au hasard et recommence.

Chaque mot est un vrai mot. Leur combinaison est nouvelle.

## Mesures (amorce « It was a dark and stormy night », 4 essais de 400 caractères)

| | Valeur |
|---|---|
| Taille du modèle | 4 048 144 octets (4,05 Mo) |
| Paramètres appris | 0 |
| Temps de « construction » | 13 s (compression seulement) |
| Chargement | 0,2 s |
| Vitesse | environ 80 caractères par seconde sur un cœur |
| Suites de 5 mots inédites | 79 à 84 % |
| Plus long passage recopié tel quel | 35 à 42 caractères |

## Exemple

> It was a dark and stormy night stand up And show of love as I myself do, as thou hast
> part, And each sleeping moonlit villages, that dwell safely, all those stars of the sky,
> and over the thirty; and the young ladies would have been different.

## Limites (honnêtement)

- La grammaire tient localement, sur quelques mots, mais **le sens dérive** d'une phrase
  à l'autre. Écho n'a aucune représentation du sens et ne peut ni répondre à une question
  ni raisonner.
- Il ne connaît que le style de sa bibliothèque : de l'anglais littéraire ancien (Austen,
  Melville, Shakespeare, la Bible, des discours présidentiels américains).
- Ce n'est pas une invention sortie de nulle part : c'est une variante des modèles
  « ∞-gram » et des modèles par compression, recherchés depuis longtemps. Sa
  particularité est de tout faire avec `str.find` sur un texte brut, sans index, sans
  comptage préalable et sans aucune dépendance.

## Utilisation

```bash
python3 construire.py                       # crée memoire.xz et validation.txt
python3 echo.py "It was a dark and stormy night" -n 400 --mesures
python3 echo.py -n 300 --diversite 3        # plus inventif, moins cohérent
```

Changer de langue revient à changer la bibliothèque : `python3 construire.py mon_corpus.txt`.
