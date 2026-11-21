## 6. Ce que le moteur a découvert (lecture de la formule du champion)

Le champion (voir `champion.py`, et `lignee.md` pour son histoire) fait, à chaque ordre k :

```
m = n[k] + d[k]
e = d[k] * (une constante proche de 2,2, propre à chaque ordre) + u1[k]
p = si N[k] > 0.25 :  (n[k] + (e + m) * p) / N[k]     sinon p
...
p = p ** 0.649
```

En clair :

1. **Il a donné beaucoup plus de poids aux ordres courts.** Dans le départ (PPM-C), la masse envoyée vers l'ordre inférieur vaut `d` (le nombre de symboles distincts vus). Dans le champion elle vaut environ `3,2·d + u1 + n`. Le départ faisait trop confiance aux longs contextes vus peu de fois. Retirer ce terme fait remonter le bpc de A de 1,628 à 1,749 (`analyse_champion.json`).
2. **Il a ajouté une puissance finale (0,65).** Comme le juge renormalise les scores, élever à une puissance < 1 aplatit la distribution : c'est un réglage de « température ». Retirer cette ligne coûte environ 0,10 bpc sur A (1,628 à 1,729). C'est la plus grosse pièce isolée.
3. **Il a changé le dénominateur** : `N[k]` seul, au lieu de `N[k] + e`. Comme les scores ne sont pas des probabilités exactes, le juge les renormalise ; ce n'est donc pas une erreur, mais ce n'est plus du PPM au sens strict.
4. **Les autres pièces sont neutres.** Le terme `u1` (symboles vus une seule fois, un clin d'oeil à Good-Turing) et les constantes différentes par ordre n'ont aucun effet mesurable : A passe de 1,6276 à 1,6289 sans `u1` et à 1,6272 sans constantes par ordre. Ce sont des restes de la dérive aléatoire.
5. **Il n'a PAS trouvé de décompte à la Kneser-Ney.** Il n'y a ni soustraction d'une remise D sur les comptes, ni comptes de continuation. Le `np.maximum(n[k], 0.0)` est inutile (n est déjà positif). L'idée de la « remise absolue » demande plusieurs mutations coordonnées, que la recherche n'a pas assemblées avec ce budget.

**Point d'honnêteté important.** Une référence écrite à la main, avec seulement deux réglages choisis sur A (`e = 16·d` et aucune puissance), fait mieux que le champion : voir le tableau de la section 1 (et `analyse_champion.json`). Le moteur a donc surtout redécouvert que « le génome de départ envoie trop peu de masse vers l'ordre inférieur et qu'il faut aplatir ». Il ne l'a pas fait de façon plus efficace qu'un réglage manuel à une constante. Les mutations de constantes de pas ~30 % montent lentement de 1 vers 16 ; c'est une limite de ce moteur à budget de quelques milliers d'évaluations.
