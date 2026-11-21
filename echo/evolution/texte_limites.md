## 8. Limites et précautions

- **Un seul domaine de texte.** Bibliothèque et validation viennent des mêmes sources (romans anglais, textes bibliques, etc.). Rien ne dit que le champion vaut autant sur un autre type de texte.
- **A sert à la fitness, B à la sélection, C a servi une seule fois.** Mais A, B et C sont trois morceaux consécutifs d'un même fichier : les trois peuvent se ressembler plus que des textes réellement différents. Les bpc de A, B et C ne sont pas comparables entre eux (textes différents).
- **Le champion est choisi sur B parmi le top 10 de A.** Le biais de sélection est faible, mais il existe. C seul donne le chiffre honnête.
- **Les comptages sont figés** (aucune adaptation au texte en cours) et l'ordre maximal est 10. Le modèle de départ n'est donc pas Écho-P : il est plus pauvre (pas d'exclusion, ordre 10 au lieu de 16).
- **Le juge normalise les scores.** Le génome n'a pas à produire de vraies probabilités. Une grande partie du gain vient d'ailleurs de cette liberté (la puissance finale, la division par N seul).
- **Écho-P et le sous-ensemble.** Écho-P a été évalué sur un sous-ensemble de C (blocs répartis), pas sur C entier : la comparaison porte sur les mêmes positions pour tous les modèles, mais l'intervalle d'erreur est plus large.
- **LZMA conditionnel** compresse biblio + C d'un bloc. C'est une référence grossière, pas un modèle de même nature.
- **Peu de graines.** L'ablation compare 5 graines par mode ; la différence éventuelle est à lire avec le test de permutation indiqué.
- **Mutations aléatoires, pas de gradient.** Le moteur ne descend pas les constantes de façon fine : voir la comparaison avec la référence à la main.
