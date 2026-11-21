# Chirurgie des poids d'un vrai modèle de langage (SmolLM2-135M), sans gradient

Reproduction réduite de « Sparks of Weight-Space Programming » (MIT CSAIL, 2026).
Défaut ciblé : en décodage glouton, le modèle tourne en boucle. Diagnostic par passes
avant (40 amorces de diagnostic), direction « boucle − progrès » dans les sorties MLP,
retrait de cette direction de colonnes de `down_proj`. Mesures sur 40 amorces jamais
vues au diagnostic, 96 jetons générés, perplexité sur 40 paragraphes tenus à l'écart.

## Étape 1 : réglage fixé d'avance (couches 18 et 20, 64 colonnes chacune)

| | Boucles | Répétition (4-grammes) | Perplexité |
|---|---|---|---|
| Avant | 50 % | 0,559 | 29,65 |
| Après chirurgie (73 728 poids, 0,055 %) | **35 %** | 0,530 | 29,66 |
| Témoin, direction aléatoire de même taille | 47,5 % | 0,550 | 29,65 |

## Étape 2 : 8 réglages comparés sur les amorces de diagnostic seulement

Garde : perplexité de sélection (20 paragraphes distincts) au plus +2 %. Retenu :
4 couches, toutes les colonnes, intensité 1 (3,5 M poids, 2,6 %). Les deux réglages
les plus forts réduisaient davantage les boucles au diagnostic (27,5 %) mais violaient la garde.

| Évaluation unique (amorces jamais vues) | Boucles | Répétition | Perplexité |
|---|---|---|---|
| Avant | 50 % | 0,559 | 29,65 |
| Après chirurgie réglée | **35 %** | **0,501** | 29,99 (+1,1 %) |
| Témoin, direction aléatoire de même taille | 45 % | 0,543 | 29,68 |

## Conclusion honnête

- Effet réel mais modeste : −15 points de boucles contre −2,5 à −5 pour le témoin.
  Sur 40 amorces, l'écart n'est pas statistiquement établi.
- Le réglage n'a pas amélioré le taux de boucles sur les amorces jamais vues (35 % dans
  les deux cas) ; il réduit un peu plus la répétition, au prix de +1,1 % de perplexité.
- Bien plus faible que le résultat du MIT (MATH-500 de 7,9 % à 48,1 %). Hypothèse : les
  boucles gloutonnes d'un petit modèle de base sont un comportement diffus, pas un
  « verrou » localisé — exactement le cas où l'étude rapporte elle aussi des échecs.
