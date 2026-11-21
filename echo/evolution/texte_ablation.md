Lecture, en phrases simples :

- **Pas de différence démontrée.** Avec 5 graines, la différence entre méta-génome automatique et fixe n'est pas significative (p = 0,98 sur A, 0,16 sur B). L'écart entre graines est plus grand que l'écart entre modes.
- Au début (250 évaluations), le mode automatique est un peu meilleur en moyenne sur A. À 2000 évaluations, le mode fixe est un peu meilleur, surtout sur B. On ne peut pas conclure que l'auto-adaptation aide dans ce budget.
- **Ce que le méta-génome a fait quand même :** dans les 5 runs automatiques, le nombre moyen de mutations par enfant est descendu de 2,0 à environ 1,3 – 1,7. Les poids des opérateurs et le pas des constantes ont dérivé (voir `meta_moyen_final` dans `resultats_evolution.json`). Le moteur « préfère » donc des changements plus petits. C'est un signal réel, mais il ne se traduit pas en meilleur score mesuré.
- Les courbes sont des « meilleur jusqu'ici » : elles ne descendent jamais. Les longs paliers montrent que la recherche stagne vite après le premier gain.
