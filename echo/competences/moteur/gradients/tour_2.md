# Gradient semantique, tour 2 (reserve = lod3_10 + base4 : 3 bits pour w < 0,10)

## Ce qui a ete accepte au tour 1
Les trois candidates passaient (gain 5,0 / 7,4 / 6,9 pts, ppl GATE +0 a +0,5 % vs 4 bits simple, accord 97 %). Le juge retient cand_2 (w < 0,10 -> 3 bits) : 92,6 % du naif. Sparsite d'entree/down par blocs : rejetee par les traces TRAIN (voir tour 1).

## Ce qui echoue maintenant
- Il reste 92,6 % des octets : tous les experts de poids >= 0,10 sont lus en 4 bits denses. Reserve de ppl : +0,5 % vs 4 bits simple sur GATE, soit 1,5 point de marge de qualite inutilisee.
- Les emplacements de poids tres faible (< 0,03) sont lus en 3 bits alors que leur contribution est quasi nulle : le culling de la phase 1 montrait +0,15 % de ppl a 0,04.
- Le down n'est pas assez sensible pour justifier du 4 bits partout (trace du tour 1 : le 3 bits sur gate/up de w faible est gratuit).

## Attribution a posteriori
Les octets restants se concentrent sur les poids moyens/forts (w >= 0,10) qui portent l'essentiel de la sortie ; la marge de ppl est donc a depenser sur les poids faibles (ecart, 2 bits) et sur le down des poids moyens.

## Quoi changer
cand_1 : ajouter l'ecartement w < 0,03. cand_2 : remplacer lod3_10 par une echelle (ecart < 0,025, 2 bits < 0,05, 3 bits < 0,14). cand_3 : down en 3 bits pour w < 0,2.
