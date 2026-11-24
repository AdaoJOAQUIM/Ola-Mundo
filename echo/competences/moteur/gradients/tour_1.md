# Gradient semantique, tour 1 (reserve = base 4 bits dense seule)

Traces TRAIN (1 024 premiers jetons de TRAIN, passes d'exploration, pas le juge). Nombres seulement.

## Ce qui echoue (E/S + qualite de la reserve actuelle)
- La base lit 100 % du naif (3 538 944 o par expert, 128 experts par token). Aucun octet n'est evite.
- Sparsite d'entree par blocs (idee TEAL) : a seuil 0,5 / 1,0 du rms, seuls 1,7 % / 2,1 % des blocs de gate/up sont ignores (energie perdue 0,3-0,4 %), donc 1,1-1,4 pt d'octets seulement ; les blocs de 32 canaux (une page de 4 Kio impose cette granularite) sont trop gros : l'activite des canaux est uniforme apres RMSNorm, la permutation par energie ne cree pas de queue morte. Rejet : gain trop faible.
- Sparsite de down (blocs de 32 canaux intermediaires) : seuil 0,5 -> -4,9 pts d'octets mais +5,8 % de ppl ; seuil 1,0 -> -23,9 pts mais +76,5 % de ppl. Mauvais echange : la queue d'activite de down est portee par peu de blocs mais pas negligeable.
- Precision 3 bits (troncature des plans) pour les experts de poids normalise < 0,08 : -3,2 pts d'octets, ppl -0,07 %, accord glouton 96,8 % avec la base (limite 95 %).
- Echelle 2/3 bits (w<0,05 -> 2 bits, w<0,12 -> 3 bits) : -8,7 pts, ppl +0,64 %, mais accord glouton 91,9 % : echouerait la region de confiance (95 %).

## Attribution a posteriori
Le cout en qualite vient surtout de l'accord glouton, pas de la perplexite : toute perturbation retourne ~5-8 % des argmax (la quantification de base seule en retourne 13 % vs ligne-majeur). La region de confiance a 95 % limite donc chaque pas a des changements doux. La perte de ppl du 3 bits est nulle parce que ces experts pesent peu dans la somme.

## Quoi changer
Pas 1 : n'agir que sur les emplacements de faible poids, avec 3 bits (troncature) ; trois seuils/echelles candidats. Le juge retient la meilleure qui passe les trois conditions. Les pas suivants viseront les experts de rang bas avec ecartement (culling) et le 2 bits, une fois la reserve mise a jour.

## Candidates
cand_1 : w < 0,08 -> 3 bits. cand_2 : w < 0,10 -> 3 bits. cand_3 : w < 0,04 -> gate/up 2 bits, down 3 bits ; 0,04 <= w < 0,09 -> 3 bits.
