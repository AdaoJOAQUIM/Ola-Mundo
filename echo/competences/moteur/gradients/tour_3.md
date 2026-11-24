# Gradient semantique, tour 3 (reserve inchangee : lod3_10 + base4, 92,6 % du naif)

## Ce qui a echoue au tour 2 (aucune acceptation)
- cand_1 (ecarter w < 0,03) : gain 1,2 pt seulement (< 2) : peu d'emplacements sont si faibles.
- cand_2 (echelle ecart < 0,025 / 2 bits < 0,05 / 3 bits < 0,14) : gain 4,8 pts mais ppl +2,79 % vs 4 bits simple (> 2 %), accord 95,7 %. Le 2 bits par troncature et la tranche 0,10-0,14 en 3 bits depensent plus de qualite que prevu.
- cand_3 (down 3 bits pour w < 0,2) : ppl +4,5 %, et octets *plus* hauts (93,7 %) : defaut d'ecriture, la candidate passait avant lod3_10 et remettait gate/up en 4 bits pour w < 0,10. Hindsight : l'ordre de la reserve compte ; une competence qui se superpose doit se restreindre a w >= 0,10. Et le down est sensible : 3 bits sur down coute cher (+4 % de ppl pour peu d'octets).

## Attribution a posteriori
La perte de ppl vient de (i) la tranche 0,10-0,14 en 3 bits et (ii) du 2 bits par troncature. Le gain d'octets vient pour l'essentiel de la precision reduite des poids faibles ; le culling ajoute peu.

## Quoi changer
Rester sur gate/up (down garde 4 bits), elargir moderement : cand_1 ecarte w < 0,02 + 3 bits < 0,12 ; cand_2 ajoute 2 bits (gate/up) pour 0,02-0,035 ; cand_3 2 bits (gate/up) < 0,05, 3 bits < 0,12. Chacune remplace lod3_10.
