# Mini-circuit simulé par une seule formule REDUCE (exécuté dans Google Sheets)

Classeur Google Sheets « Mini-cerveau de mouche – test LAMBDA » (Drive du propriétaire du dépôt).
Données : `sheets_conn.json` (794 connexions, positions 1 à 151, synapses signées). Gain 1,1.

| Feuille | Cellule | Formule |
|---|---|---|
| W | A1 | `=MAKEARRAY(151,151,LAMBDA(r,c,0.275*1.1*SUMIFS(Data!$C$1:$C$794,Data!$A$1:$A$794,r,Data!$B$1:$B$794,c)))` |
| S | A1 | `=MAKEARRAY(20,151,LAMBDA(r,c,IF(INDEX(Data!$E$1:$E$20,r)=c,1,0)))` |
| K | A1 | `=ARRAYFORMULA(--(WRAPROWS(SCAN(Sim!$B$4,SEQUENCE(Sim!$B$3*20),LAMBDA(x,i,MOD(x*48271,2147483647))),20)/2147483647<0.01))` |
| Sim | A7 | `=MAKEARRAY(1,151,LAMBDA(r,c,IF(COUNTIF(Data!$E$1:$E$20,c)>0,0,22)))` |
| Sim | A10 | voir ci-dessous |

```
=LET(nbpas,$B$3,poids,W!$A$1:$EU$151,carte,S!$A$1:$EU$20,refr,$A$7:$EU$7,tirs,K!$A$1:$T$10000,repos,-52,seuil,-45,dmem,EXP(-0.1/20),dsyn,EXP(-0.1/5),coefa,5/(5-20),depart,MAKEARRAY(22,151,LAMBDA(r,c,IF(r=1,-52,IF(r=3,-1000000,0)))),fin,REDUCE(depart,SEQUENCE(nbpas,1,0),LAMBDA(etat,pas,LET(tens,CHOOSEROWS(etat,1),cour,CHOOSEROWS(etat,2),dern,CHOOSEROWS(etat,3),cpt,CHOOSEROWS(etat,4),arrive,CHOOSEROWS(etat,5),libre,ARRAYFORMULA(((pas-dern)>=refr)*1),tensa,ARRAYFORMULA(libre*(repos+(tens-repos-coefa*cour)*dmem+coefa*cour*dsyn)+(1-libre)*tens),coura,ARRAYFORMULA(libre*cour*dsyn+(1-libre)*cour),pic,ARRAYFORMULA((tensa>seuil)*libre),courb,ARRAYFORMULA(coura+libre*MMULT(arrive,poids)),tensb,ARRAYFORMULA(tensa+MMULT(CHOOSEROWS(tirs,pas+1),carte)*68.75),VSTACK(ARRAYFORMULA(pic*repos+(1-pic)*tensb),ARRAYFORMULA((1-pic)*courb),ARRAYFORMULA(pic*pas+(1-pic)*dern),ARRAYFORMULA(cpt+pic),CHOOSEROWS(etat,SEQUENCE(17,1,6)),pic)))),CHOOSEROWS(fin,4))
```

L'état du REDUCE est un tableau 22 × 151 : tension, courant, dernier spike, compteur, puis les 18 derniers
vecteurs de spikes (délai synaptique de 1,8 ms).

## Résultat, exécuté dans Google Sheets (graine 1)

| Durée | Spikes Sheets | Spikes Python | MN9 Sheets / Python | Neurones en écart |
|---|---|---|---|---|
| 10 ms | 31 | 31 | 0 / 0 | 0 sur 151 |
| 100 ms | 438 | 438 | 7 / 7 | 0 sur 151 |
| 1 s | 4 676 | 4 676 | 71 / 71 | 0 sur 151 |

Références Python : `sheets_attendu.json` (même circuit, même générateur, simulation dense).
Non testé : Excel 365 (même famille de fonctions, comportement non vérifié).
