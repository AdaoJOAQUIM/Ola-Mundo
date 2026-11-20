# Résultats du solveur ARC (recherche de programmes, stdlib seule)

Budget : 10 s CPU par tâche, 2 processus. Règle ARC-AGI : 2 tentatives par tâche, toutes les entrées de test doivent être exactes.

| Jeu | 2 tentatives | 1 tentative | CPU moyen | CPU max | Timeouts (budget épuisé) |
|---|---|---|---|---|---|
| training | 129/400 (32.2 %) | 119/400 (29.8 %) | 5.96 s | 11.58 s | 273 |
| evaluation | 63/400 (15.8 %) | 61/400 (15.2 %) | 7.78 s | 21.99 s | 321 |

Un « timeout » = budget de 10 s épuisé sans arrêt anticipé : le solveur explore jusqu'au bout quand aucun programme n'est trouvé (ce n'est pas un échec en soi pour les tâches résolues).

Note : le budget de 10 s est vérifié de façon coopérative entre deux appels de primitives (plus une alarme SIGPROF à budget+1,5 s) ; quelques tâches le dépassent (CPU max ci-dessus, jusqu'à 22 s sur evaluation, probablement des appels coûteux dans le piège d'exception de `appliquer`). Ces dépassements sont comptés tels quels.

## Protocole

Développement et réglages uniquement sur training/. evaluation/ n'a été ni ouvert ni inspecté ; elle a été exécutée UNE seule fois avec le code final gelé (après la dernière modification du code, training a été relancé avec le même code). Aucun changement après le run d'évaluation.

## Les 20 primitives / apprenants les plus utiles (tâches résolues, train + eval, 1er programme correct)

| # | Primitive | Tâches |
|---|---|---|
| 1 | `table_de_couleurs` | 50 |
| 2 | `crop_bbox` | 14 |
| 3 | `regle_relative[rel_rayons_hv]` | 11 |
| 4 | `logique_ou[sep]` | 9 |
| 5 | `dedoublonner` | 8 |
| 6 | `echanger_2` | 6 |
| 7 | `fractal` | 5 |
| 8 | `remplir_trous` | 5 |
| 9 | `completer_symetrie[masque=0]` | 5 |
| 10 | `supprimer_fond_lc` | 5 |
| 11 | `superposer[sep]` | 5 |
| 12 | `miroir4` | 4 |
| 13 | `regle_relative[rel_n8]` | 4 |
| 14 | `upscale2` | 4 |
| 15 | `regle_relative[rel_n4]` | 3 |
| 16 | `pavage1x2` | 3 |
| 17 | `logique_xor[sep]` | 3 |
| 18 | `remplir_bbox[m8]` | 3 |
| 19 | `rayons_diag` | 3 |
| 20 | `rot180` | 3 |

## Exemples de programmes trouvés (training)

- `00d62c1b` : `remplir_trous ; couleurs{10>4}`
- `0520fde7` : `logique_et[sep] ; couleurs{10>2}`
- `0b148d64` : `retirer_dominante ; crop_bbox`
- `0d3d703e` : `identite ; couleurs{1>5,2>6,3>4,4>3,5>1,6>2,8>9,9>8}`
- `1190e5a7` : `dedoublonner ; supprimer_fond_lc`
- `1b2d62fb` : `logique_ou[sep] ; couleurs{0>8,10>0}`

## Tâches résolues

training : 007bbfb7, 00d62c1b, 025d127b, 0520fde7, 08ed6ac7, 0b148d64, 0d3d703e, 0dfd9992, 1190e5a7, 1b2d62fb, 1c786137, 1cf80156, 1e0a9b12, 1f85a75f, 2013d3e2, 22168020, 2281f1f4, 23b5c85d, 25ff71a9, 272f95fa, 28bf18c6, 2dc579da, 2dee498d, 3428a4f5, 3618c87e, 3631a71a, 3906de3d, 39a8645d, 3aa6fb7a, 3ac3eb23, 3af2c5a8, 3c9b0459, 4258a5f9, 42a50994, 4347f46a, 445eab21, 44f52bb0, 46442a0e, 48d8fb45, 496994bd, 4c4377d9, 50cb2852, 54d9e175, 5614dbcf, 5bd6f4ac, 60b61512, 6150a2bd, 623ea044, 62c24649, 6430c8c4, 67a3c6ac, 67a423a3, 67e8384a, 68b16354, 6d0aefbc, 6d75e8bb, 6e82a1ae, 6f8cd79b, 6fa7a44f, 72ca375d, 7468f01a, 746b3537, 74dd1130, 7b6016b9, 7fe24cdd, 80af3007, 810b9b61, 88a62173, 8be77c9e, 8d510a79, 8f2ea7aa, 90c28cc7, 9172f3a0, 94f9d214, 9565186b, 99b1bc43, 9dfd6313, 9ecd008a, 9f236235, a416b8f3, a5313dff, a61f2674, a68b268e, a699fb00, a740d043, a79310a0, ae4f1146, b1948b0a, b2862040, b6afb2da, b8825c91, b94a9452, ba26e723, ba97ae07, bdad9b1f, be94b721, c3f564a4, c59eb873, c8f0f002, c909285e, c9e6f938, ce22a75a, ce4f8723, cf98881b, d037b0a7, d511f180, d5d6de2d, d9fac9be, dae9d2b5, dbc1a6ce, dc0a314f, dc1df850, ded97339, e3497940, e8593010, e98196ab, ea32f347, ea786f4a, eb5a1d5d, ed36ccf7, f25fbde4, f25ffba3, f2829549, f76d97a5, f8b3ba0a, f8ff0b80, f9012d9b, fafffa47, ff805c23

evaluation : 0692e18c, 0c786b71, 0c9aba6e, 12eac192, 195ba7dc, 1a2e2828, 1c0d0a4b, 21f83797, 31d5ba1a, 32e9702f, 332efdb3, 34b99a2b, 37d3e8b2, 45bbe264, 47996f11, 48131b3c, 506d28a5, 59341089, 5b6cbef5, 5d2a5c43, 60c09cac, 642d658d, 66e6c45b, 66f2d22f, 67b4a34d, 68b67ca3, 6ea4a07e, 7039b2d7, 73182012, 73ccf9c2, 833dafe3, 84f2aca1, 8e2edd66, 903d1b4a, 929ab4e9, 981571dc, 9a4bb226, 9ddd00f0, a406ac07, a8610ef7, aa18de87, ae58858e, af22c60d, bbb1b8b6, be03b35f, c3202e5a, c48954c1, ca8f78db, cd3c21df, ce039d91, ce8d95cc, d19f7514, d282b262, d56f2372, e133d23d, e1baa8a4, e21a174a, e345f17b, e66aafb8, e872b94a, e9c9d9a1, f4081712, f823c43c

## Limites (honnêteté)

- Le DSL est fixe : toute tâche exigeant une primitive absente (dessin de formes, comptage complexe, analogies entre objets, raisonnement sur des relations entre objets, motifs dépendant de la position relative de plusieurs objets) échoue.
- La recherche est de profondeur <= 3 et limitée par le budget CPU : la plupart des tâches non résolues épuisent les 10 s sans programme valide.
- Risque de sur-ajustement : les règles locales (tables de voisinage) peuvent vérifier l'entraînement tout en étant fausses sur le test ; l'écart entre « programme trouvé » et « tâche résolue » le mesure. Seuls 2 candidats sont soumis.
- Fond = couleur majoritaire (0 pour quelques primitives) : tâches à fond ambigu mal gérées.
- Pas d'apprentissage entre tâches, pas de composition conditionnelle (si/alors), pas de boucles.

## Comparaison

L'évaluation publique d'ARC-AGI-1 est le jeu de référence standard, sur lequel les systèmes de pointe publient leurs résultats. Aucun chiffre externe n'est cité ici ; ce solveur est un système symbolique minimal, sans apprentissage, et ses scores ne sont comparables qu'avec cette réserve.
