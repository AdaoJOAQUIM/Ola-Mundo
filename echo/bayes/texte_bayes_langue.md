
## 5. Lecture : les cinq principes face aux mesures

### Lire = mettre à jour, sans entraînement (a marché surtout hors domaine)

L'expert à mise à jour ajoute au comptage de la bibliothèque (pondéré par β) les comptes du texte déjà lu : c'est la mise à jour conjuguée de Dirichlet-multinomiale, ordre par ordre, puis la même interpolation qu'avant. Aucun paramètre n'est appris par descente de gradient ; lire suffit.

- **En domaine (E)** : le gain est petit. Meilleur expert à mise à jour choisi sur A : 1,6453, contre 1,6489 pour la référence à la main (−0,0036). L'oracle (β = 0,3) fait 1,6420 (−0,007). La bibliothèque contient déjà presque tout ce que E peut apprendre.
- **Hors domaine (français)** : le champion sans mise à jour fait 5,14 bpc et le meilleur expert statique (oracle) 4,33. Les experts à mise à jour tombent à 2,46 (oracle, β = 0,001), et la moyenne bayésienne atteint 2,45 sans savoir quel β choisir. LZMA conditionnel : 2,80 en moyenne par fenêtre. Sur les 24 fenêtres de 2000 caractères, la moyenne bayésienne (fixed-share) bat LZMA dans les 24 (écart moyen environ 0,35 bit par caractère). Réserve : une partie du gain du « statique » à « mise à jour » vient de ce que les caractères propres au français (accents, guillemets) n'existent pas dans la bibliothèque ; l'alphabet étendu leur donne la probabilité uniforme tant qu'ils ne sont pas lus. Cette part n'a pas été mesurée séparément. Les fenêtres 13 à 15 (≈ 0,65 bpc) ressemblent à des paragraphes presque répétés dans l'échantillon (le corpus n'est dédupliqué que sur les lignes identiques) : tous les lecteurs de flux en profitent, LZMA aussi.

### Prédire = moyenner sur tous les modèles (a marché, avec une nuance)

- Le **BMA simple** ne fait que retrouver le meilleur expert : 1,6421 contre 1,6420 pour l'oracle. Son regret est borné par log2(429) = 8,74 bits au total, ce qui est négligeable sur 56 195 caractères, mais il ne peut pas faire mieux que le meilleur membre. Il s'est verrouillé sur un seul expert (β = 0,3, ordre 10) après environ 1000 caractères.
- **Fixed-share** fait mieux que l'oracle d'un seul expert : 1,6336 sur E (−0,0084 contre l'oracle, −0,0153 contre la référence à la main, −0,0236 contre le champion ; P(meilleur que le champion) = 1,000, intervalle crédible à 95 % [−0,0306 ; −0,0183]). α est réglé sur A (3e-4 avec Optuna, 1e-5 sans : au bord de la grille). Le gain vient d'une alternance entre experts voisins, pas d'un modèle nouveau, et il est modeste (0,5 %).
- **C'est sur le décalage de domaine que la moyenne sert vraiment** : le meilleur expert choisi sur A (β = 1) fait 3,49 sur le français, l'oracle 2,46 ; le mélange fait 2,45 sans connaître le domaine. Un choix unique fait sur A ne peut pas prévoir le changement de langue ; le postérieur le corrige en moins de 2000 caractères.

### S'améliorer = optimisation bayésienne (efficace en échantillons, pas en généralisation)

Optuna TPE (300 essais, 62 s de CPU) trouve A = 1,6066 et B = 1,4506. Le meilleur A des runs d'évolution est 1,6077 (run long, 15 000 évaluations) et la référence à la main 1,6084. **Sur E, le gagnant fait 1,6488, soit exactement la référence à la main (1,6489)** : le gain de 0,010 sur B est de la sélection (meilleur B parmi les 10 meilleurs sur A, 11 constantes réglées sur 15 000 caractères), il ne se transfère pas. Le champion de l'évolution fait 1,6572 sur E, moins bien que la référence à la main.

Efficacité en échantillons (voir tableau) : Optuna atteint B ≤ 1,50 en 16 essais, 1,48 en 31, 1,46 en 75 ; le moteur d'évolution a besoin d'une médiane de 256 évaluations pour 1,50 (point de mesure le plus fin : 128), de 504 pour 1,48, de 520 pour 1,46, et seuls 7 runs sur 22 atteignent 1,46. Optuna est donc 7 à 16 fois plus économe, en médiane. **Réserves** : (i) l'espace d'Optuna est la famille paramétrique qui contient la référence à la main, alors que le moteur cherche dans des programmes ; ce n'est pas la même tâche. (ii) Les courbes du moteur sont échantillonnées tous les ~250 évaluations : « évals min = 128 » n'est que le premier point. (iii) Les dix meilleurs essais ont tous l'ordre 10, γ ≈ 0,96 et u1 : l'optimum est plat, les c_k irréguliers (0,7 à 124) sont du bruit d'ajustement.

### Convergence = concentration du postérieur (mesurée)

Entropie du BMA sur E : 8,74 bits au départ (429 experts), 4,6 après 100 caractères, 1,1 après 500, 0,05 après 1000, puis 0. Sur le français : 2,9 experts effectifs après 100 caractères, 1,0 après 2000. Fixed-share garde 1 à 5 experts effectifs (entropie jamais nulle) : il reste capable de changer d'avis, c'est le prix de l'agilité.

### Émergence = rasoir d'Occam bayésien (signal réel, lecture prudente)

- Sur E, la masse va à l'ordre 10 (le plus complexe) et à β = 0,3 : pas de pénalité de complexité visible, parce que la bibliothèque fournit assez de données pour que les longs contextes paient. Les ordres 2 à 8 tombent à moins de 0,001 en 1000 caractères (BMA).
- Sur le français, la masse va à β = 0,001 en moins de 2000 caractères (le flux pèse beaucoup plus que l'a priori anglais), puis le BMA passe de l'ordre 10 à l'ordre 8 entre 12 000 et 20 000 caractères. Fixed-share oscille entre les ordres 6, 8 et 10. C'est une sélection pilotée par la vraisemblance, cohérente avec « moins de paramètres quand les données du domaine sont rares ». Ce n'est pas démontré comme du rasoir d'Occam : la grille n'est pas emboîtée au sens du volume a priori, l'a priori uniforme sur 429 experts donne 65 % de masse à la grille statique et 21 % à chaque ordre, et la préférence finale repose sur quelques dixièmes de bit.

## 6. Ce qui a marché, ce qui n'a pas marché

**A marché** : mise à jour de Dirichlet hors domaine (5,14 → 2,45 bpc) ; moyenne bayésienne qui égale l'oracle sans le connaître ; fixed-share qui le dépasse un peu (−0,008) ; TPE qui retrouve en 16 à 75 essais le niveau de la référence à la main ; postérieur qui se concentre en quelques centaines de caractères.

**N'a pas marché ou ne se généralise pas** : le BMA seul n'apporte rien en domaine ; le gagnant d'Optuna ne bat pas la référence à la main sur E ; Pitman-Yor (1,6993 sur E) reste moins bon que les interpolations à escape proportionnel au nombre de symboles distincts ; le champion évolué est moins bon que la référence à la main sur E (+0,008) ; le choix d'un expert sur A échoue au changement de langue.

## 7. Limites et protocole

- E vient des mêmes livres que la bibliothèque (paragraphes voisins non retirés), donc il est plus facile qu'un texte d'un autre auteur ; il n'est pas séparé des auteurs. Il a été lu une seule fois par `final.py` (témoin), après une répétition générale sur B et A (déjà utilisés) pour déboguer le code sans toucher à E ni au français.
- Le texte français n'est écrit dans aucun fichier : le JSON ne contient que des nombres et des noms d'experts (test automatique sur l'absence de phrases). Il compte 48 295 caractères (échantillon d'un paragraphe sur 150), pas 50 000 ; 24 fenêtres de 2000 caractères (la dernière, partielle, est écartée du tableau par fenêtre).
- Les 429 experts partagent les mêmes comptages : la diversité réelle est faible. Les masses par ordre ou par β dépendent du dénombrement de la grille.
- Les α de fixed-share viennent de A ; Pitman-Yor aussi. Rien n'a été réglé sur E ni sur le français.
- La bibliothèque réduite change A et B de moins de 4e-5 bpc (champion : +0,00003) : les anciens résultats restent comparables.
- Coût : dev.py 90 s de CPU (dont Optuna 62 s), final.py 483 s (dont 27 compressions LZMA ≈ 6 min), tests 2 s.

## 8. Fichiers

`donnees_bayes.py` (bibliothèque réduite, E, tableaux), `experts.py`, `melange.py`, `optimisation.py`, `efficacite.py`, `dev.py` (A et B), `final.py` (E et français, une fois), `rapport.py`, `test_bayes_langue.py`, `resultats_tests_bayes_langue.txt`, `resultats_bayes_langue.json`.
