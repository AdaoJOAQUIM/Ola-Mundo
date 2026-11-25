# Résultats J3-bis — couche de schémas causaux locaux (non vérifié)

Critères pré-enregistrés dans `criteres.md` (commit ad2e51a), inchangés. Données : `resultats_j3bis.json`, `runs/j3bis_*.json`.

## Verdict
**J3-bis NON atteint.** Seuil : >= 25 % des niveaux des 5 jeux DEV à 500 actions. Mesure : **4 niveaux sur 33 = 12,1 %** (J3 : 3 sur 33 = 9,1 %).
Les parties sont déterministes (J3 l'a montré : les graines 0, 1, 2 donnent les mêmes niveaux) : graine d'agent 0 seule.

| | niveaux (500 actions) | détail |
|---|---|---|
| DEV-train (cn04, ls20, sc25) | 3 / 19 = 15,8 % | cn04 1/6 (niveau 1 en 62 coups, référence humaine 29) ; ls20 2/7 (comme avant) ; sc25 0/6 |
| DEV-val (tn36, ka59), validation affaiblie | 1 / 14 = 7,1 % | tn36 1/7 (niveau 1 en 120 coups contre 368 en J3) ; ka59 0/7 |

DEV-val a déjà été joué en J3 et ses causes d'échec publiées : c'est une validation **affaiblie**. Elle a été jouée une seule fois ici, agent gelé (commit 856924d), sans réglage ensuite. Budget large (5 000 actions, DEV-train seulement) : cn04 1, ls20 2, sc25 0 niveaux.

## Régression sur les mondes
Mêmes 100 graines (`random.Random(11)`), 500 actions : 438 -> **440 niveaux sur 562 (77,9 % -> 78,3 %, +0,36 point)** : seuil tenu. Une première version (passage au schéma après 25 actions sans plan) faisait perdre 14,6 points : le compteur de plans ne mesure pas bien l'ajustement du programme sur les mondes. Corrigé en n'appliquant ce passage rapide que si le programme n'a fait aucun plan et presque aucune exploration, et en ajoutant un plafond de 150 coups par niveau pour le programme (99 % des niveaux résolus des mondes le sont en moins de 130).

## Ce que la couche a appris, par jeu
- cn04 : corps rigide en plusieurs couleurs ; translation de 3 px (4 actions) ; quart de tour (action 5) avec table exacte (orientation, décalage) ; le clic bascule la couleur du corps commandé ; but du niveau 1 = pointes de même couleur appariées avec celles d'un second objet (assemblage), retrouvé sans identifiant ni primitive propre au jeu.
- ls20 : avatar à pas de 5 px ; un contact avec une tuile change une zone lointaine (effet à distance) ; le but « toucher l'objet de même gabarit et palette que la zone changée, après n déclencheurs » résout le niveau 1 quand la couche joue seule (non utilisé en intégré : le programme gagne déjà plus vite).
- sc25 : aucun corps réactif ; rien appris.
- tn36 : pas de corps ; une phase de clics puis l'exploration v1 résolvent le niveau 1 plus vite que le filet.
- ka59 : un avatar est détecté ; aucun but n'aboutit.

## Causes d'échec restantes
- **cn04 niveau 2** : quatre corps sélectionnables par clic ; l'assemblage « toutes les pointes appariées » a une solution unique (vérifiée hors ligne), mais le suivi des corps non commandés échoue quand ils se recouvrent ou changent de couleur de pointe, et l'exécution n'aboutit pas.
- **ls20 niveau 3** : le registre change aussi de palette ; la couche atteint la phase de déclencheurs mais ne gagne pas dans le budget restant.
- **sc25** : jeu à clics et actions de rotation / glissement dont le but n'a pas été compris (aucun corps, aucun but plausible).
- **ka59, tn36 niveaux 2+** : but non trouvé ; budget épuisé en exploration.
- Le filet v2 n'ajoute aucun niveau ; sa part est plafonnée à 35 % du budget et la v1 passe d'abord (120 coups par niveau).

## Écarts et honnêteté
- Copie privée du générateur avec corps rigides / rotation / recoloration : **non faite** (le temps est allé aux vrais jeux de développement) ; la couche n'a donc pas été entraînée ni évaluée seule sur des mondes à ces mécaniques. Seuls un jeu jouet à corps rigide (test) et les mondes existants (via la régression) l'exercent.
- Seule DEV-train a servi à la mise au point (cn04, ls20, sc25) ; deux idées abandonnées faute de temps : palette du registre, jeux à clics sans corps.
- Le « gain d'information » des expériences est approché par un classement a priori / coût des hypothèses, avec réouverture après un effet ; pas de calcul d'entropie explicite.
- Le sondage coûte 5 à 9 coups par corps ; un game over en milieu de plan est géré par un redémarrage du niveau (connaissances du jeu conservées).
- Aucun fichier du dossier des jeux n'a été ouvert ; les 20 jeux HELD n'ont pas été touchés (garde du lanceur inchangé) ; aucun réseau ni modèle de langue (test de garde de socket, test d'identifiants étendu à `retro/schemas`).
- Non vérifié de façon indépendante.
