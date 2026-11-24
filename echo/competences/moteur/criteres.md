# Competences du Moteur Monde Infini (Skill-Pro) : criteres pre-enregistres

Ce fichier est commite AVANT toute experience. Aucune mesure n'a ete faite avant lui : seuls les fichiers de la phase 1 ont ete lus.

## Principe
Le moteur (OLMoE-1B-7B, 4 coeurs, experts streames depuis le disque) apprend ses politiques de chargement sous forme de « competences » : modules Python deterministes verifies par une liste blanche AST. Les poids du modele ne sont pas modifies (le format disque colonne-majeur et la permutation de canaux sont une disposition des memes poids, calculee une fois sur TRAIN, pas un apprentissage). Je suis l'ecrivain : a chaque tour j'analyse les traces TRAIN, j'ecris une note de gradient (gradients/tour_k.md) puis au plus 3 candidates. Un juge epingle (SHA-256 dans empreintes.json) evalue sur GATE ; il ne peut plus etre modifie apres son premier commit.

## Donnees (tranches disjointes de echo/validation.txt, tokenisation OLMoE du fichier entier = 34 712 tokens)
- TRAIN : 3 fenetres de 512 tokens commencant aux tokens 0, 2500, 5000.
- GATE : 4 fenetres de 512 tokens commencant aux tokens 9000, 10500, 12000, 13500.
- TEST : 6 fenetres de 512 tokens commencant aux tokens 20000, 22500, 25000, 27500, 30000, 32000. Touche une seule fois, a la fin, par un seul script.
- Seuls des nombres sont ecrits, jamais du texte.

## Mesures (definitions fixes)
- Octets/token : pour chaque token d'une fenetre, octets d'experts qu'une lecture token par token (sans cache, sans partage entre tokens) demanderait, comptes a la granularite de pages de 4 KiB (union des pages touchees par les extensions lues). Mesure secondaire : secteurs de 512 o. Le noyau (attention, routeur, embeddings) est en RAM et n'est pas compte, comme en phase 1.
- Reference « naif 4 bits » : 452 984 832 o/token (128 experts x 3 538 944 o), identique a la phase 1.
- Perplexite : sur les fenetres de la tranche (chaque fenetre est une sequence independante), mesuree dans le meme simulateur. Reference « 4 bits simple » = le monde 4 bits de la phase 1 (format ligne-majeur) passe dans le meme code ; la contrainte est ppl <= reference x 1,02.
- Accord des tokens gloutons : fraction des positions de GATE ou l'argmax du candidat egale l'argmax de la reserve courante (forcage de l'enseignant).

## Regle du juge (PPO Gate, region de confiance)
Parmi les candidates d'un tour, le juge retient la meilleure (moins d'octets/token) parmi celles qui satisfont les trois conditions, et rien sinon :
1. octets/token (GATE) <= octets de la reserve courante - 2 points de pourcentage du naif ;
2. ppl (GATE) <= 1,02 x ppl du 4 bits simple ;
3. accord glouton avec la reserve courante >= 95 %.
Reserve initiale : la competence de base « 4 bits dense » seule (100 % des octets). Les mecanismes de la phase 1 (LOD, culling) ne sont pas dans la reserve initiale : ils sont des idees que l'ecrivain peut reecrire comme candidates.

## Maintenance
Score bayesien par competence : posterieur normal-normal sur les octets economises (points de % du naif, par fenetre, attribues par emplacement d'expert). Elagage si la moyenne posterieure < 0,2 point ou si P(mu > 0) < 0,8 ; reserve plafonnee a 12 competences.

## Budget
Au plus 6 tours, 3 h d'horloge, CPU seulement, au plus 8 Go de disque nouveau. Un autre agent charge le CPU : octets et perplexite n'en dependent pas ; seuls les temps mesures en dependent.

## Criteres A1 a A4 (sur TEST, reserve finale)
- (A1) octets/token <= 70 % du naif 4 bits, a ppl <= +2 % vs 4 bits simple.
- (A2) pic de RSS <= 4 Go (mesure sur un vrai decodage en streaming avec la reserve finale).
- (A3) rapporter la frequence a laquelle une competence acceptee perd sa marge de GATE sur TEST (marge = les 3 conditions de la regle du juge, rejouees sur TEST le long de la lignee).
- (A4) comparer au meilleur de la phase 1 : 88 % des octets a +1,79 % de ppl (mesure sur d'autres donnees ; la config phase 1 est aussi rejouee sur TEST).

Rapport de temps par token : somme (E/S + calcul) et max(E/S, calcul), pour NVMe 3,5 et 7 Go/s, SATA 0,5 Go/s, USB 1,0 Go/s. Extrapolation K3.1 (base verifiee 52 Go/token a 4,25 bits) : calcul de reduction, etiquete extrapolation.

## Echec accepte
Si A1 n'est pas atteint, le rapport le dit tel quel avec les chiffres. Aucun seuil n'est modifie apres coup. Toute deviation sera listee dans le rapport.
