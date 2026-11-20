# Résultats : objectifs (planificateur) et aiguillage à coût minimal

Généré par `banc_aiguilleur.py` (graine 2024, Python 3.13.18, Linux-6.18.44-fc-v726-x86_64-with-glibc2.39). Python standard uniquement : aucun réseau de neurones, aucun poids.

## Avertissement

La charge et les mondes sont **générés par nous**, dans le français contrôlé que le système comprend ; la vérité terrain vient de code du banc indépendant des modules testés (simulateur du monde, fermeture des règles, entiers Python, recherche à coût uniforme sur l'état complet). Ces chiffres montrent le coût et la correction des modules sur ces charges, **pas** une intelligence générale. **Énergie = ESTIMATION** : temps CPU (`time.process_time`) x 10 W, hypothèse de 10 W par cœur occupé ; aucun wattmètre.

## 1. Charge mixte et stratégies

Charge : **400 requêtes**, composition : calcul 33, generation 6, illisible 9, insertion 144, objectif 50, q_inconnue 13, q_logique 89, q_monde 56. Requêtes répétant exactement une requête déjà vue avec le même état de la base (borne haute des succès du cache) : **89**.

Définitions : (a) cache (question normalisée exacte + version de la base) puis raisonneur, planificateur, génération, en s'arrêtant au premier résultat accepté ; (c) pareil sans cache ; (b) **tous** les modules lancés sur **toutes** les requêtes (génération incluse, 40 caractères d'amorce par requête non générative), sans cache, on garde le meilleur résultat vérifié ; (b') tous les modules dont la forme syntaxique convient (planificateur seulement pour « Objectif », génération seulement pour « Écris/Continue »).

Mesures : médiane de 5 exécutions complètes pour (a), (c), (b') ; (b) : 1 exécution(s) (très coûteuse en CPU). Le temps de chargement de `memoire.xz` (paresseux) est compté dans la première requête de génération.

### Coût total

| Stratégie | CPU total (s) | CPU moyen / requête (µs) | CPU hors génération (ms) | CPU moyen hors génération (µs) | Énergie estimée totale (J) | Énergie estimée hors génération (mJ) |
|---|---|---|---|---|---|---|
| (a) aiguilleur : le moins cher + cache | 9,233 | 23 082,9 | 57,66 | 146,3 | 92,331 | 576,61 |
| (c) le moins cher, sans cache | 9,155 | 22 887,2 | 71,58 | 181,7 | 91,549 | 715,76 |
| (b') « toujours tout » limité aux formes adaptées | 9,375 | 23 436,6 | 81,10 | 205,8 | 93,747 | 811,01 |
| (b) « toujours tout » (tous les modules, toutes les requêtes) | 274,659 | 686 648,5 | 265 580,14 | 674 061,3 | 2 746,594 | 2 655 801,43 |

**Lecture honnête** : la génération (Écho) coûte des centaines de milliers de fois plus qu'une réponse du raisonneur ; le total est donc dominé par elle dès qu'on la lance, et c'est exactement ce que « toujours tout » fait. La colonne « hors génération » isole l'effet du cache et de l'ordre des modules sur le reste de la charge.

### Exactitude (vérité terrain indépendante)

| Stratégie | Justes | Exactitude | Hallucinations (réponse affirmée fausse) | Dépassements de budget |
|---|---|---|---|---|
| (a) aiguilleur : le moins cher + cache | 400/400 | 100,0 % (400/400) | 0 | 0 |
| (c) le moins cher, sans cache | 400/400 | 100,0 % (400/400) | 0 | 0 |
| (b') « toujours tout » limité aux formes adaptées | 400/400 | 100,0 % (400/400) | 0 | 0 |
| (b) « toujours tout » (tous les modules, toutes les requêtes) | 400/400 | 100,0 % (400/400) | 0 | 0 |

« Juste » : insertion acceptée/refusée comme attendu ; réponse exacte avec preuve vérifiée ; abstention quand aucune réponse n'existe ; refus des phrases illisibles ; objectif : plan vérifié, rejoué par le banc avec ses propres règles, de **coût égal à l'optimum de l'oracle**, ou « impossible » quand l'oracle le dit ; génération : texte non vide étiqueté « non vérifié » (on ne juge pas sa qualité).

### Par catégorie (stratégie (a))

| Catégorie | Requêtes | Justes | CPU moyen (µs) |
|---|---|---|---|
| calcul | 33 | 33 | 61,8 |
| generation | 6 | 6 | 1 529 246,9 |
| illisible | 9 | 9 | 43,0 |
| insertion | 144 | 144 | 57,1 |
| objectif | 50 | 50 | 684,3 |
| q_inconnue | 13 | 13 | 33,2 |
| q_logique | 89 | 89 | 77,1 |
| q_monde | 56 | 56 | 98,2 |

Aucune erreur pour (a) sur cette charge.

### Qui a répondu (nombre de requêtes par module)

| Stratégie | 0 cache | 1 raisonneur | 2 planificateur | 3 génération | aucun (abstention/refus) |
|---|---|---|---|---|---|
| (a) aiguilleur : le moins cher + cache | 67 | 281 | 37 | 6 | 9 |
| (c) le moins cher, sans cache | 0 | 335 | 50 | 6 | 9 |
| (b') « toujours tout » limité aux formes adaptées | 0 | 335 | 50 | 6 | 9 |
| (b) « toujours tout » (tous les modules, toutes les requêtes) | 0 | 335 | 50 | 6 | 9 |

Cache de (a) : 67 succès, 333 échecs de recherche (taux de succès 16,8 % (67/400)). Colonne « aucun » : requêtes qu'aucun module n'a su traiter (phrases illisibles refusées). Les abstentions honnêtes (« Je ne sais pas ») sont comptées sous le module 1 ; elles ne sont pas mises en cache, d'où un écart avec la borne haute des répétitions ci-dessus.

**Gain du cache** (hors génération) : 19,4 % de CPU en moins que (c) (57,66 ms contre 71,58 ms).

**Aiguilleur contre « toujours tout »** : 9,233 s contre 274,7 s de CPU, soit un facteur 29,7 ; exactitude 100,0 % (400/400) contre 100,0 % (400/400). L'économie vient du fait que la génération n'est jamais lancée sans que la requête la demande ; elle ne dit rien de la qualité d'un modèle de langage neuronal.

## 2. Planificateur (A*) sur 200 mondes aléatoires

200 mondes : 2 à 4 personnes, 2 à 5 lieux, 1 à 4 objets, historique aléatoire de 2 à 12 événements après un premier déplacement (85 % des personnes), buts aléatoires dont environ 15 % d'impossibles volontaires ; la moitié des mondes ont des multiplicateurs de coût par personne (1, 2 ou 3), 30 % des coûts d'action modifiés (prendre = 2, donner = 3). L'oracle est une recherche à coût uniforme sur l'état **complet** (tous les objets, plafond 250000 nœuds).

| Mesure | Valeur |
|---|---|
| Instances faisables selon l'oracle | 152 |
| Instances impossibles selon l'oracle | 48 |
| Oracle à plafond (non comparable) | 0 |
| Plans trouvés (parmi les faisables) | 100,0 % (152/152) |
| Plans vérifiés par le raisonneur frais + vérificateur | 100,0 % (152/152) |
| Plans valides au rejeu indépendant du banc | 100,0 % (152/152) |
| Optimalité (coût A* = coût oracle) | 100,0 % (152/152) |
| Impossibles détectés correctement | 100,0 % (48/48) |
| « Impossible » déclaré à tort | 0 |
| Plan renvoyé alors que l'oracle dit impossible | 0 |
| Budget de nœuds épuisé | 0 |
| Plans non vérifiés (refusés) | 0 |
| Nœuds développés, moyenne / maximum | 0,97 / 9 |
| CPU total (s) | 0,052 |
| CPU moyen / maximum par instance (ms) | 0,26 / 0,65 |
| Énergie estimée totale (mJ) | 523,17 |

Aucun écart d'optimalité sur les instances comparables.

## 3. Limites (à lire)

- **Banc écrit par nous.** Charges et mondes utilisent les tournures que le parseur connaît ; 100 % ici ne dit rien d'une charge réelle ni de textes naturels.
- **La génération n'est pas « vérifiée ».** Écho produit du texte anglais littéraire ancien (sa bibliothèque), sans sens ; l'aiguilleur se contente de l'étiqueter « texte généré, non vérifié ». La comparaison de coût avec « toujours tout » met en scène un module lent de façon volontairement naïve (il est lancé sur tout) : c'est le point de l'expérience, pas une mesure de ce que ferait un vrai gros modèle.
- **Pas de comparaison avec un modèle neuronal.** Aucun LLM n'a été mesuré ; aucun gain d'énergie n'est revendiqué par rapport à un LLM. L'énergie est une estimation (CPU x 10 W).
- **Le cache n'aide que sur les répétitions exactes** (au caractère près après normalisation bon marché) et se vide de fait à chaque nouvelle phrase pour les questions sur le monde. Pour les réponses très bon marché (dizaines de µs), le gain est petit ; il est net sur les plans et les calculs.
- **Budget CPU coopératif** : vérifié avant chaque module, dans A* et par blocs de 40 caractères de génération ; le raisonneur lui-même n'est pas interruptible (un dépassement est signalé, pas empêché).
- **Une abstention n'est pas escaladée** : « Je ne sais pas » reste « Je ne sais pas » ; aucun module plus puissant n'est appelé pour deviner.
- **Planificateur : modèle du monde minimal.** Une personne a au plus un lieu, pas de portes, pas de capacités de transport, pas de temps. Les autres objets sont ignorés dans A* (ils n'aident jamais) ; l'oracle les garde tous, et les deux coïncident sur les instances comparées, ce qui valide l'hypothèse sur ces instances seulement.
- **Nos règles d'actions sont plus strictes que celles du raisonneur** (il accepterait « X prend O » sans que X soit là où est O). On exige la co-localisation : un plan est donc valide pour le raisonneur et pour nos règles.
- **« Impossible » n'est pas certifié par le vérificateur** : le vérificateur ne prouve que des conclusions positives. L'impossibilité vient soit d'une entité inconnue, soit de l'épuisement de l'espace d'états (donc dépend du modèle d'actions ci-dessus).
- **Plafond de l'oracle** : les instances où l'oracle dépasse son plafond ne sont pas comparées (comptées à part). L'optimalité n'est donc établie que sur des instances petites.
- **Vérificateur partageant l'analyseur** : comme pour le raisonneur, la vérification garantit que la conclusion découle des phrases telles que lues, pas que les phrases ont été bien lues.
