# Solveur + auto-amélioration (idée SOAR) : résultats

## Méthode (sans LLM, sans paramètre appris par gradient)
SOAR alterne (1) recherche de programmes et (2) apprentissage rétrospectif (hindsight) sur les tentatives de la recherche, pour améliorer la recherche suivante. Version « lite » ici :
1. **Recherche** : le solveur inchangé (`solveur.py`, BFS de compositions de primitives, profondeur 3, acceptation seulement si toutes les sorties d'entraînement sont reproduites). La bibliothèque est injectée par `soar_lite/bib.py` (primitives réordonnées, macros ajoutées).
2. **Hindsight** : (a) programmes réussis du solveur (poids 1) ; (b) relabeling a posteriori : sur chaque paire d'entraînement prise isolément (3 premières paires, y compris pour les tâches non résolues), tout programme de profondeur <= 2 qui produit exactement sa sortie est traité comme une « tâche résolue » (poids 0,5).
3. **Apprentissage** : (a) a priori = ordre des primitives par fréquence d'usage pondérée ; (b) macros = sous-compositions de 2-3 primitives vues dans >= 2 tâches distinctes (score >= 1), ajoutées comme primitives (jusqu'à 30 x ronde ; seules 15 ont atteint le seuil).
4. **Anti-fuite** : apprentissage sur `training` uniquement ; `evaluation` n'est lue qu'une fois, à la fin, avec la bibliothèque figée (`bib_finale.json`).

## Courbe d'auto-amélioration (training, 400 tâches, 10 s CPU, 2 tentatives, 1 processus)
| Ronde | Résolues (pass@2) |
|---|---|
| 0 (solveur de base, `res_training.json`) | 129 |
| 1 | 130 |
| 2 | 131 |
| 3 | 131 (15 macros, plateau) |

Pass@1 training : 121 dès la ronde 1 (base : 119). Une ronde apprend sur les résultats de la précédente ; la ronde 0 est le résultat existant, non rejoué.

## Évaluation unique (evaluation, 400 tâches, 10 s, 2 tentatives, 1 processus, 50 min)
- **pass@2 : 64/400** (base : 63/400) ; **pass@1 : 62/400** (base : 61/400).
- Aucune tâche de base perdue, une gagnée. Sur les 20 tâches du sous-ensemble de fusion : 2 résolues.

## Honnêteté
- Le gain est **marginal** (+2 en training, +1 en evaluation) : de l'ordre du bruit sur 400 tâches. Il n'y a pas d'amélioration substantielle à revendiquer.
- Raisons probables : (i) les programmes réussis sont presque tous de profondeur 1-2 et très hétérogènes, donc peu de sous-compositions récurrentes (15 macros) ; (ii) sans LLM, le « hindsight » se limite à la profondeur 2 sur une paire, donc très pauvre face au relabeling de SOAR ; (iii) les macros élargissent le branchement et consomment du budget, ce qui compense en partie le gain de profondeur ; (iv) les échecs viennent surtout des familles de solveurs hors DSL (règles locales, objets), que ni les macros ni l'ordre ne touchent.
- Les tâches du sous-ensemble de fusion sans programme vérifié sont absentes de `fusion/soar_lite.json` (149 tâches y figurent, programmes macros développés en primitives de base).

## Fichiers
`bib.py` (bibliothèque, hindsight, apprentissage), `boucle.py` (rondes + évaluation), `exporter.py`, `rondes/` (résultats par ronde, cache hindsight), `bib_finale.json`, `eval_finale.json`, `../fusion/soar_lite.json`.
