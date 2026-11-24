# Critères pré-enregistrés (écrits et commités AVANT toute expérience)

Méthode : Skill-Pro (arXiv 2602.01869) adaptée à ARC-AGI-3, enseignant (Claude, qui lit
les trajectoires et écrit des compétences) / élève (cerveau sans LLM, qui exécute des
compétences Python déterministes à pleine vitesse).

## Règle d'honnêteté
Aucun fichier source de jeu n'est ouvert, lu, grepé, imprimé ni inspecté ; aucun état interne
du jeu. Les compétences n'utilisent que : les images observées, les actions disponibles,
`levels_completed` et `state` de l'API officielle. (AGENTS.md de Prime Agent.)
Les jeux held-out ne sont jamais observés par l'enseignant avant le test final.

## Découpe pré-enregistrée (stratifiée sur le résultat publié de la baseline : un jeu où elle gagne un niveau de chaque côté)
- DEV-train : ls20, cn04, ka59 (l'enseignant analyse ces trajectoires).
- DEV-gate  : tn36, sc25 (jamais analysés image par image ; le juge seul les joue).
- HELD-OUT  : les 20 autres jeux, joués une seule fois à la fin.

## Protocole de jeu (identique à la baseline des 7/150)
500 actions par jeu, graine d'environnement 0, graine du cerveau 0, prieurs gelés
`echo/arc3/prieurs_geles.json`, courtier et garde de Prime Agent via `echo/arc3/harnais.py`.
Les budgets sont des nombres d'actions, jamais du temps mur.

## Juge (épinglé par SHA-256 dans empreintes.json, jamais modifié après son premier commit)
- Évalue un bassin candidat contre le bassin courant sur DEV-gate, graines du cerveau 0..4, 500 actions.
- Différences appariées par (jeu, graine) : n = 10 paires.
- ACCEPTE seulement si (a) ET (b) :
  (a) borne inférieure unilatérale à 90 % du gain en niveaux (moyenne - 1,383 * ecart-type/racine(n)) > 0 ;
      OU, à niveaux totaux égaux (>= 1 niveau résolu), les actions cumulées dans les niveaux résolus baissent d'au moins 10 % ;
  (b) aucun niveau (jeu, indice de niveau) résolu par le bassin courant dans au moins une graine
      n'est perdu (non résolu dans toutes les graines) par le candidat.
- Au plus 3 compétences nouvelles ou révisées par ronde, au plus 6 rondes.

## Maintenance
Postérieur Beta(1,1) par compétence sur « succès après activation » (succès = un niveau gagné pendant
l'exécution ou dans les 25 actions qui suivent son début de terminaison). Après >= 10 activations,
élagage si la moyenne a posteriori < 0,05. Bassin <= 20 compétences. L'élagage passe aussi par le juge
(le retrait ne doit perdre aucun niveau sur DEV-gate).

## Critères du test final (une seule exécution, 20 jeux held-out)
- B1 : niveaux held-out >= 14/150 (le double de la baseline : 7/150).
- B2 : actions par niveau terminé, comparées à la baseline (somme des actions dans les niveaux résolus / niveaux résolus ; et sur les niveaux résolus par les deux).
- B3 : l'exécuteur n'a aucun appel réseau ni appel à un modèle de langage au moment du jeu : analyse statique des modules (aucun import socket/urllib/http/requests/anthropic/openai/subprocess) + test dynamique (socket.socket remplacé par une fonction qui lève, un jeu entier est joué).
- B4 : nombre de compétences acceptées qui s'activent réellement au moins une fois sur des jeux held-out (mesure du transfert) et nombre de niveaux gagnés pendant une compétence.
Tout échec d'un critère est rapporté tel quel.
