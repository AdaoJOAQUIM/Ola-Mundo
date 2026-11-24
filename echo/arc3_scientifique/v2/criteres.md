# Le Scientifique v2 : critères pré-enregistrés (commités AVANT toute expérience)

Ce fichier ne sera pas modifié après son commit. Toute déviation sera signalée dans resultats_v2.md.

## Protocole
- Jeux : DEV seulement = cn04, ka59, ls20, sc25, tn36 (33 niveaux au total ; 99 niveaux = 3 graines x 33). HELD n'est jamais lancé.
- Graines de l'agent : 0, 1, 2. Graine d'environnement : 0 (l'environnement est déterministe).
- Aucun LLM, aucun réseau au moment du jeu (test de garde-socket), CPU seulement, aucun code spécifique à un jeu (test qui cherche les 25 ids), dossier des jeux jamais lu ni listé.
- Référence : v1 = 6/99 sur DEV à 500 actions (même chose que le cerveau de base echo/arc3).
- Les jeux DEV servent au développement de v2 : les scores ne sont pas indépendants. Aucun réglage sur les mondes de test D4.

## Critères
- **D1** : budget 500 actions par jeu, 3 graines : DEV >= 18/99 niveaux (trois fois v1). Succès / échec rapporté tel quel.
- **D2** : même chose avec 5000 actions par jeu. Rapporter niveaux et actions par niveau contre `niveaux_reference` (références humaines) si l'adaptateur les expose. Si le CPU ne permet pas 3 graines, on le dit et on rapporte la graine 0 seule.
- **D3** : ablations à 500 actions, mêmes graines : sans modèle d'effet d'action (CNN), sans graphe, sans théorie (v1), sans modèle de valeur. Tableau de niveaux DEV.
- **D4** : si `echo/mondes_arc3` expose `MondeEnv(seed, difficulte)` et un évaluateur utilisable : niveaux résolus sur 200 mondes jamais vus (graines non utilisées pour le pré-entraînement ni le réglage) pour v2, v1 et la base aléatoire. Sinon : D4 non évalué, et on le dit.

## Engagements d'honnêteté
- Résultats rapportés tels quels, y compris les échecs. Nombres petits : le bruit est signalé.
- Les « hypothèses de but » (tout collecter, tout effacer, motif de référence, transformation puis porte) sont des signaux de progrès génériques sur les images ; aucune n'est liée à un jeu.
- Le pré-entraînement sur mondes procéduraux, s'il a lieu, utilise des graines distinctes de celles de D4.
