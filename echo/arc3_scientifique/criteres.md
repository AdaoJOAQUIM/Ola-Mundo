# Le Scientifique : critères pré-enregistrés

Ce fichier est commité AVANT toute expérience. Il ne sera pas modifié ensuite (toute déviation sera signalée dans resultats_scientifique.md).

## Protocole
- Découpe : celle de `echo/arc3/lancer.py` : DEV = `random.Random(2026).sample(sorted(25 ids), 5)` = cn04, ls20, sc25, tn36, ka59. HELD = les 20 autres.
- Budget par jeu : 500 actions (RESET et game-over compris), graine d'environnement 0, courtier + garde de Prime Agent (comme `lancer.py`).
- Aucun LLM, aucun réseau au moment du jeu, CPU seulement.
- Aucun code spécifique à un jeu : aucun id de jeu dans le code de l'agent (test automatique), aucune coordonnée/couleur/règle codée en dur par jeu. Tout est appris pendant la partie, ou vient de priors appris automatiquement sur DEV.
- Le dossier des jeux n'est jamais lu ni listé ; on n'interagit qu'avec le SDK et les observations du harnais.
- HELD est touché une seule fois, à la fin, agent gelé (priors, bibliothèque de gabarits et hyperparamètres figés avant).
- Référence : cerveau bayésien de echo/arc3 = 7/150 niveaux sur HELD ; aléatoire = 1.67/150.

## Critères
- **M1 : HELD >= 30/150 niveaux** (quatre fois la référence). Succès = atteint, échec sinon.
- **M2 : HELD >= 75/150 niveaux.** Rapporté.
- Horizon : 150/150, rapporté.
- Rapportés aussi : actions par niveau terminé contre la référence (cerveau echo/arc3) et contre les références humaines du SDK (`level_baseline_actions`) ; courbe d'apprentissage intra-jeu (actions utilisées au niveau k en fonction de k).
- Ablations sur DEV (mêmes graines, mêmes budgets) : (a) exploration aléatoire à la place du gain d'information ; (b) jeu réactif (pas de planification en imagination) ; (c) sans prior appris (prior de types de règles uniforme). Rapport des niveaux DEV pour chaque variante.
- Le juge du sommeil (SHA-256 dans empreintes.json) n'est jamais modifié après son premier commit ; un gabarit n'est gardé que s'il améliore la prédiction sur des transitions DEV tenues de côté.

## Engagements d'honnêteté
- Les résultats sont rapportés tels quels, même si M1 n'est pas atteint. Aucun réglage après l'ouverture de HELD.
- Les jeux DEV servent au développement : leurs scores ne sont pas indépendants.

## Amendement du 2026-11-24 (consigne du coordinateur, avant toute ouverture de HELD)
- L'évaluation HELD est REPORTÉE à une évaluation finale unique, après plusieurs tours d'amélioration. Elle n'est pas lancée dans cette session ; `lancer_sci.py held` reste inutilisé.
- M1 (HELD >= 30/150) sera jugé à ce moment-là, selon les critères ci-dessus, sans changement.
- Dans cette session : travail sur DEV uniquement (résultats par jeu, ablations, courbe d'apprentissage, gabarits du sommeil), agent gelé en fin de session, point d'entrée documenté `run.py`, et accès à l'environnement derrière une petite classe adaptatrice.
