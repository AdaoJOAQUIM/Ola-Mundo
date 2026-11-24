# Vérification indépendante de J1 — Le Rétro-Créateur (commit f9bd261)

Instantané : worktree à f9bd261, arbre propre. Aucun fichier de l'agent ni du générateur modifié. Aucun accès aux vrais jeux.

## Verdict
Les trois chiffres annoncés sont reproduits (75,7 % exact, 82,0 % niveaux, 1,637x l'optimum) et les contrôles d'isolation et de protocole passent. Réserves : la métrique « programme exact » est plus faible que son libellé, et les a priori de rendu risquent de casser sur de vrais jeux.

## 1. Tests : 10/10 passent
Dont `test_isolation_pas_d_import_des_mondes`, `test_isolation_a_l_execution`, `test_aucun_identifiant_de_jeu_dans_le_code`, `test_garde_de_socket`.

## 2. Isolation : OK
- Aucun import de `mondes_arc3` (ni generateur, moteur, rendu, env) dans `retro/` (AST, test statique et test d'exécution en `-I`).
- Chemin de l'évaluateur : `evaluation/j1.py::un_monde` appelle `env.jouer(seed, d, f)` ; `jouer` crée `CourtierMonde` et `pol = fabrique()`, puis `action = pol.choisir(obs)` avec `obs = garde.observe()` / `garde.act([action])`. Le courtier ne renvoie que `{game_id, state, levels_completed, available_actions, frame, actions_taken, actions_remaining, terminal}` (`env.py::_observation`). Le vrai programme (`Monde`, `lv.sol`, `lv.opt`) reste dans `courtier.handle.env`, lu seulement après la partie (`mo.optimum()`).
- L'agent ne lit que `obs["frame"][-1]`, `available_actions`, `levels_completed`, `actions_taken` (`agent.py` l.245-248), jamais `game_id` (qui contient la graine).
- Test de socket : `socket.socket` remplacé par une classe qui lève ; 40 actions jouées sans appel. Test d'identifiants : 25 identifiants de jeux absents du code.
- Réserve mineure : le 2e `assert` du test d'import se termine par `or True` (inopérant), mais l'assertion suivante couvre le même cas.

## 3. Protocole : OK
- Graines J1 : 9 000 000 à 9 000 213, 300 mondes distincts. Développement : `choisir(1, d, n)`, graines observées ≤ 154 dans tous les `runs/dev*.json` (limite autorisée 2999). Disjoints. Aucun script de développement ne cite de graine ≥ 9 000 000.
- Les 300 mondes régénérés avec `Monde(seed, d)` : 0 mécanique scellée (cibles, ennemis, téléporteurs), 0 but « cibles » (buts : 157 sortie, 143 panneau). Les mécaniques régénérées sont identiques à celles des runs.
- Chronologie : `retro/` est identique entre d021ac7 (21:48, « avant l'essai J1 ») et f9bd261. Le premier run Rétro-Créateur sur J1 date de 21:52. Réserve : les baselines Cerveau et v1 ont été jouées sur les graines J1 dès 21:16-21:18, avant les derniers commits de l'agent ; ce sont d'autres agents et aucune fuite n'est démontrée.

## 4. Rejeu de 60 mondes (random.Random(7), 500 actions, runner du snapshot avec validation)
- 60 mondes : 0 différence sur niveaux (297/347), drapeau exact (45/60), actions totales et actions par niveau.
- L'échantillon contient 1 monde où la garde de temps CPU s'est déclenchée : identique aussi.
- Contrôle supplémentaire, les 6 mondes à garde CPU de la mesure d'origine : 5 identiques ; 1 (9000104, d3) a le même total (500), les mêmes 3 niveaux, le même verdict inexact, mais une répartition des actions par niveau différente (compteur de dépassement 291 contre 298). La garde dépend du temps CPU : déterminisme non garanti pour ces mondes (2 % des mondes), sans effet sur les agrégats observés.
- Note : les résultats par monde (`runs/`) sont ignorés par git ; la comparaison s'appuie sur `runs/j1_retro_500.json` du dossier principal, dont les agrégats recalculés (1367/1667, 227/300, 18949/11577) coïncident avec resultats_j1.md.

## 5. Métrique
- Optimum : `generateur.oracle` (A* à heuristique admissible), jamais exposé à l'agent. Aucun niveau résolu en moins d'actions que l'optimum. Ratio recalculé 1,637 sur les niveaux résolus (RESET compris dans les actions).
- « Programme exact » : implémenté comme 40 séquences aléatoires de 1 à 4 actions depuis des états visités, une erreur = inexact, vérité = images du vrai env. Mais : (a) la comparaison porte sur l'état perçu par la propre perception de l'agent (types, position, couleur de l'avatar, drapeaux), pas sur l'état interne du vrai env ; (b) le contenu caché sous l'avatar et l'énergie restante ne sont pas comparés ; (c) les états de départ sont ceux que l'agent a visités ; (d) la fonction de comparaison est celle de l'agent. Le seuil de 70 % est donc passé avec une marge mince (75,7 %) sur une mesure plus indulgente que « 100 % des transitions ».

## 6. A priori de rendu (évaluation seule, rien modifié)
- **Plateau carré centré** (`reseau.py` fin de `construire_reseau`) : n = (64 − 2·px)//c, étendue forcée à 0..n−1 dès que l'étendue observée y tient. Cela encode la mise en page du générateur (plateau n×n centré, 56 lignes). Dépend en outre de l'hypothèse plus large d'un réseau régulier de cases, déjà propre au générateur. Risque de casse sur de vrais jeux : élevé.
- **Jauges fines ignorées** (`scene.py::_lignes_hud`, `compteur.py`) : bandes unies d'au moins 20 px de large et d'au plus 3 lignes ; ce critère est générique (pas de ligne codée en dur) et les jeux réels ont souvent un indicateur de pas, donc il peut aider. Risque : murs ou lignes fines longues effacés du plateau, et compteur d'énergie supposé être une bande perdant des pixels. Risque moyen.
- Aucune ablation n'a été faite : l'effet chiffré sur J1 est inconnu.
