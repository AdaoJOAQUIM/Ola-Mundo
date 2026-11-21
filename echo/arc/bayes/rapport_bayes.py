"""Écrit resultats_bayes_arc.md à partir de resultats_bayes_arc.json (aucun chiffre saisi à la main dans les tableaux)."""
import json, os
ICI = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(ICI, "resultats_bayes_arc.json")))
P1, P2, T = r["plis"]["pli1"], r["plis"]["pli2"], r["total40"]
f = lambda x: f"{x:.3f}"
def lig(nom, cle, var="complet"):
    return f"| {nom} | {P1[var]['agregats'][cle]} / 20 | {P2[var]['agregats'][cle]} / 20 | {T[var]['agregats'][cle]} / 40 |"
def cal(var):
    out = []
    for nom, k in (("toutes les candidates", "calibration_toutes_candidates"), ("meilleur candidat (top-1)", "calibration_top1")):
        a, b, c = P1[var][k], P2[var][k], T[var][k]
        out.append(f"| {nom} | {a['n']} | {f(a['brier'])} / {f(a['ece'])} | {b['n']} | {f(b['brier'])} / {f(b['ece'])} | {c['n']} | {f(c['brier'])} / {f(c['ece'])} |")
        out.append(f"| (référence : a priori seul) | | {f(a['brier_prior_seul'])} / – | | {f(b['brier_prior_seul'])} / – | | {f(c['brier_prior_seul'])} / – |")
    return "\n".join(out)
def voc(p, nom):
    v = r["voc"][p]; m = v["modele"]
    lignes = [f"**{nom}** (λ choisi sur l'autre pli : {v['lambda_choisi']:.2e} ; coût TRM moyen mesuré sur le pli de calibration : {v['cout_trm_moyen_s_calibration']:.0f} s CPU ; "
              f"gain espéré par groupe de candidats DSL 0 / 1 / 2 : {f(m['gain']['0'])} / {f(m['gain']['1'])} / {f(m['gain']['2'])} ; p_après = {f(m['p_apres'])})\n",
              "| Politique | pass@2 | CPU total (s) | Tâches où TRM est lancé |", "|---|---|---|---|",
              f"| DSL seul (jamais TRM) | {v['jamais_TRM_DSL_seul']['pass2']} / 20 | {v['jamais_TRM_DSL_seul']['cpu_s']} | 0 |",
              f"| **VoC, λ choisi** | **{v['politique']['pass2']} / 20** | **{v['politique']['cpu_s']}** | {v['politique']['tâches_avec_TRM']} |",
              f"| Toujours TRM + fusion bayésienne | {v['toujours_TRM']['pass2']} / 20 | {v['toujours_TRM']['cpu_s']} | 20 |"]
    lignes.append("\nCourbe (tous les λ de la grille, politiques distinctes) :\n")
    lignes += ["| λ | pass@2 | CPU (s) | TRM lancé |", "|---|---|---|---|"]
    lignes += [f"| {c['lambda']:.2e} | {c['pass2']} | {c['cpu_s']} | {c['tâches_avec_TRM']} |" for c in v["courbe"]]
    return "\n".join(lignes)
pr1, pr2 = P1["complet"]["params"], P2["complet"]["params"]
tab = lambda pr, k: ", ".join(f"{x:+.2f}" for x in pr["log_lr"][k])
md = f"""# Fusion bayésienne TRM + DSL sur ARC, et valeur du calcul

## Ce qui a été fait
Remplacer la règle de fusion fixée à l'avance (« programme DSL cohérent d'abord », 43 % de justesse) par de la décision bayésienne,
avec des méthodes standard uniquement : calibration bêta–binomiale (a priori Beta(1,1) ; Dirichlet(1,…,1) pour les variables à
plusieurs modalités), combinaison naïve de Bayes, choix à utilité espérée, puis un routeur « valeur du calcul » (Russell & Wefald 1991).

**Nouvelles données de calibration.** 20 nouvelles tâches d'évaluation, tirées avec la graine 2026 parmi les tâches à une seule entrée de test,
disjointes des 20 anciennes (`sous_ensemble_bayes.json`), passées dans le TRM officiel exactement comme avant (K = 8, même inférence et même
vote ; `build_subset.py` et `infer.py` étendus par des paramètres optionnels, comportement par défaut inchangé ; 2 threads au plus). Les sorties
attendues n'ont été lues qu'après la fin de l'inférence. Résultats bruts : `res_K8_bayes.json`, `eval_K8_bayes.log`. TRM seul sur ces 20 : pass@1
{P2['complet']['agregats']['trm_top1']}, pass@2 {P2['complet']['agregats']['trm']} / 20 ; ~{r['voc']['pli1']['cout_trm_moyen_s_calibration']:.0f} s CPU par tâche.

## Modèle
Candidats d'une tâche : top-5 de TRM (votes sur 8, q) ∪ tentatives DSL cohérentes avec les exemples. P(juste | évidence) par naïf de Bayes sur :
(1) classe de votes TRM {{absent, 1, 2, 3-4, 5+}} ; (2) rang TRM {{absent, 1, 2, 3+}} ; (3) produit par le DSL (oui/non) ; (4) accord TRM–DSL (le candidat est dans les deux).
- Vraisemblances TRM (1, 2, 4) et probabilité a priori : estimées sur l'**autre pli** (comptes + 1)/(N + K).
- Vraisemblances DSL (3) : estimées sur **toutes les tâches d'évaluation hors pli de test** (380) : P(DSL | juste) = (tâches dont la solution est produite par le DSL + 1)/(380 + 2) ;
  P(DSL | faux) = (candidats DSL faux + 1)/(taille estimée du lot de faux candidats + 2), où ce lot = faux DSL + 380 × (faux TRM par tâche, mesuré sur l'autre pli).
- Les paramètres sont écrits dans `params_pli1_complet.json` / `params_pli2_complet.json` (empreinte SHA-256) **avant** la notation du pli de test, relus puis vérifiés ; `Verite` interdit en code la lecture des solutions du pli de test pendant l'ajustement (testé).
- Choix : les 2 candidats de plus forte probabilité a posteriori.
- Variante « sobre » (secondaire, non pré-déclarée comme principale) : seulement classe de votes + DSL, pour mesurer l'effet de la dépendance entre variables que le naïf de Bayes ignore.

Paramètres appris, pli 1 (test = anciennes 20) : a priori {f(pr1['prior'])} ; log-RV votes (absent,1,2,3-4,5+) = {tab(pr1,'bucket')} ; rang = {tab(pr1,'rang')} ; accord = {tab(pr1,'accord')} ; DSL (non, oui) = {tab(pr1,'dsl')}.
Pli 2 (test = nouvelles 20) : a priori {f(pr2['prior'])} ; votes = {tab(pr2,'bucket')} ; rang = {tab(pr2,'rang')} ; accord = {tab(pr2,'accord')} ; DSL = {tab(pr2,'dsl')}.

## Résultats de fusion (pass@2 ; pli 1 = anciennes 20, pli 2 = nouvelles 20)
| Méthode | Pli 1 | Pli 2 | Total 40 |
|---|---|---|---|
{lig('TRM seul (2 premières tentatives)', 'trm')}
{lig('DSL seul', 'dsl')}
{lig('Ancienne règle (DSL cohérent d\'abord)', 'vieille_regle')}
{lig('**Fusion bayésienne (4 variables, principale)**', 'bayes')}
{lig('Fusion bayésienne « sobre » (votes + DSL)', 'bayes', 'sobre')}
{lig('Plafond (une bonne réponse dans TRM top-5 ∪ DSL)', 'plafond')}

Contrôle : la règle ancienne recalculée (sans CompressARC) donne {r['controle_vieille_regle']['recalcule_ancien_20']} / 20 sur le pli 1, comme le chiffre publié ({r['controle_vieille_regle']['publie_ancien_20']}).
Pass@1 : TRM {T['complet']['agregats']['trm_top1']} / 40, fusion bayésienne {T['complet']['agregats']['bayes_top1']} / 40.

## Calibration des posteriors sur les plis de test (Brier / ECE, `ece()` de `soi.py`, 10 seaux)
Variante principale :

| Population | n (pli 1) | Brier / ECE | n (pli 2) | Brier / ECE | n (40) | Brier / ECE |
|---|---|---|---|---|---|---|
{cal('complet')}

Variante sobre : Brier / ECE sur toutes les candidates, 40 tâches = {f(T['sobre']['calibration_toutes_candidates']['brier'])} / {f(T['sobre']['calibration_toutes_candidates']['ece'])} ; top-1 = {f(T['sobre']['calibration_top1']['brier'])} / {f(T['sobre']['calibration_top1']['ece'])}.

## Valeur du calcul (DSL d'abord, TRM seulement si gain espéré > λ × coût)
Gain espéré de TRM pour une tâche = max(0, p_après − p_maintenant(g)), où g = nombre de candidats DSL distincts (0, 1 ou 2) ;
p_maintenant(g) = taux de réussite du DSL (Beta(1,1)) sur les tâches d'évaluation hors pli de test de même groupe (0 si g = 0) ; p_après = taux pass@2 de TRM (Beta(1,1)) sur le pli de calibration.
Règle de choix de λ fixée avant de voir le pli de test : le plus grand λ dont le pass@2 sur le pli de calibration reste à moins d'une tâche de « toujours TRM ».
Coûts : DSL = CPU mesuré (≈ 7,5 s en moyenne, ≤ 12 s) ; TRM = CPU mesuré par tâche. Pli de test 1 : calibration sur les nouvelles 20 ; pli de test 2 : sur les anciennes 20.
Dans la branche « TRM lancé », la réponse est celle de la fusion bayésienne figée.

{voc('pli1', 'Pli de test 1 (anciennes 20)')}

{voc('pli2', 'Pli de test 2 (nouvelles 20)')}

## Lecture honnête
COMMENTAIRE

## Réserves
- 20 + 20 tâches : écart-type d'environ ±8 points sur chaque taux ; une tâche = 5 points. Aucune différence entre méthodes ici n'est significative.
- TRM a été entraîné sur les paires de démonstration de ces tâches d'évaluation (protocole officiel) : seules les sorties de test sont inédites. Ses votes sont donc plus fiables ici que sur des tâches jamais vues.
- Le naïf de Bayes double-compte des variables dépendantes (votes/rang/accord/DSL) ; la variante « sobre » le montre. Le côté DSL est estimé sur toutes les solutions, alors que le lot de candidats ne contient la solution que si une méthode l'a trouvée : approximation notée dans le code.
- La courbe VoC du pli de test est descriptive ; seul le point « λ choisi » respecte la règle fixée d'avance.

## Fichiers
`bayes_arc.py` (modèle), `lancer.py` (protocole : ajustement → gel → notation), `rapport_bayes.py`, `test_bayes_arc.py` (8 tests), `sous_ensemble_bayes.json`, `res_K8_bayes.json`, `eval_K8_bayes.log`, `params_pli*_{{complet,sobre}}.json`, `resultats_bayes_arc.json`.
Modifs de code existant (comportement par défaut inchangé) : `../trm_officiel/build_subset.py` (arguments optionnels), `../trm_officiel/infer.py` (variables TRM_PKL, TRM_OUT). Aucun fichier de résultats existant n'a été modifié.
"""
com = open(os.path.join(ICI, "commentaire.txt")).read().strip() if os.path.exists(os.path.join(ICI, "commentaire.txt")) else "(à écrire)"
open(os.path.join(ICI, "resultats_bayes_arc.md"), "w").write(md.replace("COMMENTAIRE", com))
print("ok")
