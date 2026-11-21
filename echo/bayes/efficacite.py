"""Efficacité en échantillons : évaluations nécessaires pour atteindre un niveau de bpc sur B.

Moteur d'évolution : points (évaluations d'enfants sur A, meilleur A, B du champion) lus dans
evolution/archive/*_resume.json. Optuna : un essai = une évaluation sur A. Les deux sont comptés en
évaluations sur A ; la sélection sur B coûte en plus quelques évaluations B pour les deux méthodes.
"""
import glob
import json
import os

import donnees_bayes as db

ARCHIVE = os.path.join(db.ECHO, "evolution", "archive")


def courbes_moteur():
    """{nom_run: {'points': [(evals, B)], 'seed_B', 'budget'}} pour tous les runs ayant des points."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ARCHIVE, "*_resume.json"))):
        r = json.load(open(f))
        pts = [(p["evals"], p["best_B"]) for p in r.get("points", []) if p.get("best_B") is not None]
        if pts:
            out[os.path.basename(f)[:-len("_resume.json")]] = {"points": pts, "seed_B": r.get("seed_B"),
                                                              "budget": r.get("budget")}
    return out


def evals_pour_niveau(points, niveau):
    for e, b in points:
        if b <= niveau:
            return e
    return None


def comparer(courbe_optuna, niveaux, decalage=0.0):
    """Pour chaque niveau (échelle de B de l'ancienne bibliothèque) : évaluations par run du moteur
    et pour Optuna. `decalage` = B_nouveau - B_ancien (champion), appliqué aux niveaux d'Optuna."""
    runs = courbes_moteur()
    res = []
    for niv in niveaux:
        par_run = {nom: evals_pour_niveau(c["points"], niv) for nom, c in runs.items()}
        atteints = sorted(v for v in par_run.values() if v is not None)
        opt = None
        for c in courbe_optuna:
            if c["B_champion"] <= niv + decalage:
                opt = c["essais"]
                break
        res.append({"niveau_B": niv, "moteur_runs_total": len(par_run), "moteur_runs_atteignant": len(atteints),
                    "moteur_evals_min": atteints[0] if atteints else None,
                    "moteur_evals_mediane": atteints[len(atteints) // 2] if atteints else None,
                    "optuna_essais": opt, "par_run": par_run})
    return res
