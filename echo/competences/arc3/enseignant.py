"""Outils de l'enseignant : résumés compacts de trajectoires (jeux DEV-train uniquement)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent


def fenetres(trace):
    """Regroupe les actions de compétence consécutives : [(nom, n_debut, n_fin, nb_actions)]."""
    out = []
    for (n, nom, act) in trace:
        if out and out[-1][0] == nom and n == out[-1][2] + 1:
            out[-1][2] = n
            out[-1][3] += 1
        else:
            out.append([nom, n, n, 1])
    return out


def resumer(res, chemin_traj=None):
    """Résumé d'un résultat jouer._worker : niveaux, activations, attribution a posteriori."""
    gagnes = []
    cum = 0
    la = res["niveaux_actions"] or []
    for i in range(res["niveaux"]):
        cum += la[i]
        gagnes.append(cum)               # numéro d'action où le niveau i est gagné
    fen = fenetres(res.get("trace", []))
    attrib = []
    for g in gagnes:
        cause = "cerveau"
        for nom, a, b, k in fen:
            if a <= g <= b + 25:
                cause = nom
        attrib.append(cause)
    out = {"jeu": res["prefixe"], "graine": res["graine"], "niveaux": res["niveaux"],
           "etat": res["etat"], "niveaux_actions": la[:res["niveaux"]],
           "gagnes_a_l_action": gagnes, "attribution_gain": attrib,
           "actions_competences": res["actions_competences"], "actions_cerveau": res["actions_cerveau"],
           "competences": res["competences"],
           "fenetres": [tuple(f) for f in fen[:40]],
           "nb_fenetres": len(fen)}
    if chemin_traj and Path(chemin_traj).exists():
        d = np.load(chemin_traj)
        F = d["frames"]
        ch = np.array([int((F[i] != F[i + 1]).sum()) for i in range(len(F) - 1)])
        plancher = int(np.percentile(ch, 5)) if len(ch) else 0
        out["traj"] = {"images_distinctes": len({f.tobytes() for f in F}), "n_images": len(F),
                       "part_sans_effet": round(float((ch <= plancher).mean()), 3),
                       "types_actions": dict(Counter(d["acts"].tolist()))}
    return out
