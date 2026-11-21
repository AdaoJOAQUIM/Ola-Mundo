"""Émergence : reconstruit l'archive MAP-Elites d'un run et teste si les structures « nouvelles » servent.

Pour chaque structure absente du génome de départ, on compte les élites qui l'ont et, quand c'est
possible (u1, u2, constantes par ordre), on la NEUTRALISE (u1[...] -> 0 ; liste de constantes -> sa
moyenne) et on regarde si le bpc sur A change de plus de 0,002. Jamais de C.
Usage : python3 analyse_emergence.py principal_complet
"""
import json
import os
import re
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import composants  # noqa: E402
import donnees  # noqa: E402
import juge  # noqa: E402
import moteur  # noqa: E402

SEUIL = 0.002


def neutraliser(source, car):
    if car in ("utilise_u1", "utilise_u2"):
        nm = car[-2:]
        return re.sub(nm + r"\[[^\]]*\]", "0.0", source)
    if car == "constantes_par_ordre":
        def moy(m):
            vals = [float(v) for v in m.group(1).split(",")]
            return repr(round(sum(vals) / len(vals), 5))
        return re.sub(r"\[((?:-?[0-9.]+, ){2,}-?[0-9.]+)\]\[k\]", moy, source)
    return None


def main(nom):
    ab = donnees.charger_ab()
    archive = [json.loads(l) for l in open(os.path.join(ICI, "archive", nom + "_final.jsonl"))]
    cartes = composants.MapElites()
    for g in sorted(archive, key=lambda g: g["fit_eff"]):
        cartes.inserer(g, g["fit_eff"])
    rap = moteur.rapport_emergence(archive, cartes)
    par_id = {g["id"]: g for g in archive}
    for car, e in rap["caracteristiques_absentes_du_depart"].items():
        deltas = []
        for i in cartes.ids():
            g = par_id[i]
            if car in composants.caracteristiques(g["source"]):
                s2 = neutraliser(g["source"], car)
                if s2 and s2 != g["source"]:
                    r = juge.juger(s2, ab["A"])
                    deltas.append(r["bpc"] - g["fit_A"] if r["ok"] else 9.9)
        if deltas:
            e["neutralisation"] = {"n_testes": len(deltas), "delta_A_median": float(np.median(deltas)),
                                   "part_neutres": float(np.mean([abs(d) < SEUIL for d in deltas]))}
    # élites avec remise : on garde leur source pour lecture à la main
    rap["elites_avec_remise"] = [{"id": par_id[i]["id"], "fit_A": par_id[i]["fit_A"], "source": par_id[i]["source"]}
                                 for i in cartes.ids() if "remise" in composants.caracteristiques(par_id[i]["source"])]
    with open(os.path.join(ICI, "archive", nom + "_emergence.json"), "w") as f:
        json.dump(rap, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rap.items() if k != "meilleur_avec_remise"}, indent=1, ensure_ascii=False)[:5000])


if __name__ == "__main__":
    main(sys.argv[1])
