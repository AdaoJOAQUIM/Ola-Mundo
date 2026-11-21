"""Ablation : plusieurs graines x {meta auto, meta fixe}, même budget, même départ.

Écrit archive/ablation.json avec best-A et best-B tous les 250 évaluations.
"""
import json
import os
import sys
import time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import moteur  # noqa: E402


def main(graines, budget):
    t0 = time.time()
    sortie = {"budget": budget, "graines": graines, "runs": []}
    for g in graines:
        for mode in ("auto", "fixe"):
            r = moteur.executer("abl_%s_s%d" % (mode, g), budget, 1000 + g, mode, verbeux=False)
            sortie["runs"].append({"meta": mode, "graine": 1000 + g, "points": r["points"],
                                   "champion_A": r["champion"]["fit_A"], "champion_B": r["champion"]["fit_B"],
                                   "best_A_final": r["points"][-1]["best_A"],
                                   "rejets": r["rejets"], "rejets_garde_fous": r["rejets_garde_fous"],
                                   "temps_mur_s": r["temps_mur_s"], "taille_archive": r["taille_archive"],
                                   "meta_moyen_final": r["meta_moyen_final"]})
            print("fini", mode, g, r["champion"]["fit_A"], r["champion"]["fit_B"], r["temps_mur_s"], flush=True)
            with open(os.path.join(ICI, "archive", "ablation.json"), "w") as f:
                json.dump(sortie, f, indent=1)
    sortie["temps_total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(ICI, "archive", "ablation.json"), "w") as f:
        json.dump(sortie, f, indent=1)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    main(list(range(1, n + 1)), b)
