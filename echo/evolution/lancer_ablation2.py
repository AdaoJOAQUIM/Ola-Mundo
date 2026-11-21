"""Ablation « moteur complet » contre « moteur simple », à CPU égal.

Le budget est le temps CPU cumulé (travailleurs + parent), pas le nombre d'évaluations :
le moteur complet saute des évaluations, il peut donc en faire d'autres avec le même CPU.
Écrit archive/ablation_complet.json.
"""
import json
import os
import sys
import time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import moteur  # noqa: E402


def main(graines, budget_cpu):
    t0 = time.time()
    sortie = {"budget_cpu_s": budget_cpu, "runs": []}
    for g in graines:
        for mode in ("simple", "complet"):
            r = moteur.executer("abc_%s_s%d" % (mode, g), 10**6, 2000 + g, "auto", verbeux=False,
                                complet=(mode == "complet"), budget_cpu=budget_cpu)
            sortie["runs"].append({
                "mode": mode, "graine": 2000 + g, "points": r["points"], "evals": r["evals"],
                "cpu_total_s": r["cpu_total_s"], "best_A": min(p["best_A"] for p in r["points"]),
                "champion_A": r["champion"]["fit_A"], "champion_B": r["champion"]["fit_B"],
                "gain_par_cpu_s": r["gain_par_cpu_s"], "soi": r.get("soi"),
                "elites": {k: r["elites"][k] for k in ("couverture", "cases_remplies")} if "elites" in r else None})
            print("fini", mode, g, r["champion"]["fit_A"], r["champion"]["fit_B"], r["evals"], r["cpu_total_s"], flush=True)
            sortie["temps_total_s"] = round(time.time() - t0, 1)
            with open(os.path.join(ICI, "archive", "ablation_complet.json"), "w") as f:
                json.dump(sortie, f, indent=1)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    b = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    main(list(range(1, n + 1)), b)
