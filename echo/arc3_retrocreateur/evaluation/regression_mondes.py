"""Régression sur les mondes : 100 mondes J1 tirés par random.Random(11) parmi les 300, budget 500, comparés à f9bd261
(runs/j1_retro_500.json, mêmes mondes). Critère : la part de niveaux ne baisse pas de plus de 5 points.
  python regression_mondes.py [sortie.json] [procs]"""
import json, logging, random, sys, time
from multiprocessing import get_context
from pathlib import Path
ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
sys.path.insert(0, str(RACINE)); sys.path.insert(0, str(RACINE.parent / "mondes_arc3")); sys.path.insert(0, str(ICI))
logging.disable(logging.CRITICAL)


def un(a):
    seed, d = a
    logging.disable(logging.CRITICAL)
    from env import jouer
    from retro.agent import RetroCreateur
    res, pol = jouer(seed, d, lambda: RetroCreateur(0), max_actions=500)
    res["stats"] = pol.stats
    return res


if __name__ == "__main__":
    sortie = sys.argv[1] if len(sys.argv) > 1 else str(RACINE / "runs" / "regression_mondes.json")
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ref = json.load(open(RACINE / "runs" / "j1_retro_500.json"))
    idx = random.Random(11).sample(range(len(ref)), 100)
    ref = [ref[i] for i in idx]
    t = time.time()
    with get_context("fork").Pool(procs) as p:
        out = p.map(un, [(r["seed"], r["difficulte"]) for r in ref], chunksize=1)
    a = sum(r["niveaux"] for r in ref); b = sum(r["niveaux"] for r in out); n = sum(r["niveaux_total"] for r in ref)
    rs = {"n_mondes": 100, "niveaux_f9bd261": a, "niveaux_actuel": b, "niveaux_total": n,
          "part_f9bd261": round(a / n, 4), "part_actuel": round(b / n, 4), "ecart_points": round(100 * (b - a) / n, 2),
          "ok": (a - b) / n <= 0.05, "graines": [r["seed"] for r in ref], "mur_s": round(time.time() - t)}
    print(rs)
    json.dump({"resume": rs, "mondes": out}, open(sortie, "w"), default=str)
