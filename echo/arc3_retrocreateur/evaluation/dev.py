"""Lot de développement : python dev.py N [budget] [procs] [--d 1,2,3]  (graines 1..2999, mécaniques scellées exclues)."""
import sys, time, json, logging, os
from pathlib import Path
from multiprocessing import get_context
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent)); sys.path.insert(0, str(ICI.parent.parent / "mondes_arc3")); sys.path.insert(0, str(ICI))
logging.disable(logging.CRITICAL)
from choisir_seeds import choisir

def un(args):
    seed, d, budget = args
    logging.disable(logging.CRITICAL)
    from env import jouer
    from retro.agent import RetroCreateur
    t = time.time()
    res, pol = jouer(seed, d, lambda: RetroCreateur(0), max_actions=budget)
    res["mur"] = round(time.time() - t, 1)
    res["stats"] = pol.stats
    res["trace"] = pol.trace[-1:] 
    return res

if __name__ == "__main__":
    n = int(sys.argv[1]); budget = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    ds = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["1", "2", "3"])]
    taches = []
    for d in ds:
        for s in choisir(1, d, n):
            taches.append((s, d, budget))
    with get_context("fork").Pool(procs) as p:
        out = p.map(un, taches, chunksize=1)
    tot = sum(r["niveaux"] for r in out); nt = sum(r["niveaux_total"] for r in out)
    for r in out:
        print(r["jeu"], r["but"][:3], f"{r['niveaux']}/{r['niveaux_total']}", r["actions"], "opt", sum(r["niveaux_reference"][:max(r['niveaux'],1)]), r["niveaux_actions"][:r['niveaux']+1], ",".join(m for m in r["mecaniques"] if m not in ("deplacement","murs")), r["mur"], r["stats"].get("fallback"))
    print("TOTAL", tot, nt, round(tot / nt, 3))
    json.dump(out, open(ICI.parent / "runs" / "dev_last.json", "w"), default=str)
