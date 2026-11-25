"""Runner J3 : python j3.py JEUX SEEDS BUDGET SORTIE.json   (ex. j3.py cn04,ls20,sc25 0,1,2 500 out.json)
Tout jeu HELD est refusé par jeux_reels.garde. Aucun réseau ni LLM à la partie."""
import sys, json, os, logging
from pathlib import Path
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent)); sys.path.insert(0, str(ICI))
import jeux_reels as J


def un(jeu, graine, budget, opts=None):
    from retro.agent import RetroCreateur
    rec = ICI.parent / "runs" / f"rec_{os.getpid()}"
    os.makedirs(rec, exist_ok=True)
    cwd = os.getcwd(); os.chdir(rec)
    try:
        opts = dict(opts or {}); opts.setdefault('cpu_max', 60.0 if budget <= 500 else 400.0)
        res, ag = J.jouer(jeu, lambda: RetroCreateur(graine, opts), budget)
    finally:
        os.chdir(cwd)
        import shutil
        shutil.rmtree(rec, ignore_errors=True)   # enregistrements du SDK : jamais gardés (volumineux)
    res.update({"prefixe": jeu, "graine": graine, "budget": budget, "stats": getattr(ag, "stats", {})})
    return res


if __name__ == "__main__":
    jeux = sys.argv[1].split(","); graines = [int(x) for x in sys.argv[2].split(",")]
    budget = int(sys.argv[3]); sortie = sys.argv[4]
    for j in jeux:
        J.garde(j)
    out = []
    for j in jeux:
        for s in graines:
            r = un(j, s, budget)
            out.append(r)
            print(j, s, r["niveaux"], "/", r["niveaux_total"], r["actions"], r["niveaux_actions"], r["cpu_s"], r["stats"], flush=True)
    json.dump(out, open(sortie, "w"), default=str)
