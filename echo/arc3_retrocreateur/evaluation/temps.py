import sys, time, logging
from pathlib import Path
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent)); sys.path.insert(0, str(ICI.parent.parent / "mondes_arc3")); sys.path.insert(0, str(ICI))
logging.disable(logging.CRITICAL)
from choisir_seeds import choisir
from env import jouer
from retro.agent import RetroCreateur
for d in (1, 2, 3):
    for s in choisir(1, d, int(sys.argv[1])):
        t = time.time()
        res, pol = jouer(s, d, lambda: RetroCreateur(0), max_actions=500)
        dt = time.time() - t
        if dt > 5:
            print(res["jeu"], round(dt, 1), res["niveaux"], "/", res["niveaux_total"], ",".join(m for m in res["mecaniques"] if m not in ("deplacement", "murs")), flush=True)
print("fin")
