"""Essai de développement : python essai.py SEED D [budget]  (graines de développement 1..2999 seulement)."""
import sys, time, json, logging
from pathlib import Path
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent))
sys.path.insert(0, str(ICI.parent.parent / "mondes_arc3"))
logging.disable(logging.CRITICAL)
from env import jouer
from retro.agent import RetroCreateur

def main():
    seed, d = int(sys.argv[1]), int(sys.argv[2])
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    t = time.time()
    res, pol = jouer(seed, d, lambda: RetroCreateur(0), max_actions=budget)
    print({k: res[k] for k in ("jeu", "etat", "niveaux", "niveaux_total", "actions", "niveaux_actions", "niveaux_reference", "mecaniques", "but")}, round(time.time() - t, 1), "s")
    print(pol.stats, pol.modele.stats)
    for tr in pol.trace[-3:]:
        print(tr)
    p = pol.modele.prog
    print("dirmap", p.dirmap, "solid", p.solid, "push", p.push, "kill", p.kill)
    print("fx", p.fx, "sync", p.sync, "win", p.win, "clk", p.clk)

main()
