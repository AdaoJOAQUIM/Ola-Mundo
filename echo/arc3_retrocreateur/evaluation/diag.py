import sys, copy, random, logging
from pathlib import Path
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent)); sys.path.insert(0, str(ICI.parent.parent / "mondes_arc3")); sys.path.insert(0, str(ICI))
logging.disable(logging.CRITICAL)
import j1
from env import jouer
seed, d = int(sys.argv[1]), int(sys.argv[2])
holder = {}
def f():
    r = j1.Enregistreur(j1.fabrique("retro")()); holder["r"] = r; return r
res, pol = jouer(seed, d, f, max_actions=500)
rec = holder["r"]; ag = rec.pol
print(res["niveaux"], "/", res["niveaux_total"])
# rejoue la validation en affichant les écarts
orig = ag.valider
def v(k, a, fr, fl):
    ok, j = orig(k, a, fr, fl)
    if not ok:
        print("ECART k", k, [x["name"] + (str((x["data"]["x"], x["data"]["y"])) if x["name"] == "ACTION6" else "") for x in a], "flags", fl, "j", j, getattr(ag, "dernier_ecart", None))
        ag.dernier_ecart = None
    return ok, j
ag.valider = v
print(j1.valider_monde(seed, d, rec, ag))
mo = ag.modele; p = mo.prog
print("dirmap", p.dirmap, "solid", p.solid, "push", p.push, "kill", p.kill, "undo", p.undo, "caps", ag.cap_niveau)
print("fx", p.fx, "clk", p.clk, "sync", p.sync, "win", p.win)
