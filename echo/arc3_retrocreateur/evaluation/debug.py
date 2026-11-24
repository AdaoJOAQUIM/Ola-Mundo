import sys, logging
from pathlib import Path
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent)); sys.path.insert(0, str(ICI.parent.parent / "mondes_arc3"))
logging.disable(logging.CRITICAL)
from env import MondeEnv
from retro.agent import RetroCreateur
from actions import GameAction
seed,d,n=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
env=MondeEnv(seed,d)
ag=RetroCreateur(0)
obs=None
def mk(f):
    return {"frame":[f.frame[0]],"available_actions":f.available_actions,"levels_completed":f.levels_completed,"actions_taken":env.n_actions,"state":f.state.name}
fr=env.reset(); o=mk(fr)
for i in range(n):
    a=ag.choisir(o)
    s=ag.snap
    g=ag.geo
    print(i,a,"pos",s[1] if s else None,"avc",s[2] if s else None, "win",ag.modele.prog.win, "lvl",env.finis)
    if a["name"]=="RESET": fr=env.reset()
    else: fr=env.step(GameAction[a["name"]], a["data"] or None)
    o=mk(fr)
print(ag.modele.prog.dirmap, ag.modele.prog.fx, ag.modele.prog.clk)
g=ag.geo
print(g.R,g.C,g.gx0,g.gy0,g.net.c,g.net.px,g.net.py)
M=ag.snap[0]
for r in range(g.R): print(' '.join('%2d'%M[r*g.C+c] for c in range(g.C)))
print(ag.types.inv)
