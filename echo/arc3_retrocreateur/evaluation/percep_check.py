"""Compare la position d'avatar perçue à la vraie (développement) : python percep_check.py SEED D [N]"""
import sys, logging
from pathlib import Path
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent)); sys.path.insert(0, str(ICI.parent.parent / "mondes_arc3"))
logging.disable(logging.CRITICAL)
from env import MondeEnv
from actions import GameAction, EtatJeu
from retro.agent import RetroCreateur
seed, d = int(sys.argv[1]), int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 500
env = MondeEnv(seed, d); ag = RetroCreateur(0)
def mk(f, extra=0):
    return {"frame": [f.frame[0]], "available_actions": f.available_actions, "levels_completed": f.levels_completed,
            "actions_taken": env.n_actions, "state": f.state.name}
fr = env.reset(); o = mk(fr)
n_bad = 0
for i in range(N):
    a = ag.choisir(o)
    g = ag.geo
    # position vraie dans les coordonnées de la grille de l'agent
    n = env.lv.n
    p = env.s[0][env.s[1]]
    ty, tx = divmod(p, n)
    vrai = (ty - g.gy0, tx - g.gx0) if g else None
    perc = divmod(ag.snap[1], g.C) if ag.snap and ag.snap[1] >= 0 else None
    if perc != vrai and n_bad < 8:
        n_bad += 1
        print(f"step {i} niv {env.k} perçu {perc} vrai {vrai} avc {ag.snap[2]} cols {ag.scene.av_colors} act {a['name']}")
    prev_over = False
    fr = env.step(GameAction[a["name"]], a["data"] or None)
    o = mk(fr)
    if fr.state is EtatJeu.GAME_OVER:
        fr = env.reset(); env.n_actions += 0; o = mk(fr); o["actions_taken"] = env.n_actions + 1
    if fr.state is EtatJeu.WIN:
        break
print("fin niveau", env.finis)
