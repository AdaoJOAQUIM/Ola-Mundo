"""Maintenance bayesienne apres un tour : met a jour les observations par fenetre (GATE) des competences de la reserve."""
import json, os, sys
import numpy as np
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import juge, pool as pl, donnees

def maj():
    r = pl.Reserve(os.path.join(ICI, "pool.json"))
    src = [open(os.path.join(ICI, "competences", f)).read() for f in r.fichiers]
    key = juge.cle_pool(src, "GATE")
    z = np.load(os.path.join(donnees.CACHE, "eval_%s.npz" % key), allow_pickle=True)["res"].item()
    saved_win = np.array(z["skills"]["saved_win_pct"])           # (fenetres, competences + base)
    import garde_skills as gs
    for i, f in enumerate(r.fichiers):
        nom = gs.Competence(open(os.path.join(ICI, "competences", f)).read()).name
        if nom in r.d["obs"]:
            r.d["obs"][nom]["obs"] = [float(x) for x in saved_win[:, i]]
    out = r.elaguer()
    r.sauver()
    sc = r.scores()
    json.dump(dict(scores={k: dict(moyenne=v[0], ecart_type=v[1], p_pos=v[2]) for k, v in sc.items()}, elagues=out,
                   reserve=r.fichiers), open(os.path.join(ICI, "scores_bayes.json"), "w"), indent=1)
    print(json.dumps(sc, indent=1), out)

if __name__ == "__main__":
    maj()
