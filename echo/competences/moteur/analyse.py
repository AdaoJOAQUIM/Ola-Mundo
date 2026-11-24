"""Outils de l'ecrivain (non epingles) : explorer TRAIN, resumer les traces E/S + qualite, ecrire des nombres seulement.

  python3 analyse.py explorer NOM fichier_competence1.py [fichier2.py ...]   -> passe sur TRAIN, trace ecrite en cache/trace_NOM.json
"""
import json, os, sys, time
import numpy as np
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import donnees, garde_skills as gs, juge, sim


def trace_resume(res, ref=None, base=None):
    """Resume numerique d'une passe : octets, ppl, par couche, par competence, et perte par token vs une passe de base."""
    d = dict(pct=res["pct_naive"], pct_secteur=res["pct_naive_sector"], ppl=res["ppl"], nll=res["nll"],
             faults=res["faults"], wall_s=res["wall_s"])
    if ref:
        d["ratio_ppl_vs_4bits"] = res["ppl"] / ref["ppl"]
    if base is not None:
        d["accord_vs_base"] = float((res["argmax"] == base["argmax"]).mean())
        dn = res["nll_tok"] - base["nll_tok"]
        d["delta_nll_moyen"] = float(dn.mean())
        d["delta_nll_p90"] = float(np.percentile(dn, 90))
    if "layer" in res:
        d["layer"] = res["layer"]
        d["skills"] = res["skills"]
    return d


def explorer(eng, nom_tranche, sources, only_layers=None):
    return juge.evaluer(eng, sources, nom_tranche, only_layers=only_layers)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "explorer":
        nom = sys.argv[2]
        srcs = [open(f).read() for f in sys.argv[3:]] + [gs.BASE4]
        eng = sim.Sim("cm")
        ref = juge.reference(eng, "TRAIN")
        res = juge.evaluer(eng, srcs, "TRAIN")
        t = trace_resume(res, ref, juge.evaluer(eng, [gs.BASE4], "TRAIN"))
        json.dump(t, open(os.path.join(donnees.CACHE, "trace_%s.json" % nom), "w"), indent=1)
        print(json.dumps({k: v for k, v in t.items() if k not in ("layer", "skills")}, indent=1))
