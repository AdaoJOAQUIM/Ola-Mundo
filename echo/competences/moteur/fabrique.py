"""Outil de l'ecrivain (non epingle) : fabrique le texte source d'une competence a partir de parametres, et explore TRAIN.
Les competences produites passent la liste blanche (verifiee a la fabrication)."""
import json, os, sys, time
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import garde_skills as gs


def source(nom, parent="base4", mode="add", tables="", applies="True", plan="(4, 4, 0.0, 0.0)", fallback="False", doc=""):
    s = ('"""%s"""\n' % doc if doc else "") + 'NAME = "%s"\nPARENT = "%s"\nMODE = "%s"\n%s' % (nom, parent, mode, tables)
    s += "def applies(L, e, w, r):\n    return %s\n" % applies
    s += "def plan(L, e, w, r):\n    return %s\n" % plan
    s += "def fallback(L, e, w, r, perte):\n    return %s\n" % fallback
    ok, c, d = gs.verifier(s)
    assert ok, (c, d, s)
    return s


def explorer(eng, variantes, tranche="TRAIN", limite=None, pool_base=None):
    """variantes : dict nom -> liste de sources (la reserve, sans base4). Imprime pct, ppl vs base4, accord."""
    import donnees, juge
    from analyse import trace_resume
    base_src = [gs.BASE4]
    w = donnees.fenetres(tranche, limite)
    eng.backend = "cm"
    out = {}
    base = eng.run(w, [gs.Competence(s) for s in (pool_base or []) + base_src])
    out["_base"] = dict(pct=base["pct_naive"], ppl=base["ppl"], wall=base["wall_s"])
    print("base", out["_base"], flush=True)
    for nom, srcs in variantes.items():
        r = eng.run(w, [gs.Competence(s) for s in srcs + (pool_base or []) + base_src])
        d = dict(pct=r["pct_naive"], pct_sec=r["pct_naive_sector"], dppl=100 * (r["ppl"] / base["ppl"] - 1),
                 accord=float((r["argmax"] == base["argmax"]).mean()), faults=r["faults"], wall=r["wall_s"],
                 saved=r["skills"]["saved_pct_naive"], keep_gu=sum(r["layer"]["keep_gu"]) / 16,
                 keep_dn=sum(r["layer"]["keep_dn"]) / 16, lost=sum(r["layer"]["lost_gu"]) / 16)
        out[nom] = d
        print(nom, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "saved"}), flush=True)
    return out
