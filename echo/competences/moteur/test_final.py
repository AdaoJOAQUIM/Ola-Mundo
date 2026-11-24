"""Evaluation TEST (une seule fois, a la fin) : lignee des reserves, 4 bits simple, config phase 1.   python3 test_final.py"""
import collections, json, os, sys
import numpy as np
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import donnees, garde_skills as gs, juge, sim, layout as lay

FICHIER = os.path.join(ICI, "resultats_test.json")
assert not os.path.exists(FICHIER), "TEST deja touche : ne pas rejouer"
eng = sim.Sim("cm")
W = donnees.fenetres("TEST")
lignee = json.load(open(os.path.join(ICI, "lignee.json")))        # liste de reserves (listes de fichiers), de P0 a Pn
out = dict(fenetres=len(W), jetons=sum(len(w) - 1 for w in W))
eng.backend = "rm"
ref = eng.run(W, (), collect=False)
out["ref_4bits_simple"] = dict(ppl=ref["ppl"])
# config phase 1 (LOD 0,08 + culling 0,06, q2 dedie) : octets par comptage des emplacements
eng.lod_thr, eng.cull_thr = 0.08, 0.06
eng.st.clear()
p1 = eng.run(W, (), collect=False)
n = p1["n_tokens"]
b = (eng.st["slots_q2"] * 1966080 + eng.st["slots_q4"] * 3538944) / n
out["phase1_lod008_cull006"] = dict(ppl=p1["ppl"], ratio_ppl=p1["ppl"] / ref["ppl"], pct=100 * b / (lay.NAIVE_SLOT * 128),
                                    accord_vs_4bits=float((p1["argmax"] == ref["argmax"]).mean()))
eng.lod_thr = eng.cull_thr = 0.0
eng.backend = "cm"
res = []
prev = None
for i, fichiers in enumerate(lignee):
    src = [open(os.path.join(ICI, "competences", f)).read() for f in fichiers]
    r = eng.run(W, [gs.Competence(s) for s in src])
    d = dict(reserve=fichiers, pct=r["pct_naive"], pct_secteur=r["pct_naive_sector"], ppl=r["ppl"], ratio_ppl=r["ppl"] / ref["ppl"],
             accord_vs_4bits=float((r["argmax"] == ref["argmax"]).mean()), faults=r["faults"], wall_s=r["wall_s"],
             par_fenetre_ppl=None)
    if prev is not None:
        d["gain_pts"] = prev["pct"] - r["pct_naive"]
        d["accord_vs_precedente"] = float((r["argmax"] == prev["argmax"]).mean())
        d["marge_gate_conservee"] = bool(d["gain_pts"] >= juge.SEUIL_OCTETS_PTS and d["ratio_ppl"] <= juge.SEUIL_PPL
                                         and d["accord_vs_precedente"] >= juge.SEUIL_ACCORD)
    d["layer"], d["skills"] = r.get("layer"), r.get("skills")
    res.append(d)
    prev = dict(pct=r["pct_naive"], argmax=r["argmax"])
out["lignee"] = res
json.dump(out, open(FICHIER, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "lignee"}, indent=1))
for d in res:
    print({k: v for k, v in d.items() if k not in ("layer", "skills")})
