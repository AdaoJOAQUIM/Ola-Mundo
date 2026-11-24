import sys, json; sys.path.insert(0,'.')
import sim, fabrique as fb
eng = sim.Sim("cm")
V = {}
V["ladder"] = [fb.source("ladder", applies="w < 0.12", plan="(2, 2, 0.0, 0.0) if w < 0.05 else (3, 3, 0.0, 0.0)")]
V["dn3"] = [fb.source("dn3", applies="True", plan="(4, 3, 0.0, 0.0)")]
V["gu3"] = [fb.source("gu3", applies="True", plan="(3, 4, 0.0, 0.0)")]
V["lod3_15"] = [fb.source("lod3b", applies="w < 0.15", plan="(3, 3, 0.0, 0.0)")]
out = fb.explorer(eng, V, "TRAIN", 2)
json.dump(out, open("cache/explo2.json","w"), indent=1)
