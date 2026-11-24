import sys, json; sys.path.insert(0,'.')
import sim, fabrique as fb
eng = sim.Sim("cm")
V = {}
for t in (0.5, 1.0):
    V["gu%.1f"%t] = [fb.source("gu", applies="True", plan="(4, 4, %s, 0.0)"%t, fallback="perte > 0.3")]
    V["dn%.1f"%t] = [fb.source("dn", applies="True", plan="(4, 4, 0.0, %s)"%t)]
V["lod3"] = [fb.source("lod", applies="w < 0.08", plan="(3, 3, 0.0, 0.0)")]
out = fb.explorer(eng, V, "TRAIN", 2)
json.dump(out, open("cache/explo1.json","w"), indent=1)
