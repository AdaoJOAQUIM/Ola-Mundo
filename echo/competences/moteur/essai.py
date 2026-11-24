import sys, time, numpy as np
sys.path.insert(0, '.')
import sim, donnees, garde_skills as gs
eng = sim.Sim("cm")
w = donnees.fenetres("TRAIN", 1)
t = time.time()
eng.backend = "rm"; r0 = eng.run(w, (), collect=False); print("rm", r0["ppl"], r0["wall_s"], flush=True)
eng.backend = "cm"; r1 = eng.run(w, [gs.Competence(gs.BASE4)]); print("cm base4", r1["ppl"], r1["pct_naive"], r1["wall_s"], r1["faults"], flush=True)
print("accord", (r0["argmax"] == r1["argmax"]).mean())
