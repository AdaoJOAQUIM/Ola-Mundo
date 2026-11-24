"""Verification du vrai moteur (streaming reel) avec la reserve finale, sur 16 tokens de GATE (pas de TEST).
Compare logits/octets au simulateur ; mesure pic RSS, lectures, temps E/S et calcul.   python3 reel_run.py [N]"""
import json, os, resource, sys, time
import numpy as np, torch
ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import donnees, garde_skills as gs, reel, sim, layout as lay

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
pool = json.load(open(os.path.join(ICI, "pool.json")))
comps = [gs.Competence(open(os.path.join(ICI, "competences", f)).read()) for f in pool["competences"]]
ids = donnees.fenetres("GATE", 1)[0][:N + 1]
eng = reel.Reel(max_ctx=64)
lg, pages, logi, ios, tots = eng.decoder(ids, comps)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
s = sim.Sim("cm", max_ctx=64)
r = s.run([ids], comps)
out = dict(tokens=N, rss_pic_mo=rss, octets_pages_reel=float(np.mean(pages)), octets_pages_sim=float(r["bytes_per_token"]),
           octets_logiques_reel=float(np.mean(logi)), lectures_par_token=eng.read_ops / N,
           io_s_par_token=float(np.mean(ios)), total_s_par_token=float(np.mean(tots)),
           calcul_s_par_token=float(np.mean(np.array(tots) - np.array(ios))),
           ecart_logit_max=float((lg - torch.as_tensor(np.zeros(1))).abs().max()) if False else None,
           argmax_identique=bool((lg.argmax(-1).numpy() == r["argmax"]).all()))
# ecart de logits : recalculer les logits du simulateur token par token n'etant pas expose, on compare nll
nll_reel = float(torch.nn.functional.cross_entropy(lg, torch.tensor(ids[1:N + 1])))
out["nll_reel"], out["nll_sim"] = nll_reel, r["nll"]
json.dump(out, open(os.path.join(ICI, "resultats_reel.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
