"""Experiences du Moteur Monde Infini (un processus par configuration : pic RSS propre).

  python3 bench.py decode NOM '{"cache_frac":0.01,...}'   octets/token, taux de succes (N_DEC pas de decodage)
  python3 bench.py ppl    NOM '{"lod_thr":0.1,...}'       perplexite sur les ~20k premiers caracteres
Les resultats vont dans RES/*.json (jamais le texte, seulement des nombres).
"""
import json, os, pickle, resource, sys, time
import torch
from tokenizers import Tokenizer

import moteur as mt

WORLD = "/home/user/monde_data/monde"
SRC = "/home/user/monde_data/olmoe"
VALID = "/home/user/Ola-Mundo/echo/validation.txt"
RES = "/home/user/Ola-Mundo/echo/monde/resultats_bruts"
TRACES = "/home/user/monde_data/traces"
N_DEC = 128
NCHAR = 20000
os.makedirs(RES, exist_ok=True)
os.makedirs(TRACES, exist_ok=True)


def token_ids():
    t = open(VALID, encoding="utf-8", errors="replace").read()[:NCHAR]
    return Tokenizer.from_file(f"{SRC}/tokenizer.json").encode(t).ids


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    kind, name, kw = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    ids = token_ids()
    out = dict(name=name, kind=kind, params=kw, n_tokens_total=len(ids))
    if kind == "decode":
        m = mt.Moteur(WORLD, max_ctx=N_DEC + 8, trace=True, **kw)
        t0 = time.time()
        nll, per = m.decode_text(ids[:N_DEC + 1])
        wall = time.time() - t0
        b = [x[0] for x in per]
        o = [x[1] for x in per]
        st = dict(m.st)
        out.update(
            steps=len(per), bytes_per_token=sum(b) / len(b), reads_per_token=sum(o) / len(o),
            total_bytes=sum(b), total_reads=sum(o), wall_s=wall, io_wait_s=m.io_wait,
            compute_s_per_token=(wall - m.io_wait) / len(per), nll_mean=nll,
            uses=st.get("uses", 0), cache_hits=st.get("cache_hits", 0), pf_hits=st.get("pf_hits", 0),
            disk_miss=st.get("disk_miss", 0), uses_q4=st.get("uses_q4", 0), uses_q2=st.get("uses_q2", 0),
            routed=st.get("routed", 0), culled=st.get("culled", 0),
            hit_rate_cache=st.get("cache_hits", 0) / max(1, st.get("uses", 1)),
            hit_rate_cache_ou_prefetch=(st.get("cache_hits", 0) + st.get("pf_hits", 0)) / max(1, st.get("uses", 1)),
            pred_hit_rate=(st["pred_hit"] / st["pred_total"]) if st.get("pred_total") else None,
            pred_precision=(st["pred_hit"] / st["pred_issued"]) if st.get("pred_issued") else None,
            prefetch_submitted=m.pf.submitted if m.pf else 0,
            cache_max_used_mb=m.cache.max_used / 1e6, cache_budget_mb=m.cache.budget / 1e6,
            budget_moteur_mb=m.budget.total() / 1e6, rss_peak_mb=rss_mb(), direct=m.direct,
            chunk_sizes=m.store.size,
        )
        pickle.dump(m.trace, open(f"{TRACES}/{name}.pkl", "wb"))
        m.close()
    else:
        kw = dict(kw)
        m = mt.Moteur(WORLD, max_ctx=2048, drop=False, **kw)
        t0 = time.time()
        tot, n = m.nll_batch(ids, 2048)
        st = dict(m.st)
        slots = st.get("slots_q4", 0) + st.get("slots_q2", 0)
        out.update(nll_total=tot, n_scored=n, nll=tot / n, ppl=float(torch.exp(torch.tensor(tot / n))),
                   wall_s=time.time() - t0, rss_peak_mb=rss_mb(),
                   frac_culled=st.get("culled", 0) / max(1, st.get("routed", 1)),
                   frac_q2_of_used=st.get("slots_q2", 0) / max(1, slots),
                   cache_max_used_mb=m.cache.max_used / 1e6, budget_moteur_mb=m.budget.total() / 1e6)
        m.close()
    json.dump(out, open(f"{RES}/{kind}_{name}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "chunk_sizes"}))


if __name__ == "__main__":
    main()
