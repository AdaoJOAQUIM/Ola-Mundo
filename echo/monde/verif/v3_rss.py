"""Verif (b) : mode toutes mecanismes (cache 1% + prefetch + LOD 0.08 + culling 0.06), 16 tokens de decodage.
Le pic RSS est mesure de l'exterieur (VmHWM via /proc, sondage 20 Hz, et /usr/bin/time -v dans le shell)."""
import sys, json, time
sys.path.insert(0, "/home/user/Ola-Mundo/echo/monde")
import torch
from tokenizers import Tokenizer
import moteur as mt
SRC = "/home/user/monde_data/olmoe"; WORLD = "/home/user/monde_data/monde"
ids = Tokenizer.from_file(f"{SRC}/tokenizer.json").encode(
    "The history of the city is long and complicated, and many people have written about it over the centuries, in many languages.").ids[:17]
assert len(ids) == 17
m = mt.Moteur(WORLD, mode="q4", cache_frac=0.01, prefetch=True, lod_thr=0.08, cull_thr=0.06, max_ctx=32)
t0 = time.time(); nll, per = m.decode_text(ids)
m.close()
json.dump(dict(n_tokens=len(per), bytes_per_token=sum(b for b, _ in per) / len(per), wall_s=time.time() - t0,
               cache_max_mb=m.cache.max_used / 1e6, cache_budget_mb=m.cache.budget / 1e6, budget_moteur_mb=m.budget.total() / 1e6),
          open("/home/user/Ola-Mundo/echo/monde/verif/v3_engine_side.json", "w"), indent=1)
