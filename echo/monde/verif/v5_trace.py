"""Verif 5 : journal du routage sur 128 tokens (q4 naif, teacher forcing). Sauve (couche, expert) par token. Nombres seulement."""
import sys, json
sys.path.insert(0, "/home/user/Ola-Mundo/echo/monde")
import torch
from tokenizers import Tokenizer
import moteur as mt
SRC = "/home/user/monde_data/olmoe"; WORLD = "/home/user/monde_data/monde"
t = open("/home/user/Ola-Mundo/echo/validation.txt", encoding="utf-8", errors="replace").read()[:20000]
ids = Tokenizer.from_file(f"{SRC}/tokenizer.json").encode(t).ids[:129]
m = mt.Moteur(WORLD, mode="q4", cache_frac=0.0, max_ctx=136, trace=True)
nll, per = m.decode_text(ids)
tr = [(k[0], k[1]) for k in m.trace]
assert len(tr) == 128 * len(per) and len(per) == 128
json.dump(dict(n_tokens=len(per), routing=tr, bytes_per_token=sum(b for b, _ in per) / len(per)), open("v5_routing.json", "w"))
print("ok", len(tr))
