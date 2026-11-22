"""Verif (a) cote moteur : mode bf16 sans perte, 1 prompt, 16 tokens gloutons. Sauve tokens+logits."""
import sys, time, json, resource
sys.path.insert(0, "/home/user/Ola-Mundo/echo/monde")
import torch
from tokenizers import Tokenizer
import moteur as mt
SRC = "/home/user/monde_data/olmoe"; WORLD = "/home/user/monde_data/monde"
ids = Tokenizer.from_file(f"{SRC}/tokenizer.json").encode("The capital of France is").ids
m = mt.Moteur(WORLD, mode="bf16", cache_frac=0.01, prefetch=False, max_ctx=64)
t0 = time.time(); toks, logs = m.generate(ids, 16)
torch.save(dict(ids=ids, toks=toks, logits=logs), "/tmp/verif_engine.pt")
print("engine ok", time.time() - t0, "s", toks, "rss_mb", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
