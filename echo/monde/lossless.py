"""Critere (a) : mode bf16 depuis le disque, sans LOD ni culling, vs reference transformers fp32.

usage : python3 lossless.py moteur   puis   python3 lossless.py ref
"""
import json, resource, sys, time
import torch
from tokenizers import Tokenizer

WORLD = "/home/user/monde_data/monde"
SRC = "/home/user/monde_data/olmoe"
TMP = "/home/user/monde_data/lossless_moteur.pt"
OUT = "/home/user/Ola-Mundo/echo/monde/resultats_lossless.json"
PROMPTS = ["The capital of France is", "Once upon a time, in a small village by the sea,",
           "def fibonacci(n):\n    "]
N_NEW = 32


def rss_peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


if sys.argv[1] == "moteur":
    import moteur as mt
    tk = Tokenizer.from_file(f"{SRC}/tokenizer.json")
    m = mt.Moteur(WORLD, mode="bf16", cache_frac=0.01, prefetch=False, max_ctx=128)
    res = []
    t0 = time.time()
    for p in PROMPTS:
        ids = tk.encode(p).ids
        m.reader.bytes = m.reader.ops = 0
        toks, logs = m.generate(ids, N_NEW)
        res.append(dict(prompt_ids=ids, tokens=toks, logits=logs, bytes=m.reader.bytes, ops=m.reader.ops))
    torch.save(res, TMP)
    print("moteur ok", time.time() - t0, "s ; pic RSS", rss_peak_mb(), "Mo ; cache max", m.cache.max_used / 1e6,
          "Mo / budget", m.cache.budget / 1e6, "Mo ; budget moteur", m.budget.total() / 1e6, "Mo")
    json.dump(dict(rss_peak_mb=rss_peak_mb(), cache_max_mb=m.cache.max_used / 1e6, cache_budget_mb=m.cache.budget / 1e6,
                   budget_total_mb=m.budget.total() / 1e6), open(TMP + ".json", "w"))
else:
    import ref_hf
    res = torch.load(TMP)
    seqs = [r["prompt_ids"] + r["tokens"] for r in res]
    t0 = time.time()
    refs = ref_hf.logits_reference(SRC, seqs)
    out = dict(prompts=[], max_abs_logit_diff=0.0, all_tokens_identical=True, n_tokens=N_NEW, ref_seconds=time.time() - t0)
    for r, ref in zip(res, refs):
        P = len(r["prompt_ids"])
        rl = ref[P - 1:P - 1 + N_NEW]
        ref_tokens = rl.argmax(-1).tolist()
        same = ref_tokens == r["tokens"]
        d = float((rl - r["logits"]).abs().max())
        out["prompts"].append(dict(n_prompt_tokens=P, identical=same, max_abs_logit_diff=d,
                                   bytes_lus=r["bytes"], lectures=r["ops"]))
        out["max_abs_logit_diff"] = max(out["max_abs_logit_diff"], d)
        out["all_tokens_identical"] &= same
    out["pass"] = bool(out["all_tokens_identical"] and out["max_abs_logit_diff"] <= 1e-2)
    out["moteur"] = json.load(open(TMP + ".json"))
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
