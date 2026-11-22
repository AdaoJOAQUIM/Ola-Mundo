"""Verif 5 : LRU (cache cyclique), puis cache epingle par frequence (experts les plus utilises de chaque couche). Nombres seulement."""
import json, collections
from collections import OrderedDict
d = json.load(open("v5_routing.json")); tr = [tuple(x) for x in d["routing"]]
NT = 128; per = 128; CH = 3538944; TOT = 1024 * CH
tokens = [tr[i * per:(i + 1) * per] for i in range(NT)]
assert all(len(set(t)) == 128 for t in tokens)           # 128 chunks distincts par token
def lru(trace, slots):
    c = OrderedDict(); h = 0
    for k in trace:
        if k in c: h += 1; c.move_to_end(k)
        else:
            c[k] = 1
            if len(c) > slots: c.popitem(last=False)
    return h / len(trace)
def pinned(train, test, slots, per_layer=True):
    cnt = [collections.Counter() for _ in range(16)]
    for t in train:
        for L, e in t: cnt[L][e] += 1
    pin = set()
    if per_layer:
        q, r = divmod(slots, 16)
        for L in range(16):
            n = q + (1 if L < r else 0)
            pin |= {(L, e) for e, _ in cnt[L].most_common(n)}
    else:
        allc = collections.Counter({(L, e): c for L in range(16) for e, c in cnt[L].items()})
        pin = {k for k, _ in allc.most_common(slots)}
    hits = sum(1 for t in test for k in t if k in pin); return hits / sum(len(t) for t in test)
flat = [k for t in tokens for k in t]
res = dict(distinct_chunks_per_token=128, bytes_per_token=128 * CH, frac_of_expert_bytes_per_token=128 * CH / TOT, rows=[])
# distance de reutilisation (nb de chunks distincts entre deux usages)
last = {}; dist = []
for i, k in enumerate(flat):
    if k in last: dist.append(len(set(flat[last[k] + 1:i])))
    last[k] = i
dist.sort(); res["reuse_distance_median"] = dist[len(dist) // 2]; res["reuse_fraction_of_uses"] = len(dist) / len(flat)
res["frac_experts_same_as_prev_token"] = sum(len(set(tokens[i]) & set(tokens[i - 1])) for i in range(1, NT)) / (127 * 128)
for name, frac in (("0.25%", .0025), ("1%", .01), ("5%", .05), ("25%", .25)):
    slots = int(frac * TOT // CH)
    row = dict(cache=name, slots=slots, lru_hit=lru(flat, slots),
               pinned_oracle_per_layer=pinned(tokens, tokens, slots),          # epingle sur la meme trace : borne haute
               pinned_heldout_per_layer=pinned(tokens[:64], tokens[64:], slots),  # epingle sur 64 tokens, teste sur 64 autres
               pinned_heldout_global=pinned(tokens[:64], tokens[64:], slots, False),
               pinned_heldout_per_layer_reverse=pinned(tokens[64:], tokens[:64], slots))
    row["bytes_per_token_heldout_MB"] = 128 * CH * (1 - row["pinned_heldout_per_layer"]) / 1e6
    res["rows"].append(row)
# reference uniforme : 8/64 = 12.5 % ; taux a l'aleatoire pour 'slots' epingles = slots/1024
json.dump(res, open("v5_result.json", "w"), indent=1); print(json.dumps(res, indent=1))
