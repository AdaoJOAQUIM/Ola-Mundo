"""Verif 7 : comptage de parametres et octets/token pour Kimi-Linear-48B-A3B et Kimi-K3.1-Pro,
a partir des config.json lus via les outils HF MCP (valeurs copiees ci-dessous). Nombres seulement."""
import json
GB = 1e9
MX = 4.25 / 8      # MXFP4 : 4 bits + echelle uint8 par 32 = 4.25 bits
BF = 2.0
RES = 3 * GB       # non-routes chauds gardes en RAM
SPEEDS = {"NVMe 3.5": 3.5, "NVMe 7": 7.0, "SATA 0.5": 0.5, "USB 1.0": 1.0}

def kda(d, H, hd, gate_full):
    qkv = 3 * d * H * hd; o = H * hd * d
    g = d * H * hd if gate_full else (d * hd + hd * H * hd)      # porte de sortie
    f = d * hd + hd * H * hd                                      # decroissance (rang faible)
    return qkv + o + g + f + d * H + 3 * H * hd * 4 + 2 * H * hd  # + beta, convs courtes, A_log/dt

def mla(d, H, qlora, kvlora, nope, rope, v, out_gate):
    qh = nope + rope
    q = (d * qlora + qlora * H * qh) if qlora else d * H * qh
    kv = d * (kvlora + rope) + kvlora * H * (nope + v)
    o = H * v * d
    return q + kv + o + (d * H * v if out_gate else 0)

def count(c):
    d = c["d"]; Lm = c["L"] - c["dense"]
    n = {}
    n["routed_experts"] = Lm * c["E"] * 3 * c["r"] * c["I"]          # r = dimension d'entree des experts (latent si present)
    n["latent_proj"] = Lm * 2 * d * c["r"] if c["r"] != d else 0
    n["shared_experts"] = Lm * c["S"] * 3 * d * c["I"]
    n["router"] = Lm * c["E"] * d
    n["dense_mlp"] = c["dense"] * 3 * d * c["Idense"]
    n["attn_kda"] = c["nkda"] * kda(d, c["H"], c["khd"], c["gate_full"])
    n["attn_mla"] = c["nmla"] * mla(d, c["H"], c["qlora"], c["kvlora"], 128, 64, 128, c["out_gate"])
    n["dsa_indexer"] = c["ndsa"] * (c["qlora"] * c["ih"] * c["ihd"] + d * c["ihd"] + d * c["ih"]) if c.get("ndsa") else 0
    n["embeddings"] = c["V"] * d
    n["lm_head"] = c["V"] * d
    return n

K31 = dict(name="Kimi-K3.1-Pro", d=7168, L=93, dense=1, Idense=33792, E=896, k=16, I=3072, r=3584, S=2,
           H=96, khd=128, gate_full=True, nkda=69, nmla=24, qlora=1536, kvlora=512, out_gate=True,
           ndsa=24, ih=32, ihd=128, V=163840, vision=401e6)
KL = dict(name="Kimi-Linear-48B-A3B", d=2304, L=27, dense=1, Idense=9216, E=256, k=8, I=1024, r=2304, S=1,
          H=32, khd=128, gate_full=False, nkda=20, nmla=7, qlora=None, kvlora=512, out_gate=False,
          ndsa=0, V=163840, vision=0)

def report(c):
    n = count(c); Lm = c["L"] - c["dense"]
    total = sum(n.values()) + c["vision"]
    routed = n["routed_experts"]
    nonrouted = total - routed - c["vision"] - n["embeddings"] + 0      # lu a chaque token (hors table d'embedding : 1 ligne)
    active_routed = routed * c["k"] / c["E"]
    active = active_routed + nonrouted
    out = dict(model=c["name"], params_by_type_B={k: round(v / 1e9, 3) for k, v in n.items()},
               vision_B=c["vision"] / 1e9, total_B=round(total / 1e9, 2),
               nonrouted_read_every_token_B=round(nonrouted / 1e9, 2), routed_active_B=round(active_routed / 1e9, 2),
               active_B=round(active / 1e9, 2), expert_params_M=3 * c["r"] * c["I"] / 1e6,
               experts_read_per_token=Lm * c["k"])
    fm = {}
    for fmt, (bro, bnr) in {"publie (MXFP4 routes + bf16 reste)": (MX, BF), "tout a 4.25 bits": (MX, MX), "tout bf16": (BF, BF)}.items():
        r = active_routed * bro / GB
        nr = nonrouted * bnr / GB
        nr_read = max(0.0, nr - RES / GB)
        tot = r + nr_read
        fm[fmt] = dict(routed_GB=round(r, 2), nonrouted_GB=round(nr, 2), nonrouted_lu_GB=round(nr_read, 2), total_GB=round(tot, 2),
                       s_par_token={k: round(tot / v, 2) for k, v in SPEEDS.items()},
                       s_routes_seuls={k: round(r / v, 2) for k, v in SPEEDS.items()})
    out["formats"] = fm
    return out

res = dict(K31=report(K31), KL=report(KL))
# somme des fichiers safetensors K3.1 (listing HF)
a = [2341216112] + [16990911504] * 7 + [16582379656] * 2
big = 16990916912; small = 16582385056
files = a + [small if i % 4 == 0 else big for i in range(11, 93)] + [small, 4697664072, 92289328, 802448352]
assert len(files) == 96
res["K31"]["somme_fichiers_GB"] = round(sum(files) / GB, 1)
routed_mx = res["K31"]["formats"]["publie (MXFP4 routes + bf16 reste)"]
est = count(K31)
est_bytes = (est["routed_experts"] * MX + (sum(est.values()) - est["routed_experts"]) * BF + 401e6 * BF) / GB
res["K31"]["taille_estimee_depuis_params_GB"] = round(est_bytes, 1)
# comparaison avec l'agent precedent
res["agent_precedent"] = dict(K31_octets_GB=26.4, K31_NVMe35_s=7.7, KL_octets_GB=0.8, KL_NVMe35_s=0.3,
    expert_K31_Mparams=33.7, note="33.7M = 2.78T/(92*896) : retro-calcule depuis le total, sans non-routes ; la valeur lue (latent 3584 x 3072 x 3) est 33.03M")
res["K31"]["total_si_expert_33.7M"] = round(92 * 896 * 33.7e6 / 1e12, 3)
json.dump(res, open("v7_extrapolation.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
