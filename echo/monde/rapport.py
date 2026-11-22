"""Agrege les resultats bruts -> resultats_monde.json (+ tableaux utilises dans le .md)."""
import glob, json, os, pickle
import moteur as mt

RES = "/home/user/Ola-Mundo/echo/monde/resultats_bruts"
TR = "/home/user/monde_data/traces"
OUTJ = "/home/user/Ola-Mundo/echo/monde/resultats_monde.json"
FRACS = [0.0025, 0.01, 0.05, 0.25]
GB = 1e9
# disques (debit, latence). NVMe : donnees de la consigne. Latences SATA/USB : HYPOTHESES (non fournies).
DISQUES = {"NVMe": (3.5 * GB, 0.1e-3), "SATA": (0.5 * GB, 0.2e-3), "USB": (1.0 * GB, 0.5e-3)}


def load(kind, name):
    p = f"{RES}/{kind}_{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def temps_token(bytes_, ops, compute, disque, recouvrement=False):
    bw, lat = DISQUES[disque]
    io = bytes_ / bw + ops * lat
    return max(io, compute) if recouvrement else io + compute


def sim_cache(name, sizes, total_bytes, fracs=FRACS):
    tr = pickle.load(open(f"{TR}/{name}.pkl", "rb"))
    n_tok = len(tr) // max(1, 1)  # remplace plus bas
    out = {}
    for f in fracs:
        hits, uses, rd = mt.simulate_lru(tr, sizes, f * total_bytes)
        out[str(f)] = dict(budget_mb=f * total_bytes / 1e6, hit_rate=hits / uses, bytes_lus=rd, acces=uses)
    return out


def main():
    R = {"lossless": json.load(open("/home/user/Ola-Mundo/echo/monde/resultats_lossless.json"))}
    dec = {os.path.basename(p)[7:-5]: json.load(open(p)) for p in glob.glob(f"{RES}/decode_*.json")}
    ppl = {os.path.basename(p)[4:-5]: json.load(open(p)) for p in glob.glob(f"{RES}/ppl_*.json")}
    R["decode"], R["ppl"] = dec, ppl
    sizes = dec["naif"]["chunk_sizes"]
    total_q4 = 16 * 64 * sizes["q4"]
    R["total_expert_bytes_q4"] = total_q4
    R["sim_cache"] = {}
    for name, d in dec.items():
        steps = d["steps"]
        s = sim_cache(name.replace("/", "_"), sizes, total_q4)
        for v in s.values():
            v["bytes_par_token"] = v["bytes_lus"] / steps
        R["sim_cache"][name] = s
    json.dump(R, open(OUTJ, "w"), indent=1)
    return R


if __name__ == "__main__":
    main()
