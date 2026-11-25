"""Agrège runs/j3_{train,val}_{500,5000}_{jeu}.json en resultats_j3.json (le texte est écrit à la main dans resultats_j3.md)."""
import json, statistics
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "runs"
GROUPES = {"train": ["cn04", "ls20", "sc25"], "val": ["tn36", "ka59"]}


def charger(grp, budget):
    out = {}
    for j in GROUPES[grp]:
        f = R / f"j3_{grp}_{budget}_{j}.json"
        if f.exists() and f.stat().st_size:
            out[j] = json.load(open(f))
    return out


def agreger(d):
    res = {"jeux": {}}
    tn = tt = 0
    rap = []
    for j, runs in d.items():
        par = {r["graine"]: {"niveaux": r["niveaux"], "total": r["niveaux_total"], "actions": r["actions"],
                             "niveaux_actions": r["niveaux_actions"], "cpu_s": r["cpu_s"]} for r in runs}
        n = sum(r["niveaux"] for r in runs); t = sum(r["niveaux_total"] for r in runs)
        res["jeux"][j] = {"par_graine": par, "niveaux": n, "total": t, "part": round(n / t, 4) if t else None}
        tn += n; tt += t
        for r in runs:
            ref = r.get("niveaux_reference") or []
            for i in range(r["niveaux"]):
                if i < len(ref) and ref[i]:
                    rap.append(r["niveaux_actions"][i] / ref[i])
    res["niveaux"], res["total"] = tn, tt
    res["part"] = round(tn / tt, 4) if tt else None
    res["actions_sur_reference_humaine"] = {"n_niveaux_resolus": len(rap),
        "rapport_agrege": round(sum(rap) / len(rap), 3) if rap else None,
        "mediane": round(statistics.median(rap), 3) if rap else None}
    return res


if __name__ == "__main__":
    out = {}
    for b in (500, 5000):
        for g in ("train", "val"):
            d = charger(g, b)
            if d:
                out[f"{g}_{b}"] = agreger(d)
        # DEV total
        tr, va = out.get(f"train_{b}"), out.get(f"val_{b}")
        if tr and va:
            n, t = tr["niveaux"] + va["niveaux"], tr["total"] + va["total"]
            out[f"dev_{b}"] = {"niveaux": n, "total": t, "part": round(n / t, 4)}
    out["seuil_j3"] = 0.5
    p = Path(__file__).resolve().parent.parent / "resultats_j3.json"
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: (v.get("part") if isinstance(v, dict) else v) for k, v in out.items()}, ensure_ascii=False))
