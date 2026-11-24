"""Agrège runs/campagne/*.json en resultats_v2.json (et tableaux pour resultats_v2.md)."""
import json
from collections import defaultdict
from pathlib import Path

ICI = Path(__file__).resolve().parent
C = ICI / "runs" / "campagne"
res = {}
for f in sorted(C.glob("*.json")):
    out = json.loads(f.read_text())
    par_jeu = defaultdict(list)
    for r in out:
        par_jeu[r["prefixe"]].append(r["niveaux"])
    ref = {}
    apl = defaultdict(list)
    for r in out:
        ref[r["prefixe"]] = r.get("niveaux_reference")
        na = r.get("niveaux_actions") or []
        for k, a in enumerate(na):
            if k < r["niveaux"] and a:
                apl[r["prefixe"]].append(a)
    res[f.stem] = {"total": sum(r["niveaux"] for r in out), "niveaux_possibles": sum(r["niveaux_total"] or 0 for r in out),
                   "par_jeu": {k: v for k, v in par_jeu.items()} if not f.stem.startswith("monde") else None,
                   "mondes_resolus_>=1": sum(1 for r in out if r["niveaux"] > 0) if f.stem.startswith("monde") else None,
                   "n_parties": len(out), "references": ref if not f.stem.startswith("monde") else None,
                   "actions_niveaux_resolus": dict(apl) if not f.stem.startswith("monde") else None,
                   "cpu_s_moyen": round(sum(r.get("cpu_s", 0) for r in out) / max(len(out), 1), 1)}
(ICI / "resultats_v2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
for k, v in res.items():
    print(k, v["total"], "/", v["niveaux_possibles"], v["par_jeu"] or v["mondes_resolus_>=1"])
