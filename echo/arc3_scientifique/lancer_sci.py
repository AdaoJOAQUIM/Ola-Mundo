"""Exécuteur propre au Scientifique (n'édite rien dans echo/arc3 ; réutilise son harnais, courtier et garde).

Protocole identique à echo/arc3/lancer.py : même découpe DEV/HELD, 500 actions, graine d'environnement 0,
2 processus, enregistrements du SDK dans un dossier runs/ local (ignoré par git).

Usage (python du venv arc3) :
  python lancer_sci.py essai [jeux...]      # une partie par jeu DEV, résumé
  python lancer_sci.py dev [variante]       # 5 jeux DEV (variantes : plein, alea, reactif, sans_prior)
  python lancer_sci.py held                 # 20 jeux HELD, agent gelé, UNE seule fois
"""
from __future__ import annotations

import json
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
sys.path.insert(0, str(ICI.parent / "arc3"))      # harnais (courtier + garde) : réutilisé tel quel
RUNS = ICI / "runs"
RUNS.mkdir(exist_ok=True)

JEUX = sorted("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30".split())
DEV = random.Random(2026).sample(JEUX, 5)         # = lancer.py
HELD = [j for j in JEUX if j not in DEV]
NB_PROCS = 2

VARIANTES = {
    "plein": {"options": {}, "prior": True},
    "alea": {"options": {"explo": "alea"}, "prior": True},
    "reactif": {"options": {"plan": False}, "prior": True},
    "sans_prior": {"options": {}, "prior": False},
}


def charger_prior(utiliser=True):
    from theorie import Prior
    p = ICI / "prior_gele.json"
    if utiliser and p.exists():
        return Prior.depuis_json(json.loads(p.read_text()))
    return Prior()


def _worker(args):
    import run
    jeu, variante, graine = args
    return run._un((jeu, 500, graine, variante, str(ICI / "prior_gele.json"), "adapter:AdaptateurARC3", str(RUNS)))


def lancer(taches):
    with Pool(NB_PROCS) as p:
        return p.map(_worker, taches, chunksize=1)


def resume(res):
    return {"niveaux": sum(r["niveaux"] for r in res), "total": sum(r["niveaux_total"] or 0 for r in res)}


def ecrire(nom, obj):
    (ICI / nom).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "essai"
    if cmd == "essai":
        jeux = sys.argv[2:] or DEV
        out = lancer([(j, "plein", 0) for j in jeux])
        for r in out:
            print(r["prefixe"], r["etat"], f"{r['niveaux']}/{r['niveaux_total']}", r["actions"], r["niveaux_actions"],
                  "cpu", r["cpu_s"], r["stats_agent"], r["types_map"])
    elif cmd == "dev":
        var = sys.argv[2] if len(sys.argv) > 2 else "plein"
        graines = (0, 1, 2)
        out = lancer([(j, var, g) for j in DEV for g in graines])
        ecrire(f"res_dev_{var}.json", out)
        print(var, resume(out))
    elif cmd == "held":
        raise SystemExit("HELD reporté à l'évaluation finale (voir criteres.md, amendement).")
        if (ICI / "res_held.json").exists():
            raise SystemExit("HELD déjà ouvert : refus de rejouer.")
        out = lancer([(j, "plein", 0) for j in HELD])
        ecrire("res_held.json", out)
        print("HELD", resume(out))
