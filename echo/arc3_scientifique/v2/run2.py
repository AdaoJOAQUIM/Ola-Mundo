"""Point d'entrée v2 :
  python run2.py --jeux JEU1 JEU2 --budget 500 --graines 0 1 2 --variante plein --sortie res.json [--env adapter:AdaptateurARC3]
Variantes : plein, sans_effet, sans_graphe, sans_theorie, sans_valeur, v1 (agent v1 inchangé), alea.
Python : celui du système (numpy+torch) ; chemin.py ajoute les paquets du SDK en fin de sys.path.
Ne choisit jamais HELD : l'appelant donne les jeux.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import chemin  # noqa: E402,F401

VARIANTES = {
    "plein": {},
    "sans_effet": {"effet": False},
    "sans_graphe": {"graphe": False},
    "sans_theorie": {"theorie": False},
    "sans_valeur": {"valeur": False},
}


def _un(args):
    jeu, budget, graine, variante, env_spec, rundir, pretrain = args
    os.makedirs(rundir, exist_ok=True)
    os.chdir(rundir)
    import logging
    logging.disable(logging.CRITICAL)
    import adapter
    mod, cls = env_spec.split(":")
    env = getattr(importlib.import_module(mod), cls)(jeu, budget)
    if variante == "v1":
        import agent as agent1
        from theorie import Prior
        prior = Prior()
        pp = Path(chemin.V1) / "prior_gele.json"
        if pp.exists():
            prior = Prior.depuis_json(json.loads(pp.read_text()))
        ag = agent1.Scientifique(graine, prior, {})
    elif variante == "alea":
        ag = Alea(graine)
    else:
        from agent2 import Scientifique2
        modele = None
        if pretrain and variante != "sans_effet":
            from effet import ModeleEffet
            import torch
            modele = ModeleEffet(graine)
            modele.charger(torch.load(pretrain))
        ag = Scientifique2(graine, dict(VARIANTES[variante]), modele)
    res = adapter.jouer(env, ag)
    res.update({"prefixe": jeu, "variante": variante, "graine": graine, "budget": budget,
                "stats_agent": getattr(ag, "stats", {}), "niveaux_actions_agent": getattr(ag, "niveaux_actions", [])})
    return res


class Alea:
    def __init__(self, graine):
        import numpy as np
        self.rng = np.random.default_rng(graine)

    def choisir(self, obs):
        av = [int(a) for a in obs["available_actions"] if int(a) != 0]
        a = int(self.rng.choice(av))
        if a == 6:
            return {"name": "ACTION6", "data": {"x": int(self.rng.integers(64)), "y": int(self.rng.integers(64))}}
        return {"name": f"ACTION{a}", "data": {}}


def lancer(jeux, budget, graines, variante, env="adapter:AdaptateurARC3", procs=2, rundir=None, pretrain=None):
    rundir = rundir or str(ICI / "runs" / f"{variante}_{budget}")
    taches = [(j, budget, g, variante, env, rundir, pretrain) for g in graines for j in jeux]
    with get_context("spawn").Pool(procs) as p:
        return p.map(_un, taches, chunksize=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jeux", nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--graines", type=int, nargs="+", default=[0])
    ap.add_argument("--variante", default="plein", choices=list(VARIANTES) + ["v1", "alea"])
    ap.add_argument("--sortie", default=None)
    ap.add_argument("--env", default="adapter:AdaptateurARC3")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--rundir", default=None)
    ap.add_argument("--pretrain", default=None)
    a = ap.parse_args(argv)
    out = lancer(a.jeux, a.budget, a.graines, a.variante, a.env, a.procs, a.rundir, a.pretrain)
    for r in out:
        print(r["prefixe"], r["graine"], r["etat"], f"{r['niveaux']}/{r['niveaux_total']}", r["actions"], r["niveaux_actions"], f"cpu={r['cpu_s']}")
    print("TOTAL", sum(r["niveaux"] for r in out), "/", sum(r["niveaux_total"] or 0 for r in out))
    if a.sortie:
        Path(a.sortie).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return out


if __name__ == "__main__":
    main()
