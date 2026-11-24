"""Point d'entrée documenté (agent gelé) :

  python run.py --jeux ls20 ka59 --budget 500 --graine 0 [--variante plein|alea|reactif|sans_prior]
                [--prior prior_gele.json] [--sortie res.json] [--env adaptateur.module:Classe]

--env permet de brancher un autre environnement respectant l'interface de adapter.py
(par défaut : AdaptateurARC3). Les jeux sont donnés à l'adaptateur tels quels.
Ce script ne choisit jamais HELD lui-même : c'est à l'appelant de fournir les jeux.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
VARIANTES = {"plein": {}, "alea": {"explo": "alea"}, "reactif": {"plan": False}, "sans_prior": {}}


def _un(args):
    jeu, budget, graine, variante, prior_path, env_spec, rundir = args
    os.makedirs(rundir, exist_ok=True)
    os.chdir(rundir)
    import logging
    logging.disable(logging.CRITICAL)
    import adapter
    import agent
    from theorie import Prior
    prior = Prior()
    if variante != "sans_prior" and prior_path and Path(prior_path).exists():
        prior = Prior.depuis_json(json.loads(Path(prior_path).read_text()))
    mod, cls = env_spec.split(":")
    env = getattr(importlib.import_module(mod), cls)(jeu, budget)
    ag = agent.Scientifique(graine, prior, dict(VARIANTES[variante]))
    res = adapter.jouer(env, ag)
    res.update({"prefixe": jeu, "variante": variante, "graine": graine, "stats_agent": ag.stats,
                "etiquettes": ag.exporter_etiquettes()})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jeux", nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--variante", default="plein", choices=list(VARIANTES))
    ap.add_argument("--prior", default=str(ICI / "prior_gele.json"))
    ap.add_argument("--sortie", default=None)
    ap.add_argument("--env", default="adapter:AdaptateurARC3")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--rundir", default=str(ICI / "runs"))
    a = ap.parse_args(argv)
    taches = [(j, a.budget, a.graine, a.variante, a.prior, a.env, a.rundir) for j in a.jeux]
    with Pool(a.procs) as p:
        out = p.map(_un, taches, chunksize=1)
    for r in out:
        print(r["prefixe"], r["etat"], f"{r['niveaux']}/{r['niveaux_total']}", r["actions"], r["niveaux_actions"])
    print("TOTAL", sum(r["niveaux"] for r in out), "/", sum(r["niveaux_total"] or 0 for r in out))
    if a.sortie:
        Path(a.sortie).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
