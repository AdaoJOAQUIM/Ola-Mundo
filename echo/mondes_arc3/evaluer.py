#!/usr/bin/env python3
"""Évalue un agent sur des mondes générés, avec le protocole du harnais (courtier + garde, 500 actions).

Usage :
  python evaluer.py --agent cerveau --mondes 20 --graine 0
  python evaluer.py --agent aleatoire:Aleatoire --mondes 20 --graine 0 --difficulte 2
  --agent : nom de module (cherché dans echo/arc3, echo/arc3_scientifique, ...), `module:Classe`,
            ou chemin d'un .py. La classe est construite par Classe(graine) et doit avoir .choisir(obs).
Rapporte les niveaux résolus et les actions dépensées par rapport à l'optimum de l'oracle.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
from multiprocessing import get_context
from pathlib import Path

ICI = Path(__file__).resolve().parent
ECHO = ICI.parent
if str(ICI) not in sys.path:
    sys.path.insert(0, str(ICI))
DOSSIERS_AGENTS = [ECHO / "arc3", ECHO / "arc3_scientifique", ECHO / "arc3_joueur", ICI]

from env import MAX_ACTIONS, jouer  # noqa: E402


def charger_fabrique(spec: str):
    """spec -> fonction graine -> politique."""
    nom, _, classe = spec.partition(":")
    if nom.endswith(".py") or "/" in nom:
        p = Path(nom).resolve()
        sys.path.insert(0, str(p.parent))
        mod = importlib.import_module(p.stem)
    else:
        for d in DOSSIERS_AGENTS:
            if (d / f"{nom}.py").exists():
                sys.path.insert(0, str(d))
                break
        mod = importlib.import_module(nom)
    if classe:
        cls = getattr(mod, classe)
        return lambda graine: cls(graine)
    if hasattr(mod, "fabrique"):
        return mod.fabrique
    cands = [v for v in vars(mod).values()
             if isinstance(v, type) and v.__module__ == mod.__name__ and hasattr(v, "choisir")]
    if not cands:
        raise SystemExit(f"aucune classe avec .choisir dans {spec}")
    cls = cands[0]
    return lambda graine: cls(graine)


def _tache(args):
    spec, seed, d, budget, graine_agent = args
    logging.disable(logging.CRITICAL)
    fab = charger_fabrique(spec)
    res, _ = jouer(seed, d, lambda: fab(graine_agent), max_actions=budget)
    res["agent"] = spec
    return res


def lancer(spec, seeds, difficultes, budget=MAX_ACTIONS, procs=2, graine_agent=0):
    taches = [(spec, s, d, budget, graine_agent) for s, d in zip(seeds, difficultes)]
    if procs <= 1:
        return [_tache(t) for t in taches]
    with get_context("fork").Pool(procs) as p:
        return p.map(_tache, taches, chunksize=1)


def resume(res):
    niv = sum(r["niveaux"] for r in res)
    tot = sum(r["niveaux_total"] for r in res)
    opt_res = 0
    act_res = 0
    for r in res:
        for i in range(r["niveaux"]):
            opt_res += r["niveaux_reference"][i]
            act_res += r["niveaux_actions"][i]
    return {
        "mondes": len(res),
        "niveaux_resolus": niv,
        "niveaux_total": tot,
        "taux": round(niv / tot, 4) if tot else 0.0,
        "mondes_gagnes": sum(1 for r in res if r["etat"] == "WIN"),
        "actions_sur_resolus": act_res,
        "optimum_sur_resolus": opt_res,
        "efficacite_vs_optimum": round(opt_res / act_res, 4) if act_res else None,
        "actions_totales": sum(r["actions"] for r in res),
        "cpu_s": round(sum(r["cpu_s"] for r in res), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--mondes", type=int, default=20)
    ap.add_argument("--graine", type=int, default=0, help="première graine de monde")
    ap.add_argument("--difficulte", default="mix", help="1 à 5 ou mix (cycle 1..5)")
    ap.add_argument("--budget", type=int, default=MAX_ACTIONS)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [a.graine + i for i in range(a.mondes)]
    ds = [1 + i % 5 for i in range(a.mondes)] if a.difficulte == "mix" else [int(a.difficulte)] * a.mondes
    res = lancer(a.agent, seeds, ds, a.budget, a.procs, a.graine)
    r = resume(res)
    for x in res:
        print(f"{x['jeu']} {x['but']:7s} niveaux {x['niveaux']}/{x['niveaux_total']} {x['etat']:10s} "
              f"actions {x['actions']:3d} optimum_total {sum(x['niveaux_reference'])}")
    print(json.dumps(r, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps({"resume": r, "mondes": res}, indent=1))


if __name__ == "__main__":
    main()
