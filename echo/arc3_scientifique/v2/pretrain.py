"""Pré-entraînement du modèle d'effet sur des mondes procéduraux (echo/mondes_arc3, lecture seule).

  python3 pretrain.py --graines 1000 1060 --budget 300 --sortie checkpoints/effet_mondes.pt
Les graines de pré-entraînement (1000..) ne recoupent jamais celles de D4 (100000..). Produit un petit point de contrôle (< 1 Mo).
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import get_context
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import chemin  # noqa: E402,F401


def _collecte(args):
    seed, diff, budget = args
    import logging
    logging.disable(logging.CRITICAL)
    import adapter
    from adaptateur_monde import AdaptateurMonde
    from agent2 import Scientifique2
    env = AdaptateurMonde(f"{seed}:{diff}", budget)
    ag = Scientifique2(seed, {"theorie": False})
    adapter.jouer(env, ag)
    m = ag.effet
    return [(m.images[i], a, y, x, l) for (i, a, y, x, l) in m.ex]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--graines", type=int, nargs=2, default=[1000, 1060])
    ap.add_argument("--difficultes", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--pas", type=int, default=1500)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--sortie", default=str(ICI / "checkpoints" / "effet_mondes.pt"))
    a = ap.parse_args(argv)
    taches = [(s, a.difficultes[s % len(a.difficultes)], a.budget) for s in range(*a.graines)]
    with get_context("spawn").Pool(a.procs) as p:
        lots = p.map(_collecte, taches, chunksize=1)
    import numpy as np
    import torch
    from effet import ModeleEffet
    ex = [e for lot in lots for e in lot]
    print("exemples", len(ex), "positifs", float(np.mean([e[4] for e in ex])))
    m = ModeleEffet(0, lr=2e-3)
    m.images = [e[0] for e in ex]
    m.ex = [(i, e[1], e[2], e[3], e[4]) for i, e in enumerate(ex)]
    for k in range(a.pas // 25):
        l = m.entrainer(pas=25, lot=48, force=True)
        if k % 10 == 0:
            print(k * 25, l)
    Path(a.sortie).parent.mkdir(exist_ok=True)
    torch.save(m.etat(), a.sortie)
    print("sauvé", a.sortie)


if __name__ == "__main__":
    main()
