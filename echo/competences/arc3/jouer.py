"""Lance l'élève (cerveau + bassin de compétences) sur des jeux, avec le harnais de echo/arc3.
Protocole identique à la baseline : 500 actions, graine d'environnement 0, prieurs gelés.
Les budgets sont des nombres d'actions."""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ICI = Path(__file__).resolve().parent
ARC3 = ICI.parent.parent / "arc3"
sys.path.insert(0, str(ICI))
sys.path.insert(0, str(ARC3))
RUNS = ICI / "runs"
RUNS.mkdir(exist_ok=True)
TRAJ = ICI / "traj"

DEV_TRAIN = ["ls20", "cn04", "ka59"]
DEV_GATE = ["tn36", "sc25"]
JEUX = sorted("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30".split())
HELD = [j for j in JEUX if j not in DEV_TRAIN + DEV_GATE]


def prieurs_geles():
    p = ARC3 / "prieurs_geles.json"
    return json.loads(p.read_text()) if p.exists() else None


def _worker(args):
    jeu, graine, bassin, prieurs, enreg = args
    os.chdir(RUNS)
    import logging
    logging.disable(logging.CRITICAL)
    import numpy as np
    import harnais
    import eleve

    class Enreg(eleve.Eleve):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.frames, self.acts, self.nivs = [], [], []

        def choisir(self, obs):
            self.frames.append(np.array(obs["frame"][-1], dtype=np.uint8))
            self.nivs.append(int(obs["levels_completed"]))
            r = super().choisir(obs)
            self.acts.append((r["name"], r["data"].get("x", -1), r["data"].get("y", -1)))
            return r

    fab = (lambda: Enreg(graine, prieurs, bassin)) if enreg else (lambda: eleve.Eleve(graine, prieurs, bassin))
    res, pol = harnais.jouer(jeu, fab)
    res.update({"prefixe": jeu, "graine": graine, "competences": pol.bilan(),
                "actions_competences": pol.stats["actions_competences"],
                "actions_cerveau": pol.stats["actions_cerveau"],
                "trace": pol.trace[:2000]})
    if enreg:
        TRAJ.mkdir(exist_ok=True)
        np.savez_compressed(TRAJ / f"{jeu}_s{graine}.npz", frames=np.array(pol.frames), nivs=np.array(pol.nivs),
                            acts=np.array([a[0] for a in pol.acts]),
                            xy=np.array([[a[1], a[2]] for a in pol.acts]),
                            comp=np.array([next((t[1] for t in pol.trace if t[0] == i), "") for i in range(len(pol.acts))]))
    return res


def jouer_lot(taches, nproc=2):
    """taches : liste de (jeu, graine, bassin, prieurs, enreg)."""
    with Pool(nproc) as p:
        return p.map(_worker, taches, chunksize=1)


def resume(res):
    return {"jeu": res["prefixe"], "graine": res["graine"], "niveaux": res["niveaux"], "etat": res["etat"],
            "niveaux_actions": res["niveaux_actions"], "actions": res["actions"]}
