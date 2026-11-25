"""Accès aux vrais jeux par l'adaptateur de la v1, avec garde de découpe.

La découpe DEV/HELD est IMPORTÉE de echo/arc3/lancer.py (jamais recopiée). Toute demande d'un jeu HELD lève une erreur.
Les dossiers des jeux ne sont jamais ouverts ici : seul l'adaptateur (courtier + SDK) y accède pour jouer.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
ECHO = RACINE.parent


class JeuInterdit(RuntimeError):
    pass


def _lancer():
    spec = importlib.util.spec_from_file_location("lancer_decoupe", ECHO / "arc3" / "lancer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_L = _lancer()
JEUX = tuple(_L.JEUX)
DEV = tuple(_L.DEV)
HELD = tuple(_L.HELD)
# Sous-découpe pré-enregistrée (critères J3) : entraînement / validation, parmi DEV importé.
DEV_TRAIN = tuple(j for j in DEV if j in ("cn04", "ls20", "sc25"))
DEV_VAL = tuple(j for j in DEV if j not in DEV_TRAIN)


def garde(jeu_id: str) -> str:
    """Lève JeuInterdit si `jeu_id` (préfixe ou id complet) désigne un jeu HELD ; renvoie le préfixe DEV sinon."""
    j = str(jeu_id)
    for h in HELD:
        if j.startswith(h) or h.startswith(j) and len(j) >= 2:
            raise JeuInterdit(f"jeu HELD interdit : {jeu_id}")
    if not any(j.startswith(d) for d in DEV):
        raise JeuInterdit(f"jeu hors DEV : {jeu_id}")
    return j


def ouvrir(jeu_id: str, budget: int):
    """Adaptateur v1 pour un jeu DEV (jamais HELD)."""
    garde(jeu_id)
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(ECHO / "arc3_scientifique"))
    from adapter import AdaptateurARC3
    import harnais  # noqa: F401  (ajoute le courtier à sys.path)
    import arc3_local_broker as b
    b.MAX_ACTIONS = max(b.MAX_ACTIONS, int(budget))      # budget large : plafond relevé en mémoire, aucun fichier modifié
    return AdaptateurARC3(jeu_id, budget)


def jouer(jeu_id: str, fabrique, budget: int):
    """Joue un jeu DEV avec un agent (`fabrique()` -> objet à .choisir(obs)). Renvoie le résumé de l'adaptateur."""
    import time
    env = ouvrir(jeu_id, budget)
    ag = fabrique()
    t0, w0 = time.process_time(), time.time()
    obs = env.observe()
    while obs["terminal"] == "ACTIVE":
        obs = env.act(ag.choisir(obs))
    res = env.resume()
    res["cpu_s"] = round(time.process_time() - t0, 1)
    res["mur_s"] = round(time.time() - w0, 1)
    return res, ag
