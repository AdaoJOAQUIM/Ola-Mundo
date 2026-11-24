"""Pont vers la perception de la v1 (echo/arc3_scientifique/percep.py) : chargée par chemin, sous un nom privé,
sans modifier sys.path (évite toute collision avec d'autres modules nommés « percep »)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ICI = Path(__file__).resolve().parent
V1 = _ICI.parent.parent / "arc3_scientifique"


def charger(nom: str, fichier: str):
    spec = importlib.util.spec_from_file_location(nom, V1 / fichier)
    m = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[nom] = m
    spec.loader.exec_module(m)
    return m


percep_v1 = charger("retro_v1_percep", "percep.py")
