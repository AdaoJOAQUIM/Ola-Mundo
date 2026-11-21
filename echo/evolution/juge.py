"""Juge : mesure en bits par caractère (bpc) la qualité d'un génome.

bpc = moyenne de -log2 p(symbole réel) après normalisation des scores.
Ce fichier est épinglé par SHA-256 : le moteur ne doit jamais pouvoir le modifier.
"""
import ast

import numpy as np

from garde_fous import verifier

EPS_LIGNE = 1e-30     # somme minimale d'une ligne de scores
PLANCHER_P = 1e-10    # probabilité minimale créditée au vrai symbole (perte max ~33 bits)


def div(a, b):
    """Division protégée : x / 0 = 0."""
    with np.errstate(all="ignore"):
        return np.where(b == 0, 0.0, a / np.where(b == 0, 1.0, b))


def compiler(source):
    """Compile un génome DÉJÀ vérifié et retourne la fonction `estimer`."""
    ns = {"np": np, "div": div, "__builtins__": {"range": range}}
    exec(compile(ast.parse(source), "<genome>", "exec"), ns)
    return ns["estimer"]


def juger(source, jeu, pertes=False):
    """Évalue un génome sur un jeu {'n','N','d','u1','u2','y'}.

    Retourne {'ok': True, 'bpc': x[, 'pertes': tableau]} ou {'ok': False, 'raison': ..}.
    """
    ok, code, detail = verifier(source)
    if not ok:
        return {"ok": False, "raison": code, "detail": detail}
    try:
        f = compiler(source)
        K = jeu["n"].shape[0] - 1
        with np.errstate(all="ignore"):
            p = f(jeu["n"], jeu["N"], jeu["d"], jeu["u1"], jeu["u2"], K)
    except MemoryError:
        raise
    except Exception as e:
        if type(e).__name__ == "DelaiDepasse":
            raise
        return {"ok": False, "raison": "erreur_execution", "detail": type(e).__name__}
    T, A = jeu["n"].shape[1], jeu["n"].shape[2]
    if not isinstance(p, np.ndarray) or p.shape != (T, A):
        return {"ok": False, "raison": "forme_incorrecte", "detail": str(getattr(p, "shape", type(p)))}
    if not np.all(np.isfinite(p)):
        return {"ok": False, "raison": "non_fini", "detail": ""}
    if p.min() < 0:
        return {"ok": False, "raison": "negatif", "detail": ""}
    p = p.astype(np.float64)
    s = p.sum(axis=1)
    nul = s <= EPS_LIGNE
    if nul.any():
        p[nul] = 1.0   # lignes nulles : lissage uniforme
        s = p.sum(axis=1)
    y = jeu["y"]
    pv = p[np.arange(T), y] / s
    pv = np.maximum(pv, PLANCHER_P)
    pert = -np.log2(pv)
    res = {"ok": True, "bpc": float(pert.mean())}
    if pertes:
        res["pertes"] = pert
    return res
