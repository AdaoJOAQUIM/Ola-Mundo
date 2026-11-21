"""Mélanges bayésiens d'experts, strictement causaux.

BMA : poids a posteriori w_i(t) ∝ π_i Π_{s<t} p_i(y_s). La probabilité de y_t est Σ_i w_i(t) p_i(y_t) : on
MOYENNE sur tous les modèles au lieu d'en choisir un. Regret total ≤ log2(1/π_i) bits face à n'importe quel
membre i (identité). Fixed-share (Herbster et Warmuth 1998) : après chaque symbole, w ← (1-α) w + α/M.

On enregistre en plus, pour lire la convergence : l'entropie du postérieur à chaque pas et la masse
postérieure par groupe (ordre, β, famille) tous les `pas_snap` caractères.
"""
import math

import numpy as np

PLANCHER_P = 1e-10
GRILLE_ALPHA = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def melanger(P, alpha=0.0, prior=None, groupes=None, pas_snap=500):
    """P : (T, M) probabilités du vrai symbole (lignes = positions). Retourne dict.

    pertes : -log2 du mélange, par position ; entropie : entropie (bits) du poids a priori de chaque pas ;
    masses : (n_snap, G) masse par groupe (groupes : tableau d'entiers de longueur M) ; poids_finaux.
    """
    T, M = P.shape
    w = np.full(M, 1.0 / M) if prior is None else np.asarray(prior, float) / np.sum(prior)
    pertes = np.empty(T)
    ent = np.empty(T)
    G = int(groupes.max()) + 1 if groupes is not None else 0
    masses, pos_snap = [], []
    for t in range(T):
        if groupes is not None and t % pas_snap == 0:
            masses.append(np.bincount(groupes, weights=w, minlength=G))
            pos_snap.append(t)
        wl = w[w > 0]
        ent[t] = float(-(wl * np.log2(wl)).sum())
        pt = P[t]
        mel = float(w @ pt)
        pertes[t] = -math.log2(max(mel, PLANCHER_P))
        w = w * pt / mel
        if alpha > 0.0:
            w = (1.0 - alpha) * w + alpha / M
        else:
            w = w / w.sum()
    if groupes is not None:
        masses.append(np.bincount(groupes, weights=w, minlength=G))
        pos_snap.append(T)
    return {"pertes": pertes, "entropie": ent, "masses": np.array(masses) if masses else None,
            "positions_snap": pos_snap, "poids_finaux": w}


def choisir_alpha(P_A, prior=None, grille=GRILLE_ALPHA):
    """α de fixed-share qui minimise le bpc sur A (A seulement)."""
    res = {a: float(melanger(P_A, a, prior)["pertes"].mean()) for a in grille}
    return min(res, key=res.get), res


def proba_depuis_log2(L):
    """L : (M, T) log2 p → P (T, M) float64 contigu."""
    return np.ascontiguousarray(np.exp2(L.astype(np.float64)).T)


def borne_bma(M):
    return math.log2(M)
