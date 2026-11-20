"""Évaluation sur les 1000 puzzles de test (poids EMA) + rapport.

Pour chaque modèle : exactitude exacte (81 cases correctes, donc cohérentes avec les
indices recopiés), exactitude sur cases vides, taux de grilles violant une contrainte
Sudoku, en fonction du nombre de pas de supervision S au test ; arrêt adaptatif ;
calibration de la tête d'arrêt.
"""
import json
import math
import os
import sys

import numpy as np
import torch

import donnees
from modele import TRM, nb_params, quantifier_int8, taille_export

torch.set_num_threads(2)
ICI = os.path.dirname(os.path.abspath(__file__))
S_LISTE = [1, 2, 4, 8, 16, 32]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - r), min(1.0, c + r))


def viole(grille):
    """True si la grille (81 entiers) contient un doublon dans une unité."""
    # int() : avec des tenseurs 0-d, set() comparerait les objets et non les valeurs
    return not donnees.sans_conflit([int(x) for x in grille])


def trajectoire(m, P, S, bs=250):
    """Exécute S pas de supervision ; renvoie par pas : prédictions [S,N,81], logits d'arrêt [S,N]."""
    N = P.shape[0]
    preds = np.zeros((S, N, 81), dtype=np.uint8)
    halts = np.zeros((S, N), dtype=np.float32)
    m.eval()
    with torch.no_grad():
        for a in range(0, N, bs):
            x = torch.from_numpy(P[a:a + bs].astype(np.int64))
            y, z = m.init_etat(x.shape[0])
            for s in range(S):
                y, z, lg, h = m.etape(x, y, z)
                pr = torch.where(x > 0, x, lg.argmax(-1) + 1)
                preds[s, a:a + bs] = pr.numpy().astype(np.uint8)
                halts[s, a:a + bs] = h.numpy()
    return preds, halts


def metriques_pas(preds, halts, P, Sol):
    vide = P == 0
    out = {}
    corrects = (preds == Sol[None]).all(2)               # [S,N]
    for s in S_LISTE:
        if s > preds.shape[0]:
            continue
        pr = preds[s - 1]
        ex = int(corrects[s - 1].sum())
        cell = float(((pr == Sol) & vide).sum() / vide.sum())
        nv = sum(viole(g) for g in pr)
        out[s] = {"exact": ex, "exact_ic95": wilson(ex, len(P)), "cases_vides": cell,
                  "viol_contrainte": nv / len(P)}
    return out, corrects


def adaptatif(preds, halts, Sol, smax):
    """Arrêt dès que la tête d'arrêt dit « fini » (logit>0), au plus smax pas."""
    N = preds.shape[1]
    ok, pas, dit_fini, fini_et_ok = 0, [], 0, 0
    for i in range(N):
        stop = smax
        arret = False
        for s in range(smax):
            if halts[s, i] > 0:
                stop, arret = s + 1, True
                break
        pas.append(stop)
        c = bool((preds[stop - 1, i] == Sol[i]).all())
        ok += c
        if arret:
            dit_fini += 1
            fini_et_ok += c
    return {"exact": ok, "exact_ic95": wilson(ok, N), "pas_moyens": float(np.mean(pas)),
            "n_arret_declare": dit_fini,
            "precision_quand_dit_fini": (fini_et_ok / dit_fini) if dit_fini else None}


def calibration(preds, halts, Sol, s):
    """Au pas s : table par tranche de proba d'arrêt -> fréquence réelle d'exactitude."""
    p = 1 / (1 + np.exp(-halts[s - 1]))
    c = (preds[s - 1] == Sol).all(1)
    bornes = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]
    lignes = []
    for lo, hi in zip(bornes[:-1], bornes[1:]):
        msk = (p >= lo) & (p < hi)
        lignes.append({"tranche": f"[{lo:.1f},{min(hi, 1):.1f})", "n": int(msk.sum()),
                       "proba_moy": float(p[msk].mean()) if msk.any() else None,
                       "exact_reel": float(c[msk].mean()) if msk.any() else None})
    return lignes


def evaluer_nom(nom, P, Sol, smax):
    ck = torch.load(os.path.join(ICI, "runs", f"{nom}.pt"))
    m = TRM(**ck["cfg"])
    res = {"params": nb_params(m), "export": taille_export(m), "cfg": ck["cfg"]}
    for cle in ("ema", "raw"):
        m.load_state_dict(ck[cle])
        preds, halts = trajectoire(m, P, smax)
        mp, _ = metriques_pas(preds, halts, P, Sol)
        res[cle] = {"par_pas": mp}
        if cle == "ema":
            res["ema"]["adaptatif"] = adaptatif(preds, halts, Sol, smax if smax <= 16 else 16)
            res["ema"]["calibration_pas1"] = calibration(preds, halts, Sol, 1)
            res["ema"]["calibration_pasN"] = calibration(preds, halts, Sol, min(8, smax))
            res["ema"]["viol_et_confiant"] = None
    m.load_state_dict(ck["ema"])
    q, octets = quantifier_int8(m)
    preds, halts = trajectoire(q, P, smax)
    mq, _ = metriques_pas(preds, halts, P, Sol)
    res["ema_int8"] = {"octets_export_reels": octets, "par_pas": mq}
    assert octets <= 10_000_000, "export int8 > 10 Mo"
    return res


def main(noms=("A", "B", "C"), smax=32, n=None, sortie="resultats_trm.json"):
    jeux = donnees.charger(os.path.join(ICI, "donnees_sudoku.pt"))
    P, Sol = jeux["test"]
    if n:
        P, Sol = P[:n], Sol[:n]
    out = {"n_test": int(len(P))}
    for nom in noms:
        if os.path.exists(os.path.join(ICI, "runs", f"{nom}.pt")):
            print("évaluation", nom, flush=True)
            out[nom] = evaluer_nom(nom, P, Sol, smax if nom == "A" else min(smax, 8))
    json.dump(out, open(os.path.join(ICI, sortie), "w"), indent=1)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n=n, sortie="resultats_trm.json" if n is None else "resultats_smoke.json")
