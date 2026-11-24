"""Calcule les permutations de canaux sur TRAIN (activite du modele 4 bits dense) puis construit les fichiers
colonne-majeur. Les poids ne changent pas : on ne fait que ranger les memes colonnes par activite decroissante."""
import os, sys, time, json
import numpy as np
import torch
import sim as S
import donnees, layout as lay


def permutations():
    eng = S.Sim("rm")
    c = eng.calibrer(donnees.fenetres("TRAIN"))
    perm_d = np.zeros((S.NL, lay.D_IN), np.int16)
    perm_i = np.zeros((S.NL, S.NE, lay.I_DIM), np.int16)
    for L in range(S.NL):
        perm_d[L] = torch.argsort(c["xe"][L] / max(c["xn"][L].item(), 1), descending=True).numpy()
        ae, an = c["ae"][L], c["an"][L]
        pooled = ae.sum(0) / an.sum()                       # energie moyenne par canal intermediaire, toute la couche
        for e in range(S.NE):
            # melange bayesien avec la moyenne de la couche (peu de jetons par expert)
            m = (ae[e] + 24 * pooled) / (an[e] + 24)
            perm_i[L, e] = torch.argsort(m, descending=True).numpy()
    np.savez(S.PERM, perm_d=perm_d, perm_i=perm_i)
    json.dump(dict(tokens_train=int(c["xn"][0].item()), experts_vus=[int((c["an"][L] > 0).sum()) for L in range(S.NL)],
                   jetons_par_expert_min=float(min(c["an"][L].min().item() for L in range(S.NL))),
                   jetons_par_expert_moy=float(np.mean([c["an"][L].mean().item() for L in range(S.NL)]))),
              open(os.path.join(S.ICI, "cache", "calib_info.json"), "w"), indent=1)
    return perm_d, perm_i


if __name__ == "__main__":
    os.makedirs(os.path.join(S.ICI, "cache"), exist_ok=True)
    t0 = time.time()
    permutations()
    print("permutations ecrites", S.PERM, time.time() - t0)
