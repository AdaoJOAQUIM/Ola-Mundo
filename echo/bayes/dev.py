"""Phase développement : tout ce qui se règle, se règle sur A ; B sert à choisir et à regarder. Ni E ni le français.

Sortie : cache/parametres_figes.json (lu par final.py) et cache/dev_resultats.json (repris dans le rapport).
"""
import json
import os
import time

import numpy as np

import donnees_bayes as db
import efficacite
import experts as ex
import melange as ml
import optimisation as opt


def tous_experts(jeu, py_params, params_optuna=None):
    metas, L = [], []
    for f in (lambda: ex.experts_references(jeu, py_params), lambda: ex.experts_grille(jeu),
              lambda: ex.experts_maj(jeu)):
        m, l = f()
        metas += m
        L.append(l)
    if params_optuna is not None:
        L.append(ex.log2_params_jeu(jeu, params_optuna)[None, :])
        metas.append(dict(nom="Optuna TPE (gagnant sur B)", famille="optuna", c=None, ordre=params_optuna["kmax"],
                          gamma=params_optuna["gamma"], u1=params_optuna["u1"], beta=None))
    return metas, np.concatenate(L)


def main():
    t0 = time.process_time()
    jeux = db.tableaux_dev()
    A, B = jeux["A"], jeux["B"]
    be = ex.bayes_evolution()
    py = be.ajuster_py(A)
    py_params = {"delta": py["delta"], "theta": py["theta"], "bpc_A": py["bpc_A"]}
    print("Pitman-Yor A = %.4f" % py["bpc_A"], flush=True)

    r_opt = opt.optimiser(A, B)
    pg = r_opt["params_gagnant"]
    print("Optuna : gagnant A=%.4f B=%.4f (%.0f s CPU)" % (r_opt["gagnant"]["A"], r_opt["gagnant"]["B"],
                                                           r_opt["cpu_s"]), flush=True)

    out = {"taille_A": len(A["y"]), "taille_B": len(B["y"])}
    res = {}
    for nom, jeu in (("A", A), ("B", B)):
        metas, L = tous_experts(jeu, py_params, pg)
        res[nom] = (metas, L)
        print(nom, "experts", L.shape, flush=True)
    metas = res["A"][0]
    nm_sans = [i for i, m in enumerate(metas) if m["famille"] != "optuna"]
    PA = ml.proba_depuis_log2(res["A"][1])
    PB = ml.proba_depuis_log2(res["B"][1])
    bpcA = -res["A"][1].mean(axis=1)
    bpcB = -res["B"][1].mean(axis=1)
    alphas = {}
    for etiquette, cols in (("avec_optuna", list(range(len(metas)))), ("sans_optuna", nm_sans)):
        a, table = ml.choisir_alpha(np.ascontiguousarray(PA[:, cols]))
        alphas[etiquette] = {"alpha": a, "bpc_A_par_alpha": {str(k): v for k, v in table.items()}}
        bma = ml.melanger(np.ascontiguousarray(PB[:, cols]))["pertes"].mean()
        fs = ml.melanger(np.ascontiguousarray(PB[:, cols]), a)["pertes"].mean()
        alphas[etiquette].update({"bpc_B_bma": float(bma), "bpc_B_fixed_share": float(fs)})
        print(etiquette, "alpha", a, "B bma %.4f fs %.4f" % (bma, fs), flush=True)

    idx = {m["nom"]: i for i, m in enumerate(metas)}
    refs = {n: {"A": float(bpcA[idx[n]]), "B": float(bpcB[idx[n]])}
            for n in ("départ (PPM-C)", "champion évolué", "référence à la main (e=16·d)", "Pitman-Yor (A)",
                      "Optuna TPE (gagnant sur B)")}
    # décalage de B dû à la bibliothèque réduite (anciennes valeurs publiées : champion 1.4772, main 1.4608)
    decalage = {"champion": refs["champion évolué"]["B"] - 1.4772,
                "main": refs["référence à la main (e=16·d)"]["B"] - 1.4608}
    niveaux = [1.55, 1.50, 1.49, 1.48, 1.47, 1.46]
    dec = decalage["champion"]
    eff = efficacite.comparer(r_opt["courbe"], niveaux, decalage=dec)
    meilleur_A_par_famille = {}
    for fam in ("grille", "maj", "ref", "optuna"):
        ii = [i for i, m in enumerate(metas) if m["famille"] == fam]
        if ii:
            j = min(ii, key=lambda i: bpcA[i])
            meilleur_A_par_famille[fam] = {"nom": metas[j]["nom"], "A": float(bpcA[j]), "B": float(bpcB[j])}
    fig = {"py_params": py_params, "params_optuna": pg, "alphas": alphas,
           "bpc_A_experts": {m["nom"]: float(bpcA[i]) for i, m in enumerate(metas)}}
    os.makedirs(db.CACHE, exist_ok=True)
    with open(os.path.join(db.CACHE, "parametres_figes.json"), "w") as f:
        json.dump(fig, f, indent=1)
    out.update({"references_AB": refs, "decalage_B": decalage, "meilleur_A_par_famille": meilleur_A_par_famille,
                "optuna": {"gagnant": r_opt["gagnant"], "top10_A": r_opt["top10_A"], "courbe": r_opt["courbe"],
                           "cpu_s": r_opt["cpu_s"], "params": pg},
                "efficacite": eff, "alphas": alphas, "py": py_params, "n_experts": len(metas),
                "cpu_total_s": time.process_time() - t0})
    with open(os.path.join(db.CACHE, "dev_resultats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("dev terminé, CPU %.0f s" % (time.process_time() - t0))


if __name__ == "__main__":
    main()
