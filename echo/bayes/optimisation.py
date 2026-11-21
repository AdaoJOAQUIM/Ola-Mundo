"""Auto-amélioration = optimisation bayésienne (TPE d'Optuna, Bergstra et al. 2011), sur A seulement.

Espace : c_k par ordre (échelle log, k ≤ ordre max), γ, utilisation de u1, ordre max. Objectif : bpc sur A.
300 essais, graine fixe. B est évalué à chaque essai pour la seule courbe d'efficacité (Optuna ne le voit
jamais) ; le gagnant est choisi sur B parmi les 10 meilleurs sur A.
"""
import json
import os
import time

import numpy as np
import optuna

import experts as ex

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_ESSAIS = 300
GRAINE = 0


def params_de_essai(trial):
    kmax = trial.suggest_int("kmax", 2, 10)
    cs = [trial.suggest_float("c%d" % k, 0.05, 300.0, log=True) if k <= kmax else 1.0 for k in range(11)]
    gamma = trial.suggest_float("gamma", 0.4, 1.0)
    u1 = trial.suggest_categorical("u1", [False, True])
    return {"cs": cs, "gamma": gamma, "u1": u1, "kmax": kmax}


def params_de_dict(p):
    kmax = int(p["kmax"])
    return {"cs": [p["c%d" % k] if k <= kmax else 1.0 for k in range(11)], "gamma": p["gamma"],
            "u1": bool(p["u1"]), "kmax": kmax}


def optimiser(jeu_A, jeu_B, n_essais=N_ESSAIS, graine=GRAINE):
    prA, prB = ex.Prepare(jeu_A), ex.Prepare(jeu_B)
    bpcB = {}

    def objectif(trial):
        par = params_de_essai(trial)
        a = ex.bpc_params(prA, par)
        bpcB[trial.number] = ex.bpc_params(prB, par)      # pour la courbe seulement
        return a

    etude = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=graine))
    t0 = time.process_time()
    etude.optimize(objectif, n_trials=n_essais)
    cpu = time.process_time() - t0
    essais = [{"n": t.number, "A": t.value, "B": bpcB[t.number], "params": t.params} for t in etude.trials]
    ordre_A = sorted(essais, key=lambda e: e["A"])[:10]
    gagnant = min(ordre_A, key=lambda e: e["B"])
    # courbe : après m essais, B du meilleur sur B parmi les 10 meilleurs sur A vus jusque-là
    courbe = []
    for m in range(1, len(essais) + 1):
        top = sorted(essais[:m], key=lambda e: e["A"])[:10]
        courbe.append({"essais": m, "meilleur_A": top[0]["A"], "B_champion": min(e["B"] for e in top)})
    return {"essais": essais, "top10_A": ordre_A, "gagnant": gagnant,
            "params_gagnant": params_de_dict(gagnant["params"]), "courbe": courbe, "cpu_s": cpu}
