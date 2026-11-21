"""Lance les expériences « Bayes, la dernière pierre ».

Étapes (toujours dans cet ordre) :
  selection : tout ce qui se règle sur A et B (hyperparamètres de Pitman-Yor, α de fixed-share).
  test_d    : évaluation UNIQUE sur la tranche fraîche D (témoin cache/d_utilise.json ; --rejouer pour passer outre).
  thompson  : comparaison à budget égal Thompson / moteur existant sur A et B (aucun test).
  rapport   : assemble resultats_bayes.json et resultats_bayes.md.
C (cache/c.npz) n'est jamais lu ici.
"""
import json
import os
import sys
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

import bayes  # noqa: E402
import donnees  # noqa: E402
import genome_initial  # noqa: E402

SORTIE = os.path.join(ICI, "archive_bayes")
os.makedirs(SORTIE, exist_ok=True)
NOMS = ["graine", "champion", "reference_main", "pitman_yor"]


def _ecrire(nom, obj):
    with open(os.path.join(SORTIE, nom), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def _lire(nom):
    with open(os.path.join(SORTIE, nom), encoding="utf-8") as f:
        return json.load(f)


def sources():
    return {"graine": genome_initial.SOURCE, "champion": bayes.source_champion(),
            "reference_main": bayes.source_reference_main()}


def pertes_modeles(jeu, py):
    """Pertes (bits) par caractère des 4 membres, dans l'ordre NOMS."""
    s = sources()
    L = [bayes.pertes_genome(s[k], jeu) for k in NOMS[:3]]
    p_py = bayes.py_proba_vraie(jeu, py["delta"], py["theta"])
    L.append(-np.log2(p_py))
    return np.array(L)


def selection():
    t0 = time.time()
    ab = donnees.charger_ab()
    py = bayes.ajuster_py(ab["A"])
    out = {"py": py, "temps_ajustement_s": round(time.time() - t0, 1)}
    LA, LB = pertes_modeles(ab["A"], py), pertes_modeles(ab["B"], py)
    PA, PB = 2.0 ** -LA, 2.0 ** -LB
    alpha, courbe = bayes.choisir_alpha(PA)
    out["alpha"] = alpha
    out["alpha_courbe_bpc_A"] = {str(a): v for a, v in courbe.items()}
    out["bpc"] = {}
    for nom, L, P in (("A", LA, PA), ("B", LB, PB)):
        bma = bayes.bma_en_ligne(P)
        fs = bayes.bma_en_ligne(P, alpha=alpha)
        out["bpc"][nom] = {**{k: float(L[i].mean()) for i, k in enumerate(NOMS)},
                           "bma": float(bma["pertes"].mean()), "fixed_share": float(fs["pertes"].mean()),
                           "poids_moyens_bma": bma["poids_moyens"], "poids_moyens_fs": fs["poids_moyens"]}
    # contrôle de la borne de regret sur A et B (bits, totaux)
    prior = np.full(4, 0.25)
    out["regret"] = {}
    for nom, L, P in (("A", LA, PA), ("B", LB, PB)):
        bma = bayes.bma_en_ligne(P)
        tot = L.sum(axis=1)
        out["regret"][nom] = {"perte_bma_bits": float(bma["pertes"].sum()),
                              "meilleur_membre": NOMS[int(tot.argmin())], "perte_meilleur_bits": float(tot.min()),
                              "regret_observe_bits": float(bma["pertes"].sum() - tot.min()),
                              "borne_log2_1_sur_prior": float(bayes.borne_regret_bma(prior)[int(tot.argmin())])}
    # B : probabilité que chacun batte le champion (B a servi à choisir le champion : optimiste pour lui)
    fsB = bayes.bma_en_ligne(PB, alpha=alpha)["pertes"]
    bmaB = bayes.bma_en_ligne(PB)["pertes"]
    cand = {"graine": LB[0], "reference_main": LB[2], "pitman_yor": LB[3], "bma": bmaB, "fixed_share": fsB}
    out["p_meilleur_B"] = {k: bayes.p_meilleur(v, LB[1]) for k, v in cand.items()}
    _ecrire("selection.json", out)
    print("sélection faite en %.0f s ; α = %g" % (time.time() - t0, alpha))
    print(json.dumps(out["bpc"], indent=1)[:1500])


def test_d(rejouer=False):
    if os.path.exists(bayes.TEMOIN_D) and not rejouer:
        raise SystemExit("D a déjà été utilisé (témoin %s). --rejouer pour passer outre." % bayes.TEMOIN_D)
    sel = _lire("selection.json")
    py, alpha = sel["py"], sel["alpha"]
    bayes.construire_d()
    D = bayes.charger_d()
    LD = pertes_modeles(D, py)
    PD = 2.0 ** -LD
    bma = bayes.bma_en_ligne(PD)
    fs = bayes.bma_en_ligne(PD, alpha=alpha)
    lib, _ = donnees.lire_textes()
    td, debut, nval = bayes.texte_frais()
    lz = bayes.lzma_conditionnel(lib, td)
    bpc = {**{k: float(LD[i].mean()) for i, k in enumerate(NOMS)}, "bma": float(bma["pertes"].mean()),
           "fixed_share": float(fs["pertes"].mean()), "lzma": lz["bpc"]}
    pertes = {"graine": LD[0], "champion": LD[1], "reference_main": LD[2], "pitman_yor": LD[3],
              "bma": bma["pertes"], "fixed_share": fs["pertes"]}
    contre_champion = {k: bayes.p_meilleur(pertes[k], pertes["champion"]) for k in pertes if k != "champion"}
    contre_main = {k: bayes.p_meilleur(pertes[k], pertes["reference_main"])
                   for k in ("champion", "pitman_yor", "bma", "fixed_share")}
    tot = LD.sum(axis=1)
    out = {"D": {"taille": int(len(LD[0])), "debut_dans_validation": debut, "taille_validation": nval,
                 "bpc": bpc, "p_meilleur_que_champion": contre_champion, "p_meilleur_que_reference_main": contre_main,
                 "poids_moyens_bma": bma["poids_moyens"], "poids_moyens_fs": fs["poids_moyens"],
                 "poids_finaux_fs": fs["poids_finaux"], "lzma": lz,
                 "regret": {"perte_bma_bits": float(bma["pertes"].sum()), "meilleur_membre": NOMS[int(tot.argmin())],
                            "perte_meilleur_bits": float(tot.min()),
                            "regret_observe_bits": float(bma["pertes"].sum() - tot.min()),
                            "borne_log2_1_sur_prior": 2.0}},
           "alpha": alpha}
    _ecrire("test_d.json", out)
    with open(bayes.TEMOIN_D, "w") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "note": "D lu une seule fois par lancer_bayes.py test_d"}, f)
    print(json.dumps(out["D"]["bpc"], indent=1))


def thompson(budget=1500, graines=(1, 2, 3)):
    import moteur
    res = {"budget": budget, "graines": list(graines), "base": [], "thompson": []}
    for g in graines:
        r = moteur.executer("bayes_base_s%d" % g, budget, 5000 + g, "auto", dossier=SORTIE, verbeux=False)
        res["base"].append({"graine": g, "best_A": min(p["best_A"] for p in r["points"]),
                            "champion_A": r["champion"]["fit_A"], "champion_B": r["champion"]["fit_B"],
                            "evals": r["evals"], "cpu_total_s": r["cpu_total_s"], "temps_mur_s": r["temps_mur_s"],
                            "points": [{k: p[k] for k in ("evals", "best_A", "best_B")} for p in r["points"]
                                       if p["evals"] % 250 == 0 or p["evals"] == r["evals"]]})
        print("base", g, res["base"][-1]["best_A"], res["base"][-1]["champion_B"], flush=True)
        t = bayes.executer_thompson("bayes_ts_s%d" % g, budget, 5000 + g)
        res["thompson"].append({"graine": g, "best_A": t["best_A"], "champion_A": t["champion"]["fit_A"],
                                "champion_B": t["champion"]["fit_B"], "evals": t["evals"],
                                "cpu_total_s": t["cpu_total_s"], "temps_mur_s": t["temps_mur_s"],
                                "points": t["points"], "thompson": t["thompson"]})
        print("ts  ", g, res["thompson"][-1]["best_A"], res["thompson"][-1]["champion_B"], flush=True)
        _ecrire("thompson.json", res)
    for cle in ("best_A", "champion_B"):
        res["comparaison_" + cle] = bayes.p_moyenne_inferieure([r[cle] for r in res["thompson"]],
                                                               [r[cle] for r in res["base"]])
    _ecrire("thompson.json", res)


if __name__ == "__main__":
    etape = sys.argv[1] if len(sys.argv) > 1 else "selection"
    if etape == "selection":
        selection()
    elif etape == "test_d":
        test_d("--rejouer" in sys.argv)
    elif etape == "thompson":
        thompson(int(sys.argv[2]) if len(sys.argv) > 2 else 1500)
    else:
        raise SystemExit(__doc__)
