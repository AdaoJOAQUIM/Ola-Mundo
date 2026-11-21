"""Exécute tout le protocole bayésien (à lancer APRÈS la fin de l'inférence TRM sur le nouvel échantillon).
Ordre imposé : (1) ajustement sans lire le pli de test, (2) gel des paramètres, (3) relecture + notation."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bayes_arc as B  # noqa: E402

ICI, ARC = B.ICI, B.ARC
ANCIEN = json.load(open(os.path.join(ARC, "sous_ensemble_fusion.json")))["taches"]
NOUVEAU = json.load(open(os.path.join(ICI, "sous_ensemble_bayes.json")))["taches"]
TRM = {"ancien": B.charger_trm(os.path.join(ARC, "trm_officiel", "res_K8.json")),
       "nouveau": B.charger_trm(os.path.join(ICI, "res_K8_bayes.json"))}
PLIS = {"ancien": ANCIEN, "nouveau": NOUVEAU}
# pli 1 : test = anciennes 20, calibration TRM = nouvelles 20 ; pli 2 : l'inverse
SCHEMA = {"pli1": ("ancien", "nouveau"), "pli2": ("nouveau", "ancien")}
VARIANTES = {"complet": ("bucket", "rang", "accord"), "sobre": ("bucket",)}


def main():
    assert not set(ANCIEN) & set(NOUVEAU), "plis non disjoints"
    assert all(t in TRM["nouveau"] for t in NOUVEAU), "inférence TRM incomplète"
    dsl = B.charger_dsl()
    tous = B.toutes_taches()
    pools = {nom: {t: B.pool(t, TRM[nom][t], dsl[t]) for t in PLIS[nom]} for nom in PLIS}
    res = {"plis": {}, "protocole": {"graine_nouveau": 2026, "k_votes": 8}}
    gel_avant = {}

    # ---------- (1)+(2) ajustement et gel, sans solution du pli de test
    for pli, (test, cal) in SCHEMA.items():
        for var, caract in VARIANTES.items():
            v = B.Verite(interdits=PLIS[test])
            hors_test = [t for t in tous if t not in set(PLIS[test])]
            params = B.ajuster(pools[cal], dsl, hors_test, v, caracteristiques=caract)
            assert not (v.lectures & set(PLIS[test]))
            chemin = os.path.join(ICI, f"params_{pli}_{var}.json")
            h = B.geler(params, chemin, {"pli": pli, "test": test, "calibration_trm": cal, "variante": var,
                                         "taches_test_exclues": PLIS[test]})
            gel_avant[(pli, var)] = (chemin, h)
            print("gel", chemin, h[:12], flush=True)

    # ---------- (3) notation du pli de test, avec les paramètres relus
    sorties = {}
    for pli, (test, cal) in SCHEMA.items():
        sorties[pli] = {}
        for var in VARIANTES:
            chemin, h = gel_avant[(pli, var)]
            params, meta = B.relire(chemin)
            sorties[pli][var] = (params, B.noter_pli(PLIS[test], pools[test], dsl, params, B.Verite()))

    def agr(var):
        tot = {}
        paires, top1 = [], []
        for pli in SCHEMA:
            r = sorties[pli][var][1]
            for m, x in r["agregats"].items():
                tot[m] = tot.get(m, 0) + x
            paires += r["paires"]
            top1 += r["paires_top1"]
        prior = sum(sorties[p][var][0]["prior"] for p in SCHEMA) / 2
        return tot, paires, top1, prior

    for pli, (test, cal) in SCHEMA.items():
        res["plis"][pli] = {"test": test, "calibration_trm": cal, "taches": PLIS[test]}
        for var in VARIANTES:
            params, r = sorties[pli][var]
            res["plis"][pli][var] = {
                "params": params, "agregats": r["agregats"], "detail": r["detail"],
                "calibration_toutes_candidates": B.calibration(r["paires"], params["prior"]),
                "calibration_top1": B.calibration(r["paires_top1"], params["prior"])}
    res["total40"] = {}
    for var in VARIANTES:
        tot, paires, top1, prior = agr(var)
        res["total40"][var] = {"agregats": tot, "calibration_toutes_candidates": B.calibration(paires, prior),
                               "calibration_top1": B.calibration(top1, prior)}

    # contrôle : la règle ancienne (sans CompressARC) retrouve le chiffre publié (8/20) sur le pli ancien
    pub = json.load(open(os.path.join(ARC, "fusion", "resultats_fusion.json")))
    res["controle_vieille_regle"] = {"publie_ancien_20": pub["resolus_pass2"]["fusion"],
                                     "recalcule_ancien_20": res["plis"]["pli1"]["complet"]["agregats"]["vieille_regle"]}

    # ---------- valeur du calcul
    ok_dsl = {t: res["plis"][p]["complet"]["detail"][t]["dsl"] for p, (te, _) in SCHEMA.items() for t in PLIS[te]}
    ok_trm = {t: res["plis"][p]["complet"]["detail"][t]["trm"] for p, (te, _) in SCHEMA.items() for t in PLIS[te]}
    ok_bayes = {t: res["plis"][p]["complet"]["detail"][t]["bayes"] for p, (te, _) in SCHEMA.items() for t in PLIS[te]}
    cpu_dsl = {t: dsl[t]["cpu"] for t in tous}
    cpu_trm = {t: TRM[n][t]["cpu_s"] for n in PLIS for t in PLIS[n]}
    grille = [0.0] + [10 ** (-5 + 2.6 * i / 120) for i in range(121)]
    res["voc"] = {}
    for pli, (test, cal) in SCHEMA.items():
        tcal, ttest = PLIS[cal], PLIS[test]
        hors_test = [t for t in tous if t not in set(ttest)]
        modele = B.modele_voc(dsl, hors_test, ok_trm, tcal)
        cout = sum(cpu_trm[t] for t in tcal) / len(tcal)
        lam = B.choisir_lambda(tcal, dsl, modele, cout, ok_dsl, ok_trm, cpu_dsl, cpu_trm, grille)
        # sur le pli de test : la branche « TRM lancé » utilise la fusion bayésienne figée
        courbe, vus = [], set()
        for l in grille + [1.0]:
            lance = B.politique(ttest, dsl, modele, cout, l)
            sig = tuple(sorted(lance.items()))
            if sig in vus:
                continue
            vus.add(sig)
            r, cpu, nl = B.evaluer_politique(ttest, lance, ok_dsl, ok_bayes, cpu_dsl, cpu_trm)
            courbe.append({"lambda": l, "pass2": r, "cpu_s": round(cpu, 1), "tâches_avec_TRM": nl})
        lance = B.politique(ttest, dsl, modele, cout, lam)
        r, cpu, nl = B.evaluer_politique(ttest, lance, ok_dsl, ok_bayes, cpu_dsl, cpu_trm)
        toujours = B.evaluer_politique(ttest, {t: True for t in ttest}, ok_dsl, ok_bayes, cpu_dsl, cpu_trm)
        jamais = B.evaluer_politique(ttest, {t: False for t in ttest}, ok_dsl, ok_bayes, cpu_dsl, cpu_trm)
        res["voc"][pli] = {"modele": {k: ({str(a): b for a, b in v.items()} if isinstance(v, dict) else v)
                                      for k, v in modele.items()},
                           "cout_trm_moyen_s_calibration": cout, "lambda_choisi": lam,
                           "politique": {"pass2": r, "cpu_s": round(cpu, 1), "tâches_avec_TRM": nl,
                                         "lance": lance},
                           "toujours_TRM": {"pass2": toujours[0], "cpu_s": round(toujours[1], 1)},
                           "jamais_TRM_DSL_seul": {"pass2": jamais[0], "cpu_s": round(jamais[1], 1)},
                           "courbe": courbe}
    json.dump(res, open(os.path.join(ICI, "resultats_bayes_arc.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("plis", "voc")}, indent=1, ensure_ascii=False))
    for pli in SCHEMA:
        print(pli, res["plis"][pli]["complet"]["agregats"], res["plis"][pli]["sobre"]["agregats"])
        v = res["voc"][pli]
        print(pli, "lambda", v["lambda_choisi"], v["politique"]["pass2"], v["politique"]["cpu_s"], v["toujours_TRM"],
              v["jamais_TRM_DSL_seul"], v["modele"])


if __name__ == "__main__":
    main()
