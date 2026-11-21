"""Moteur d'évolution (archive ouverte à la manière de la Darwin Gödel Machine).

  * l'archive part du génome de départ et ne jette rien ;
  * parents tirés avec p_i ~ sigmoïde(10 (z_i - 0.5)) / (1 + nb_enfants_i) ;
  * un enfant = 1 à 3 mutations de l'arbre syntaxique, choisies selon le
    MÉTA-GÉNOME du parent (poids des opérateurs, pas des constantes, nombre de
    mutations) ; le méta-génome est lui-même hérité avec du bruit log-normal
    (option --meta auto) ou figé (--meta fixe) ;
  * fitness sur A ; toutes les ~10 générations le top 10 est évalué sur B ;
  * champion = meilleur sur B (B ne sert qu'à choisir, jamais à muter).

Le moteur ne touche jamais au juge : les empreintes SHA-256 sont vérifiées au
départ puis toutes les 25 générations.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

import donnees  # noqa: E402
import composants  # noqa: E402
import garde_fous  # noqa: E402
import genome_initial  # noqa: E402
import mutations  # noqa: E402

DOSSIER_ARCHIVE = os.path.join(ICI, "archive")


class AlerteEmpreinte(SystemExit):
    pass


def controle_empreintes():
    mauvais = garde_fous.verifier_empreintes()
    if mauvais:
        raise AlerteEmpreinte("ABANDON : empreinte modifiée pour %s" % mauvais)


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def poids_parents(archive):
    fit = np.array([g["fit_eff"] for g in archive])
    meilleur = fit.min()
    ref = archive[0]["fit_eff"]
    plage = max(ref - meilleur, 0.0) + 0.1
    z = np.clip(1.0 - (fit - meilleur) / plage, 0.0, 1.0)
    enfants = np.array([g["enfants"] for g in archive], dtype=float)
    w = sigmoide(10 * (z - 0.5)) / (1.0 + enfants)
    return w / w.sum()


def executer(nom, budget, graine, mode_meta="auto", nproc=4, taille_lot=8, delai=20,
             dossier=DOSSIER_ARCHIVE, verbeux=True, K=None, complet=False, soi=None, objectif=None,
             cout=None, elites=None, patience=None, delta=0.002, cible=None, budget_cpu=None,
             source_depart=None, lambda_mdl=0.5, mu_cout=0.01):
    """Boucle d'évolution. `complet=True` active les cinq composants (soi, objectif MDL+Pareto,
    coût, MAP-Elites) ; patience/cible/budget_cpu pilotent l'arrêt."""
    soi = complet if soi is None else soi
    objectif = complet if objectif is None else objectif
    cout = complet if cout is None else cout
    elites = complet if elites is None else elites
    t0 = time.time()
    cpu0 = time.process_time()
    controle_empreintes()
    os.makedirs(dossier, exist_ok=True)
    rng = random.Random(graine)
    jeux = donnees.charger_ab()
    K = jeux["A"]["n"].shape[0] - 1
    T_A = len(jeux["A"]["y"])
    ev = composants.EvaluateurMesure({"A": jeux["A"], "B": jeux["B"]}, nproc=nproc, delai=delai)
    chemin_log = os.path.join(dossier, nom + ".jsonl")
    log = open(chemin_log, "w")
    cpu_trav = [0.0]
    cpu_evals = []

    def ecrire(obj):
        log.write(json.dumps(obj, ensure_ascii=False) + "\n")
        log.flush()

    def cpu_total():
        return cpu_trav[0] + (time.process_time() - cpu0)

    def suivre(res):
        for r in res:
            cpu_trav[0] += r.get("cpu_s", 0.0)

    src0 = source_depart or genome_initial.SOURCE
    r = ev.evaluer_lot([src0], "A")[0]
    rb = ev.evaluer_lot([src0], "B")[0]
    suivre([r, rb])
    assert r["ok"] and rb["ok"], "le génome de départ doit être valide"
    cpu_graine = max(r["cpu_s"], 1e-3)

    def fit_eff(fit_A, source, cpu_s):
        v = fit_A
        if objectif:
            v += lambda_mdl * composants.taille_mdl_bits(source) / T_A
        if cout:
            v += min(0.1, mu_cout * max(0.0, cpu_s / cpu_graine - 1.0))
        return v

    graine_g = {"id": 0, "parent": None, "mutations": [], "source": src0,
                "fit_A": r["bpc"], "fit_B": rb["bpc"], "meta": mutations.meta_defaut(),
                "enfants": 0, "gen": 0, "cpu_s": r["cpu_s"], "mdl_bits": composants.taille_mdl_bits(src0)}
    graine_g["fit_eff"] = fit_eff(r["bpc"], src0, r["cpu_s"])
    archive = [graine_g]
    vus = {src0}
    ecrire({"type": "genome", **graine_g})
    cartes = composants.MapElites() if elites else None
    if cartes:
        cartes.inserer(graine_g, graine_g["fit_eff"])
    auto_modele = composants.Soi() if soi else None
    arret = composants.ArretPrecoce(delta, patience, cible) if (patience or cible) else None
    compteur_rejets = Counter()
    depenses = 0
    gen = 0
    points = []
    nb_ecartes = 0
    front_ids = set()
    raison_arret = "budget"
    seed_A = r["bpc"]

    def top_sur_B():
        ordre = sorted(archive, key=lambda g: g["fit_eff"])[:10]
        a_faire = [g for g in ordre if g["fit_B"] is None]
        if a_faire:
            res = ev.evaluer_lot([g["source"] for g in a_faire], "B")
            suivre(res)
            for g, rr in zip(a_faire, res):
                g["fit_B"] = rr["bpc"] if rr["ok"] else float("inf")
                ecrire({"type": "B", "id": g["id"], "fit_B": g["fit_B"]})

    def champion():
        cands = [g for g in archive if g["fit_B"] is not None]
        return min(cands, key=lambda g: g["fit_B"])

    def point():
        top_sur_B()
        ch = champion()
        c = cpu_total()
        bA = min(g["fit_A"] for g in archive)
        pt = {"evals": depenses, "gen": gen, "best_A": bA, "best_B": ch["fit_B"], "champion_id": ch["id"],
              "taille_archive": len(archive), "temps_s": round(time.time() - t0, 1), "cpu_s": round(c, 1),
              "gain_par_cpu_s": (seed_A - bA) / c if c > 0 else 0.0}
        if cartes:
            pt["couverture_elites"] = cartes.couverture()
        points.append(pt)
        ecrire({"type": "point", **pt})
        if verbeux:
            print("[%s] evals=%d gen=%d bestA=%.4f bestB=%.4f archive=%d cpu=%.0fs" % (
                nom, depenses, gen, bA, ch["fit_B"], len(archive), c), flush=True)

    prochain_point = 250
    prochain_point_cpu = 25.0
    while depenses < budget:
        gen += 1
        if gen % 25 == 0:
            controle_empreintes()
        if objectif and gen % 10 == 1:
            idx = composants.front_pareto([(g["fit_A"], g["mdl_bits"]) for g in archive])
            front_ids = {archive[i]["id"] for i in idx}
        w = poids_parents(archive)
        if objectif and front_ids:
            w = w * np.array([1.5 if g["id"] in front_ids else 1.0 for g in archive])
            w = w / w.sum()
        lot = min(taille_lot, budget - depenses)
        nrg = np.random.default_rng(rng.getrandbits(32))
        fits = np.array([g["fit_A"] for g in archive])
        tri = np.sort(fits)
        sources_arch = [g["source"] for g in archive]
        ids_elites = cartes.ids() if cartes else []
        enfants = []
        essais = 0
        while len(enfants) < lot and essais < lot * 8:
            essais += 1
            if cartes and ids_elites and rng.random() < 0.5:
                par = archive[ids_elites[rng.randrange(len(ids_elites))]]
            else:
                par = archive[int(nrg.choice(len(archive), p=w))]
            meta_enf = mutations.meta_heritee(par["meta"], rng) if mode_meta == "auto" else mutations.meta_defaut()
            src, noms, ok, raison = mutations.muter(par["source"], par["meta"], rng, sources_arch, K=K, deja_vus=vus)
            if raison == "doublon":
                compteur_rejets["doublon"] += 1
                continue
            vus.add(src)
            x = p_pred = None
            if auto_modele is not None:
                rang = float(np.searchsorted(tri, par["fit_A"])) / len(tri)
                x = auto_modele.vecteur(noms, rang, len(src) - len(par["source"]))
                evaluer, p_pred = auto_modele.decider(x, rng)
                if not evaluer:
                    continue
            enfants.append((par, src, noms, meta_enf, x, p_pred))
        if enfants:
            res = ev.evaluer_lot([e[1] for e in enfants], "A")
            suivre(res)
            for (par, src, noms, meta_enf, x, p_pred), rr in zip(enfants, res):
                depenses += 1
                cpu_evals.append(rr.get("cpu_s", 0.0))
                gagne = rr["ok"] and rr["bpc"] < par["fit_A"] - 1e-4
                if auto_modele is not None:
                    auto_modele.observer(x, p_pred, noms, gagne)
                if not rr["ok"]:
                    compteur_rejets[rr["raison"]] += 1
                    ecrire({"type": "rejet", "parent": par["id"], "mutations": noms, "raison": rr["raison"],
                            "detail": rr.get("detail", ""), "source": src})
                    continue
                if rr["bpc"] > archive[0]["fit_A"] + 0.3:
                    nb_ecartes += 1   # valide mais bien pire que le départ : non gardé
                    continue
                par["enfants"] += 1
                g = {"id": len(archive), "parent": par["id"], "mutations": noms, "source": src,
                     "fit_A": rr["bpc"], "fit_B": None, "meta": meta_enf, "enfants": 0, "gen": gen,
                     "cpu_s": rr.get("cpu_s", 0.0), "mdl_bits": composants.taille_mdl_bits(src)}
                g["fit_eff"] = fit_eff(g["fit_A"], src, g["cpu_s"])
                archive.append(g)
                if cartes:
                    cartes.inserer(g, g["fit_eff"])
                ecrire({"type": "genome", **g})
        else:
            depenses += lot // 2 + 1  # sécurité : évite une boucle sans fin
        if auto_modele is not None:
            auto_modele.reentrainer()
        if gen % 10 == 0:
            top_sur_B()
            if arret is not None and arret.mise_a_jour(depenses, champion()["fit_B"]):
                raison_arret = arret.raison
                break
        c = cpu_total()
        if depenses >= prochain_point or c >= prochain_point_cpu:
            if depenses >= prochain_point:
                prochain_point = (depenses // 250 + 1) * 250
            prochain_point_cpu = (c // 25 + 1) * 25.0
            point()
        if budget_cpu is not None and c >= budget_cpu:
            raison_arret = "budget_cpu (%.0f s)" % budget_cpu
            break
    if not points or points[-1]["evals"] != depenses or points[-1]["cpu_s"] != round(cpu_total(), 1):
        point()
    ch = champion()
    ev.fermer()
    log.close()
    controle_empreintes()
    cpu_final = cpu_total()
    resume = {
        "nom": nom, "graine": graine, "meta": mode_meta, "budget": budget, "evals": depenses,
        "generations": gen, "taille_archive": len(archive), "ecartes_trop_mauvais": nb_ecartes,
        "rejets": dict(compteur_rejets), "rejets_garde_fous": dict(ev.rejets),
        "points": points, "seed_A": seed_A, "seed_B": archive[0]["fit_B"],
        "champion": {"id": ch["id"], "fit_A": ch["fit_A"], "fit_B": ch["fit_B"], "source": ch["source"]},
        "temps_mur_s": round(time.time() - t0, 1),
        "temps_cpu_parent_s": round(time.process_time() - cpu0, 1),
        "cpu_total_s": round(cpu_final, 1), "cpu_travailleurs_s": round(cpu_trav[0], 1),
        "cpu_par_evaluation_s": {"moyenne": float(np.mean(cpu_evals)) if cpu_evals else None,
                                 "mediane": float(np.median(cpu_evals)) if cpu_evals else None,
                                 "max": float(np.max(cpu_evals)) if cpu_evals else None},
        "gain_bpc_A": seed_A - min(g["fit_A"] for g in archive),
        "gain_par_cpu_s": (seed_A - min(g["fit_A"] for g in archive)) / cpu_final,
        "options": {"soi": bool(soi), "objectif": bool(objectif), "cout": bool(cout), "elites": bool(elites),
                    "patience": patience, "delta": delta, "cible": cible, "budget_cpu": budget_cpu,
                    "lambda_mdl": lambda_mdl, "mu_cout": mu_cout},
        "raison_arret": raison_arret,
        "meta_moyen_final": meta_moyenne(archive),
        "journal_rejets": ev.journal[:200],
    }
    if auto_modele is not None:
        resume["soi"] = auto_modele.rapport()
    if objectif:
        idx = composants.front_pareto([(g["fit_A"], g["mdl_bits"]) for g in archive])
        resume["front_pareto"] = sorted([{"id": archive[i]["id"], "fit_A": archive[i]["fit_A"],
                                          "mdl_bits": archive[i]["mdl_bits"]} for i in idx],
                                        key=lambda d: d["fit_A"])
    if cartes:
        resume["elites"] = rapport_emergence(archive, cartes)
    with open(os.path.join(dossier, nom + "_resume.json"), "w") as f:
        json.dump(resume, f, indent=1, ensure_ascii=False)
    with open(os.path.join(dossier, nom + "_final.jsonl"), "w") as f:
        for g in archive:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    return resume


TECHNIQUES_CONNUES = {
    "remise": "remise absolue / Kneser-Ney (soustraire une remise D du compte) : à vérifier à la main",
    "utilise_u1": "nombre de symboles vus une fois : idée de Good-Turing / Witten-Bell / PPM-D",
    "utilise_u2": "nombre de symboles vus deux fois : idée de Good-Turing / Kneser-Ney modifié",
    "puissance": "aplatissement (température) des scores : technique connue de calibration",
    "constantes_par_ordre": "poids différents par ordre : technique connue (interpolation à poids par ordre)",
    "variable_intermediaire": "pas une technique : simple réécriture",
}


def rapport_emergence(archive, cartes):
    """Structures présentes dans les élites et absentes du génome de départ."""
    car0 = composants.caracteristiques(archive[0]["source"])
    par_id = {g["id"]: g for g in archive}
    elites = [par_id[i] for i in cartes.ids()]
    emergent = {}
    for g in elites:
        for c in composants.caracteristiques(g["source"]) - car0:
            e = emergent.setdefault(c, {"nb_elites": 0, "meilleur_A": 1e9, "meilleur_B": None})
            e["nb_elites"] += 1
            e["meilleur_A"] = min(e["meilleur_A"], g["fit_A"])
            if g["fit_B"] is not None:
                e["meilleur_B"] = g["fit_B"] if e["meilleur_B"] is None else min(e["meilleur_B"], g["fit_B"])
    for c, e in emergent.items():
        e["technique_connue"] = TECHNIQUES_CONNUES.get(c, "fonction utilitaire (clip, abs, ...) : pas une technique nommée")
    # recherche dans toute l'archive (pas seulement les élites)
    avec_remise = [g for g in archive if "remise" in composants.caracteristiques(g["source"])]
    return {"cases_remplies": len(cartes.cases), "cases_total": composants.NB_CASES_TOTAL,
            "couverture": cartes.couverture(), "nb_elites": len(elites),
            "caracteristiques_absentes_du_depart": emergent,
            "genomes_avec_remise_dans_archive": len(avec_remise),
            "meilleur_avec_remise": (min(avec_remise, key=lambda g: g["fit_A"])["source"]
                                     if avec_remise else None),
            "meilleur_avec_remise_fit_A": (min(g["fit_A"] for g in avec_remise) if avec_remise else None)}


def point_fixe(nom, source, graine, budget=600, delta=0.002, **kw):
    """Relance l'évolution depuis le champion : s'il n'y a plus de gain > delta, c'est un point fixe."""
    r = executer(nom, budget, graine, "auto", source_depart=source, verbeux=False, **kw)
    gain_A = r["seed_A"] - min(p["best_A"] for p in r["points"])
    return {"budget": budget, "A_depart": r["seed_A"], "A_apres": min(p["best_A"] for p in r["points"]),
            "gain_A": gain_A, "delta": delta, "point_fixe": gain_A <= delta,
            "B_depart": r["seed_B"], "B_apres": r["champion"]["fit_B"]}


def meta_moyenne(archive):
    dern = archive[-max(1, len(archive) // 5):]
    out = {"sigma": float(np.mean([g["meta"]["sigma"] for g in dern])),
           "nmut": float(np.mean([g["meta"]["nmut"] for g in dern])), "poids": {}}
    for op in mutations.OPERATEURS:
        out["poids"][op] = float(np.mean([g["meta"]["poids"][op] for g in dern]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=int, default=6000, help="nombre d'évaluations (enfants)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--meta", choices=["auto", "fixe"], default="auto")
    ap.add_argument("--nom", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--complet", action="store_true", help="active soi, objectif MDL+Pareto, coût, MAP-Elites")
    ap.add_argument("--patience", type=int, default=None, help="arrêt précoce : évaluations sans gain > delta sur B")
    ap.add_argument("--delta", type=float, default=0.002)
    ap.add_argument("--cible", type=float, default=None, help="arrêt dès que bpc B <= cible")
    ap.add_argument("--budget-cpu", type=float, default=None, help="arrêt à ce temps CPU cumulé (s)")
    ap.add_argument("--point-fixe", action="store_true", help="relance depuis le champion pour tester le point fixe")
    a = ap.parse_args()
    nom = a.nom or "run_%s_s%d" % (a.meta, a.seed)
    r = executer(nom, a.budget, a.seed, a.meta, nproc=a.workers, complet=a.complet, patience=a.patience,
                 delta=a.delta, cible=a.cible, budget_cpu=a.budget_cpu)
    if a.point_fixe:
        pf = point_fixe(nom + "_pf", r["champion"]["source"], a.seed + 77, delta=a.delta)
        r["point_fixe"] = pf
        with open(os.path.join(DOSSIER_ARCHIVE, nom + "_resume.json")) as f:
            res = json.load(f)
        res["point_fixe"] = pf
        with open(os.path.join(DOSSIER_ARCHIVE, nom + "_resume.json"), "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        print("point fixe :", pf)
    print(json.dumps({k: r[k] for k in ("nom", "evals", "seed_A", "seed_B", "champion", "temps_mur_s")}, indent=1)[:1500])
