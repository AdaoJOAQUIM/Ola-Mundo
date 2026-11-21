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
    fit = np.array([g["fit_A"] for g in archive])
    meilleur = fit.min()
    ref = archive[0]["fit_A"]
    plage = max(ref - meilleur, 0.0) + 0.1
    z = np.clip(1.0 - (fit - meilleur) / plage, 0.0, 1.0)
    enfants = np.array([g["enfants"] for g in archive], dtype=float)
    w = sigmoide(10 * (z - 0.5)) / (1.0 + enfants)
    return w / w.sum()


def executer(nom, budget, graine, mode_meta="auto", nproc=4, taille_lot=8, delai=20,
             dossier=DOSSIER_ARCHIVE, verbeux=True, K=None):
    t0 = time.time()
    cpu0 = time.process_time()
    controle_empreintes()
    os.makedirs(dossier, exist_ok=True)
    rng = random.Random(graine)
    jeux = donnees.charger_ab()
    K = jeux["A"]["n"].shape[0] - 1
    ev = garde_fous.Evaluateur({"A": jeux["A"], "B": jeux["B"]}, nproc=nproc, delai=delai)
    chemin_log = os.path.join(dossier, nom + ".jsonl")
    log = open(chemin_log, "w")

    def ecrire(obj):
        log.write(json.dumps(obj, ensure_ascii=False) + "\n")
        log.flush()

    r = ev.evaluer_lot([genome_initial.SOURCE], "A")[0]
    rb = ev.evaluer_lot([genome_initial.SOURCE], "B")[0]
    assert r["ok"] and rb["ok"], "le génome de départ doit être valide"
    graine_g = {"id": 0, "parent": None, "mutations": [], "source": genome_initial.SOURCE,
                "fit_A": r["bpc"], "fit_B": rb["bpc"], "meta": mutations.meta_defaut(),
                "enfants": 0, "gen": 0}
    archive = [graine_g]
    vus = {genome_initial.SOURCE}
    ecrire({"type": "genome", **graine_g})
    compteur_rejets = Counter()
    depenses = 0
    gen = 0
    points = []
    nb_ecartes = 0

    def top_sur_B():
        ordre = sorted(archive, key=lambda g: g["fit_A"])[:10]
        a_faire = [g for g in ordre if g["fit_B"] is None]
        if a_faire:
            res = ev.evaluer_lot([g["source"] for g in a_faire], "B")
            for g, rr in zip(a_faire, res):
                g["fit_B"] = rr["bpc"] if rr["ok"] else float("inf")
                ecrire({"type": "B", "id": g["id"], "fit_B": g["fit_B"]})

    def champion():
        cands = [g for g in archive if g["fit_B"] is not None]
        return min(cands, key=lambda g: g["fit_B"])

    def point():
        top_sur_B()
        ch = champion()
        pt = {"evals": depenses, "gen": gen, "best_A": min(g["fit_A"] for g in archive),
              "best_B": ch["fit_B"], "champion_id": ch["id"], "taille_archive": len(archive),
              "temps_s": round(time.time() - t0, 1)}
        points.append(pt)
        ecrire({"type": "point", **pt})
        if verbeux:
            print("[%s] evals=%d gen=%d bestA=%.4f bestB=%.4f archive=%d (%.0fs)" % (
                nom, depenses, gen, pt["best_A"], pt["best_B"], len(archive), pt["temps_s"]), flush=True)

    prochain_point = 250
    while depenses < budget:
        gen += 1
        if gen % 25 == 0:
            controle_empreintes()
        w = poids_parents(archive)
        lot = min(taille_lot, budget - depenses)
        parents = [archive[i] for i in np.random.default_rng(rng.getrandbits(32)).choice(len(archive), size=lot, p=w)]
        sources_arch = [g["source"] for g in archive]
        enfants = []
        for par in parents:
            meta_enf = mutations.meta_heritee(par["meta"], rng) if mode_meta == "auto" else mutations.meta_defaut()
            src, noms, ok, raison = mutations.muter(par["source"], par["meta"], rng, sources_arch, K=K, deja_vus=vus)
            if raison == "doublon":
                compteur_rejets["doublon"] += 1
                continue
            vus.add(src)
            enfants.append((par, src, noms, meta_enf))
        if enfants:
            res = ev.evaluer_lot([e[1] for e in enfants], "A")
            for (par, src, noms, meta_enf), rr in zip(enfants, res):
                depenses += 1
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
                     "fit_A": rr["bpc"], "fit_B": None, "meta": meta_enf, "enfants": 0, "gen": gen}
                archive.append(g)
                ecrire({"type": "genome", **g})
        else:
            depenses += lot // 2 + 1  # sécurité : évite une boucle sans fin
        if gen % 10 == 0:
            top_sur_B()
        while depenses >= prochain_point and prochain_point <= budget:
            point()
            prochain_point += 250
    if not points or points[-1]["evals"] != depenses:
        point()
    ch = champion()
    ev.fermer()
    log.close()
    controle_empreintes()
    resume = {
        "nom": nom, "graine": graine, "meta": mode_meta, "budget": budget, "evals": depenses,
        "generations": gen, "taille_archive": len(archive), "ecartes_trop_mauvais": nb_ecartes,
        "rejets": dict(compteur_rejets), "rejets_garde_fous": dict(ev.rejets),
        "points": points, "seed_A": archive[0]["fit_A"], "seed_B": archive[0]["fit_B"],
        "champion": {"id": ch["id"], "fit_A": ch["fit_A"], "fit_B": ch["fit_B"], "source": ch["source"]},
        "temps_mur_s": round(time.time() - t0, 1),
        "temps_cpu_parent_s": round(time.process_time() - cpu0, 1),
        "meta_moyen_final": meta_moyenne(archive),
        "journal_rejets": ev.journal[:200],
    }
    with open(os.path.join(dossier, nom + "_resume.json"), "w") as f:
        json.dump(resume, f, indent=1, ensure_ascii=False)
    with open(os.path.join(dossier, nom + "_final.jsonl"), "w") as f:
        for g in archive:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    return resume


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
    a = ap.parse_args()
    nom = a.nom or "run_%s_s%d" % (a.meta, a.seed)
    r = executer(nom, a.budget, a.seed, a.meta, nproc=a.workers)
    print(json.dumps({k: r[k] for k in ("nom", "evals", "seed_A", "seed_B", "champion", "temps_mur_s")}, indent=1)[:1500])
