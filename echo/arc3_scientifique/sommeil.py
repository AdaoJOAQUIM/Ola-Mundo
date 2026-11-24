"""Sommeil : entre les jeux DEV, l'agent analyse ses propres échecs de prédiction et énumère des gabarits de règles
(conjonctions de <= 2 primitives de forme) ; le juge épinglé (juge.py) ne garde que ceux qui améliorent la
prédiction sur des jeux DEV tenus de côté. Aucun LLM. Produit prior_gele.json (apprentissage de l'apprentissage).

Usage : python sommeil.py res_dev_sans_prior.json
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import juge

ICI = Path(__file__).resolve().parent


def verifier_empreinte():
    emp = json.loads((ICI / "empreintes.json").read_text())
    h = hashlib.sha256((ICI / "juge.py").read_bytes()).hexdigest()
    if emp["juge.py"] != h:
        raise SystemExit("juge.py modifié : empreinte invalide")


def analyser(res):
    """Étiquettes par jeu (graine 0 uniquement, jeux DEV)."""
    par = {}
    for r in res:
        if r["graine"] == 0:        # étiquettes poolées sur les variantes d'exploration (même jeu)
            par.setdefault(r["prefixe"], []).extend(e for e in r["etiquettes"] if e["cars"])
    return par


def candidats(labels_par_jeu, base):
    """Énumération par composition : primitives présentes dans les étiquettes mal prédites par la base."""
    tous = [l for ls in labels_par_jeu.values() for l in ls]
    echecs = [l for l in tous if base[l["type"]] < 0.2]       # types rares pour le prior de base
    lits = [(f, v) for f in juge.PRIMITIVES for v in (True, False)]
    cand = []
    for k in (1, 2):
        for combo in itertools.combinations(lits, k):
            if len({f for f, _ in combo}) < k:
                continue
            pred = dict(combo)
            # pertinent s'il couvre au moins un échec, ou s'il discrimine
            if any(juge._match(pred, l["cars"]) for l in echecs) or k == 1:
                cand.append(pred)
    return cand


def dormir(res):
    verifier_empreinte()
    par = analyser(res)
    tous = [l for ls in par.values() for l in ls]
    base = juge._base(tous)
    cand = candidats(par, base)
    gardes, journal = [], []
    courant = []
    for tour in range(4):
        meilleur, mres = None, None
        for pred in cand:
            if pred in courant:
                continue
            r = juge.evaluer(par, courant + [pred])
            if mres is None or r["gain_moyen"] > mres["gain_moyen"]:
                meilleur, mres = pred, r
        if meilleur is None:
            break
        ok = juge.accepte(mres)
        journal.append({"tour": tour, "gabarit": meilleur, "resultat": mres, "garde": ok})
        if not ok:
            break
        courant.append(meilleur)
    final = juge.ajuster(tous, courant)
    # le prior brut (frequences DEV) rend « win » très improbable et bride la recherche de l'objectif :
    # on le tempère à moitié vers l'uniforme (choix fait sur DEV, voir resultats).
    base_t = [0.5 * x + 0.5 * 0.2 for x in base]
    prior = {"base": base_t, "base_brut": base, "gabarits": [{"pred": p, "probs": q} for p, q in final]}
    (ICI / "prior_gele.json").write_text(json.dumps(prior, indent=1))
    (ICI / "sommeil_journal.json").write_text(json.dumps(
        {"n_etiquettes": len(tous), "par_jeu": {j: len(v) for j, v in par.items()}, "base": base,
         "frequences_types": dict(zip(["pass", "block", "lethal", "collect", "win"], base)),
         "candidats_testes": len(cand), "journal": journal}, indent=1))
    return prior, journal


if __name__ == "__main__":
    res = [r for f in sys.argv[1:] for r in json.loads(Path(f).read_text())]
    prior, journal = dormir(res)
    print(json.dumps(journal, indent=1))
