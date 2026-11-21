"""Orchestration : découpe dev/held-out, baseline aléatoire, harnais continu, rapports.

Usage (avec le python du venv arc3) :
  python lancer.py dev            # débogage/tuning + harnais continu sur les 5 jeux de dev
  python lancer.py aleatoire      # baseline aléatoire, 3 graines, 25 jeux
  python lancer.py held           # 20 jeux held-out, agent gelé, une seule fois
  python lancer.py rapport        # écrit resultats_arc3.json / .md
"""
from __future__ import annotations

import json
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
RUNS = ICI / "runs"
RUNS.mkdir(exist_ok=True)

JEUX = sorted("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30".split())
# Découpe pré-enregistrée : liste triée des 25 préfixes, random.Random(2026).sample(.., 5)
DEV = random.Random(2026).sample(JEUX, 5)
HELD = [j for j in JEUX if j not in DEV]
NB_PROCS = 2


def _worker(args):
    import os
    os.chdir(RUNS)                      # les enregistrements du SDK tombent dans runs/ (ignoré par git)
    import logging
    logging.disable(logging.CRITICAL)
    import harnais
    import cerveau
    import aleatoire
    jeu, genre, graine, prieurs = args
    if genre == "cerveau":
        fab = lambda: cerveau.Cerveau(graine, prieurs)
    else:
        fab = lambda: aleatoire.Aleatoire(graine)
    res, pol = harnais.jouer(jeu, fab)
    res.update({"prefixe": jeu, "genre": genre, "graine": graine})
    if genre == "cerveau":
        res["stats_cerveau"] = pol.stats
        res["comptes"] = pol.exporter_comptes()
    return res


def lancer(taches):
    with Pool(NB_PROCS) as p:
        return p.map(_worker, taches, chunksize=1)


# ------------------------------------------------------------------ harnais continu
def mettre_en_commun(liste_comptes, plafond=6.0, groupes=("A", "C")):
    """Prieurs transférables : somme des comptes excédentaires des jeux précédents,
    tempérée (pseudo-comptes a+b <= plafond par groupe et quantité)."""
    tot = {}
    for comptes in liste_comptes:
        for q, gs in comptes.items():
            for g, (a, b) in gs.items():
                if not any(g.startswith(p) for p in groupes):
                    continue
                t = tot.setdefault(q, {}).setdefault(g, [0.0, 0.0])
                t[0] += max(a, 0.0)
                t[1] += max(b, 0.0)
    for q, gs in tot.items():
        for g, ab in gs.items():
            s = ab[0] + ab[1]
            if s > plafond:
                ab[0] *= plafond / s
                ab[1] *= plafond / s
    return tot


def comparer_par_niveau(res_a, res_b):
    """Comparaison appariée niveau par niveau (candidat A vs champion B).
    Pour chaque niveau : résolu par un seul -> victoire ; les deux -> moins d'actions gagne ; sinon égalité."""
    vic = def_ = 0
    for ra, rb in zip(res_a, res_b):
        la, lb = ra["niveaux_actions"] or [], rb["niveaux_actions"] or []
        for i in range(max(ra["niveaux"], rb["niveaux"])):
            sa, sb = i < ra["niveaux"], i < rb["niveaux"]
            if sa and not sb:
                vic += 1
            elif sb and not sa:
                def_ += 1
            elif sa and sb and i < len(la) and i < len(lb):
                if la[i] < lb[i]:
                    vic += 1
                elif lb[i] < la[i]:
                    def_ += 1
    return vic, def_


def p_meilleur(vic, defa, n_mc=200000, graine=7):
    """P(theta > 0.5), theta ~ Beta(1+vic, 1+def) : probabilité que le candidat soit meilleur."""
    rng = random.Random(graine)
    return sum(rng.betavariate(1 + vic, 1 + defa) > 0.5 for _ in range(n_mc // 20)) / (n_mc // 20)


def ecrire(nom, obj):
    (ICI / nom).write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def charger(nom, defaut=None):
    p = ICI / nom
    return json.loads(p.read_text()) if p.exists() else defaut


def phase_dev():
    """Harnais continu sur les jeux de dev, dans l'ordre : v0 -> candidats R1, R2.
    Un raffinement n'est gardé que si P(meilleur) > 0.9 sur les jeux POSTÉRIEURS ; sinon retour arrière."""
    from datetime import datetime
    graines = (0, 1, 2)
    journal = []
    v0 = lancer([(j, "cerveau", g, None) for j in DEV for g in graines])
    v0_par = {(r["prefixe"], r["graine"]): r for r in v0}
    champion = ()          # groupes de prieurs transférés par la version courante
    version = "v0"
    journal.append({"version": "v0", "date": datetime.now().isoformat(timespec="seconds"),
                    "description": "prieurs Beta non informatifs, aucun transfert",
                    "decision": "base"})
    candidats = [("v1", "transfert des prieurs d'actions simples (A1-A5, A7)", ("A",)),
                 ("v2", "transfert des prieurs de clics par couleur/taille d'objet", ("C",))]
    for nom, desc, groupes in candidats:
        cand_groupes = tuple(sorted(set(champion) | set(groupes)))
        taches, champ_res = [], []
        for i, j in enumerate(DEV):
            if i == 0:
                continue                         # aucun jeu antérieur : pas de transfert possible
            avant = [v0_par[(a, 0)]["comptes"] for a in DEV[:i]]
            pri_cand = mettre_en_commun(avant, groupes=cand_groupes)
            pri_champ = mettre_en_commun(avant, groupes=champion) if champion else None
            for g in graines:
                taches.append((j, "cerveau", g, pri_cand))
                taches.append((j, "cerveau", g, pri_champ))
        out = lancer(taches)
        cand_res, champ_res = out[0::2], out[1::2]
        vic, dfa = comparer_par_niveau(cand_res, champ_res)
        p = p_meilleur(vic, dfa)
        ok = p > 0.9
        journal.append({"version": nom, "date": datetime.now().isoformat(timespec="seconds"),
                        "description": desc, "jeux_test": DEV[1:], "graines": list(graines),
                        "victoires_candidat": vic, "defaites_candidat": dfa, "P_meilleur": round(p, 4),
                        "niveaux_candidat": sum(r["niveaux"] for r in cand_res),
                        "niveaux_champion": sum(r["niveaux"] for r in champ_res),
                        "decision": "GARDE" if ok else "RETOUR ARRIERE"})
        if ok:
            champion, version = cand_groupes, nom
    gel = mettre_en_commun([v0_par[(a, 0)]["comptes"] for a in DEV], groupes=champion) if champion else None
    ecrire("journal_versions.json", {"champion": version, "groupes_transfert": list(champion),
                                     "journal": journal})
    ecrire("prieurs_geles.json", gel or {})
    ecrire("res_dev_v0.json", [{k: v for k, v in r.items() if k != "comptes"} for r in v0])
    print(json.dumps(journal, indent=1, ensure_ascii=False))


def phase_held():
    gel = charger("prieurs_geles.json") or None
    out = lancer([(j, "cerveau", 0, gel) for j in HELD])
    # jeux de dev avec l'agent gelé, graine 0 (pour le tableau, signalés « dev »)
    dev = lancer([(j, "cerveau", 0, gel) for j in DEV])
    for r in out + dev:
        r.pop("comptes", None)
    ecrire("res_held.json", out)
    ecrire("res_dev_gele.json", dev)


def phase_aleatoire():
    out = lancer([(j, "aleatoire", g, None) for j in JEUX for g in (0, 1, 2)])
    ecrire("res_aleatoire.json", out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "essai":      # tuning rapide : v0 sur les jeux de dev uniquement
        out = lancer([(j, "cerveau", 0, None) for j in DEV])
        for r in out:
            print(r["prefixe"], r["etat"], f"{r['niveaux']}/{r['niveaux_total']}", r["actions"], r["niveaux_actions"], r["cpu_s"], r["stats_cerveau"])
    elif cmd == "dev":
        phase_dev()
    elif cmd == "held":
        phase_held()
    elif cmd == "aleatoire":
        phase_aleatoire()
    elif cmd == "rapport":
        from rapport import ecrire_rapport
        ecrire_rapport()
