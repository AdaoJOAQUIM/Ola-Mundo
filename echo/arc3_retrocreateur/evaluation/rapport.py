"""Agrège runs/j1_*.json en resultats_j1.json et resultats_j1.md (français simple).

python rapport.py --retro500 runs/j1_retro_500.json --retro5000 runs/j1_retro_5000.json
                  --cerveau runs/j1_cerveau_500.json --v1 runs/j1_v1_500.json"""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent

MECAS = ["annuler", "blocs_poussables", "cles_portes", "collectibles", "couleurs", "dangers", "energie", "gravite",
         "interrupteurs", "lumieres", "motif", "selection"]


def resume(res, avec_exact=False):
    niv = sum(r["niveaux"] for r in res)
    tot = sum(r["niveaux_total"] for r in res)
    act = opt = 0
    ratios = []
    for r in res:
        for i in range(r["niveaux"]):
            a, o = r["niveaux_actions"][i], r["niveaux_reference"][i]
            act += a
            opt += o
            ratios.append(a / o)
    out = {"mondes": len(res), "niveaux_resolus": niv, "niveaux_total": tot, "taux_niveaux": round(niv / tot, 4) if tot else None,
           "mondes_termines": sum(1 for r in res if r["etat"] == "WIN"),
           "actions_sur_resolus": act, "optimum_sur_resolus": opt,
           "ratio_actions_sur_optimum": round(act / opt, 3) if opt else None,
           "ratio_median_par_niveau": round(statistics.median(ratios), 3) if ratios else None}
    if avec_exact:
        ex = [bool(r.get("validation", {}).get("exact")) for r in res]
        out["programme_exact"] = sum(ex)
        out["taux_programme_exact"] = round(sum(ex) / len(res), 4)
    return out


def par_meca(res, avec_exact):
    tab = {}
    cles = [(m, lambda r, m=m: m in r["mecaniques"]) for m in MECAS]
    cles += [("but = sortie", lambda r: r["but"] == "sortie"), ("but = panneau", lambda r: r["but"] == "panel")]
    for nom, f in cles:
        sel = [r for r in res if f(r)]
        if sel:
            tab[nom] = resume(sel, avec_exact)
    return tab


def par_difficulte(res, avec_exact):
    return {f"d{d}": resume([r for r in res if r["difficulte"] == d], avec_exact) for d in (1, 2, 3)}


def charger(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else None


def ligne(nom, s, exact):
    e = f" | {s['taux_programme_exact']:.0%}" if exact else " | —"
    return (f"| {nom} | {s['mondes']} | {s['taux_niveaux']:.0%} ({s['niveaux_resolus']}/{s['niveaux_total']}) | "
            f"{s['ratio_actions_sur_optimum']} | {s['ratio_median_par_niveau']}{e} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retro500"); ap.add_argument("--retro5000"); ap.add_argument("--cerveau"); ap.add_argument("--v1")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    r500, r5000, cer, v1 = charger(a.retro500), charger(a.retro5000), charger(a.cerveau), charger(a.v1)
    J = {}
    if r500:
        J["retro_500"] = {"global": resume(r500, True), "par_mecanique": par_meca(r500, True), "par_difficulte": par_difficulte(r500, True)}
        g = J["retro_500"]["global"]
        J["j1"] = {"taux_programme_exact": g["taux_programme_exact"], "seuil_exact": 0.70,
                   "taux_niveaux_500": g["taux_niveaux"], "seuil_niveaux": 0.70,
                   "ratio_actions_sur_optimum": g["ratio_actions_sur_optimum"], "seuil_ratio": 2.0,
                   "atteint": bool(g["taux_programme_exact"] >= 0.70 and g["taux_niveaux"] >= 0.70 and g["ratio_actions_sur_optimum"] <= 2.0)}
    if r5000:
        J["retro_5000"] = {"global": resume(r5000, True), "par_mecanique": par_meca(r5000, True)}
    if cer:
        J["cerveau_500"] = {"global": resume(cer), "par_mecanique": par_meca(cer, False)}
    if v1:
        J["v1_500"] = {"global": resume(v1), "par_mecanique": par_meca(v1, False)}
    (RACINE / "resultats_j1.json").write_text(json.dumps(J, indent=1, ensure_ascii=False))
    L = ["# Résultats J1 — Le Rétro-Créateur, phase 1", ""]
    if "j1" in J:
        j = J["j1"]
        L += ["## Verdict", "",
              f"- Programme exact retrouvé : **{j['taux_programme_exact']:.1%}** (seuil {j['seuil_exact']:.0%}) -> {'OK' if j['taux_programme_exact'] >= j['seuil_exact'] else 'NON atteint'}",
              f"- Niveaux résolus (500 actions) : **{j['taux_niveaux_500']:.1%}** (seuil {j['seuil_niveaux']:.0%}) -> {'OK' if j['taux_niveaux_500'] >= j['seuil_niveaux'] else 'NON atteint'}",
              f"- Actions par niveau résolu / optimum : **{j['ratio_actions_sur_optimum']}** (seuil <= {j['seuil_ratio']}) -> {'OK' if j['ratio_actions_sur_optimum'] <= j['seuil_ratio'] else 'NON atteint'}",
              f"- **J1 : {'ATTEINT' if j['atteint'] else 'NON ATTEINT'}**", ""]
    if r500:
        perdus, inexacts = {}, {}
        tot_perdu = sum(r["niveaux_total"] - r["niveaux"] for r in r500)
        tot_inex = sum(1 for r in r500 if not r.get("validation", {}).get("exact"))
        for r in r500:
            for m in MECAS:
                if m in r["mecaniques"]:
                    perdus[m] = perdus.get(m, 0) + r["niveaux_total"] - r["niveaux"]
                    inexacts[m] = inexacts.get(m, 0) + (0 if r.get("validation", {}).get("exact") else 1)
        J["causes"] = {"niveaux_perdus_total": tot_perdu, "mondes_inexacts_total": tot_inex,
                       "niveaux_perdus_par_mecanique": dict(sorted(perdus.items(), key=lambda x: -x[1])),
                       "mondes_inexacts_par_mecanique": dict(sorted(inexacts.items(), key=lambda x: -x[1]))}
        L += ["## Où les niveaux et les programmes exacts se perdent (500 actions)", "",
              f"Niveaux perdus au total : {tot_perdu} ; mondes au programme inexact : {tot_inex}. Un monde compte pour chaque mécanique qu'il contient (les causes se recouvrent).", "",
              "| Mécanique | Niveaux perdus dans les mondes qui la contiennent | Mondes inexacts qui la contiennent |", "|---|---|---|"]
        for m in J["causes"]["niveaux_perdus_par_mecanique"]:
            L.append(f"| {m} | {perdus[m]} | {inexacts[m]} |")
        L.append("")
    L += [a.notes, ""]
    L += ["## Comparaison sur les mêmes 300 mondes (500 actions)", "",
          "| Agent | Mondes | Niveaux résolus | Actions / optimum (agrégé) | Médiane par niveau | Programme exact |", "|---|---|---|---|---|---|"]
    if r500:
        L.append(ligne("Rétro-Créateur", J["retro_500"]["global"], True))
    if r5000:
        L.append(ligne("Rétro-Créateur (5000 actions)", J["retro_5000"]["global"], True))
    if cer:
        L.append(ligne("Cerveau bayésien (echo/arc3)", J["cerveau_500"]["global"], False))
    if v1:
        L.append(ligne("Scientifique v1", J["v1_500"]["global"], False))
    L.append("")
    if r500:
        L += ["## Par difficulté (Rétro-Créateur, 500 actions)", "", "| Difficulté | Mondes | Niveaux | Actions/optimum | Médiane | Exact |", "|---|---|---|---|---|---|"]
        for d, s in J["retro_500"]["par_difficulte"].items():
            L.append(ligne(d, s, True))
        L += ["", "## Par mécanique (un monde compte pour chaque mécanique qu'il contient)", ""]
        cols = [("Rétro-Créateur 500", J["retro_500"]["par_mecanique"], True)]
        if r5000:
            cols.append(("Rétro-Créateur 5000", J["retro_5000"]["par_mecanique"], True))
        if cer:
            cols.append(("Cerveau", J["cerveau_500"]["par_mecanique"], False))
        if v1:
            cols.append(("v1", J["v1_500"]["par_mecanique"], False))
        L.append("| Mécanique | Mondes | Exact (500) | Niveaux (500) | Actions/optimum (500) | " + " | ".join(f"Niveaux {n}" for n, _, _ in cols[1:]) + " |")
        L.append("|---|---|---|---|---|" + "---|" * (len(cols) - 1))
        for nom in J["retro_500"]["par_mecanique"]:
            s = J["retro_500"]["par_mecanique"][nom]
            autres = " | ".join((f"{c[1][nom]['taux_niveaux']:.0%}" if nom in c[1] else "—") for c in cols[1:])
            L.append(f"| {nom} | {s['mondes']} | {s['taux_programme_exact']:.0%} | {s['taux_niveaux']:.0%} | {s['ratio_actions_sur_optimum']} | {autres} |")
        L.append("")
    (RACINE / "resultats_j1.md").write_text("\n".join(L))
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
