"""Analyse du champion sur A et B seulement (jamais C).

1. Retire un à un les éléments de la formule du champion.
2. Cherche, par grille sur A, la meilleure « référence à la main » à deux réglages :
   e = c * d  (masse d'échappement) et une puissance finale g (aplatissement).
Écrit analyse_champion.json.
"""
import json
import os
import re
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import donnees  # noqa: E402
import genome_initial  # noqa: E402
import juge  # noqa: E402

MODELE = '''def estimer(n, N, d, u1, u2, K):
    p = 0 * n[0] + 1.0 / n.shape[-1]
    for k in range(K + 1):
        e = d[k] * %s
        p = np.where(N[k] > 0, div(n[k] + e * p, N[k] + e), p)
    p = p ** %s
    return p
'''


def main(run="principal"):
    ab = donnees.charger_ab()
    champ = json.load(open(os.path.join(ICI, "archive", run + "_resume.json")))["champion"]["source"]

    def ev(s):
        a, b = juge.juger(s, ab["A"]), juge.juger(s, ab["B"])
        return {"A": round(a["bpc"], 4), "B": round(b["bpc"], 4)}

    sortie = {"depart": ev(genome_initial.SOURCE), "champion": ev(champ), "ablations_du_champion": {}}
    v = {}
    v["sans la puissance finale"] = re.sub(r"    p = p \*\* [0-9.]+\n", "", champ)
    v["sans les constantes par ordre (une seule valeur, 2.2)"] = re.sub(r"\[2\.17814.*?\]\[k\]", "2.2", champ)
    v["sans le terme u1"] = champ.replace(" + u1[k]\n", "\n")
    v["sans le terme (e + m) : seulement m"] = champ.replace("(e + m) * p", "m * p")
    v["sans le terme m : seulement e"] = champ.replace("(e + m) * p", "e * p")
    for k, s in v.items():
        if s != champ:
            sortie["ablations_du_champion"][k] = ev(s)
    meilleur = None
    for c in (2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64):
        for g in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0):
            a = juge.juger(MODELE % (c, g), ab["A"])["bpc"]
            if meilleur is None or a < meilleur[0]:
                meilleur = (a, c, g)
    src = MODELE % (meilleur[1], meilleur[2])
    sortie["reference_a_la_main"] = {"c": meilleur[1], "gamma": meilleur[2], "source": src, **ev(src)}
    with open(os.path.join(ICI, "analyse_champion.json"), "w", encoding="utf-8") as f:
        json.dump(sortie, f, indent=1, ensure_ascii=False)
    print(json.dumps(sortie, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
