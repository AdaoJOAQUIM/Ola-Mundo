"""Choisit sur TUNE seulement le format d'invite et la normalisation du score ; ecrit format_gele.json."""
import json, os, sys
import analyse as A

ORDRE = [("cloze", "sum"), ("cloze", "mean"), ("liste", "sum"), ("liste", "mean")]


def choisir(systeme):
    res = []
    for fmt, sc in ORDRE:
        nom = f"{systeme}_tune_{fmt}"
        if os.path.exists(os.path.join(A.RUNS, nom + ".jsonl.meta.json")):
            c = A.correct(A.charger(nom), sc)
            res.append((fmt, sc, float(c.mean()), len(c)))
    best = max(res, key=lambda r: r[2])   # max renvoie le premier en cas d'egalite : l'ordre = simplicite
    return dict(fmt=best[0], score=best[1], acc_tune=best[2], n_tune=best[3],
                toutes=[dict(fmt=f, score=s, acc=a, n=n) for f, s, a, n in res])


if __name__ == "__main__":
    systemes = sys.argv[1:] or ["A", "B", "C"]
    g = {s: choisir(s) for s in systemes}
    g["D"] = dict(g["A"], note="D reprend le format choisi pour A")
    g["Ctrl"] = dict(g["C"], note="le controle reprend le format de C")
    g["Bsans"] = dict(g["B"], note="invite sans enonce ; normalisation de B")
    json.dump(g, open(os.path.join(A.ICI, "format_gele.json"), "w"), indent=1)
    for s, v in g.items():
        print(s, v["fmt"], v["score"], round(v["acc_tune"], 3), v["n_tune"])
