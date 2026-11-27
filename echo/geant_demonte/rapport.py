"""Assemble resultats.json et les chiffres de resultats.md a partir de results/runs (TEST)."""
import json, os
import numpy as np
import analyse as A, bibliotheque as bib, bm25, fuite, invites as I

ICI = A.ICI
G = json.load(open(os.path.join(ICI, "format_gele.json")))


def run(sys_, split="test"):
    g = G[sys_]
    nom = f"{sys_}_{split}_{g['fmt']}"
    if not os.path.exists(os.path.join(A.RUNS, nom + ".jsonl.meta.json")):
        return None
    rows = A.charger(nom)
    meta = json.load(open(os.path.join(A.RUNS, nom + ".jsonl.meta.json")))
    c = A.correct(rows, g["score"])
    lo, hi = A.bootstrap_ic(c)
    return dict(nom=nom, fmt=g["fmt"], score=g["score"], n=len(c), acc=float(c.mean()), ic95=[lo, hi],
                secs_par_question=float(np.mean([r["secs"] for r in rows])),
                secs_mediane=float(np.median([r["secs"] for r in rows])),
                rss_pic_mo=meta["rss_pic_mo"], chargement_s=meta["chargement_s"], _c=c, _rows=rows)


def main():
    R = {s: run(s) for s in ("A", "B", "C", "D", "Ctrl", "Bsans")}
    out = dict(hasard=0.25, graine_questions=20260801, formats_geles=G)
    for s, r in R.items():
        if r:
            out[s] = {k: v for k, v in r.items() if not k.startswith("_")}
    qs = I.charger_questions("test")
    out["n_test"] = len(qs)
    out["n_commandes_test"] = len({q["cmd"] for q in qs})
    out["n_commandes_total"] = len({q["cmd"] for q in I.charger_questions()})
    # differences appariees (memes questions)
    def diff(a, b):
        ra, rb = R.get(a), R.get(b)
        if not ra or not rb or len(ra["_c"]) != len(rb["_c"]):
            return None
        d = float(ra["_c"].mean() - rb["_c"].mean())
        return dict(diff=d, ic95=list(A.bootstrap_diff(ra["_c"], rb["_c"])))
    out["diff"] = {"C-A": diff("C", "A"), "C-B": diff("C", "B"), "C-Ctrl": diff("C", "Ctrl"),
                   "A-B": diff("A", "B"), "B-Bsans": diff("B", "Bsans")}
    # retrieval
    if R["C"]:
        ret = [r["ret"] for r in R["C"]["_rows"]]
        c = R["C"]["_c"]
        ha = np.array([x["hit_answer"] for x in ret])
        out["retrieval"] = dict(
            rappel_drapeau_reponse=float(ha.mean()), rappel_alias=float(np.mean([x["hit_alias"] for x in ret])),
            bonne_page=float(np.mean([x["hit_page"] for x in ret])),
            distracteurs_moyens_par_question=float(np.mean([x["n_distr_hit"] for x in ret])),
            acc_C_si_reponse_dans_passages=float(c[ha].mean()) if ha.any() else None,
            acc_C_sinon=float(c[~ha].mean()) if (~ha).any() else None, n_hit=int(ha.sum()))
    if R["Ctrl"]:
        ret = [r["ret"] for r in R["Ctrl"]["_rows"]]
        out["retrieval_controle"] = dict(rappel_drapeau_reponse=float(np.mean([x["hit_answer"] for x in ret])),
                                         bonne_page=float(np.mean([x["hit_page"] for x in ret])))
    # fuite
    out["fuite"] = dict(
        ligne_lexicale_test=fuite.precision_lexicale(qs),
        ligne_lexicale_tune=fuite.precision_lexicale(I.charger_questions("tune")),
        part_reponse_avec_recouvrement_test=fuite.part_avec_signal(qs),
        B_sans_enonce=out.get("Bsans"))
    # tailles
    t = bib.taille_bibliotheque()
    out["tailles"] = dict(bibliotheque=t, index_pickle_octets=os.path.getsize(bm25.INDEX_PATH),
                          n_pages=t["n_pages"])
    # familles
    fam = lambda c: "cargo/rust" if c.startswith(("cargo", "rust")) else ("jdk" if c.startswith("j") or c in
                                                                                ("keytool", "rmiregistry", "serialver") else "postgresql")
    out["par_famille"] = {}
    for s in ("A", "B", "C"):
        if R[s]:
            d = {}
            for q, ok in zip(qs, R[s]["_c"]):
                d.setdefault(fam(q["cmd"]), []).append(ok)
            out["par_famille"][s] = {k: dict(n=len(v), acc=float(np.mean(v))) for k, v in d.items()}
    # critere
    if R["A"] and R["C"]:
        d = out["diff"]["C-A"]
        out["critere"] = dict(
            c_bat_a_de_10_points=bool(d["diff"] >= 0.10), diff_points=100 * d["diff"],
            rss_c_sous_4go=bool(R["C"]["rss_pic_mo"] <= 4096), rss_c_mo=R["C"]["rss_pic_mo"],
            reussi=bool(d["diff"] >= 0.10 and R["C"]["rss_pic_mo"] <= 4096),
            borne_basse_diff_au_dessus_de_0=bool(d["ic95"][0] > 0), borne_basse_diff_au_dessus_de_10=bool(d["ic95"][0] > 0.10))
    json.dump(out, open(os.path.join(ICI, "resultats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ("A", "B", "C", "D", "Ctrl", "diff", "retrieval", "critere", "fuite")}, indent=1))


if __name__ == "__main__":
    main()
