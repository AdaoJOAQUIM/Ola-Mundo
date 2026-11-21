"""Script final : SEUL module autorisé à lire E (paragraphes retenus de la bibliothèque) et le français.

Usage :
  python3 final.py --essai   # répétition générale : E := B et « français » := A (tranches déjà utilisées), rien n'est écrit
  python3 final.py           # exécution unique sur E et le français ; refuse de tourner deux fois (témoin)

Les paramètres (α de fixed-share, Pitman-Yor, gagnant d'Optuna) sont figés par dev.py à partir de A et B.
Le texte français n'est ni écrit ni mis en cache : seuls des nombres sortent.
"""
import json
import lzma
import math
import os
import sys
import time
from multiprocessing.pool import ThreadPool

import numpy as np

import donnees_bayes as db
import experts as ex
import melange as ml
import dev as dv

TEMOIN = os.path.join(db.CACHE, "final_utilise.json")
SORTIE_JSON = os.path.join(db.ICI, "resultats_bayes_langue.json")
FENETRE = 2000
BETAS_ETIQ = ["statique"] + ["β=%g" % b for b in ex.BETAS] + ["références", "Optuna"]
ORDRES_ETIQ = ["ordre %d" % o for o in ex.ORDRES_GRILLE] + ["références", "Optuna"]


def groupes_de(metas):
    g_ordre, g_beta = [], []
    for m in metas:
        if m["famille"] in ("ref", "optuna"):
            lab = "références" if m["famille"] == "ref" else "Optuna"
            g_ordre.append(ORDRES_ETIQ.index(lab))
            g_beta.append(BETAS_ETIQ.index(lab))
        else:
            g_ordre.append(ex.ORDRES_GRILLE.index(m["ordre"]))
            g_beta.append(0 if m["beta"] is None else 1 + ex.BETAS.index(m["beta"]))
    return np.array(g_ordre), np.array(g_beta)


def lzma_taille(s):
    return len(lzma.compress(s.encode("utf-8"), preset=9 | lzma.PRESET_EXTREME))


def lzma_fenetres(lib, texte, n_fen, fen=FENETRE):
    """bpc par fenêtre de `fen` caractères : (compress(lib+préfixe+fenêtre) - compress(lib+préfixe)) · 8 / fen.
    Les tailles successives sont réutilisées (n_fen + 1 compressions)."""
    chaines = [lib + texte[:fen * i] for i in range(n_fen + 1)]
    with ThreadPool(2) as tp:
        tailles = tp.map(lzma_taille, chaines)
    return [8.0 * (tailles[i + 1] - tailles[i]) / fen for i in range(n_fen)], tailles[0]


def resume_melange(r, metas, g_ordre, g_beta, positions):
    ent = r["entropie"]
    pos = [p for p in positions if p < len(ent)]
    return {"bpc": float(r["pertes"].mean()),
            "entropie_bits": {str(p): float(ent[p]) for p in pos},
            "nb_experts_effectif": {str(p): float(2 ** ent[p]) for p in pos},
            "poids_final_max": float(r["poids_finaux"].max()),
            "expert_final_dominant": metas[int(np.argmax(r["poids_finaux"]))]["nom"]}


def masses_tableau(r_ord, r_beta):
    return {"positions": r_ord["positions_snap"],
            "par_ordre": {ORDRES_ETIQ[j]: r_ord["masses"][:, j].tolist() for j in range(len(ORDRES_ETIQ))},
            "par_beta": {BETAS_ETIQ[j]: r_beta["masses"][:, j].tolist() for j in range(len(BETAS_ETIQ))}}


def analyser(nom, jeu, fig, lib, texte, n_fen_lzma, fen_pertes, pas_snap):
    """Experts + mélanges sur un flux. Retourne (dict de mesures, L (M,T), metas, pertes des mélanges)."""
    t0 = time.time()
    metas, L = dv.tous_experts(jeu, fig["py_params"], fig["params_optuna"])
    print("[%s] experts calculés : %s (%.0f s)" % (nom, L.shape, time.time() - t0), flush=True)
    P = ml.proba_depuis_log2(L)
    g_ordre, g_beta = groupes_de(metas)
    sans = [i for i, m in enumerate(metas) if m["famille"] != "optuna"]
    res = {"n_experts": len(metas), "T": int(L.shape[1])}
    bpc = -L.mean(axis=1)
    idx = {m["nom"]: i for i, m in enumerate(metas)}
    res["bpc_experts_cles"] = {n: float(bpc[idx[n]]) for n in
                               ("départ (PPM-C)", "champion évolué", "référence à la main (e=16·d)",
                                "Pitman-Yor (A)", "Optuna TPE (gagnant sur B)")}
    mel = {}
    for etiq, cols in (("avec_optuna", list(range(len(metas)))), ("sans_optuna", sans)):
        a = fig["alphas"][etiq]["alpha"]
        Pc = np.ascontiguousarray(P[:, cols])
        gc_o, gc_b = g_ordre[cols], g_beta[cols]
        rb_o = ml.melanger(Pc, 0.0, None, gc_o, pas_snap, len(ORDRES_ETIQ))
        rb_b = ml.melanger(Pc, 0.0, None, gc_b, pas_snap, len(BETAS_ETIQ))
        rf_o = ml.melanger(Pc, a, None, gc_o, pas_snap, len(ORDRES_ETIQ))
        rf_b = ml.melanger(Pc, a, None, gc_b, pas_snap, len(BETAS_ETIQ))
        mc = [metas[i] for i in cols]
        pos_ent = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 40000, L.shape[1] - 1]
        mel[etiq] = {"alpha": a,
                     "bma": dict(resume_melange(rb_o, mc, gc_o, gc_b, pos_ent), masses=masses_tableau(rb_o, rb_b)),
                     "fixed_share": dict(resume_melange(rf_o, mc, gc_o, gc_b, pos_ent),
                                         masses=masses_tableau(rf_o, rf_b))}
        mel[etiq]["_pertes_bma"] = rb_o["pertes"]
        mel[etiq]["_pertes_fs"] = rf_o["pertes"]
        mel[etiq]["_cols"] = cols
        print("[%s] %s : BMA %.4f, fixed-share %.4f" % (nom, etiq, rb_o["pertes"].mean(), rf_o["pertes"].mean()),
              flush=True)
    res["melanges"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in mel.items()}
    # oracle : meilleur expert isolé a posteriori
    j = int(np.argmin(bpc))
    res["meilleur_expert_oracle"] = {"nom": metas[j]["nom"], "bpc": float(bpc[j]),
                                     "avertissement": "ORACLE : choisi en connaissant le flux évalué"}
    # meilleur expert choisi sur A (non oracle) par famille
    for fam in ("grille", "maj"):
        ii = [i for i, m in enumerate(metas) if m["famille"] == fam]
        ja = min(ii, key=lambda i: fig["bpc_A_experts"][metas[i]["nom"]])
        jo = min(ii, key=lambda i: bpc[i])
        res["meilleur_%s_choisi_sur_A" % fam] = {"nom": metas[ja]["nom"], "bpc": float(bpc[ja])}
        res["meilleur_%s_oracle" % fam] = {"nom": metas[jo]["nom"], "bpc": float(bpc[jo])}
    return res, L, metas, mel, bpc


def fenetres(vec, fen=FENETRE):
    n = len(vec) // fen
    return [float(np.mean(vec[i * fen:(i + 1) * fen])) for i in range(n)]


def calculer(E, F, texte_E, texte_F, lib, fig, n_fen_lzma_fr=None, lzma_E=True):
    t0 = time.process_time()
    sortie = {"protocole": {"taille_E": len(texte_E), "taille_francais": len(texte_F),
                            "fenetre": FENETRE, "pas_E": db.PAS_E}}
    # ---- E
    rE, LE, metasE, melE, bpcE = analyser("E", E, fig, lib, texte_E, 0, 0, 1000)
    if lzma_E:
        a = lzma_taille(lib)
        b = lzma_taille(lib + texte_E)
        rE["lzma_conditionnel_bpc"] = 8.0 * (b - a) / len(texte_E)
    be = ex.bayes_evolution()
    idx = {m["nom"]: i for i, m in enumerate(metasE)}
    pc = -LE[idx["champion évolué"]].astype(np.float64)
    rE["p_meilleur_que_champion"] = {}
    for etiq in ("avec_optuna", "sans_optuna"):
        for k in ("bma", "fs"):
            r = be.p_meilleur(melE[etiq]["_pertes_" + k], pc, 500, 4000, 0)
            rE["p_meilleur_que_champion"]["%s_%s" % (k, etiq)] = {kk: r[kk] for kk in
                                                               ("p_meilleur_bayes", "diff_bpc_moyenne", "ic95_bayes")}
    r = be.p_meilleur(-LE[idx["Optuna TPE (gagnant sur B)"]].astype(np.float64), pc, 500, 4000, 0)
    rE["p_meilleur_que_champion"]["optuna"] = {kk: r[kk] for kk in ("p_meilleur_bayes", "diff_bpc_moyenne", "ic95_bayes")}
    sortie["E"] = rE
    del LE
    # ---- français
    rF, LF, metasF, melF, bpcF = analyser("français", F, fig, lib, texte_F, 0, 0, FENETRE)
    n_fen = len(F["y"]) // FENETRE
    idxF = {m["nom"]: i for i, m in enumerate(metasF)}
    cols_maj = [i for i, m in enumerate(metasF) if m["famille"] == "maj"]
    j_or = rF["meilleur_maj_oracle"]["nom"]
    j_A = rF["meilleur_maj_choisi_sur_A"]["nom"]
    bpcE_nom = {m["nom"]: float(bpcE[i]) for i, m in enumerate(metasE)}
    j_E = min((metasF[i]["nom"] for i in cols_maj), key=lambda n: bpcE_nom[n])
    fen = {"champion": fenetres(-LF[idxF["champion évolué"]]),
           "maj_oracle": fenetres(-LF[idxF[j_or]]),
           "maj_choisi_sur_A": fenetres(-LF[idxF[j_A]]),
           "maj_choisi_sur_E": fenetres(-LF[idxF[j_E]]),
           "fixed_share_avec_optuna": fenetres(melF["avec_optuna"]["_pertes_fs"]),
           "fixed_share_sans_optuna": fenetres(melF["sans_optuna"]["_pertes_fs"]),
           "bma_avec_optuna": fenetres(melF["avec_optuna"]["_pertes_bma"]),
           "optuna": fenetres(-LF[idxF["Optuna TPE (gagnant sur B)"]])}
    n_fen_lzma = n_fen if n_fen_lzma_fr is None else n_fen_lzma_fr
    lz, taille_lib = lzma_fenetres(lib, texte_F, n_fen_lzma)
    fen["lzma"] = lz
    rF["fenetres_bpc"] = fen
    rF["fenetres_noms"] = {"maj_oracle": j_or + " (ORACLE)", "maj_choisi_sur_A": j_A, "maj_choisi_sur_E": j_E}
    rF["moyennes_fenetres"] = {k: float(np.mean(v)) for k, v in fen.items()}
    rF["moyennes_fenetres_sur_lzma"] = {k: float(np.mean(v[:n_fen_lzma])) for k, v in fen.items()}
    sortie["francais"] = rF
    sortie["cpu_final_s"] = time.process_time() - t0
    return sortie


def temoin_present():
    return os.path.exists(TEMOIN)


def main():
    essai = "--essai" in sys.argv
    fig = json.load(open(os.path.join(db.CACHE, "parametres_figes.json")))
    if essai:
        jeux = db.tableaux_dev(verbeux=False)
        ta, tb = db.textes_A_B()
        lib = db.bibliotheque_reduite()
        out = calculer(jeux["B"], jeux["A"], tb, ta, lib, fig, n_fen_lzma_fr=2, lzma_E=False)
        out["essai"] = True
        with open(os.path.join(db.CACHE, "final_essai.json"), "w") as f:
            json.dump(out, f, indent=1)
        print("essai terminé (E := B, français := A) ; rien d'officiel n'est écrit")
        return
    if temoin_present():
        sys.exit("final.py a déjà été exécuté (cache/final_utilise.json) : E et le français ne se relisent pas.")
    donnees = db.tableaux_final()
    with open(TEMOIN, "w") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "taille_E": len(donnees["texte_E"]),
                   "taille_francais": len(donnees["texte_F"])}, f)
    out = calculer(donnees["E"], donnees["F"], donnees["texte_E"], donnees["texte_F"], donnees["lib"], fig)
    with open(SORTIE_JSON, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("final terminé ; résultats dans", SORTIE_JSON)


if __name__ == "__main__":
    main()
