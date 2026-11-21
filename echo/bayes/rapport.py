"""Génère resultats_bayes_langue.md à partir de resultats_bayes_langue.json et cache/dev_resultats.json.

Les tableaux sont produits par le code ; le texte d'interprétation est dans texte_bayes_langue.md (écrit après
lecture des chiffres) et inséré à la fin.
"""
import json
import os

ICI = os.path.dirname(os.path.abspath(__file__))


def f(x, n=4):
    return "-" if x is None else ("%." + str(n) + "f") % x


def table(entetes, lignes):
    s = "| " + " | ".join(entetes) + " |\n|" + "---|" * len(entetes) + "\n"
    for l in lignes:
        s += "| " + " | ".join(str(c) for c in l) + " |\n"
    return s


def masses_lignes(m, cle, positions_voulues):
    pos = m["positions"]
    etiqs = [e for e in m[cle] if max(m[cle][e]) > 0.005]
    lignes = []
    for p in positions_voulues:
        if p in pos:
            i = pos.index(p)
            lignes.append([p] + ["%.3f" % m[cle][e][i] for e in etiqs])
    return ["position"] + etiqs, lignes


def main():
    R = json.load(open(os.path.join(ICI, "resultats_bayes_langue.json"), encoding="utf-8"))
    D = json.load(open(os.path.join(ICI, "cache", "dev_resultats.json"), encoding="utf-8"))
    E, F = R["E"], R["francais"]
    o = []
    o.append("# Bayes comme principe unique : lire, prédire, s'améliorer\n")
    o.append("Document généré par `rapport.py` (tableaux) à partir de `resultats_bayes_langue.json` (E et français, "
             "`final.py`, une seule exécution) et de `cache/dev_resultats.json` (A et B, `dev.py`). "
             "Texte d'interprétation : voir les sections « Lecture ».\n")
    # --- protocole
    p = R["protocole"]
    o.append("## 1. Protocole\n")
    o.append("- **E, test frais en domaine** : un paragraphe de la bibliothèque sur %d (indices %d, 2·%d−1, ...), soit %d "
             "caractères, retirés de `memoire.xz`. Les comptages de A, B, E et du français viennent de la **bibliothèque "
             "réduite** (pipeline `evolution/donnees.py`, importé sans modification). E n'est lu que par `final.py`, une fois "
             "(témoin `cache/final_utilise.json`). Les paragraphes viennent des mêmes livres que ceux restés dans la "
             "bibliothèque : E est « en domaine », pas « hors auteur »." % (p["pas_E"], p["pas_E"] - 1, p["pas_E"],
                                                                          p["taille_E"]))
    o.append("- **Français** : %d caractères (un paragraphe sur 150 du corpus français local `echo/data/`, non redistribuable ; "
             "jamais écrit dans un fichier suivi, seuls des nombres sont publiés). Alphabet étendu aux caractères du "
             "français ; bibliothèque anglaise réduite comme a priori." % p["taille_francais"])
    o.append("- **Experts** : %d. (a) grille statique 7 c × 5 ordres × 4 γ × {sans, avec u1} = 280 ; (b) mise à jour "
             "bayésienne 6 β × 3 c × 4 ordres × 2 γ = 144 ; (c) départ, champion, référence à la main, Pitman-Yor (A) ; "
             "(d) gagnant d'Optuna. A priori uniforme sur les experts. Tout est causal : la probabilité en t n'utilise "
             "que les caractères avant t.\n" % E["n_experts"])
    # --- E
    o.append("## 2. Résultats sur E (bpc, plus petit = mieux)\n")
    r = E["bpc_experts_cles"]
    mel = E["melanges"]
    lignes = [
        ["Départ (PPM-C)", f(r["départ (PPM-C)"])],
        ["Champion évolué", f(r["champion évolué"])],
        ["Référence à la main (e = 16·d)", f(r["référence à la main (e=16·d)"])],
        ["Pitman-Yor (réglé sur A)", f(r["Pitman-Yor (A)"])],
        ["Optuna TPE (gagnant choisi sur B)", f(r["Optuna TPE (gagnant sur B)"])],
        ["LZMA conditionnel (biblio réduite + E, moins biblio réduite)", f(E.get("lzma_conditionnel_bpc"))],
        ["**Meilleur expert isolé, a posteriori (ORACLE : `%s`)**" % E["meilleur_expert_oracle"]["nom"],
         f(E["meilleur_expert_oracle"]["bpc"])],
        ["Meilleur expert à mise à jour choisi sur A", f(E["meilleur_maj_choisi_sur_A"]["bpc"]) + " (`%s`)" % E["meilleur_maj_choisi_sur_A"]["nom"]],
        ["Meilleur expert de la grille statique choisi sur A", f(E["meilleur_grille_choisi_sur_A"]["bpc"]) + " (`%s`)" % E["meilleur_grille_choisi_sur_A"]["nom"]],
        ["BMA, tous les experts sauf Optuna", f(mel["sans_optuna"]["bma"]["bpc"])],
        ["Fixed-share (α = %g), sauf Optuna" % mel["sans_optuna"]["alpha"], f(mel["sans_optuna"]["fixed_share"]["bpc"])],
        ["BMA, avec Optuna", f(mel["avec_optuna"]["bma"]["bpc"])],
        ["**Fixed-share (α = %g), avec Optuna**" % mel["avec_optuna"]["alpha"], "**" + f(mel["avec_optuna"]["fixed_share"]["bpc"]) + "**"],
    ]
    o.append(table(["Modèle", "bpc sur E"], lignes))
    o.append("\nα est choisi sur A dans {1e-5, ..., 0,1}. L'oracle est donné pour situer le plafond d'un choix de modèle ; "
             "il n'est pas atteignable sans connaître E.\n")
    o.append("### Probabilité postérieure de battre le champion (bootstrap bayésien, blocs de 500)\n")
    pm = E["p_meilleur_que_champion"]
    o.append(table(["Modèle", "P(meilleur)", "différence moyenne (bpc)", "intervalle crédible 95 %"],
                   [[k, f(v["p_meilleur_bayes"], 3), "%+.4f" % v["diff_bpc_moyenne"],
                     "[%+.4f ; %+.4f]" % tuple(v["ic95_bayes"])] for k, v in pm.items()]))
    o.append("\n### Convergence : entropie du postérieur au fil de la lecture (BMA et fixed-share, avec Optuna)\n")
    bm, fs = mel["avec_optuna"]["bma"], mel["avec_optuna"]["fixed_share"]
    ks = list(bm["entropie_bits"].keys())
    o.append(table(["caractères lus"] + ks,
                   [["entropie BMA (bits)"] + [f(bm["entropie_bits"][k], 2) for k in ks],
                    ["experts effectifs 2^H, BMA"] + [f(bm["nb_experts_effectif"][k], 1) for k in ks],
                    ["entropie fixed-share (bits)"] + [f(fs["entropie_bits"][k], 2) for k in ks],
                    ["experts effectifs 2^H, fixed-share"] + [f(fs["nb_experts_effectif"][k], 1) for k in ks]]))
    o.append("\nDominant en fin de lecture : BMA `%s` (poids %.2f) ; fixed-share `%s` (poids %.2f). Entropie initiale "
             "= log2(%d) = %.2f bits.\n" % (bm["expert_final_dominant"], bm["poids_final_max"],
                                            fs["expert_final_dominant"], fs["poids_final_max"], E["n_experts"],
                                            __import__("math").log2(E["n_experts"])))
    o.append("### Occam : masse postérieure par ordre et par β sur E (BMA, avec Optuna ; position 0 = a priori)\n")
    posE = [0, 1000, 2000, 5000, 10000, 20000, 40000, 56000]
    h, l = masses_lignes(bm["masses"], "par_ordre", posE)
    o.append("Par ordre maximal :\n\n" + table(h, l))
    h, l = masses_lignes(bm["masses"], "par_beta", posE)
    o.append("\nPar β (statique = grille sans lecture du flux) :\n\n" + table(h, l))
    h, l = masses_lignes(fs["masses"], "par_ordre", posE)
    o.append("\nFixed-share, par ordre :\n\n" + table(h, l))
    h, l = masses_lignes(fs["masses"], "par_beta", posE)
    o.append("\nFixed-share, par β :\n\n" + table(h, l))
    # --- français
    o.append("\n## 3. Changement de domaine : français (a priori anglais, mise à jour sur le flux français)\n")
    r = F["bpc_experts_cles"]
    m = F["melanges"]
    mf = F["moyennes_fenetres"]
    nf = len(F["fenetres_bpc"]["champion"])
    o.append(table(["Modèle", "bpc sur tout le flux (%d caractères)" % F["T"],
                    "moyenne des %d fenêtres de 2000 (%d caractères)" % (nf, nf * 2000)],
                   [["Champion (sans mise à jour)", f(r["champion évolué"]), f(mf["champion"])],
                    ["Référence à la main (statique)", f(r["référence à la main (e=16·d)"]), "-"],
                    ["Optuna (statique)", f(r["Optuna TPE (gagnant sur B)"]), f(mf["optuna"])],
                    ["LZMA conditionnel (biblio + français déjà lu)", "-", f(mf["lzma"])],
                    ["Meilleur expert statique, ORACLE (`%s`)" % F["meilleur_grille_oracle"]["nom"], f(F["meilleur_grille_oracle"]["bpc"]), "-"],
                    ["Meilleur expert à mise à jour, **ORACLE** (`%s`)" % F["meilleur_maj_oracle"]["nom"], f(F["meilleur_maj_oracle"]["bpc"]), f(mf["maj_oracle"])],
                    ["Meilleur expert à mise à jour choisi sur A (`%s`)" % F["meilleur_maj_choisi_sur_A"]["nom"], f(F["meilleur_maj_choisi_sur_A"]["bpc"]), f(mf["maj_choisi_sur_A"])],
                    ["Meilleur expert à mise à jour choisi sur E (`%s`)" % F["fenetres_noms"]["maj_choisi_sur_E"], "-", f(mf["maj_choisi_sur_E"])],
                    ["BMA (avec Optuna)", f(m["avec_optuna"]["bma"]["bpc"]), f(mf["bma_avec_optuna"])],
                    ["Fixed-share α = %g (avec Optuna)" % m["avec_optuna"]["alpha"], f(m["avec_optuna"]["fixed_share"]["bpc"]), f(mf["fixed_share_avec_optuna"])],
                    ["Fixed-share (sans Optuna)", f(m["sans_optuna"]["fixed_share"]["bpc"]), f(mf["fixed_share_sans_optuna"])]]))
    fw = F["fenetres_bpc"]
    n = len(fw["champion"])
    o.append("\n### bpc par fenêtre de %d caractères\n" % R["protocole"]["fenetre"])
    o.append("LZMA : taille(biblio + français déjà lu + fenêtre) − taille(biblio + français déjà lu). Meilleur expert à "
             "mise à jour = `%s` (ORACLE, choisi sur ce flux).\n" % F["meilleur_maj_oracle"]["nom"])
    o.append(table(["fenêtre", "champion", "LZMA", "meilleur maj (oracle)", "maj choisi sur A", "fixed-share", "BMA"],
                   [[i + 1, f(fw["champion"][i], 3), f(fw["lzma"][i], 3), f(fw["maj_oracle"][i], 3),
                     f(fw["maj_choisi_sur_A"][i], 3), f(fw["fixed_share_avec_optuna"][i], 3),
                     f(fw["bma_avec_optuna"][i], 3)] for i in range(n)]))
    bm, fs = m["avec_optuna"]["bma"], m["avec_optuna"]["fixed_share"]
    ks = list(bm["entropie_bits"].keys())
    o.append("\n### Convergence sur le français\n")
    o.append(table(["caractères lus"] + ks,
                   [["experts effectifs, BMA"] + [f(bm["nb_experts_effectif"][k], 1) for k in ks],
                    ["experts effectifs, fixed-share"] + [f(fs["nb_experts_effectif"][k], 1) for k in ks]]))
    posF = [0, 2000, 4000, 8000, 12000, 20000, 30000, 40000, 50000]
    for nom, mm in (("BMA", bm), ("Fixed-share", fs)):
        h, l = masses_lignes(mm["masses"], "par_beta", posF)
        o.append("\n%s : masse postérieure par β au fil du français\n\n" % nom + table(h, l))
        h, l = masses_lignes(mm["masses"], "par_ordre", posF)
        o.append("\n%s : masse postérieure par ordre\n\n" % nom + table(h, l))
    # --- Optuna
    o.append("\n## 4. Auto-amélioration : optimisation bayésienne (Optuna TPE, graine 0, 300 essais, objectif bpc sur A)\n")
    og = D["optuna"]
    pr = og["params"]
    o.append("Espace : c_k par ordre (log, 0,05 à 300, k ≤ ordre max), γ ∈ [0,4 ; 1], u1 ∈ {non, oui}, ordre max ∈ [2 ; 10]. "
             "Gagnant (meilleur B parmi les 10 meilleurs sur A) : A = %.4f, B = %.4f ; ordre max %d, γ = %.3f, u1 = %s, "
             "c_k = %s. CPU : %.0f s.\n" % (og["gagnant"]["A"], og["gagnant"]["B"], pr["kmax"], pr["gamma"], pr["u1"],
                                           ", ".join("%.3g" % c for c in pr["cs"][:pr["kmax"] + 1]), og["cpu_s"]))
    ref = D["references_AB"]
    o.append(table(["Modèle", "A", "B", "E"],
                   [[k, f(v["A"]), f(v["B"]), f(E["bpc_experts_cles"].get(k))] for k, v in ref.items()]))
    o.append("\nDécalage de B dû à la bibliothèque réduite (nouveau − ancien) : champion %+.5f, main %+.5f (négligeable).\n"
             % (D["decalage_B"]["champion"], D["decalage_B"]["main"]))
    o.append("### Efficacité en échantillons : évaluations (sur A) pour atteindre un niveau de B\n")
    o.append("Moteur d'évolution : %d runs de `evolution/archive/*_resume.json` (points tous les ~250 évaluations, "
             "pas plus fin ; B du champion choisi parmi les meilleurs sur A). Optuna : après m essais, B du meilleur "
             "parmi les 10 meilleurs sur A (B calculé à chaque essai pour la courbe seulement ; Optuna ne le voit pas). "
             "Niveaux de B de l'ancienne échelle, décalage appliqué aux niveaux d'Optuna.\n" % D["efficacite"][0]["moteur_runs_total"])
    o.append(table(["niveau B", "runs du moteur qui l'atteignent", "évals min (moteur)", "évals médiane (moteur)", "essais Optuna"],
                   [[e["niveau_B"], "%d / %d" % (e["moteur_runs_atteignant"], e["moteur_runs_total"]),
                     e["moteur_evals_min"], e["moteur_evals_mediane"], e["optuna_essais"]] for e in D["efficacite"]]))
    o.append("\n### Dix meilleurs essais sur A\n")
    o.append(table(["rang", "essai", "A", "B", "ordre max", "γ", "u1"],
                   [[i + 1, e["n"], f(e["A"]), f(e["B"]), e["params"]["kmax"], "%.3f" % e["params"]["gamma"], e["params"]["u1"]]
                    for i, e in enumerate(og["top10_A"])]))
    o.append("\n### Meilleur de chaque famille sur A (et son B)\n")
    o.append(table(["famille", "expert", "A", "B"],
                   [[k, "`%s`" % v["nom"], f(v["A"]), f(v["B"])] for k, v in D["meilleur_A_par_famille"].items()]))
    o.append("\nα de fixed-share choisi sur A : %s (avec Optuna), %s (sans). Sur B (regard) : BMA %.4f, fixed-share %.4f.\n"
             % (D["alphas"]["avec_optuna"]["alpha"], D["alphas"]["sans_optuna"]["alpha"],
                D["alphas"]["avec_optuna"]["bpc_B_bma"], D["alphas"]["avec_optuna"]["bpc_B_fixed_share"]))
    chemin = os.path.join(ICI, "texte_bayes_langue.md")
    if os.path.exists(chemin):
        o.append("\n" + open(chemin, encoding="utf-8").read())
    open(os.path.join(ICI, "resultats_bayes_langue.md"), "w", encoding="utf-8").write("\n".join(o))
    print("resultats_bayes_langue.md écrit")


if __name__ == "__main__":
    main()
