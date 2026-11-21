"""Assemble resultats_bayes.json et resultats_bayes.md à partir de archive_bayes/*.json."""
import json
import math
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import bayes  # noqa: E402

SORTIE = os.path.join(ICI, "archive_bayes")


def lire(nom):
    with open(os.path.join(SORTIE, nom), encoding="utf-8") as f:
        return json.load(f)


def f4(x):
    return "-" if x is None else "%.4f" % x


def main():
    sel, tst, ts = lire("selection.json"), lire("test_d.json")["D"], lire("thompson.json")
    cdc = json.load(open(bayes.CHEMIN_CDC))
    py = sel["py"]
    K1 = len(py["delta"])
    nom_aff = {"graine": "Départ (PPM-C)", "champion": "Champion évolué", "reference_main": "Référence à la main (e = 16·d)",
               "pitman_yor": "Pitman-Yor (A)", "bma": "BMA en ligne", "fixed_share": "Fixed-share (α = %g)" % sel["alpha"],
               "lzma": "LZMA conditionnel"}
    ordre = ["graine", "champion", "reference_main", "pitman_yor", "bma", "fixed_share", "lzma"]
    bA, bB, bD = sel["bpc"]["A"], sel["bpc"]["B"], tst["bpc"]
    pc = tst["p_meilleur_que_champion"]
    L = []
    w = L.append
    w("# Bayes, la dernière pierre")
    w("")
    w("Ce document est généré par `rapport_bayes.py` à partir de `archive_bayes/*.json`. Code : `bayes.py` (méthodes), "
      "`lancer_bayes.py` (expériences), `test_bayes.py` (tests). Rien d'inventé : Pitman-Yor hiérarchique (Teh 2006 ; "
      "Wood et al. 2009), moyenne bayésienne de modèles, fixed-share (Herbster et Warmuth 1998 ; Volf et Willems 1998), "
      "échantillonnage de Thompson, bootstrap bayésien (Rubin 1981).")
    w("")
    w("## 1. Protocole et donnée fraîche")
    w("")
    w("- Tout ce qui se règle (δ_k, θ_k de Pitman-Yor ; α de fixed-share) l'est sur **A** seulement. **B** sert à regarder. "
      "Le test final se fait **une seule fois** sur D.")
    w("- **Du texte frais existait** : `validation.txt` compte %d caractères ; A+B+C en prennent %d. La tranche **D** = les %d caractères "
      "restants (positions %d à %d). Elle est construite par le même pipeline de comptage que A, B et C (`donnees.table_ordre`, "
      "`Empreintes`, mêmes sels), avec des comptages issus de la seule bibliothèque `memoire.xz`, ordres 0 à 10."
      % (tst["taille_validation"], tst["debut_dans_validation"], tst["taille"], tst["debut_dans_validation"], tst["taille_validation"]))
    w("- **C n'a pas été rouvert** : ni lu ni utilisé. Un témoin (`cache/d_utilise.json`) empêche de relancer le test sur D par accident.")
    w("- D ne ressemble pas forcément à A, B, C : les bpc de D (autour de 1,9) sont plus hauts que ceux de B (1,46 à 1,48) parce que le texte est plus difficile, "
      "pas parce que les modèles sont moins bons. **Ne comparer que des lignes d'une même colonne.**")
    w("")
    w("## 2. Résultats (bpc, plus petit = mieux)")
    w("")
    w("| Modèle | bpc A (réglage) | bpc B (regard) | **bpc D (test unique, frais)** | P(meilleur que le champion) sur D |")
    w("|---|---|---|---|---|")
    for k in ordre:
        if k == "lzma":
            w("| %s | - | - | %s | - |" % (nom_aff[k], f4(bD[k])))
            continue
        p = "-" if k == "champion" else "%.3f" % pc[k]["p_meilleur_bayes"]
        w("| %s | %s | %s | **%s** | %s |" % (nom_aff[k], f4(bA[k]), f4(bB[k]), f4(bD[k]), p))
    w("")
    w("Sur B, le champion a servi à la sélection (l'écart est donc un peu optimiste pour lui) ; la référence à la main a eu ses deux réglages choisis sur A ; "
      "Pitman-Yor est ajusté sur A et donc légèrement avantagé sur A. LZMA n'est calculé que sur D (bibliothèque + D moins bibliothèque, comme `tester.py`).")
    w("")
    w("### Probabilité postérieure de battre le champion, sur D")
    w("")
    w("P(meilleur) = P(bpc du modèle < bpc du champion), par **bootstrap bayésien** (poids de Dirichlet sur les %d blocs de 500 caractères), "
      "avec en contrôle le bootstrap par blocs. Différence de bpc moyenne (modèle moins champion, négatif = le modèle gagne) et intervalle crédible à 95 %%."
      % pc["graine"]["nb_blocs"])
    w("")
    w("| Modèle | P(meilleur), bayésien | P(meilleur), blocs | différence moyenne | intervalle à 95 % |")
    w("|---|---|---|---|---|")
    for k in ("graine", "reference_main", "pitman_yor", "bma", "fixed_share"):
        r = pc[k]
        w("| %s | %.3f | %.3f | %+.4f | [%+.4f ; %+.4f] |" % (nom_aff[k], r["p_meilleur_bayes"], r["p_meilleur_blocs"],
                                                                 r["diff_bpc_moyenne"], r["ic95_bayes"][0], r["ic95_bayes"][1]))
    w("")
    pm = tst["p_meilleur_que_reference_main"]
    w("Contre la **référence à la main** (D) : champion P = %.3f, Pitman-Yor P = %.3f, BMA P = %.3f, fixed-share P = %.3f. "
      "Le champion et la référence à la main sont statistiquement indiscernables sur D (différence %+.4f bpc)."
      % (pm["champion"]["p_meilleur_bayes"], pm["pitman_yor"]["p_meilleur_bayes"], pm["bma"]["p_meilleur_bayes"],
         pm["fixed_share"]["p_meilleur_bayes"], pc["reference_main"]["diff_bpc_moyenne"]))
    w("")
    w("## 3. Estimateur de Pitman-Yor : hyperparamètres ajustés sur A")
    w("")
    w("Formule : `p_k = (max(n_k − δ_k, 0) + (θ_k + δ_k·d_k)·p_{k−1}) / (N_k + θ_k)`, base uniforme, `p_k = p_{k−1}` si `N_k = 0`. "
      "Les n_k sont les comptages bruts de la bibliothèque (pas de comptes de continuation). Recherche par coordonnées "
      "(section dorée sur δ_k, puis sur log(θ_k+δ_k), 6 balayages, 4 points de départ ; %.0f s en tout). "
      "Les 4 départs finissent à %s bpc sur A : la vraisemblance est plate dans plusieurs directions." %
      (sel["temps_ajustement_s"], ", ".join("%.4f" % x for x in py["departs_bpc_A"])))
    w("")
    w("| ordre k | δ_k | erreur std. δ (approx.) | θ_k | remise de Ney n1/(n1+2·n2) de la bibliothèque | D1 / D2 (Chen-Goodman) |")
    w("|---|---|---|---|---|---|")
    for k in range(K1):
        c = cdc[str(k)]
        dn = "-" if c is None else "%.3f" % c["D_ney"]
        d12 = "-" if c is None or c.get("D2") is None else "%.2f / %.2f" % (c["D1"], c["D2"])
        w("| %d | %.3f | %s | %.3g | %s | %s |" % (k, py["delta"][k],
                                                     "-" if math.isnan(py["se_delta"][k]) else "%.3f" % py["se_delta"][k],
                                                     py["theta"][k], dn, d12))
    w("")
    ident = [k for k in range(K1) if py["delta"][k] > 0.01 and py["theta"][k] < 1e3]
    w("Lecture :")
    w("")
    w("- **Ça ne reproduit pas partout des remises de Kneser-Ney de 0,7 à 0,9.** Seuls les ordres %s ont un δ strictement positif et un θ fini : "
      "δ = %s. L'ordre le plus long (10) a δ = %.2f, proche du bas de la fourchette 0,7 à 0,9 ; les ordres 4 à 6 sont plus bas (0,4 à 0,5)."
      % (", ".join(map(str, ident)), ", ".join("%.2f" % py["delta"][k] for k in ident), py["delta"][10]))
    w("- Aux ordres 0 à 3 (et 7 à 9), δ = 0 et θ n'est pas identifiable : les contextes courts ont des comptes énormes (N de l'ordre de 10⁵ à 10⁶) "
      "et δ ou θ y changent à peine la probabilité (erreurs standard très grandes ou θ collé à une borne). Ce sont des valeurs sans signification, pas des mesures.")
    w("- Aux ordres 8 et 9, θ est collé à la borne haute (10⁴) : le modèle choisit d'**ignorer** ces ordres (`p_k ≈ p_{k−1}`). "
      "Les comptages bruts de longs contextes sont trop peu fiables dans cette bibliothèque.")
    w("- Les remises de Ney calculées sur les comptes de comptes de la bibliothèque donnent la valeur « attendue » (colonne 5). Elles sont élevées "
      "(0,5 à 0,8, croissantes avec l'ordre ; 0,78 à l'ordre 10) aux ordres moyens et longs ; le δ ajusté est en dessous. Cela peut venir de ce que A est un texte d'évaluation "
      "(pas la distribution de la bibliothèque), et surtout de ce que nous utilisons des comptes bruts à tous les ordres : "
      "le vrai Kneser-Ney remplace les comptes des ordres inférieurs par des comptes de continuation, ce que le jeu de données ne fournit pas. "
      "La comparaison est donc indicative, pas une mesure de Kneser-Ney.")
    w("- Pitman-Yor ainsi réglé est **moins bon que le champion et que la référence à la main** : %s bpc sur A, %s sur B, %s sur D. "
      "L'écart sur D est de %+.4f bpc contre le champion (P(meilleur) = %.3f). La raison probable : le champion et la référence "
      "gagnent beaucoup en aplatissant les scores (puissance 0,65) ou en envoyant beaucoup de masse vers l'ordre inférieur (e = 16·d), deux réglages "
      "que la forme Pitman-Yor à θ constant ne peut pas imiter ; et les comptages bruts ne sont pas des comptes de continuation."
      % (f4(bA["pitman_yor"]), f4(bB["pitman_yor"]), f4(bD["pitman_yor"]), pc["pitman_yor"]["diff_bpc_moyenne"],
         pc["pitman_yor"]["p_meilleur_bayes"]))
    w("")
    w("## 4. Moyenne bayésienne de modèles et fixed-share")
    w("")
    w("Membres : départ, champion, référence à la main, Pitman-Yor (4 membres, a priori uniforme 1/4). En ligne : poids a posteriori "
      "∝ a priori × vraisemblance des caractères déjà vus, remis à l'a priori au début de chaque tranche. Fixed-share : après chaque caractère, "
      "w ← (1−α)·w + α/M, avec α choisi sur A dans {1e-5, ..., 0,1} : **α = %g**." % sel["alpha"])
    w("")
    w("**Borne de regret connue.** Pour le BMA simple, la perte cumulée en bits vérifie, pour tout membre i : "
      "`perte_BMA ≤ perte_i + log2(1/π_i)` (c'est une identité : la perte du mélange est −log2 Σ π_i Π p_i). Avec 4 membres et a priori uniforme, "
      "le regret ne dépasse jamais **2 bits** au total sur toute la tranche, quelle que soit sa longueur. Pour fixed-share, contre la meilleure "
      "suite de membres avec m changements : `≤ log2 M + m·log2(M/α) + (T−1−m)·log2(1/(1−α))` (forme Herbster-Warmuth / Volf-Willems, "
      "le dernier terme est le prix de l'agilité). Les deux bornes sont vérifiées par les tests.")
    w("")
    r = tst["regret"]
    w("Regret observé sur D du BMA contre le meilleur membre (%s) : **%.4f bits** au total (borne : %.0f bits), soit environ %.1e bit par caractère. "
      "Sur A et B : %.4f et %.4f bits." % (r["meilleur_membre"], r["regret_observe_bits"], r["borne_log2_1_sur_prior"],
                                          r["regret_observe_bits"] / tst["taille"], sel["regret"]["A"]["regret_observe_bits"],
                                          sel["regret"]["B"]["regret_observe_bits"]))
    w("")
    w("Poids moyens sur D (départ / champion / référence / Pitman-Yor) : BMA = %s ; fixed-share = %s."
      % (" / ".join("%.3f" % x for x in tst["poids_moyens_bma"]), " / ".join("%.3f" % x for x in tst["poids_moyens_fs"])))
    w("")
    w("Lecture : le BMA simple **ne gagne rien** (D : %s bpc contre %s pour le meilleur membre) : il se verrouille très vite sur un seul "
      "membre, et son regret est borné. Fixed-share fait mieux que tous les membres : %s bpc sur D, soit %+.4f contre le champion "
      "(P(meilleur) = %.3f, intervalle à 95 %% [%+.4f ; %+.4f]). Il peut changer de membre quand le texte change (sur B le gain est bien plus grand : %s contre %s), "
      "mais le gain vient d'un mélange entre quatre prédicteurs très semblables (mêmes comptages) : c'est un petit gain sûr, pas un nouveau modèle."
      % (f4(bD["bma"]), f4(min(bD["champion"], bD["reference_main"])), f4(bD["fixed_share"]),
         pc["fixed_share"]["diff_bpc_moyenne"], pc["fixed_share"]["p_meilleur_bayes"],
         pc["fixed_share"]["ic95_bayes"][0], pc["fixed_share"]["ic95_bayes"][1], f4(bB["fixed_share"]), f4(bB["reference_main"])))
    w("")
    w("Attention à un biais : α et les hyperparamètres de Pitman-Yor sont réglés sur A, où Pitman-Yor est ajusté (légèrement avantagé). "
      "Les poids d'A ne sont pas réutilisés (les poids repartent de l'a priori à chaque tranche).")
    w("")
    w("## 5. Thompson sampling pour choisir les opérateurs de mutation")
    w("")
    w("Un bandit de Bernoulli par opérateur (9 opérateurs de `mutations.py`) : succès = l'enfant bat son parent sur A. Postérieur Beta(a, b) "
      "avec a priori Beta(1, 1). Pour chaque mutation on tire θ_op ~ Beta et on prend l'argmax. **Il remplace les poids du méta-génome** ; "
      "le nombre de mutations (1,5) et le pas des constantes (0,3) sont fixes. Même boucle, même sélection des parents, même fitness A et "
      "même choix du champion sur B que `moteur.py` ; seule la façon de choisir l'opérateur change. "
      "Comparaison à budget égal : %d évaluations d'enfants par run, %d graines (les mêmes pour les deux modes), moteur existant = méta-génome automatique." % (ts["budget"], len(ts["graines"])))
    w("")
    w("| Graine | meilleur A, existant | meilleur A, Thompson | B du champion, existant | B du champion, Thompson | CPU (s) existant / Thompson |")
    w("|---|---|---|---|---|---|")
    for b, t in zip(ts["base"], ts["thompson"]):
        w("| %d | %s | %s | %s | %s | %.0f / %.0f |" % (b["graine"], f4(b["best_A"]), f4(t["best_A"]), f4(b["champion_B"]),
                                                   f4(t["champion_B"]), b["cpu_total_s"], t["cpu_total_s"]))
    import numpy as np
    mA = [np.mean([r["best_A"] for r in ts[k]]) for k in ("base", "thompson")]
    mB = [np.mean([r["champion_B"] for r in ts[k]]) for k in ("base", "thompson")]
    sA = [np.std([r["best_A"] for r in ts[k]], ddof=1) for k in ("base", "thompson")]
    sB = [np.std([r["champion_B"] for r in ts[k]], ddof=1) for k in ("base", "thompson")]
    w("| **moyenne ± écart-type** | %.4f ± %.4f | %.4f ± %.4f | %.4f ± %.4f | %.4f ± %.4f | |" % (mA[0], sA[0], mA[1], sA[1], mB[0], sB[0], mB[1], sB[1]))
    w("")
    ca, cb = ts["comparaison_best_A"], ts["comparaison_champion_B"]
    w("Probabilité postérieure que Thompson soit meilleur (bpc plus bas), modèle de Student sur les moyennes, a priori plat : "
      "**sur A : %.2f** (différence %+.4f, intervalle à 95 %% [%+.4f ; %+.4f]) ; **sur B : %.2f** (différence %+.4f, intervalle [%+.4f ; %+.4f])."
      % (ca["p_x_meilleur"], ca["diff_moyenne"], ca["ic95"][0], ca["ic95"][1], cb["p_x_meilleur"], cb["diff_moyenne"], cb["ic95"][0], cb["ic95"][1]))
    w("")
    mo = ts["thompson"][0]["thompson"]
    w("Opérateurs préférés par Thompson (graine %d), moyenne postérieure du taux de succès / nombre de choix :" % ts["thompson"][0]["graine"])
    w("")
    w("| Opérateur | taux de succès postérieur | nombre de choix |")
    w("|---|---|---|")
    for o in sorted(mo["moyennes_posterieures"], key=lambda o: -mo["moyennes_posterieures"][o]):
        w("| %s | %.3f | %d |" % (o, mo["moyennes_posterieures"][o], mo["nb_choix"][o]))
    w("")
    w("Avec 3 graines, l'incertitude est grande : lire ce résultat comme une indication, pas une preuve. Le budget (1500 évaluations) est plus court que "
      "celui des runs précédents (2000 à 15000).")
    w("")
    w("## 6. Limites")
    w("")
    w("- D vient du même fichier `validation.txt` que A, B, C : même corpus, mêmes sources. Elle n'est « fraîche » que pour les décisions prises ici (aucun modèle ni réglage n'y a été ajusté), pas pour la distribution.")
    w("- Le champion et la référence à la main ont été sélectionnés avec A et B (et le champion avec C pour un test déjà publié) ; leur avance éventuelle sur D est donc honnête, mais leur comparaison à Pitman-Yor porte sur des modèles qui ont eu des réglages « puissance » ou « masse d'échappement » que Pitman-Yor n'a pas.")
    w("- Les comptes sont bruts à tous les ordres, ce qui n'est pas Kneser-Ney (comptes de continuation) : l'interprétation des δ_k est indicative. Le critère d'ajustement est la vraisemblance prédictive de A (Bayes empirique de type II), pas une évidence marginale complète ; les erreurs standard (hessienne diagonale, texte non indépendant) sont optimistes.")
    w("- Le BMA et fixed-share mélangent des modèles presque identiques (mêmes comptages). Le gain de fixed-share vient de la variation locale du meilleur membre ; il se limite à quelques millièmes de bit par caractère sur D.")
    w("- Thompson : 3 graines, 1500 évaluations, un seul réglage (pas d'oubli, a priori Beta(1,1)). Les opérateurs ne sont pas indépendants (une mutation réussit ou échoue avec les autres du même enfant : le crédit est attribué à tous les opérateurs utilisés).")
    w("- Incident d'exécution : le run Thompson a été lancé deux fois par erreur ; le doublon a été tué après environ 12 minutes, "
      "l'autre a terminé seul. Les résultats en bpc ne dépendent pas du temps (graines fixes, évaluations déterministes), mais les temps CPU "
      "du tableau sont gonflés par cette contention et ne sont pas comparables ; pendant ces minutes il y a eu plus de 4 processus de calcul. "
      "Les journaux `archive_bayes/*.jsonl` des deux copies se sont écrasés mutuellement : seul `thompson.json` fait foi.")
    w("- LZMA conditionnel est une référence grossière (pas un modèle de même nature).")
    w("")
    w("## 7. Fichiers")
    w("")
    w("`bayes.py`, `lancer_bayes.py`, `rapport_bayes.py`, `test_bayes.py`, `archive_bayes/*.json`, `cache/d.npz` (tranche D), `cache/bayes_comptes_de_comptes.json`, `cache/d_utilise.json` (témoin). Aucun fichier épinglé n'a été modifié.")
    with open(os.path.join(ICI, "resultats_bayes.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    out = {"protocole": {"D": {k: tst[k] for k in ("taille", "debut_dans_validation", "taille_validation")},
                         "C_rouvert": False, "texte_frais": True},
           "selection_A_B": sel, "test_D": tst, "thompson": ts, "comptes_de_comptes_bibliotheque": cdc}
    with open(os.path.join(ICI, "resultats_bayes.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
