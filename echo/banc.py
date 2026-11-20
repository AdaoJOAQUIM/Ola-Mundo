"""Banc d'essai d'Écho-P (predicteur.py) contre lzma, bz2 et zlib.

  A. courbe d'apprentissage : mémoires de 0,1 %, 1 %, 10 % et 100 % de la
     bibliothèque anglaise, évaluées sur le MÊME échantillon de validation ;
  B. apprentissage continu : on ajoute une mémoire française à la mémoire
     anglaise, sans rien recalculer, et on mesure l'oubli et l'apprentissage ;
  C. énergie : secondes de CPU par caractère et ESTIMATION en millijoules ;
  D. taille : mémoire anglaise + mémoire française compressées <= 10 000 000 octets.

Usage : python3 banc.py            (banc complet)
        python3 banc.py --rapide   (test de fumée, 300 caractères)
Écrit resultats_banc.md et resultats_banc.json. Uniquement la bibliothèque standard.
"""
import argparse
import bz2
import json
import lzma
import multiprocessing as mp
import os
import random
import sys
import time
import zlib

from predicteur import Memoire, bits_par_caractere

ICI = os.path.dirname(os.path.abspath(__file__))
CHEMIN_EN = os.path.join(ICI, "memoire.xz")
CHEMIN_VALID = os.path.join(ICI, "validation.txt")
CHEMIN_FR = os.path.join(ICI, "data", "francais.txt")
SORTIE_XZ_FR = os.path.join(ICI, "memoire_fr.xz")
SORTIE_MD = os.path.join(ICI, "resultats_banc.md")
SORTIE_JSON = os.path.join(ICI, "resultats_banc.json")

GRAINE = 20240607
FRACTIONS = [0.001, 0.01, 0.1, 1.0]
WATTS = 10.0           # hypothèse d'estimation : 10 W par cœur de CPU occupé
LIMITE_OCTETS = 10_000_000
PRESET_XZ = 9 | lzma.PRESET_EXTREME

# Données partagées avec les processus fils (héritées par fork).
G = {}


# ---------------------------------------------------------------------------
# Préparation des données
# ---------------------------------------------------------------------------
def paragraphes(texte):
    lignes = texte.split("\n")
    if lignes and lignes[-1] == "":
        lignes.pop()
    return lignes


def echantillon_eval(lignes, n_car, graine, memoire=None):
    """~n_car premiers caractères d'une sélection aléatoire (graine fixe) de paragraphes.

    Si `memoire` est donné, on écarte les paragraphes qui y figurent déjà mot pour mot
    (doublons, p. ex. versets de la Bible) : sinon l'évaluation serait de la recopie.
    """
    lignes = [l for l in lignes if l.strip() and (memoire is None or l not in memoire)]
    random.Random(graine).shuffle(lignes)
    return "\n".join(lignes)[:n_car]


def memoire_fraction(paras, fraction):
    """Échantillon aléatoire (graine fixe) de paragraphes, remis dans l'ordre d'origine."""
    k = max(1, round(len(paras) * fraction))
    idx = sorted(random.Random(GRAINE + 1).sample(range(len(paras)), k))
    return "".join(paras[i] + "\n" for i in idx)


def preparer(n_car, avec_fr):
    with lzma.open(CHEMIN_EN, "rt", encoding="utf-8") as f:
        G["en_paras"] = paragraphes(f.read())
    with open(CHEMIN_VALID, encoding="utf-8") as f:
        valid = paragraphes(f.read())
    texte_en_complet = "".join(p + "\n" for p in G["en_paras"])
    G["doublons_en"] = sum(1 for l in valid if l.strip() and l in texte_en_complet)
    G["eval_en"] = echantillon_eval(valid, n_car, GRAINE + 2, texte_en_complet)
    G["n_car"] = n_car
    if avec_fr:
        with open(CHEMIN_FR, encoding="utf-8") as f:
            fr = paragraphes(f.read())
        tenues = [l for i, l in enumerate(fr) if i % 100 == 99]
        memo = [l for i, l in enumerate(fr) if i % 100 != 99]
        G["fr_memoire"] = "".join(l + "\n" for l in memo)
        G["fr_paras_memoire"] = len(memo)
        G["fr_paras_tenues"] = len(tenues)
        G["eval_fr"] = echantillon_eval(tenues, n_car, GRAINE + 3,
                                        G["fr_memoire"] + texte_en_complet)


# ---------------------------------------------------------------------------
# Tâches (exécutées dans des processus fils)
# ---------------------------------------------------------------------------
def texte_en(fraction):
    if fraction >= 1.0:
        return "".join(p + "\n" for p in G["en_paras"])
    return memoire_fraction(G["en_paras"], fraction)


def tache(spec):
    genre = spec[0]
    t0 = time.time()
    if genre == "echo_a":       # courbe d'apprentissage, Écho-P
        _, fraction = spec
        mem_txt = texte_en(fraction)
        m = Memoire(mem_txt)
        bpc, cpu = bits_par_caractere(m, G["eval_en"], G["n_car"])
        r = {"genre": genre, "fraction": fraction, "caracteres_memoire": len(mem_txt),
             "bpc": bpc, "cpu_s": cpu}
    elif genre == "base":       # lignes de base : compression conditionnelle
        _, fraction, algo = spec
        mem = texte_en(fraction).encode("utf-8")
        ev = G["eval_en"].encode("utf-8")
        comp = {"lzma": lambda b: lzma.compress(b, preset=PRESET_XZ),
                "bz2": lambda b: bz2.compress(b, 9),
                "zlib": lambda b: zlib.compress(b, 9)}[algo]
        c_mem = len(comp(mem))
        c_tout = len(comp(mem + ev))
        r = {"genre": genre, "fraction": fraction, "algo": algo,
             "bpc": 8.0 * (c_tout - c_mem) / len(G["eval_en"]),
             "octets_memoire_comprimee": c_mem, "octets_tout_comprime": c_tout}
    elif genre == "cont":       # apprentissage continu, Écho-P
        _, cond = spec          # a_en, b_fr, c_en, c_fr
        mem_txt = texte_en(1.0)
        m = Memoire(mem_txt)
        t_ajout = None
        if cond.startswith("c"):
            t1 = time.perf_counter()
            c1 = time.process_time()
            m.ajouter(G["fr_memoire"])
            t_ajout = {"mur_s": time.perf_counter() - t1,
                       "cpu_s": time.process_time() - c1}
        ev = G["eval_en"] if cond.endswith("en") else G["eval_fr"]
        bpc, cpu = bits_par_caractere(m, ev, G["n_car"])
        r = {"genre": genre, "cond": cond, "bpc": bpc, "cpu_s": cpu,
             "caracteres_memoire": m.taille(), "ajout": t_ajout,
             "caracteres_eval": min(G["n_car"], len(ev))}
    else:
        raise ValueError(genre)
    r["duree_mur_s"] = time.time() - t0
    if "cpu_s" in r:
        n = min(G["n_car"], len(G["eval_en"]) if genre != "cont" or r["cond"].endswith("en")
                else len(G["eval_fr"]))
        r["cpu_s_par_car"] = r["cpu_s"] / n
        r["mJ_par_car"] = r["cpu_s_par_car"] * WATTS * 1000.0
    return r


def lancer(taches, nb_proc):
    # les plus lourdes d'abord, pour bien remplir les cœurs
    with mp.get_context("fork").Pool(nb_proc) as pool:
        return list(pool.imap_unordered(tache, taches))


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
def pct(f):
    return ("%g" % (f * 100)).replace(".", ",") + " %"


def fnum(x, d=3):
    return ("%." + str(d) + "f") % x


def rapport(res, duree_totale, rapide):
    L = []
    n_en = len(G["eval_en"])
    L.append("# Résultats du banc d'essai Écho-P")
    L.append("")
    if rapide:
        L.append("> MODE RAPIDE (test de fumée) : échantillon réduit, ne pas citer ces chiffres.")
        L.append("")
    L.append("Écho-P = `predicteur.py` : PPM (échappement de méthode C, avec exclusion), ordres 16 à 0,"
             " au plus 64 occurrences par ordre, uniquement `str.find` sur le texte brut,"
             " 0 paramètre appris. bpc = bits par caractère (plus bas = mieux).")
    L.append("")
    L.append("Échantillon d'évaluation anglais : %d caractères (paragraphes de `validation.txt` tirés"
             " au hasard avec une graine fixe, joints par un saut de ligne)." % n_en)
    L.append("")
    # -- A
    L.append("## A. Courbe d'apprentissage (efficacité en données)")
    L.append("")
    L.append("Mémoire = échantillon aléatoire de paragraphes de la bibliothèque anglaise (graine fixe)."
             " Lignes de base : bpc de compression *conditionnelle* = 8 x (taille(mémoire + éval) -"
             " taille(mémoire)) / nb de caractères de l'éval. lzma : préset 9|EXTREME ; bz2 et zlib : niveau 9.")
    L.append("")
    L.append("| Mémoire | Caractères | Écho-P (bpc) | lzma (bpc) | bz2 (bpc) | zlib (bpc) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    A = {}
    for r in res:
        if r["genre"] == "echo_a":
            A.setdefault(r["fraction"], {})["echo"] = r
        elif r["genre"] == "base":
            A.setdefault(r["fraction"], {})[r["algo"]] = r
    fr_list = sorted(A)
    for f in fr_list:
        a = A[f]
        L.append("| %s | %s | %s | %s | %s | %s |" % (
            pct(f), format(a["echo"]["caracteres_memoire"], ",").replace(",", " "),
            fnum(a["echo"]["bpc"]), fnum(a["lzma"]["bpc"]), fnum(a["bz2"]["bpc"]), fnum(a["zlib"]["bpc"])))
    L.append("")
    L.append("Précautions de lecture : la fenêtre de zlib n'est que de 32 Ko et les blocs de bz2 de"
             " 900 Ko ; ces deux compresseurs « oublient » donc la quasi-totalité d'une mémoire de"
             " plusieurs mégaoctets et n'en profitent presque pas. Seul lzma (dictionnaire de 64 Mo au"
             " préset 9) voit toute la mémoire : c'est la seule vraie comparaison. Ces trois chiffres"
             " ne sont pas un classement de « méthodes » mais une mesure de ce que chaque outil retient"
             " d'une longue mémoire.")
    L.append("")
    L.append("Écho-P contre lzma, écart de bpc (négatif = Écho-P meilleur) :")
    L.append("")
    L.append("| Mémoire | Écho-P - lzma (bpc) |")
    L.append("|---|---:|")
    for f in fr_list:
        a = A[f]
        L.append("| %s | %s |" % (pct(f), "%+.3f" % (a["echo"]["bpc"] - a["lzma"]["bpc"])))
    L.append("")
    # -- B
    B = {r["cond"]: r for r in res if r["genre"] == "cont"}
    if B:
        L.append("## B. Apprentissage continu (anglais, puis ajout du français)")
        L.append("")
        L.append("Mémoire anglaise = 100 %% de la bibliothèque. Français : %d paragraphes en mémoire,"
                 " %d paragraphes tenus à l'écart (1 ligne sur 100) ; échantillon d'évaluation français :"
                 " %d caractères. L'ajout se fait par `Memoire.ajouter`, sans aucun recalcul."
                 % (G["fr_paras_memoire"], G["fr_paras_tenues"], len(G["eval_fr"])))
        L.append("")
        L.append("| Condition | Mémoire | Évaluation | bpc |")
        L.append("|---|---|---|---:|")
        L.append("| (a) | anglais seul | anglais | %s |" % fnum(B["a_en"]["bpc"]))
        L.append("| (b) | anglais seul | français | %s |" % fnum(B["b_fr"]["bpc"]))
        L.append("| (c) | anglais + français | anglais | %s |" % fnum(B["c_en"]["bpc"]))
        L.append("| (c) | anglais + français | français | %s |" % fnum(B["c_fr"]["bpc"]))
        L.append("")
        oubli = B["c_en"]["bpc"] - B["a_en"]["bpc"]
        gain = B["b_fr"]["bpc"] - B["c_fr"]["bpc"]
        L.append("- **Oubli** = c_en - a_en = **%+.3f bpc** (positif = l'anglais s'est dégradé ; 0 = aucun oubli)." % oubli)
        L.append("- **Apprentissage immédiat** = b_fr - c_fr = **%+.3f bpc** (positif = le français est mieux prédit)." % gain)
        aj = B["c_en"]["ajout"]
        L.append("- Durée de `ajouter` (%s caractères) : %.4f s de temps réel, %.4f s de CPU."
                 % (format(len(G["fr_memoire"]), ",").replace(",", " "), aj["mur_s"], aj["cpu_s"]))
        L.append("- Pour comparaison, la mémoire anglaise compressée se charge en une dizaine de secondes"
                 " au plus ; aucune phase d'entraînement n'existe.")
        L.append("")
    else:
        L.append("## B. Apprentissage continu")
        L.append("")
        L.append("Non réalisé : `data/francais.txt` est absent.")
        L.append("")
    # -- C
    L.append("## C. Énergie (ESTIMATION)")
    L.append("")
    L.append("**Estimation, pas une mesure** : énergie = temps CPU (`time.process_time`) x %g W, en"
             " supposant %g W par cœur de CPU occupé. Aucune mesure de puissance réelle n'a été faite."
             " Les temps CPU ont été relevés avec plusieurs processus en parallèle sur 4 cœurs."
             % (WATTS, WATTS))
    L.append("")
    L.append("| Mesure | CPU (s) | CPU (s / caractère) | Énergie estimée (mJ / caractère) |")
    L.append("|---|---:|---:|---:|")
    for f in fr_list:
        r = A[f]["echo"]
        L.append("| A, mémoire %s | %s | %.6f | %.2f |" % (pct(f), fnum(r["cpu_s"], 2), r["cpu_s_par_car"], r["mJ_par_car"]))
    for c, nom in [("a_en", "B (a) anglais / mém. anglaise"), ("b_fr", "B (b) français / mém. anglaise"),
                   ("c_en", "B (c) anglais / mém. EN+FR"), ("c_fr", "B (c) français / mém. EN+FR")]:
        if c in B:
            r = B[c]
            L.append("| %s | %s | %.6f | %.2f |" % (nom, fnum(r["cpu_s"], 2), r["cpu_s_par_car"], r["mJ_par_car"]))
    L.append("")
    L.append("Le coût par caractère croît avec la taille de la mémoire, car chaque `str.find` balaie"
             " davantage de texte.")
    L.append("")
    # -- D
    L.append("## D. Taille")
    L.append("")
    D = G.get("taille")
    if D:
        L.append("| Fichier | Octets |")
        L.append("|---|---:|")
        L.append("| memoire.xz (anglais) | %s |" % format(D["en"], ",").replace(",", " "))
        L.append("| memoire_fr.xz (français, lzma 9\\|EXTREME) | %s |" % format(D["fr"], ",").replace(",", " "))
        L.append("| **Somme** | **%s** |" % format(D["somme"], ",").replace(",", " "))
        L.append("")
        L.append("Limite %s octets : **%s**." % (format(LIMITE_OCTETS, ",").replace(",", " "),
                 "RESPECTÉE" if D["somme"] <= LIMITE_OCTETS else "DÉPASSÉE"))
        L.append("")
    else:
        L.append("Non réalisé (pas de corpus français).")
        L.append("")
    L.append("## Durée")
    L.append("")
    L.append("Durée totale du banc : %.0f s (%.1f min)." % (duree_totale, duree_totale / 60))
    L.append("")
    L.append("## Limites et honnêteté")
    L.append("")
    L.append("- Un seul échantillon d'évaluation de ~%d caractères par langue : l'incertitude est de"
             " l'ordre de quelques centièmes de bpc." % n_en)
    L.append("- L'évaluation prédit chaque caractère à partir du texte d'évaluation déjà lu"
             " (contexte <= 16 caractères).")
    L.append("- Fuite écartée : %d paragraphes de `validation.txt` figurent mot pour mot dans la mémoire"
             " anglaise (versets bibliques en double). Ils sont exclus des échantillons d'évaluation"
             " (anglais et français), sinon Écho-P et lzma les recopieraient." % G["doublons_en"])
    L.append("- Écho-P n'est pas un compresseur complet : on calcule -log2 P sans coder réellement.")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def attendre_francais(rapide):
    if os.path.exists(CHEMIN_FR):
        return True
    if rapide:
        return False
    print("data/francais.txt absent : attente (30 s, 20 min au plus)...", flush=True)
    for _ in range(40):
        time.sleep(30)
        if os.path.exists(CHEMIN_FR) and os.path.getsize(CHEMIN_FR) > 0:
            time.sleep(10)  # laisser finir une écriture éventuelle
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Banc d'essai d'Écho-P")
    ap.add_argument("--rapide", action="store_true", help="test de fumée (300 caractères)")
    ap.add_argument("--processus", type=int, default=4)
    args = ap.parse_args()
    t_debut = time.time()
    n_car = 300 if args.rapide else 3000
    fractions = [0.001, 0.01, 0.1] if args.rapide else FRACTIONS

    preparer(n_car, avec_fr=False)
    print("Partie A : %d caractères d'éval, mémoires %s" % (len(G["eval_en"]), fractions), flush=True)
    taches = []
    for f in sorted(fractions, reverse=True):
        for a in ["lzma", "bz2", "zlib"]:
            taches.append(("base", f, a))
        taches.append(("echo_a", f))
    res = lancer(taches, args.processus)
    print("Partie A terminée (%.0f s)" % (time.time() - t_debut), flush=True)

    # -- B, D
    if attendre_francais(args.rapide):
        preparer(n_car, avec_fr=True)
        print("Partie B : français %d paragraphes (mémoire), %d tenus" %
              (G["fr_paras_memoire"], G["fr_paras_tenues"]), flush=True)
        res += lancer([("cont", c) for c in ["c_en", "c_fr", "a_en", "b_fr"]], args.processus)
        xz_fr = lzma.compress(G["fr_memoire"].encode("utf-8"), preset=PRESET_XZ)
        with open(SORTIE_XZ_FR, "wb") as f:
            f.write(xz_fr)
        en = os.path.getsize(CHEMIN_EN)
        G["taille"] = {"en": en, "fr": len(xz_fr), "somme": en + len(xz_fr)}
    else:
        print("Pas de corpus français : parties B et D ignorées.", flush=True)

    duree = time.time() - t_debut
    md = rapport(res, duree, args.rapide)
    chemin_md, chemin_json = SORTIE_MD, SORTIE_JSON
    if args.rapide:  # ne pas écraser les vrais résultats
        chemin_md = SORTIE_MD.replace(".md", "_rapide.md")
        chemin_json = SORTIE_JSON.replace(".json", "_rapide.json")
    with open(chemin_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump({"rapide": args.rapide, "n_car": n_car, "watts_hypothese": WATTS,
                   "duree_totale_s": duree, "taille": G.get("taille"), "resultats": res},
                  f, ensure_ascii=False, indent=2)
    print(md)
    print("Écrit :", chemin_md, chemin_json)


if __name__ == "__main__":
    main()
