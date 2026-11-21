"""Test final sur C : à lancer UNE SEULE FOIS.

C'est le seul fichier autorisé à charger cache/c.npz. Il compare :
  * le génome de départ et le champion (bpc sur C, gain relatif, intervalle par blocs) ;
  * Écho-P d'origine (predicteur.py), sur un sous-ensemble déclaré de C
    (mêmes positions que le départ et le champion) ;
  * la compression LZMA conditionnelle : taille(biblio + C) - taille(biblio), en bits par caractère de C.
Il écrit aussi champion.py, lignee.md et resultats_test.json.
Un fichier-témoin empêche de relancer le test par accident (option --rejouer pour passer outre).
"""
import argparse
import difflib
import json
import lzma
import multiprocessing as mp
import os
import sys
import time
from multiprocessing.pool import ThreadPool

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
ECHO = os.path.dirname(ICI)
sys.path.insert(0, ICI)
sys.path.insert(0, ECHO)

import donnees  # noqa: E402
import garde_fous  # noqa: E402
import genome_initial  # noqa: E402
import juge  # noqa: E402

TEMOIN = os.path.join(ICI, "cache", "c_utilise.json")


def categorie(c):
    if c == "\n":
        return "retour_ligne"
    if c == " ":
        return "espace"
    if c.isdigit():
        return "chiffre"
    if c.isalpha() and c.islower():
        return "minuscule"
    if c.isalpha() and c.isupper():
        return "majuscule"
    if not c.isalnum() and not c.isspace():
        return "ponctuation"
    return "autre"


# ---- Écho-P sur un sous-ensemble -------------------------------------------------
_MEM = None
_TEXTE_C = None


def _bloc_echo(args):
    debut, longueur = args
    import math
    om = _MEM.ordre_max
    depart = None
    pertes = []
    for i in range(debut, debut + longueur):
        ctx = _TEXTE_C[max(0, i - om):i]
        p, ordre = _MEM.proba(ctx, _TEXTE_C[i], depart)
        pertes.append(-math.log2(p))
        depart = ordre + 1
    return debut, pertes


def echo_p_sous_ensemble(lib, texte_c, nb_blocs, longueur, nproc=4):
    global _MEM, _TEXTE_C
    from predicteur import Memoire
    _MEM = Memoire(lib)
    _TEXTE_C = texte_c
    debuts = [int(x) for x in np.linspace(0, len(texte_c) - longueur - 1, nb_blocs)]
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc) as pool:
        res = pool.map(_bloc_echo, [(d, longueur) for d in debuts], chunksize=1)
    pos, pert = [], []
    for d, p in res:
        pos.extend(range(d, d + longueur))
        pert.extend(p)
    return np.array(pos), np.array(pert)


def lzma_conditionnel(lib, texte_c):
    b_lib = lib.encode("utf-8")
    b_tot = (lib + texte_c).encode("utf-8")
    preset = 9 | lzma.PRESET_EXTREME
    with ThreadPool(2) as tp:
        a, b = tp.map(lambda x: len(lzma.compress(x, preset=preset)), [b_lib, b_tot])
    return a, b, 8.0 * (b - a) / len(texte_c)


def bootstrap_blocs(pa, pb, taille_bloc=500, n=2000, graine=0):
    """Gain relatif de pb par rapport à pa, avec rééchantillonnage de blocs consécutifs."""
    rng = np.random.default_rng(graine)
    nb = len(pa) // taille_bloc
    sa = pa[:nb * taille_bloc].reshape(nb, taille_bloc).sum(1)
    sb = pb[:nb * taille_bloc].reshape(nb, taille_bloc).sum(1)
    g = []
    for _ in range(n):
        idx = rng.integers(0, nb, nb)
        g.append(1 - sb[idx].sum() / sa[idx].sum())
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def ecrire_lignee(archive, champ, chemin):
    par_id = {g["id"]: g for g in archive}
    chaine = []
    g = champ
    while g is not None:
        chaine.append(g)
        g = par_id.get(g["parent"])
    chaine.reverse()
    L = ["# Lignée du champion", "",
         "Chaque étape : le parent, les mutations appliquées, le score sur A (fitness) et sur B (sélection), "
         "puis les lignes de code qui changent. Les scores sont en bits par caractère (plus petit = mieux).", ""]
    for i, g in enumerate(chaine):
        fb = "non évalué" if g["fit_B"] is None else "%.4f" % g["fit_B"]
        if i == 0:
            L += ["## Étape 0 : génome de départ (PPM-C interpolé)", "", "- A = %.4f, B = %s" % (g["fit_A"], fb), ""]
            continue
        p = par_id[g["parent"]]
        L += ["## Étape %d : génome n°%d (parent n°%d, génération %d)" % (i, g["id"], p["id"], g["gen"]), "",
              "- mutations : %s" % ", ".join(g["mutations"]),
              "- A : %.4f -> %.4f (%+.4f) ; B = %s" % (p["fit_A"], g["fit_A"], g["fit_A"] - p["fit_A"], fb), "",
              "```diff"]
        diff = difflib.unified_diff(p["source"].splitlines(), g["source"].splitlines(), lineterm="", n=0)
        L += [l for l in diff if not l.startswith(("---", "+++", "@@"))]
        L += ["```", ""]
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return chaine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="principal")
    ap.add_argument("--echo-blocs", type=int, default=24)
    ap.add_argument("--echo-long", type=int, default=400)
    ap.add_argument("--rejouer", action="store_true")
    a = ap.parse_args()
    if os.path.exists(TEMOIN) and not a.rejouer:
        sys.exit("Le test sur C a déjà été lancé (%s). Il ne doit tourner qu'une fois." % TEMOIN)
    mauvais = garde_fous.verifier_empreintes()
    if mauvais:
        sys.exit("ABANDON : empreintes modifiées : %s" % mauvais)
    t0 = time.time()
    with open(os.path.join(ICI, "archive", a.run + "_resume.json")) as f:
        resume = json.load(f)
    archive = [json.loads(l) for l in open(os.path.join(ICI, "archive", a.run + "_final.jsonl"))]
    champ = next(g for g in archive if g["id"] == resume["champion"]["id"])
    for nom, src in (("départ", genome_initial.SOURCE), ("champion", champ["source"])):
        ok, code, _ = garde_fous.verifier(src)
        assert ok, (nom, code)

    # --- chargement de C (seul endroit du projet) ---
    z = np.load(os.path.join(ICI, "cache", "c.npz"))
    jeu = {c: z[c + "C"] for c in ("n", "N", "d", "u1", "u2", "y")}
    alphabet = "".join(z["alphabet"].tolist())
    lib, val = donnees.lire_textes()
    _, _, texte_c = donnees.decouper(val)
    assert len(texte_c) == len(jeu["y"]) and all(alphabet[i] == c for i, c in zip(jeu["y"][:200], texte_c[:200]))

    r0 = juge.juger(genome_initial.SOURCE, jeu, pertes=True)
    r1 = juge.juger(champ["source"], jeu, pertes=True)
    assert r0["ok"] and r1["ok"]
    p0, p1 = r0["pertes"], r1["pertes"]
    # référence « à la main » (2 réglages choisis sur A par analyse_champion.py)
    ref = json.load(open(os.path.join(ICI, "analyse_champion.json")))["reference_a_la_main"]
    assert garde_fous.verifier(ref["source"])[0]
    r2 = juge.juger(ref["source"], jeu, pertes=True)
    p2 = r2["pertes"]
    gain = 1 - r1["bpc"] / r0["bpc"]
    ic = bootstrap_blocs(p0, p1)
    print("C : départ %.4f, champion %.4f, gain %.2f %%" % (r0["bpc"], r1["bpc"], 100 * gain), flush=True)

    # --- catégories ---
    cats = {}
    gain_total = p0.sum() - p1.sum()
    for i, c in enumerate(texte_c):
        cats.setdefault(categorie(c), []).append(i)
    diag = {}
    for nom, idx in cats.items():
        idx = np.array(idx)
        diag[nom] = {"n": int(len(idx)), "part_des_caracteres": float(len(idx) / len(texte_c)),
                     "bpc_depart": float(p0[idx].mean()), "bpc_champion": float(p1[idx].mean()),
                     "part_du_gain_total": float((p0[idx].sum() - p1[idx].sum()) / gain_total)}

    # --- LZMA conditionnel ---
    t1 = time.time()
    tl_lib, tl_tot, bpc_lzma = lzma_conditionnel(lib, texte_c)
    print("LZMA conditionnel : %.4f bpc (%.0f s)" % (bpc_lzma, time.time() - t1), flush=True)

    # --- Écho-P sur sous-ensemble ---
    t2 = time.time()
    pos, pert_echo = echo_p_sous_ensemble(lib, texte_c, a.echo_blocs, a.echo_long)
    cpu_echo = time.time() - t2
    sous = {"positions": int(len(pos)), "blocs": a.echo_blocs, "longueur_bloc": a.echo_long,
            "echo_p_bpc": float(pert_echo.mean()), "depart_bpc": float(p0[pos].mean()),
            "champion_bpc": float(p1[pos].mean()), "reference_main_bpc": float(p2[pos].mean()),
            "duree_s": round(cpu_echo, 1)}
    print("Écho-P sous-ensemble:", sous, flush=True)

    sortie = {
        "taille_C": len(texte_c), "alphabet": len(alphabet),
        "C": {"depart_bpc": r0["bpc"], "champion_bpc": r1["bpc"], "gain_relatif": gain,
              "gain_relatif_IC95_blocs500": ic},
        "A_B": {"depart_A": resume["seed_A"], "depart_B": resume["seed_B"],
                "champion_A": champ["fit_A"], "champion_B": champ["fit_B"]},
        "reference_a_la_main": {"c": ref["c"], "gamma": ref["gamma"], "bpc_C": r2["bpc"],
                                "gain_relatif_sur_depart": 1 - r2["bpc"] / r0["bpc"],
                                "champion_moins_reference_bpc": r1["bpc"] - r2["bpc"]},
        "sous_ensemble_echo_p": sous,
        "lzma_conditionnel": {"taille_biblio_octets": tl_lib, "taille_biblio_plus_C_octets": tl_tot, "bpc": bpc_lzma},
        "categories": diag, "champion_id": champ["id"], "duree_totale_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(ICI, "resultats_test.json"), "w") as f:
        json.dump(sortie, f, indent=1, ensure_ascii=False)
    with open(os.path.join(ICI, "champion.py"), "w", encoding="utf-8") as f:
        f.write('"""Génome champion de l\'évolution (génome n°%d, bpc sur A = %.4f, sur B = %.4f).\n\n'
                'Fichier généré par tester.py ; le texte de la fonction est exactement celui que le juge a évalué.\n"""\n'
                "import numpy as np\n\nfrom juge import div  # noqa: F401\n\n\n%s" % (champ["id"], champ["fit_A"], champ["fit_B"], champ["source"]))
    ecrire_lignee(archive, champ, os.path.join(ICI, "lignee.md"))
    with open(TEMOIN, "w") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "run": a.run}, f)
    print(json.dumps(sortie, indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
