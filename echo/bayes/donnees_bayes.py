"""Données du paradigme bayésien : bibliothèque réduite, tranche fraîche E, tableaux de comptage.

Principe : on retire de la bibliothèque (memoire.xz) un paragraphe sur PAS_E (déterministe : les lignes
d'indice PAS_E-1, 2·PAS_E-1, ...). Ces paragraphes forment E, texte de test en domaine (même livres,
mais jamais comptés). Tous les comptages (A, B, E, français) viennent de la bibliothèque RÉDUITE seule,
avec le pipeline de evolution/donnees.py (empreintes glissantes, np.searchsorted), importé sans modification.

Règle de lecture : seul final.py a le droit d'appeler `texte_E`, `tableaux_final` et de lire le français.
Le mode « développement » (dev.py, optimisation.py, ...) ne manipule que A et B.
"""
import lzma
import os
import sys
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
ECHO = os.path.dirname(ICI)
sys.path.insert(0, os.path.join(ECHO, "evolution"))

import donnees  # noqa: E402  (module épinglé, importé tel quel)

CACHE = os.path.join(ICI, "cache")
CHEMIN_AB = os.path.join(CACHE, "ab_reduit.npz")
K_MAX = donnees.K_MAX
PAS_E = 250          # un paragraphe sur 250 : ~56 000 caractères (cible 40 000 à 60 000)
PAS_FR = 150         # un paragraphe français sur 150
TAILLE_FR = 50000


def _lignes_biblio():
    with lzma.open(os.path.join(ECHO, "memoire.xz"), "rt", encoding="utf-8") as f:
        return f.read().split("\n")


def _indices_E(n_lignes, pas=PAS_E):
    return set(range(pas - 1, n_lignes, pas))


def bibliotheque_reduite(pas=PAS_E):
    """Bibliothèque sans les paragraphes réservés à E (le texte retiré n'est PAS renvoyé)."""
    lignes = _lignes_biblio()
    ret = _indices_E(len(lignes), pas)
    return "\n".join(l for i, l in enumerate(lignes) if i not in ret)


def texte_E(pas=PAS_E):
    """Texte de E. RÉSERVÉ à final.py."""
    lignes = _lignes_biblio()
    return "\n".join(lignes[i] for i in sorted(_indices_E(len(lignes), pas)))


def textes_A_B():
    """Tranches A et B de validation.txt (identiques à donnees.decouper)."""
    with open(os.path.join(ECHO, "validation.txt"), encoding="utf-8") as f:
        val = f.read()
    ta, tb, _tc = donnees.decouper(val)
    return ta, tb


def alphabet_anglais():
    """Même alphabet que le pipeline d'origine (bibliothèque complète + validation)."""
    lib, val = donnees.lire_textes()
    return donnees.alphabet_de(lib, val)


def texte_francais():
    """~50 000 caractères de français (un paragraphe sur PAS_FR). RÉSERVÉ à final.py. Jamais écrit sur disque."""
    chemin = os.path.join(ECHO, "data", "francais.txt")
    with open(chemin, encoding="utf-8") as f:
        lignes = f.read().split("\n")
    pris = [l for i, l in enumerate(lignes) if i % PAS_FR == 0 and l]
    return "\n".join(pris)[:TAILLE_FR]


def calculer_tableaux(lib, textes, alphabet, K=K_MAX, verbeux=True):
    """{nom: dict(n, N, d, u1, u2, y)} pour chaque texte, comptages de `lib` seule (pipeline donnees.py)."""
    t0 = time.time()
    A = len(alphabet)
    codes_lib = donnees.encoder(lib, alphabet)
    emp_lib = donnees.Empreintes(codes_lib)
    sel = donnees.sels(A)
    codes = {nom: donnees.encoder(t, alphabet) for nom, t in textes.items()}
    emps = {nom: donnees.Empreintes(c) for nom, c in codes.items()}
    n_tab = {nom: np.zeros((K + 1, len(c), A), dtype=np.float32) for nom, c in codes.items()}
    for k in range(K + 1):
        uniques, comptes = donnees.table_ordre(emp_lib, codes_lib, k, sel)
        for nom, c in codes.items():
            T = len(c)
            tt = np.arange(k, T, dtype=np.int64)
            if len(tt) == 0:
                continue
            h = emps[nom].fenetre(tt - k, tt)
            with np.errstate(over="ignore"):
                q = (h[:, None] + sel[None, :]).ravel()
            pos = np.minimum(np.searchsorted(uniques, q), len(uniques) - 1)
            trouve = uniques[pos] == q
            v = np.where(trouve, comptes[pos], 0.0).astype(np.float32).reshape(-1, A)
            n_tab[nom][k][k:] = v
        del uniques, comptes
        if verbeux:
            print("  ordre %d fait (%.0f s)" % (k, time.time() - t0), flush=True)
    res = {}
    for nom, n in n_tab.items():
        res[nom] = dict(n=n, N=n.sum(axis=2, keepdims=True),
                        d=(n > 0).sum(axis=2, keepdims=True).astype(np.float32),
                        u1=(n == 1).sum(axis=2, keepdims=True).astype(np.float32),
                        u2=(n == 2).sum(axis=2, keepdims=True).astype(np.float32),
                        y=codes[nom])
    return res


def tableaux_dev(verbeux=True):
    """A et B recomptés sur la bibliothèque réduite (mis en cache). Ne touche jamais à E ni au français."""
    if os.path.exists(CHEMIN_AB):
        z = np.load(CHEMIN_AB)
        out = {"alphabet": "".join(z["alphabet"].tolist())}
        for s in ("A", "B"):
            out[s] = {c: z[c + s] for c in ("n", "N", "d", "u1", "u2", "y")}
        return out
    alphabet = alphabet_anglais()
    ta, tb = textes_A_B()
    res = calculer_tableaux(bibliotheque_reduite(), {"A": ta, "B": tb}, alphabet, verbeux=verbeux)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(CHEMIN_AB, alphabet=np.array(list(alphabet)),
             **{k + "A": v for k, v in res["A"].items()}, **{k + "B": v for k, v in res["B"].items()})
    return {"alphabet": alphabet, "A": res["A"], "B": res["B"]}


def tableaux_final(verbeux=True):
    """E (alphabet anglais) et français (alphabet étendu). RÉSERVÉ à final.py ; un seul passage sur les ordres.

    Bibliothèque réduite pour les deux. Le français est lu ici et nulle part ailleurs ; son texte n'est
    ni écrit ni mis en cache.
    """
    alphabet = alphabet_anglais()
    tE = texte_E()
    tF = texte_francais()
    alphabet_fr = "".join(sorted(set(alphabet) | set(tF)))
    lib = bibliotheque_reduite()
    # même passage de comptage pour les deux alphabets : on traite E puis le français séparément
    resE = calculer_tableaux(lib, {"E": tE}, alphabet, verbeux=verbeux)["E"]
    resF = calculer_tableaux(lib, {"F": tF}, alphabet_fr, verbeux=verbeux)["F"]
    return {"alphabet": alphabet, "alphabet_fr": alphabet_fr, "E": resE, "F": resF, "lib": lib,
            "texte_E": tE, "texte_F": tF}
