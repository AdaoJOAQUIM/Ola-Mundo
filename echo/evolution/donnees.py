"""Données et statistiques de comptage pour le moteur d'évolution.

Les comptages viennent UNIQUEMENT du texte de la bibliothèque (statique, sans
adaptation). Pour chaque position t d'une tranche et chaque ordre k = 0..K, le
contexte est formé des k caractères précédents de la MÊME tranche.

Méthode rapide : empreintes glissantes uint64 des contextes de la bibliothèque,
clés triées (empreinte du contexte + sel du symbole suivant), comptage par
np.searchsorted.

Seul tester.py a le droit de charger cache/c.npz. Ici, on ne fait que l'écrire.
"""
import json
import lzma
import os
import sys
import time

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
ECHO = os.path.dirname(ICI)
CACHE = os.path.join(ICI, "cache")
CHEMIN_AB = os.path.join(CACHE, "ab.npz")
CHEMIN_C = os.path.join(CACHE, "c.npz")

K_MAX = 10
TAILLE_A = 15000
TAILLE_B = 15000
TAILLE_C_MAX = 60000


BASE = 0x9E3779B97F4A7C15 | 1  # impair, donc inversible modulo 2**64
BASE_INV = pow(BASE, -1, 1 << 64)


def lire_textes():
    """Retourne (bibliothèque, validation) sous forme de str."""
    with lzma.open(os.path.join(ECHO, "memoire.xz"), "rt", encoding="utf-8") as f:
        lib = f.read()
    with open(os.path.join(ECHO, "validation.txt"), encoding="utf-8") as f:
        val = f.read()
    return lib, val


def alphabet_de(lib, val):
    return "".join(sorted(set(lib) | set(val)))


def decouper(val):
    """Trois tranches consécutives et disjointes : A (fitness), B (sélection), C (test)."""
    n = len(val)
    if n >= TAILLE_A + TAILLE_B + 1000:
        a, b = TAILLE_A, TAILLE_B
        c = min(n - a - b, TAILLE_C_MAX)
    else:
        a, b = int(n * 0.25), int(n * 0.25)
        c = n - a - b
    return val[:a], val[a:a + b], val[a + b:a + b + c]


def encoder(texte, alphabet):
    idx = {c: i for i, c in enumerate(alphabet)}
    return np.fromiter((idx[c] for c in texte), dtype=np.int64, count=len(texte))


def _puissances(base, n):
    """base**0 .. base**(n-1) modulo 2**64 (uint64 qui déborde)."""
    p = np.full(n, base, dtype=np.uint64)
    p[0] = 1
    with np.errstate(over="ignore"):
        return np.cumprod(p, dtype=np.uint64)


class Empreintes:
    """Empreinte de fenêtre [a, b) : somme (s_j+1) * BASE**(j-a) modulo 2**64."""

    def __init__(self, codes):
        n = len(codes)
        self.bp = _puissances(BASE, n + 1)
        self.ip = _puissances(BASE_INV, n + 1)
        with np.errstate(over="ignore"):
            terme = (codes.astype(np.uint64) + np.uint64(1)) * self.bp[:n]
            self.H = np.zeros(n + 1, dtype=np.uint64)
            np.cumsum(terme, dtype=np.uint64, out=self.H[1:])

    def fenetre(self, a, b):
        with np.errstate(over="ignore"):
            return (self.H[b] - self.H[a]) * self.ip[a]


def sels(A):
    rng = np.random.default_rng(12345)
    return rng.integers(1, 2**63, size=A, dtype=np.uint64) * np.uint64(2) + np.uint64(1)


def table_ordre(emp_lib, codes_lib, k, sel):
    """Clés triées uniques et comptes pour l'ordre k sur la bibliothèque."""
    L = len(codes_lib)
    i = np.arange(k, L, dtype=np.int64)
    with np.errstate(over="ignore"):
        cle = emp_lib.fenetre(i - k, i) + sel[codes_lib[k:]]
    cle.sort()
    flag = np.empty(len(cle), dtype=bool)
    flag[0] = True
    flag[1:] = cle[1:] != cle[:-1]
    debuts = np.flatnonzero(flag)
    uniques = cle[debuts]
    comptes = np.diff(np.append(debuts, len(cle))).astype(np.float32)
    return uniques, comptes


def precalculer(K=K_MAX, verbeux=True):
    t0 = time.time()
    lib, val = lire_textes()
    alphabet = alphabet_de(lib, val)
    A = len(alphabet)
    ta, tb, tc = decouper(val)
    codes_lib = encoder(lib, alphabet)
    emp_lib = Empreintes(codes_lib)
    sel = sels(A)
    tranches = {"A": encoder(ta, alphabet), "B": encoder(tb, alphabet), "C": encoder(tc, alphabet)}
    sortie = {nom: [None] * 5 for nom in tranches}
    for k in range(K + 1):
        tab = table_ordre(emp_lib, codes_lib, k, sel)
        for nom, codes in tranches.items():
            sortie[nom][0] = sortie[nom][0] if sortie[nom][0] is not None else np.zeros(
                (K + 1, len(codes), A), dtype=np.float32)
            T = len(codes)
            emp = Empreintes(codes)
            t = np.arange(T, dtype=np.int64)
            ok = t >= k
            tt = t[ok]
            h = emp.fenetre(tt - k, tt)
            with np.errstate(over="ignore"):
                q = (h[:, None] + sel[None, :]).ravel()
            pos = np.minimum(np.searchsorted(tab[0], q), len(tab[0]) - 1)
            trouve = tab[0][pos] == q
            v = np.where(trouve, tab[1][pos], 0.0).astype(np.float32).reshape(-1, A)
            sortie[nom][0][k][ok] = v
        del tab
        if verbeux:
            print("ordre %d fait (%.0f s)" % (k, time.time() - t0), flush=True)
    res = {}
    for nom, codes in tranches.items():
        n = sortie[nom][0]
        N = n.sum(axis=2, keepdims=True)
        d = (n > 0).sum(axis=2, keepdims=True).astype(np.float32)
        u1 = (n == 1).sum(axis=2, keepdims=True).astype(np.float32)
        u2 = (n == 2).sum(axis=2, keepdims=True).astype(np.float32)
        res[nom] = dict(n=n, N=N, d=d, u1=u1, u2=u2, y=codes)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(CHEMIN_AB, alphabet=np.array(list(alphabet)),
             **{k + "A": v for k, v in res["A"].items()},
             **{k + "B": v for k, v in res["B"].items()})
    np.savez(CHEMIN_C, alphabet=np.array(list(alphabet)), **{k + "C": v for k, v in res["C"].items()})
    if verbeux:
        print("précalcul terminé en %.0f s ; A=%d symboles ; tailles A/B/C = %d/%d/%d" % (
            time.time() - t0, A, len(ta), len(tb), len(tc)))
    return time.time() - t0


def charger_ab():
    """Charge les jeux A et B (jamais C). Retourne {'A': dict, 'B': dict, 'alphabet': str}."""
    z = np.load(CHEMIN_AB)
    out = {"alphabet": "".join(z["alphabet"].tolist())}
    for s in ("A", "B"):
        out[s] = {c: z[c + s] for c in ("n", "N", "d", "u1", "u2", "y")}
    return out


def comptage_brut(lib, ctx, sym):
    """Comptage brut (occurrences qui se chevauchent) de ctx+sym dans lib, pour les tests."""
    mot = ctx + sym
    c, i = 0, lib.find(mot)
    while i != -1:
        c += 1
        i = lib.find(mot, i + 1)
    return c


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else K_MAX
    precalculer(K)
