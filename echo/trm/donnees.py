"""Données Sudoku 9x9 : solveur exact à masques de bits, générateur, augmentation.

Le solveur (propagation de contraintes + retour sur trace, heuristique MRV) est un
algorithme exact : il sert de référence symbolique et de vérificateur d'unicité
pour le générateur. Les grilles sont des listes de 81 entiers (0 = case vide,
1..9 = chiffre), ligne par ligne.
"""
import random
import sys
import time

import numpy as np
import torch

sys.setrecursionlimit(10000)

# ---------------------------------------------------------------- géométrie
UNITES = ([[9 * r + c for c in range(9)] for r in range(9)]
          + [[9 * r + c for r in range(9)] for c in range(9)]
          + [[9 * (3 * br + r) + 3 * bc + c for r in range(3) for c in range(3)]
             for br in range(3) for bc in range(3)])
UNITES_DE = [[u for u in UNITES if i in u] for i in range(81)]
PAIRS = [sorted({j for u in UNITES_DE[i] for j in u if j != i}) for i in range(81)]
POPCOUNT = [bin(v).count("1") for v in range(512)]


# ---------------------------------------------------------------- solveur
def _eliminer(c, i, b):
    """Retire le(s) candidat(s) b de la case i ; propage. False si contradiction."""
    if not c[i] & b:
        return True
    c[i] &= ~b
    v = c[i]
    if v == 0:
        return False
    if v & (v - 1) == 0:  # un seul candidat : on l'élimine chez les pairs
        for p in PAIRS[i]:
            if not _eliminer(c, p, v):
                return False
    for u in UNITES_DE[i]:  # « single caché » : un chiffre n'a plus qu'une place
        places = [j for j in u if c[j] & b]
        if not places:
            return False
        if len(places) == 1 and not _affecter(c, places[0], b):
            return False
    return True


def _affecter(c, i, b):
    autres = c[i] & ~b
    while autres:
        bas = autres & -autres
        if not _eliminer(c, i, bas):
            return False
        autres ^= bas
    return True


def _initialiser(grille):
    c = [511] * 81
    for i, v in enumerate(grille):
        if v and not _affecter(c, i, 1 << (v - 1)):
            return None
    return c


def _chercher(c, limite, sols, rng=None):
    meilleur, bn = -1, 10
    for i in range(81):
        n = POPCOUNT[c[i]]
        if 1 < n < bn:
            meilleur, bn = i, n
            if n == 2:
                break
    if meilleur < 0:
        sols.append(c)
        return
    bits = [1 << k for k in range(9) if c[meilleur] >> k & 1]
    if rng is not None:
        rng.shuffle(bits)
    for b in bits:
        c2 = c[:]
        if _affecter(c2, meilleur, b):
            _chercher(c2, limite, sols, rng)
            if len(sols) >= limite:
                return


def _vers_grille(c):
    return [v.bit_length() for v in c]


def compter_solutions(grille, limite=2):
    """Nombre de solutions, plafonné à `limite`."""
    c = _initialiser(grille)
    if c is None:
        return 0
    sols = []
    _chercher(c, limite, sols)
    return len(sols)


def resoudre(grille):
    """Une solution (liste de 81 entiers) ou None."""
    c = _initialiser(grille)
    if c is None:
        return None
    sols = []
    _chercher(c, 1, sols)
    return _vers_grille(sols[0]) if sols else None


def resolu_par_propagation(grille):
    """True si la propagation seule (sans hypothèse) résout la grille."""
    c = _initialiser(grille)
    return c is not None and all(POPCOUNT[v] == 1 for v in c)


def grille_valide(g):
    """Grille complète sans doublon dans une ligne/colonne/bloc (chiffres 1..9)."""
    return all(sorted(g[i] for i in u) == list(range(1, 10)) for u in UNITES)


def sans_conflit(g):
    """Aucun doublon de chiffre non nul dans une unité (grille éventuellement partielle)."""
    for u in UNITES:
        vs = [g[i] for i in u if g[i]]
        if len(vs) != len(set(vs)):
            return False
    return True


# ---------------------------------------------------------------- générateur
def grille_complete(rng):
    c = [511] * 81
    sols = []
    _chercher(c, 1, sols, rng)
    return _vers_grille(sols[0])


def generer_puzzle(rng, indices_min=22, indices_max=28):
    """Retire des indices tant que la solution reste unique, jusqu'à une cible
    tirée dans [indices_min, indices_max] (ou jusqu'à ne plus pouvoir retirer)."""
    sol = grille_complete(rng)
    cible = rng.randint(indices_min, indices_max)
    puz = sol[:]
    nb = 81
    ordre = list(range(81))
    rng.shuffle(ordre)
    for i in ordre:
        if nb <= cible:
            break
        v = puz[i]
        puz[i] = 0
        if compter_solutions(puz, 2) == 1:
            nb -= 1
        else:
            puz[i] = v
    return puz, sol


# ---------------------------------------------------------------- symétries
def forme_canonique_legere(g):
    """Invariant bon marché sous renommage des chiffres et transposition :
    on renomme les chiffres par ordre de première apparition (lecture ligne par
    ligne), pour la grille et sa transposée, et on prend le minimum."""
    def renom(h):
        m, out = {}, []
        for v in h:
            if v and v not in m:
                m[v] = len(m) + 1
            out.append(m.get(v, 0))
        return tuple(out)
    t = [g[9 * r + c] for c in range(9) for r in range(9)]
    return min(renom(g), renom(t))


def construire_jeux(n_train=1000, n_test=1000, graine=2024, verbeux=True):
    """Génère train/test disjoints (puzzles ET solutions distincts, y compris
    modulo renommage des chiffres/transposition)."""
    rng = random.Random(graine)
    vus_puz, vus_sol = set(), set()
    jeux = {"train": [], "test": []}
    t0 = time.time()
    for nom, n in (("train", n_train), ("test", n_test)):
        while len(jeux[nom]) < n:
            puz, sol = generer_puzzle(rng)
            kp, ks = forme_canonique_legere(puz), forme_canonique_legere(sol)
            if kp in vus_puz or ks in vus_sol:
                continue
            vus_puz.add(kp)
            vus_sol.add(ks)
            jeux[nom].append((puz, sol))
            if verbeux and len(jeux[nom]) % 200 == 0:
                print(nom, len(jeux[nom]), f"{time.time() - t0:.0f}s", flush=True)
    return jeux


def stats_indices(puzzles):
    nb = [sum(1 for v in p if v) for p in puzzles]
    dist = {}
    for k in nb:
        dist[k] = dist.get(k, 0) + 1
    return {"min": min(nb), "max": max(nb), "moyenne": sum(nb) / len(nb),
            "distribution": dict(sorted(dist.items()))}


def sauver(jeux, chemin):
    """Format sans dépendance : torch.save de tenseurs uint8 [N,81]."""
    d = {k: {"puzzles": torch.tensor([p for p, _ in v], dtype=torch.uint8),
             "solutions": torch.tensor([s for _, s in v], dtype=torch.uint8)}
         for k, v in jeux.items()}
    torch.save(d, chemin)


def charger(chemin):
    d = torch.load(chemin)
    return {k: (v["puzzles"].numpy(), v["solutions"].numpy()) for k, v in d.items()}


# ---------------------------------------------------------------- augmentation
def augmenter(puz, sol, rng):
    """Transformation préservant la validité : permutation des chiffres, des
    lignes dans les bandes, des bandes, des colonnes dans les piles, des piles,
    et transposition (la même pour puzzle et solution).
    `puz`, `sol` : np.uint8 [81]. `rng` : np.random.Generator."""
    bandes = rng.permutation(3)
    lignes = np.array([3 * bandes[k] + p for k in range(3) for p in rng.permutation(3)])
    bandes = rng.permutation(3)
    cols = np.array([3 * bandes[k] + p for k in range(3) for p in rng.permutation(3)])
    chiffres = np.concatenate([[0], rng.permutation(9) + 1]).astype(np.uint8)
    tr = rng.random() < 0.5
    res = []
    for g in (puz, sol):
        m = g.reshape(9, 9)[lignes][:, cols]
        if tr:
            m = m.T
        res.append(chiffres[m].reshape(81).copy())
    return res[0], res[1]


if __name__ == "__main__":
    import os
    ici = os.path.dirname(os.path.abspath(__file__))
    jeux = construire_jeux()
    sauver(jeux, os.path.join(ici, "donnees_sudoku.pt"))
    for k, v in jeux.items():
        st = stats_indices([p for p, _ in v])
        print(k, st)
        print("  résolus par propagation seule :",
              sum(resolu_par_propagation(p) for p, _ in v), "/", len(v))
