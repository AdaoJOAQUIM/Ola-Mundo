"""Écho — un modèle de langage sans paramètres, sans entraînement, sans réseau de neurones.

Le « modèle » est une bibliothèque de textes compressée (memoire.xz, < 10 Mo).
Pour écrire, Écho cherche dans la bibliothèque le plus long passage qui finit
comme le texte en cours, regarde ce qui suit ce passage à ses différentes
apparitions, et en choisit une suite au hasard. Il ne s'appuie que sur des
passages qui ont au moins `diversite` suites différentes : il est donc obligé
de choisir et de recombiner, au lieu de recopier un livre.

Uniquement la bibliothèque standard de Python.
"""
import argparse
import lzma
import os
import random
import sys
import time

ICI = os.path.dirname(os.path.abspath(__file__))


def charger(chemin=os.path.join(ICI, "memoire.xz")):
    with lzma.open(chemin, "rt", encoding="utf-8") as f:
        return f.read()


def occurrences(T, motif, limite, rng):
    """Jusqu'à `limite` positions de `motif` dans T, à partir d'un point au hasard."""
    n = len(T)
    debut = rng.randrange(n)
    res = []
    i = T.find(motif, debut)
    while i != -1 and len(res) < limite:
        res.append(i)
        i = T.find(motif, i + 1)
    if len(res) < limite:  # on fait le tour
        i = T.find(motif, 0, debut + len(motif))
        while i != -1 and i < debut and len(res) < limite:
            res.append(i)
            i = T.find(motif, i + 1, debut + len(motif))
    return res


def mot_suivant(T, p):
    """Le mot qui commence en p, espace final compris (ou jusqu'à la fin du paragraphe)."""
    e = T.find(" ", p, p + 40)
    nl = T.find("\n", p, p + 40)
    if nl != -1 and (e == -1 or nl < e):
        return T[p:nl + 1]
    return T[p:e + 1] if e != -1 else T[p:p + 40]


def ecrire(T, amorce, n=500, k_max=32, diversite=2, graine=None):
    """Génère environ `n` caractères à la suite de `amorce`, mot par mot.

    Le contexte se termine toujours à la fin d'un mot : chaque mot ajouté est
    donc un vrai mot de la bibliothèque, et seule leur combinaison est nouvelle.
    """
    rng = random.Random(graine)
    if not amorce:  # commencer au début d'un paragraphe au hasard
        p = T.find("\n", rng.randrange(len(T))) + 1
        amorce = mot_suivant(T, p)
    sortie = amorce if amorce[-1] in " \n" else amorce + " "
    k = k_max
    while len(sortie) - len(amorce) < n:
        k = min(k + 4, k_max, len(sortie))
        choix = None
        while k >= 1:
            ctx = sortie[-k:]
            pos = [p for p in occurrences(T, ctx, 48, rng) if p + k < len(T)]
            suites = {mot_suivant(T, p + k) for p in pos}
            if len(suites) >= diversite or (k <= 2 and pos):
                choix = rng.choice(pos) + k
                break
            k -= 1
        if choix is None:
            break
        sortie += mot_suivant(T, choix)
    return sortie


def nouveaute(T, texte, n=5):
    """Part des suites de `n` mots du texte qui n'existent nulle part dans la bibliothèque."""
    mots = texte.split()
    grams = [" ".join(mots[i:i + n]) for i in range(len(mots) - n + 1)]
    if not grams:
        return 0.0
    return sum(T.find(g) == -1 for g in grams) / len(grams)


def plus_long_emprunt(T, texte):
    """Longueur du plus long passage de `texte` copié tel quel de la bibliothèque."""
    best, i, j = 0, 0, 1
    while j <= len(texte):
        if T.find(texte[i:j]) != -1:
            best = max(best, j - i)
            j += 1
        else:
            i += 1
            j = max(j, i + 1)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("amorce", nargs="?", default="")
    ap.add_argument("-n", type=int, default=500, help="caractères à générer")
    ap.add_argument("--k_max", type=int, default=32, help="mémoire de contexte (caractères)")
    ap.add_argument("--diversite", type=int, default=2,
                    help="nombre minimum de mots suivants différents pour s'appuyer sur un passage "
                         "(plus haut = plus inventif, moins cohérent)")
    ap.add_argument("--graine", type=int, default=None)
    ap.add_argument("--mesures", action="store_true",
                    help="mesurer le plus long passage recopié et la part de nouveauté")
    a = ap.parse_args()

    t0 = time.time()
    T = charger()
    print(f"[bibliothèque : {len(T):,} caractères, chargée en {time.time()-t0:.1f} s]",
          file=sys.stderr)
    t0 = time.time()
    s = ecrire(T, a.amorce, a.n, a.k_max, a.diversite, a.graine)
    print(s)
    print(f"[{len(s)} caractères en {time.time()-t0:.1f} s]", file=sys.stderr)
    if a.mesures:
        print(f"[plus long passage recopié tel quel : {plus_long_emprunt(T, s)} caractères ; "
              f"suites de 5 mots inédites : {100*nouveaute(T, s):.0f} %]", file=sys.stderr)


if __name__ == "__main__":
    main()
