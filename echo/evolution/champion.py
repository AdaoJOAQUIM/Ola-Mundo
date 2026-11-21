"""Génome champion de l'évolution (génome n°3321, bpc sur A = 1.6276, sur B = 1.4772).

Fichier généré par tester.py ; le texte de la fonction est exactement celui que le juge a évalué.
"""
import numpy as np

from juge import div  # noqa: F401


def estimer(n, N, d, u1, u2, K):
    p = np.sqrt(0) + 0.92546 / n.shape[-1]
    for k in range(K + 1):
        b = div(u2[k], p)
        m = n[k] + d[k]
        e = d[k] * [2.17814, 2.23475, 2.3698, 2.43409, 2.36564, 2.30956, 2.34621, 2.18404, 1.94839, 2.20264, 2.19865][k] + u1[k]
        q = p
        p = np.where(N[k] > 0.25491, div(np.maximum(n[k], 0.0) + (e + m) * p, N[k]), p)
    p = p ** 0.64919
    return p
