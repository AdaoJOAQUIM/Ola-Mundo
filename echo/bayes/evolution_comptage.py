"""Comptage brut (str.find, occurrences qui se chevauchent) : référence lente pour les tests."""


def brut(lib, ctx, sym):
    mot = ctx + sym
    c, i = 0, lib.find(mot)
    while i != -1:
        c += 1
        i = lib.find(mot, i + 1)
    return c
