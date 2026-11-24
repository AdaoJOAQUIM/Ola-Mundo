"""Precision 3 bits (2 plans de poids fort + 1) pour les experts de faible poids du routeur"""
NAME = "lod3_08"
PARENT = "base4"
MODE = "add"
def applies(L, e, w, r):
    return w < 0.08
def plan(L, e, w, r):
    return (3, 3, 0.0, 0.0)
def fallback(L, e, w, r, perte):
    return False
