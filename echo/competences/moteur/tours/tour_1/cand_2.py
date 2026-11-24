"""Idem seuil 0.10"""
NAME = "lod3_10"
PARENT = "base4"
MODE = "add"
def applies(L, e, w, r):
    return w < 0.10
def plan(L, e, w, r):
    return (3, 3, 0.0, 0.0)
def fallback(L, e, w, r, perte):
    return False
