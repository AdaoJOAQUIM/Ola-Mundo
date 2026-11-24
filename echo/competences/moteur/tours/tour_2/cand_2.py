"""Echelle de precision elargie : ecart < 0.025, 2 bits < 0.05, 3 bits < 0.14"""
NAME = "echelle3"
PARENT = "lod3_10"
MODE = "replace"
def applies(L, e, w, r):
    return w < 0.14
def plan(L, e, w, r):
    return (0, 4, 0.0, 0.0) if w < 0.025 else ((2, 3, 0.0, 0.0) if w < 0.05 else (3, 3, 0.0, 0.0))
def fallback(L, e, w, r, perte):
    return False
