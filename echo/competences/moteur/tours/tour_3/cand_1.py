"""ecart < 0.02, 3 bits < 0.12"""
NAME = "echelle_a"
PARENT = "lod3_10"
MODE = "replace"
def applies(L, e, w, r):
    return w < 0.12
def plan(L, e, w, r):
    return (0, 4, 0.0, 0.0) if w < 0.02 else (3, 3, 0.0, 0.0)
def fallback(L, e, w, r, perte):
    return False
