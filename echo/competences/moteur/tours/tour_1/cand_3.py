"""Echelle de precision : 2 bits gate/up si w<0.04, sinon 3 bits"""
NAME = "echelle_lite"
PARENT = "base4"
MODE = "add"
def applies(L, e, w, r):
    return w < 0.09
def plan(L, e, w, r):
    return (2, 3, 0.0, 0.0) if w < 0.04 else (3, 3, 0.0, 0.0)
def fallback(L, e, w, r, perte):
    return False
