"""Ecarter les experts de poids normalise < 0.03"""
NAME = "cull03"
PARENT = "base4"
MODE = "add"
def applies(L, e, w, r):
    return w < 0.03
def plan(L, e, w, r):
    return (0, 4, 0.0, 0.0)
def fallback(L, e, w, r, perte):
    return False
