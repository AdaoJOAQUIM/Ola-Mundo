"""down en 3 bits pour les experts de poids moyen (w < 0.2) que lod3_10 ne couvre pas"""
NAME = "dn3_mid"
PARENT = "base4"
MODE = "add"
def applies(L, e, w, r):
    return w < 0.2
def plan(L, e, w, r):
    return (4, 3, 0.0, 0.0)
def fallback(L, e, w, r, perte):
    return False
