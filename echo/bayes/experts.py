"""Les experts : chacun donne, à chaque position, une distribution normalisée ; on garde log2 p(vrai symbole).

Deux familles, plus des références :
  (a) grille statique : interpolation de comptes de la bibliothèque, e_k = c·d_k (+ u1_k), ordre max K,
      puissance finale γ. Aucune lecture du texte déjà vu.
  (b) experts à mise à jour bayésienne : pseudo-comptes a posteriori de Dirichlet-multinomial,
      n'_k(ctx, s) = β · n_biblio_k(ctx, s) + n_flux_k(ctx, s), où n_flux_k ne compte que les occurrences
      ENTIÈREMENT situées avant la position t du même flux (strictement causal). β règle le poids de
      l'a priori (la bibliothèque) face à ce qui vient d'être lu. Pas d'entraînement : lire = mettre à jour.
  (c) références : départ (PPM-C), champion évolué, référence à la main (e = 16·d), Pitman-Yor.

Interpolation (Chen-Goodman / PPM-C interpolé) : p_k = (n'_k + e_k p_{k-1}) / (N'_k + e_k) si N'_k > 0, sinon
p_{k-1} ; base uniforme. Chaque p_k est normalisée (somme 1), puis p ← p^γ renormalisée.
"""
import importlib.util
import math
import os

import numpy as np

import donnees_bayes as db  # noqa: F401  (met evolution/ dans sys.path)

import donnees  # noqa: E402
import genome_initial  # noqa: E402
import juge  # noqa: E402

PLANCHER_P = juge.PLANCHER_P
C_GRILLE = [1, 2, 4, 8, 16, 32, 64]
ORDRES_GRILLE = [2, 4, 6, 8, 10]
GAMMAS_GRILLE = [0.5, 0.65, 0.8, 1.0]
BETAS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.001]
C_MAJ = [4, 16, 64]
ORDRES_MAJ = [4, 6, 8, 10]
GAMMAS_MAJ = [0.65, 1.0]


# ---------------------------------------------------------------------------
# Sources de comptes
# ---------------------------------------------------------------------------
class Statique:
    """Comptes de la bibliothèque seule."""

    def __init__(self, jeu):
        self.jeu = jeu
        self.K = jeu["n"].shape[0] - 1

    def n(self, k):
        return self.jeu["n"][k].astype(np.float64)

    def N(self, k):
        return self.jeu["N"][k].astype(np.float64)

    def d(self, k):
        return self.jeu["d"][k].astype(np.float64)

    def u1(self, k):
        return self.jeu["u1"][k].astype(np.float64)


class MiseAJour:
    """Pseudo-comptes a posteriori : β · bibliothèque + flux déjà lu (strictement avant t)."""

    def __init__(self, jeu, flux, beta):
        self.jeu, self.flux, self.beta = jeu, flux, beta
        self.K = jeu["n"].shape[0] - 1

    def n(self, k):
        return self.beta * self.jeu["n"][k].astype(np.float64) + self.flux["n"][k]

    def N(self, k):
        return self.beta * self.jeu["N"][k].astype(np.float64) + self.flux["N"][k]

    def d(self, k):
        return self.flux["d"][k]

    def u1(self, k):
        raise NotImplementedError("pas de u1 pour les experts à mise à jour")


def comptes_flux(y, A, K):
    """Comptes du flux lui-même, strictement causaux.

    Retourne dict(n (K+1,T,A) float64... en float32, N, d) : n[k][t, s] = nombre d'indices j < t tels que
    y[j-k:j] = y[t-k:t] et y[j] = s. (Une occurrence j < t n'utilise que y[:j+1], donc rien de futur.)
    """
    T = len(y)
    emp = donnees.Empreintes(y)
    n = np.zeros((K + 1, T, A), dtype=np.float32)
    for k in range(K + 1):
        tt = np.arange(k, T, dtype=np.int64)
        if len(tt) == 0:
            continue
        h = emp.fenetre(tt - k, tt)
        ordre = np.argsort(h, kind="stable")          # à contexte égal, t croissant
        pos = tt[ordre]
        hs = h[ordre]
        m = len(pos)
        un = np.zeros((m, A), dtype=np.float32)
        un[np.arange(m), y[pos]] = 1.0
        cs = np.cumsum(un, axis=0, dtype=np.float32)
        excl = cs - un                                   # occurrences strictement antérieures dans la pile
        debut = np.ones(m, dtype=bool)
        debut[1:] = hs[1:] != hs[:-1]
        idx_debut = np.flatnonzero(debut)
        base = excl[idx_debut]                           # cumul avant le groupe
        taille = np.diff(np.append(idx_debut, m))
        excl -= np.repeat(base, taille, axis=0)
        n[k][pos] = excl
    N = n.sum(axis=2, keepdims=True)
    d = (n > 0).sum(axis=2, keepdims=True).astype(np.float32)
    return {"n": n, "N": N, "d": d}


class FluxDense:
    """Enveloppe : indexation flux["n"][k] en float64, avec d' = |supp(biblio) ∪ supp(flux)|."""

    def __init__(self, jeu, y):
        A = jeu["n"].shape[2]
        K = jeu["n"].shape[0] - 1
        f = comptes_flux(y, A, K)
        self.nf = f["n"]
        self.Nf = f["N"]
        self.df = (((jeu["n"] > 0) | (f["n"] > 0)).sum(axis=2, keepdims=True)).astype(np.float64)


class _Idx:
    def __init__(self, f):
        self.f = f

    def __getitem__(self, k):
        return self.f(k)


def source_maj(jeu, fd, beta):
    s = MiseAJour(jeu, {"n": _Idx(lambda k: fd.nf[k].astype(np.float64)),
                        "N": _Idx(lambda k: fd.Nf[k].astype(np.float64)),
                        "d": _Idx(lambda k: fd.df[k])}, beta)
    return s


# ---------------------------------------------------------------------------
# Chaîne d'interpolation et lecture du vrai symbole
# ---------------------------------------------------------------------------
def chaine(src, cs, use_u1, snaps):
    """Interpolation sur les ordres 0..max(snaps). cs : c_k par ordre. Retourne {ordre_max: p (T,A)}."""
    kmax = max(snaps)
    n0 = src.n(0)
    p = np.full(n0.shape, 1.0 / n0.shape[1])
    out = {}
    for k in range(kmax + 1):
        nk = src.n(k) if k else n0
        Nk = src.N(k)
        e = cs[k] * src.d(k)
        if use_u1:
            e = e + src.u1(k)
        vide = Nk <= 0
        p = np.where(vide, p, (nk + e * p) / np.where(vide, 1.0, Nk + e))
        if k in snaps:
            out[k] = p
    return out


def distribution(p, gamma):
    """p^γ renormalisée (T,A)."""
    q = np.power(p, gamma)
    return q / q.sum(axis=1, keepdims=True)


def log2_vrai(p, y, gamma):
    """log2 de la probabilité (normalisée, plancher du juge) du vrai symbole, p^γ renormalisée."""
    q = np.power(p, gamma)
    s = q.sum(axis=1)
    pv = q[np.arange(len(y)), y] / s
    return np.log2(np.maximum(pv, PLANCHER_P))


# ---------------------------------------------------------------------------
# Grille statique
# ---------------------------------------------------------------------------
def experts_grille(jeu, avec_u1=(False, True)):
    y = jeu["y"]
    K = jeu["n"].shape[0] - 1
    src = Statique(jeu)
    metas, L = [], []
    for u in avec_u1:
        for c in C_GRILLE:
            snaps = chaine(src, [float(c)] * (K + 1), u, set(ORDRES_GRILLE))
            for kmax in ORDRES_GRILLE:
                for g in GAMMAS_GRILLE:
                    L.append(log2_vrai(snaps[kmax], y, g).astype(np.float32))
                    metas.append(dict(nom="grille c=%g ordre=%d g=%g%s" % (c, kmax, g, " +u1" if u else ""),
                                      famille="grille", c=c, ordre=kmax, gamma=g, u1=bool(u), beta=None))
            del snaps
    return metas, np.stack(L)


# ---------------------------------------------------------------------------
# Experts à mise à jour bayésienne
# ---------------------------------------------------------------------------
def experts_maj(jeu, betas=BETAS, cs=C_MAJ, ordres=ORDRES_MAJ, gammas=GAMMAS_MAJ):
    y = jeu["y"]
    K = jeu["n"].shape[0] - 1
    fd = FluxDense(jeu, y)
    metas, L = [], []
    for beta in betas:
        src = source_maj(jeu, fd, beta)
        for c in cs:
            snaps = chaine(src, [float(c)] * (K + 1), False, set(ordres))
            for kmax in ordres:
                for g in gammas:
                    L.append(log2_vrai(snaps[kmax], y, g).astype(np.float32))
                    metas.append(dict(nom="maj beta=%g c=%g ordre=%d g=%g" % (beta, c, kmax, g),
                                      famille="maj", c=c, ordre=kmax, gamma=g, u1=False, beta=beta))
            del snaps
    return metas, np.stack(L)


# ---------------------------------------------------------------------------
# Références : départ, champion, main (via le juge), Pitman-Yor (via bayes.py d'évolution)
# ---------------------------------------------------------------------------
def _charger_bayes_evolution():
    chemin = os.path.join(db.ECHO, "evolution", "bayes.py")
    spec = importlib.util.spec_from_file_location("bayes_evolution", chemin)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BE = None


def bayes_evolution():
    global _BE
    if _BE is None:
        _BE = _charger_bayes_evolution()
    return _BE


def log2_genome(source, jeu):
    r = juge.juger(source, jeu, pertes=True)
    assert r["ok"], r
    return (-r["pertes"]).astype(np.float32)


def source_champion():
    return bayes_evolution().source_champion()


def log2_py(jeu, delta, theta):
    be = bayes_evolution()
    p = be.py_distribution(jeu["n"].astype(np.float64), jeu["N"].astype(np.float64),
                           jeu["d"].astype(np.float64), np.asarray(delta), np.asarray(theta))
    p = p / p.sum(axis=1, keepdims=True)
    pv = p[np.arange(len(jeu["y"])), jeu["y"]]
    return np.log2(np.maximum(pv, PLANCHER_P)).astype(np.float32)


def distribution_py(jeu, delta, theta):
    be = bayes_evolution()
    p = be.py_distribution(jeu["n"].astype(np.float64), jeu["N"].astype(np.float64),
                           jeu["d"].astype(np.float64), np.asarray(delta), np.asarray(theta))
    return p / p.sum(axis=1, keepdims=True)


def experts_references(jeu, py_params):
    metas, L = [], []
    for nom, src in (("départ (PPM-C)", genome_initial.SOURCE), ("champion évolué", source_champion())):
        L.append(log2_genome(src, jeu))
        metas.append(dict(nom=nom, famille="ref", c=None, ordre=10, gamma=None, u1=None, beta=None))
    # référence à la main : e = 16·d, ordre 10, γ = 1 (même chose que la grille ; test d'équivalence dans test_bayes)
    st = Statique(jeu)
    p = chaine(st, [16.0] * (jeu["n"].shape[0]), False, {10})[10]
    L.append(log2_vrai(p, jeu["y"], 1.0).astype(np.float32))
    metas.append(dict(nom="référence à la main (e=16·d)", famille="ref", c=16, ordre=10, gamma=1.0, u1=False,
                      beta=None))
    L.append(log2_py(jeu, py_params["delta"], py_params["theta"]))
    metas.append(dict(nom="Pitman-Yor (A)", famille="ref", c=None, ordre=10, gamma=None, u1=None, beta=None))
    return metas, np.stack(L)


# ---------------------------------------------------------------------------
# Expert paramétré par c_k (pour Optuna) et bpc direct
# ---------------------------------------------------------------------------
class Prepare:
    """Tableaux float64 d'un jeu, convertis une fois pour les évaluations répétées d'Optuna."""

    def __init__(self, jeu):
        self.n = [jeu["n"][k].astype(np.float64) for k in range(jeu["n"].shape[0])]
        self.N = [jeu["N"][k].astype(np.float64) for k in range(jeu["n"].shape[0])]
        self.d = [jeu["d"][k].astype(np.float64) for k in range(jeu["n"].shape[0])]
        self.u1 = [jeu["u1"][k].astype(np.float64) for k in range(jeu["n"].shape[0])]
        self.y = jeu["y"]
        self.K = jeu["n"].shape[0] - 1


class _SrcPrep:
    def __init__(self, pr):
        self.pr = pr

    def n(self, k):
        return self.pr.n[k]

    def N(self, k):
        return self.pr.N[k]

    def d(self, k):
        return self.pr.d[k]

    def u1(self, k):
        return self.pr.u1[k]


def params_vers_log2(pr, params, jeu_y=None):
    """log2 p(vrai symbole) de l'expert décrit par params = dict(cs[0..10], gamma, u1, kmax)."""
    k = int(params["kmax"])
    cs = list(params["cs"]) + [1.0] * (pr.K + 1 - len(params["cs"]))
    p = chaine(_SrcPrep(pr), cs, bool(params["u1"]), {k})[k]
    return log2_vrai(p, pr.y, params["gamma"])


def bpc_params(pr, params):
    return float(-params_vers_log2(pr, params).mean())


def log2_params_jeu(jeu, params):
    return params_vers_log2(Prepare(jeu), params).astype(np.float32)
