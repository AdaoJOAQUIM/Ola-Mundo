"""Bayes, la dernière pierre : outils bayésiens pour le moteur d'évolution d'Écho.

Méthodes classiques réutilisées, rien d'inventé ici :
  1. Estimateur hiérarchique de Pitman-Yor sur la chaîne des ordres (Teh 2006 ; Sequence Memoizer,
     Wood et al. 2009 ; Kneser-Ney interpolé en est l'approximation). Hyperparamètres par ordre
     choisis par maximum de vraisemblance (Bayes empirique, « evidence » de type II) sur A seulement.
  2. Moyenne bayésienne de modèles (BMA), en ligne, et variante « fixed-share » (Herbster et Warmuth 1998 ;
     Volf et Willems 1998), avec le α choisi sur A.
  3. Outils de décision : échantillonnage de Thompson (Thompson 1933 ; Chapelle et Li 2011) avec postérieurs
     Beta pour choisir les opérateurs de mutation ; bootstrap bayésien (Rubin 1981) par blocs de 500
     caractères pour la probabilité postérieure qu'un modèle batte un autre.

Contraintes : stdlib + numpy seulement. Ce module ne lit JAMAIS cache/c.npz (réservé à tester.py).
La tranche fraîche D (texte de validation.txt au-delà de A+B+C) est construite par le même
pipeline de comptage que A, B, C, avec des comptages issus de la bibliothèque seule.
"""
import json
import lzma
import math
import os
import random
import sys
import time
from multiprocessing.pool import ThreadPool

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

import donnees  # noqa: E402
import genome_initial  # noqa: E402
import juge  # noqa: E402

CHEMIN_D = os.path.join(donnees.CACHE, "d.npz")
CHEMIN_CDC = os.path.join(donnees.CACHE, "bayes_comptes_de_comptes.json")
TEMOIN_D = os.path.join(donnees.CACHE, "d_utilise.json")
PLANCHER_P = juge.PLANCHER_P   # même plancher que le juge : 1e-10
LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# 0. Tranche fraîche D
# ---------------------------------------------------------------------------
def texte_frais():
    """Texte de validation.txt situé APRÈS A+B+C (vide s'il n'y en a pas)."""
    lib, val = donnees.lire_textes()
    ta, tb, tc = donnees.decouper(val)
    debut = len(ta) + len(tb) + len(tc)
    return val[debut:], debut, len(val)


def _discounts_ney(comptes):
    """Remises de Ney et al. / Chen-Goodman à partir des comptes de comptes n1..n4."""
    n1, n2, n3 = (float((comptes == i).sum()) for i in (1, 2, 3))
    n4 = float((comptes >= 4).sum())
    if n1 == 0 or n2 == 0:
        return None
    Y = n1 / (n1 + 2 * n2)
    out = {"n1": n1, "n2": n2, "n3": n3, "n4plus": n4, "D_ney": Y}
    out["D1"] = 1 - 2 * Y * n2 / n1
    out["D2"] = 2 - 3 * Y * n3 / n2 if n3 > 0 else None
    return out


def construire_d(verbeux=True, K=donnees.K_MAX):
    """Construit D avec exactement le pipeline de donnees.precalculer (comptages de la bibliothèque seule).

    Écrit cache/d.npz et, au passage, les comptes de comptes de la bibliothèque (remises de Ney)
    dans cache/bayes_comptes_de_comptes.json. N'écrase pas un d.npz existant.
    """
    if os.path.exists(CHEMIN_D) and os.path.exists(CHEMIN_CDC):
        return
    t0 = time.time()
    lib, val = donnees.lire_textes()
    alphabet = donnees.alphabet_de(lib, val)
    A = len(alphabet)
    td, debut, _ = texte_frais()
    assert len(td) > 1000, "pas de texte frais"
    codes_lib = donnees.encoder(lib, alphabet)
    emp_lib = donnees.Empreintes(codes_lib)
    sel = donnees.sels(A)
    codes = donnees.encoder(td, alphabet)
    T = len(codes)
    emp = donnees.Empreintes(codes)
    n_out = np.zeros((K + 1, T, A), dtype=np.float32)
    cdc = {}
    for k in range(K + 1):
        uniques, comptes = donnees.table_ordre(emp_lib, codes_lib, k, sel)
        cdc[str(k)] = _discounts_ney(comptes)
        t = np.arange(T, dtype=np.int64)
        ok = t >= k
        tt = t[ok]
        h = emp.fenetre(tt - k, tt)
        with np.errstate(over="ignore"):
            q = (h[:, None] + sel[None, :]).ravel()
        pos = np.minimum(np.searchsorted(uniques, q), len(uniques) - 1)
        trouve = uniques[pos] == q
        v = np.where(trouve, comptes[pos], 0.0).astype(np.float32).reshape(-1, A)
        n_out[k][ok] = v
        if verbeux:
            print("D : ordre %d fait (%.0f s)" % (k, time.time() - t0), flush=True)
    N = n_out.sum(axis=2, keepdims=True)
    d = (n_out > 0).sum(axis=2, keepdims=True).astype(np.float32)
    u1 = (n_out == 1).sum(axis=2, keepdims=True).astype(np.float32)
    u2 = (n_out == 2).sum(axis=2, keepdims=True).astype(np.float32)
    os.makedirs(donnees.CACHE, exist_ok=True)
    np.savez(CHEMIN_D, alphabet=np.array(list(alphabet)), nD=n_out, ND=N, dD=d, u1D=u1, u2D=u2, yD=codes,
             debut=np.array(debut))
    with open(CHEMIN_CDC, "w") as f:
        json.dump(cdc, f, indent=1)
    if verbeux:
        print("D construit : %d caractères (à partir de la position %d) en %.0f s" % (T, debut, time.time() - t0))


def charger_d():
    z = np.load(CHEMIN_D)
    return {c: z[c + "D"] for c in ("n", "N", "d", "u1", "u2", "y")}


# ---------------------------------------------------------------------------
# 1. Pitman-Yor hiérarchique sur la chaîne des ordres
# ---------------------------------------------------------------------------
def py_distribution(n, N, d, delta, theta):
    """Distribution complète (T, A). n: (K+1,T,A) ; N, d: (K+1,T,1).

    p_k = (max(n_k - δ_k, 0) + (θ_k + δ_k d_k) p_{k-1}) / (N_k + θ_k) ; base uniforme ; p_k = p_{k-1} si N_k = 0.
    """
    K1, T, A = n.shape
    p = np.full((T, A), 1.0 / A)
    for k in range(K1):
        num = np.maximum(n[k] - delta[k], 0.0) + (theta[k] + delta[k] * d[k]) * p
        p = np.where(N[k] > 0, num / (N[k] + theta[k]), p)
    return p


class Prep:
    """Extrait, une fois pour toutes, ce dont la chaîne a besoin pour le VRAI symbole seulement.

    p_k(y) ne dépend que de n_k(y), N_k, d_k et p_{k-1}(y) : on évite donc les tableaux (T, A).
    """

    def __init__(self, jeu):
        n, N, d, y = jeu["n"], jeu["N"], jeu["d"], jeu["y"]
        T = len(y)
        idx = np.arange(T)
        self.T = T
        self.K1 = n.shape[0]
        self.A = n.shape[2]
        self.ny = np.stack([n[k][idx, y].astype(np.float64) for k in range(self.K1)])
        self.N = N[:, :, 0].astype(np.float64)
        self.d = d[:, :, 0].astype(np.float64)


def py_chaine(pr, delta, u, depuis=0, p0=None):
    """Probabilité du vrai symbole après la chaîne ; u_k = θ_k + δ_k > 0."""
    p = np.full(pr.T, 1.0 / pr.A) if p0 is None else p0
    for k in range(depuis, pr.K1):
        th = u[k] - delta[k]
        num = np.maximum(pr.ny[k] - delta[k], 0.0) + (th + delta[k] * pr.d[k]) * p
        p = np.where(pr.N[k] > 0, num / (pr.N[k] + th), p)
    return p


def py_proba_vraie(jeu, delta, theta):
    """Probabilité (T,) du vrai symbole, avec le plancher du juge."""
    pr = Prep(jeu)
    delta = np.asarray(delta, float)
    u = np.asarray(theta, float) + delta
    return np.maximum(py_chaine(pr, delta, u), PLANCHER_P)


def bpc_de_p(p):
    return float(-np.log2(np.maximum(p, PLANCHER_P)).mean())


def _min_1d(f, lo, hi, n_grille=7, iters=16):
    """Minimum d'une fonction 1-D : grille grossière puis section dorée dans le meilleur intervalle."""
    xs = np.linspace(lo, hi, n_grille)
    fs = [f(x) for x in xs]
    i = int(np.argmin(fs))
    a, b = xs[max(i - 1, 0)], xs[min(i + 1, n_grille - 1)]
    g = (math.sqrt(5) - 1) / 2
    c, e = b - g * (b - a), a + g * (b - a)
    fc, fe = f(c), f(e)
    for _ in range(iters):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, e, fe
            e = a + g * (b - a)
            fe = f(e)
    x = (a + b) / 2
    fx = f(x)
    return (x, fx) if fx <= min(fs) else (xs[i], min(fs))


BORNE_DELTA = (0.0, 0.999)
BORNE_LOGU = (math.log(1e-3), math.log(1e4))


def ajuster_py(jeu_A, balayages=6, verbeux=False, departs=((0.75, 1.0), (0.9, 1.0), (0.5, 10.0), (0.3, 100.0))):
    """Bayes empirique : maximise la vraisemblance de A sur (δ_k, log u_k), recherche par coordonnées.

    Retourne dict(delta, theta, u, bpc_A, historique, erreurs_std). Un balayage : pour k = 0..K,
    minimisation 1-D de δ_k puis de log u_k, les autres hyperparamètres fixes.
    """
    meilleur = None
    for d0, u0 in departs:
        r = _ajuster_depuis(jeu_A, balayages, verbeux, d0, u0)
        r.setdefault("departs_bpc_A", [])
        if meilleur is None or r["bpc_A"] < meilleur["bpc_A"]:
            r["departs_bpc_A"] = (meilleur or {}).get("departs_bpc_A", []) + [r["bpc_A"]]
            meilleur = r
        else:
            meilleur["departs_bpc_A"].append(r["bpc_A"])
    return meilleur


def _ajuster_depuis(jeu_A, balayages, verbeux, d0, u0):
    pr = Prep(jeu_A)
    K1 = pr.K1
    delta = np.full(K1, d0)
    logu = np.full(K1, math.log(u0))
    hist = []

    def perte(dl, lu, depuis, p0):
        return bpc_de_p(py_chaine(pr, dl, np.exp(lu), depuis, p0))

    for b in range(balayages):
        for k in range(K1):
            u = np.exp(logu)
            # p_{k-1} : chaîne tronquée aux ordres < k
            p_prec = np.full(pr.T, 1.0 / pr.A)
            if k > 0:
                tmp = Prep.__new__(Prep)
                tmp.__dict__.update(pr.__dict__)
                tmp.K1 = k
                p_prec = py_chaine(tmp, delta, u, 0, None)

            def f_delta(x):
                dl = delta.copy()
                dl[k] = x
                return perte(dl, logu, k, p_prec)
            delta[k], _ = _min_1d(f_delta, *BORNE_DELTA)

            def f_u(x):
                lu = logu.copy()
                lu[k] = x
                return perte(delta, lu, k, p_prec)
            logu[k], _ = _min_1d(f_u, *BORNE_LOGU)
        bpc = perte(delta, logu, 0, None)
        hist.append(bpc)
        if verbeux:
            print("  balayage %d : bpc A = %.5f" % (b + 1, bpc), flush=True)
    u = np.exp(logu)
    theta = u - delta
    # erreurs standard approchées (hessienne diagonale de la perte totale en nats, différences finies)
    T = pr.T
    f0 = perte(delta, logu, 0, None) * T * LOG2
    se_delta, se_logu = np.zeros(K1), np.zeros(K1)
    for k in range(K1):
        h = 0.01
        dl_p, dl_m = delta.copy(), delta.copy()
        dl_p[k] = min(delta[k] + h, 0.9995)
        dl_m[k] = max(delta[k] - h, 0.0)
        hh = dl_p[k] - dl_m[k]
        fp, fm = perte(dl_p, logu, 0, None) * T * LOG2, perte(dl_m, logu, 0, None) * T * LOG2
        f_mid = f0
        curv = (fp - 2 * f_mid + fm) / ((hh / 2) ** 2)
        se_delta[k] = 1 / math.sqrt(curv) if curv > 1e-12 else float("nan")
        lp, lm = logu.copy(), logu.copy()
        lp[k] += 0.05
        lm[k] -= 0.05
        fp, fm = perte(delta, lp, 0, None) * T * LOG2, perte(delta, lm, 0, None) * T * LOG2
        curv = (fp - 2 * f0 + fm) / (0.05 ** 2)
        se_logu[k] = 1 / math.sqrt(curv) if curv > 1e-12 else float("nan")
    return {"delta": delta.tolist(), "theta": theta.tolist(), "u": u.tolist(), "bpc_A": hist[-1],
            "historique": hist, "se_delta": se_delta.tolist(), "se_logu": se_logu.tolist()}


# ---------------------------------------------------------------------------
# 2. Moyenne bayésienne de modèles (BMA) et fixed-share
# ---------------------------------------------------------------------------
def bma_en_ligne(P, prior=None, alpha=0.0):
    """Mélange en ligne. P : (M, T) probabilités du vrai symbole par modèle.

    Poids a posteriori w_i(t) ∝ prior_i × Π_{s<t} p_i(y_s). Avec alpha > 0 : fixed-share
    (Herbster-Warmuth ; Volf-Willems) : après chaque symbole, w ← (1-α) w + α/M.
    Retourne dict(pertes (bits par caractère), poids_moyens, poids_finaux).
    """
    M, T = P.shape
    w = np.full(M, 1.0 / M) if prior is None else np.asarray(prior, float) / np.sum(prior)
    pertes = np.empty(T)
    somme_w = np.zeros(M)
    for t in range(T):
        pt = P[:, t]
        mel = float(w @ pt)
        pertes[t] = -math.log2(max(mel, PLANCHER_P))
        somme_w += w
        w = w * pt / mel
        if alpha > 0.0:
            w = (1.0 - alpha) * w + alpha / M
    return {"pertes": pertes, "poids_moyens": (somme_w / T).tolist(), "poids_finaux": w.tolist()}


def borne_regret_bma(prior):
    """Regret maximal (bits) de BMA sur le meilleur membre i : log2(1/prior_i)."""
    return [math.log2(1.0 / p) for p in prior]


def borne_fixed_share(M, alpha, T, m):
    """Borne (bits) de fixed-share vs la meilleure suite à m changements (forme Herbster-Warmuth / Volf-Willems) :
    log2 M + m log2(M/α) + (T-1-m) log2(1/(1-α))."""
    return math.log2(M) + m * math.log2(M / alpha) + (T - 1 - m) * math.log2(1.0 / (1.0 - alpha))


GRILLE_ALPHA = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def choisir_alpha(P_A, prior=None, grille=GRILLE_ALPHA):
    """α de fixed-share qui minimise le bpc sur A (A seulement)."""
    res = {a: float(bma_en_ligne(P_A, prior, a)["pertes"].mean()) for a in grille}
    meilleur = min(res, key=res.get)
    return meilleur, res


# ---------------------------------------------------------------------------
# 3b. Probabilité postérieure qu'un modèle batte un autre
# ---------------------------------------------------------------------------
def _diff_blocs(pertes_m, pertes_ref, taille_bloc):
    nb = len(pertes_m) // taille_bloc
    dm = np.asarray(pertes_m)[:nb * taille_bloc] - np.asarray(pertes_ref)[:nb * taille_bloc]
    return dm.reshape(nb, taille_bloc).mean(axis=1)


def p_meilleur(pertes_m, pertes_ref, taille_bloc=500, n=4000, graine=0):
    """P(modèle m meilleur que ref) = P(bpc_m < bpc_ref).

    Principal : bootstrap bayésien (Rubin 1981), poids Dirichlet(1,...,1) sur les blocs consécutifs de
    500 caractères (la dépendance locale est conservée dans le bloc). Contrôle : bootstrap par blocs
    (rééchantillonnage avec remise).
    """
    db = _diff_blocs(pertes_m, pertes_ref, taille_bloc)
    nb = len(db)
    rng = np.random.default_rng(graine)
    W = rng.dirichlet(np.ones(nb), size=n)
    delta_bayes = W @ db
    idx = rng.integers(0, nb, size=(n, nb))
    delta_boot = db[idx].mean(axis=1)
    return {"p_meilleur_bayes": float((delta_bayes < 0).mean()),
            "p_meilleur_blocs": float((delta_boot < 0).mean()),
            "diff_bpc_moyenne": float(db.mean()),
            "ic95_bayes": [float(np.percentile(delta_bayes, 2.5)), float(np.percentile(delta_bayes, 97.5))],
            "nb_blocs": nb}


# ---------------------------------------------------------------------------
# 3a. Thompson sampling pour choisir les opérateurs de mutation
# ---------------------------------------------------------------------------
class ThompsonOperateurs:
    """Un bandit de Bernoulli par opérateur : succès = l'enfant bat son parent sur A.

    Postérieur Beta(a, b) par opérateur (a priori Beta(1, 1)). À chaque choix : on tire θ_op ~ Beta et on
    prend l'argmax (Thompson 1933). Option `oubli` < 1 : décroissance exponentielle des comptes
    (variante non stationnaire, Garivier et Moulines 2011) car le taux de succès baisse au fil de la recherche.
    """

    def __init__(self, operateurs, oubli=1.0, a0=1.0, b0=1.0):
        self.ops = list(operateurs)
        self.a = {o: a0 for o in self.ops}
        self.b = {o: b0 for o in self.ops}
        self.a0, self.b0, self.oubli = a0, b0, oubli
        self.nb_choix = {o: 0 for o in self.ops}

    def choisir(self, rng, permis=None):
        ops = permis or self.ops
        tirages = {o: rng.betavariate(self.a[o], self.b[o]) for o in ops}
        op = max(tirages, key=tirages.get)
        self.nb_choix[op] += 1
        return op

    def observer(self, ops_utilises, succes):
        for o in set(ops_utilises):
            if self.oubli < 1.0:
                self.a[o] = self.a0 + (self.a[o] - self.a0) * self.oubli
                self.b[o] = self.b0 + (self.b[o] - self.b0) * self.oubli
            if succes:
                self.a[o] += 1.0
            else:
                self.b[o] += 1.0

    def moyennes(self):
        return {o: self.a[o] / (self.a[o] + self.b[o]) for o in self.ops}


def muter_thompson(source, ts, rng, archive_sources=None, K=10, essais=6, deja_vus=None, nmut=1.5):
    """Comme mutations.muter, mais l'opérateur de chaque mutation est choisi par Thompson (pas de méta-génome).

    Retourne (source, [noms], ok, raison). Le pas des constantes (sigma) reste fixe à sa valeur par défaut.
    """
    import ast
    import mutations
    from garde_fous import verifier
    meta = mutations.meta_defaut()
    meta["nmut"] = nmut
    dernier = (source, [], False, "aucune_mutation")
    for _ in range(essais):
        arbre = ast.parse(source)
        f = arbre.body[0]
        nb = int(nmut)
        if rng.random() < nmut - nb:
            nb += 1
        nb = max(1, min(3, nb))
        ctx = {"rng": rng, "meta": meta, "archive_sources": archive_sources, "K": K}
        noms = []
        for _m in range(nb):
            essayes = []
            for _t in range(6):
                restants = [o for o in ts.ops if o not in essayes]
                if not restants:
                    break
                op = ts.choisir(rng, restants)
                try:
                    r = mutations.FONCTIONS[op](f, ctx)
                except (IndexError, ValueError, KeyError):
                    r = None
                if r:
                    noms.append(r)
                    break
                essayes.append(op)
        if not noms:
            continue
        try:
            ast.fix_missing_locations(arbre)
            nouvelle = ast.unparse(arbre) + "\n"
        except Exception:
            continue
        ok, code, _ = verifier(nouvelle)
        dernier = (nouvelle, noms, ok, code)
        if not ok:
            continue
        if nouvelle == source or (deja_vus is not None and nouvelle in deja_vus):
            dernier = (nouvelle, noms, False, "doublon")
            continue
        return nouvelle, noms, True, ""
    return dernier


def executer_thompson(nom, budget, graine, nproc=4, taille_lot=8, delai=20, oubli=1.0, verbeux=False):
    """Même boucle que moteur.executer (archive ouverte, parents tirés pareil, fitness A, champion sur B),
    mais SANS méta-génome : les opérateurs viennent du Thompson sampling. Même budget en évaluations d'enfants.
    """
    import composants
    import garde_fous
    import moteur
    import mutations
    moteur.controle_empreintes()
    rng = random.Random(graine)
    jeux = donnees.charger_ab()
    K = jeux["A"]["n"].shape[0] - 1
    ev = composants.EvaluateurMesure({"A": jeux["A"], "B": jeux["B"]}, nproc=nproc, delai=delai)
    ts = ThompsonOperateurs(mutations.OPERATEURS, oubli=oubli)
    t0, cpu0 = time.time(), time.process_time()
    cpu_trav = [0.0]
    src0 = genome_initial.SOURCE
    ra, rb = ev.evaluer_lot([src0], "A")[0], ev.evaluer_lot([src0], "B")[0]
    cpu_trav[0] += ra["cpu_s"] + rb["cpu_s"]
    g0 = {"id": 0, "parent": None, "mutations": [], "source": src0, "fit_A": ra["bpc"], "fit_B": rb["bpc"],
          "fit_eff": ra["bpc"], "enfants": 0, "meta": None}
    archive = [g0]
    vus = {src0}
    depenses, gen, points = 0, 0, []
    ecartes = 0

    def top_sur_B():
        ordre = sorted(archive, key=lambda g: g["fit_eff"])[:10]
        a_faire = [g for g in ordre if g["fit_B"] is None]
        if a_faire:
            res = ev.evaluer_lot([g["source"] for g in a_faire], "B")
            for g, r in zip(a_faire, res):
                cpu_trav[0] += r.get("cpu_s", 0.0)
                g["fit_B"] = r["bpc"] if r["ok"] else float("inf")

    def champion():
        return min((g for g in archive if g["fit_B"] is not None), key=lambda g: g["fit_B"])

    def point():
        top_sur_B()
        ch = champion()
        points.append({"evals": depenses, "best_A": min(g["fit_A"] for g in archive), "best_B": ch["fit_B"],
                       "cpu_s": round(cpu_trav[0] + time.process_time() - cpu0, 1)})

    prochain = 250
    while depenses < budget:
        gen += 1
        if gen % 25 == 0:
            moteur.controle_empreintes()
        w = moteur.poids_parents(archive)
        lot = min(taille_lot, budget - depenses)
        nrg = np.random.default_rng(rng.getrandbits(32))
        sources_arch = [g["source"] for g in archive]
        enfants, essais = [], 0
        while len(enfants) < lot and essais < lot * 8:
            essais += 1
            par = archive[int(nrg.choice(len(archive), p=w))]
            src, noms, ok, raison = muter_thompson(par["source"], ts, rng, sources_arch, K=K, deja_vus=vus)
            if raison == "doublon":
                continue
            vus.add(src)
            enfants.append((par, src, noms))
        if enfants:
            res = ev.evaluer_lot([e[1] for e in enfants], "A")
            for (par, src, noms), rr in zip(enfants, res):
                depenses += 1
                cpu_trav[0] += rr.get("cpu_s", 0.0)
                gagne = rr["ok"] and rr["bpc"] < par["fit_A"] - 1e-4
                ts.observer(noms, gagne)
                if not rr["ok"]:
                    continue
                if rr["bpc"] > archive[0]["fit_A"] + 0.3:
                    ecartes += 1
                    continue
                par["enfants"] += 1
                archive.append({"id": len(archive), "parent": par["id"], "mutations": noms, "source": src,
                                "fit_A": rr["bpc"], "fit_B": None, "fit_eff": rr["bpc"], "enfants": 0,
                                "meta": None})
        else:
            depenses += lot // 2 + 1
        if gen % 10 == 0:
            top_sur_B()
        if depenses >= prochain:
            prochain = (depenses // 250 + 1) * 250
            point()
    point()
    ch = champion()
    ev.fermer()
    moteur.controle_empreintes()
    return {"nom": nom, "graine": graine, "evals": depenses, "taille_archive": len(archive),
            "points": points, "seed_A": g0["fit_A"], "seed_B": g0["fit_B"],
            "champion": {"id": ch["id"], "fit_A": ch["fit_A"], "fit_B": ch["fit_B"], "source": ch["source"]},
            "best_A": min(g["fit_A"] for g in archive),
            "cpu_total_s": round(cpu_trav[0] + time.process_time() - cpu0, 1),
            "temps_mur_s": round(time.time() - t0, 1),
            "thompson": {"moyennes_posterieures": ts.moyennes(), "nb_choix": ts.nb_choix,
                         "a": ts.a, "b": ts.b}}


# ---------------------------------------------------------------------------
# Utilitaires d'évaluation des modèles existants
# ---------------------------------------------------------------------------
def source_champion():
    txt = open(os.path.join(ICI, "champion.py"), encoding="utf-8").read()
    return txt[txt.index("def estimer"):]


def source_reference_main():
    return json.load(open(os.path.join(ICI, "analyse_champion.json")))["reference_a_la_main"]["source"]


def pertes_genome(source, jeu):
    r = juge.juger(source, jeu, pertes=True)
    assert r["ok"], r
    return r["pertes"]


def lzma_conditionnel(lib, texte):
    """bits/caractère de taille(lib + texte) - taille(lib), comme tester.py."""
    preset = 9 | lzma.PRESET_EXTREME
    with ThreadPool(2) as tp:
        a, b = tp.map(lambda x: len(lzma.compress(x.encode("utf-8"), preset=preset)), [lib, lib + texte])
    return {"taille_lib": a, "taille_lib_plus_texte": b, "bpc": 8.0 * (b - a) / len(texte)}


def p_moyenne_inferieure(x, y, n=20000, graine=0):
    """P(moyenne de x < moyenne de y) pour deux petits échantillons (graines), a priori plat :
    chaque moyenne a un postérieur de Student, μ = m + s/√n · t_{n-1} (modèle normal, variance inconnue)."""
    rng = np.random.default_rng(graine)
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx = x.mean() + x.std(ddof=1) / math.sqrt(len(x)) * rng.standard_t(len(x) - 1, n)
    my = y.mean() + y.std(ddof=1) / math.sqrt(len(y)) * rng.standard_t(len(y) - 1, n)
    d = mx - my
    return {"p_x_meilleur": float((d < 0).mean()), "diff_moyenne": float(x.mean() - y.mean()),
            "ic95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
