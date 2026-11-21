"""Tests de bayes.py. Lancer : python3 test_bayes.py (ou pytest)."""
import math
import os
import random
import sys

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

import bayes  # noqa: E402
import donnees  # noqa: E402
import garde_fous  # noqa: E402
import genome_initial  # noqa: E402


def _jouet(K1=4, T=60, A=7, graine=0):
    """Comptages aléatoires (non emboîtés) : suffisant pour les identités algébriques."""
    rng = np.random.default_rng(graine)
    n = rng.poisson(0.7, size=(K1, T, A)).astype(np.float32)
    n[0] += 3                               # l'ordre 0 est toujours peuplé
    n[2][:5] = 0                            # quelques contextes jamais vus (N = 0)
    N = n.sum(axis=2, keepdims=True)
    d = (n > 0).sum(axis=2, keepdims=True).astype(np.float32)
    y = rng.integers(0, A, size=T)
    return {"n": n, "N": N, "d": d, "u1": (n == 1).sum(2, keepdims=True).astype(np.float32),
            "u2": (n == 2).sum(2, keepdims=True).astype(np.float32), "y": y}


def test_py_normalise_et_coherent_avec_le_vrai_symbole():
    J = _jouet()
    delta = np.array([0.0, 0.5, 0.8, 0.9])
    theta = np.array([0.3, 1.0, 5.0, -0.2])        # θ > -δ : -0.2 > -0.9
    P = bayes.py_distribution(J["n"], J["N"], J["d"], delta, theta)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P >= 0).all()
    pv = bayes.py_proba_vraie(J, delta, theta)
    assert np.allclose(pv, np.maximum(P[np.arange(len(J["y"])), J["y"]], bayes.PLANCHER_P))


def test_py_cas_limites():
    # N = 0 partout : uniforme
    n = np.zeros((3, 4, 5), dtype=np.float32)
    N = n.sum(2, keepdims=True)
    P = bayes.py_distribution(n, N, N.copy(), np.array([.5, .5, .5]), np.array([1., 1., 1.]))
    assert np.allclose(P, 0.2)
    # un seul ordre, calcul à la main : n = (3, 1, 0), d = 2, N = 4, δ = .5, θ = 2
    n = np.array([[[3, 1, 0]]], dtype=np.float32)
    N = n.sum(2, keepdims=True)
    d = np.array([[[2]]], dtype=np.float32)
    P = bayes.py_distribution(n, N, d, np.array([.5]), np.array([2.0]))[0]
    base = 1 / 3
    att = [(2.5 + (2 + 1.0) * base) / 6, (0.5 + 3.0 * base) / 6, (0 + 3.0 * base) / 6]
    assert np.allclose(P, att)
    # δ = 0 : on retrouve Dirichlet / m-estimate : (n + θ p) / (N + θ)
    P = bayes.py_distribution(n, N, d, np.array([0.0]), np.array([2.0]))[0]
    assert np.allclose(P, (n[0, 0] + 2.0 / 3) / 6)


def test_ajustement_ameliore_et_respecte_les_bornes():
    J = _jouet(K1=3, T=400, A=6, graine=3)
    # tire y selon un modèle réel pour que l'ajustement ait un signal
    rng = np.random.default_rng(1)
    pr = J["n"][1] + 0.01
    J["y"] = np.array([rng.choice(6, p=r / r.sum()) for r in pr])
    pr0 = bayes.Prep(J)
    ref = bayes.bpc_de_p(bayes.py_chaine(pr0, np.full(3, .75), np.full(3, 1.0)))
    f = bayes.ajuster_py(J, balayages=3, departs=((0.75, 1.0),))
    assert f["bpc_A"] <= ref + 1e-9
    assert all(0.0 <= x < 1.0 for x in f["delta"])
    assert all(t > -d for t, d in zip(f["theta"], f["delta"]))


def test_bma_borne_de_regret_exacte():
    rng = np.random.default_rng(0)
    M, T = 4, 3000
    P = rng.uniform(0.05, 0.9, size=(M, T))
    P[2] = np.clip(P[2] * 1.3, 0, 0.95)
    for prior in (np.full(M, 0.25), np.array([0.7, 0.1, 0.1, 0.1])):
        r = bayes.bma_en_ligne(P, prior)
        perte = r["pertes"].sum()
        pertes_i = -np.log2(P).sum(axis=1)
        bornes = bayes.borne_regret_bma(prior)
        for i in range(M):
            assert perte <= pertes_i[i] + bornes[i] + 1e-6
        # identité : la perte du mélange est exactement -log2 Σ prior_i Π p_i
        exact = -np.log2((prior * 2.0 ** (-pertes_i)).sum()) if pertes_i.max() < 1000 else None
        if exact is not None:
            assert abs(perte - exact) < 1e-6
        assert abs(sum(r["poids_finaux"]) - 1) < 1e-9


def test_fixed_share_alpha0_egal_bma_et_borne_un_changement():
    rng = np.random.default_rng(2)
    T = 2000
    # deux experts : le premier est bon dans la 1re moitié, le second dans la 2e (un changement)
    P = np.empty((2, T))
    P[0, :T // 2], P[1, :T // 2] = 0.8, 0.3
    P[0, T // 2:], P[1, T // 2:] = 0.3, 0.8
    P = np.clip(P + rng.normal(0, 0.01, P.shape), 0.01, 0.99)
    assert np.allclose(bayes.bma_en_ligne(P)["pertes"], bayes.bma_en_ligne(P, alpha=0.0)["pertes"])
    alpha = 0.01
    bma = bayes.bma_en_ligne(P)["pertes"].sum()
    fs = bayes.bma_en_ligne(P, alpha=alpha)["pertes"].sum()
    meilleure_suite = (-np.log2(P[0, :T // 2])).sum() + (-np.log2(P[1, T // 2:])).sum()
    assert fs <= meilleure_suite + bayes.borne_fixed_share(2, alpha, T, 1) + 1e-6
    assert fs < bma - 50          # le partage bat nettement le mélange simple quand le meilleur change


def test_choisir_alpha_sur_donnees_a_changements():
    rng = np.random.default_rng(5)
    T = 4000
    P = np.empty((2, T))
    blocs = (np.arange(T) // 200) % 2
    P[0], P[1] = np.where(blocs == 0, .8, .2), np.where(blocs == 0, .2, .8)
    a, courbe = bayes.choisir_alpha(P)
    assert a >= 1e-3 and courbe[a] == min(courbe.values())


def test_bootstrap_bayesien():
    rng = np.random.default_rng(0)
    base = rng.normal(1.5, 1.0, 20000)
    meilleur = base - 0.1
    r = bayes.p_meilleur(meilleur, base)
    assert r["p_meilleur_bayes"] > 0.999 and r["p_meilleur_blocs"] > 0.999
    r2 = bayes.p_meilleur(base, meilleur)
    assert r2["p_meilleur_bayes"] < 0.001
    bruit = rng.normal(0, 1.0, 20000)
    r3 = bayes.p_meilleur(base + bruit * 0.0, base)
    assert abs(r3["diff_bpc_moyenne"]) < 1e-12
    bruit_a = rng.normal(1.5, 1.0, 20000)
    r4 = bayes.p_meilleur(bruit_a, base)
    assert 0.02 < r4["p_meilleur_bayes"] < 0.98
    # symétrie : P(a < b) + P(b < a) = 1 (sauf égalité exacte)
    assert abs(r4["p_meilleur_bayes"] + bayes.p_meilleur(base, bruit_a)["p_meilleur_bayes"] - 1) < 1e-9


def test_p_moyenne_inferieure():
    r = bayes.p_moyenne_inferieure([1.0, 1.01, 0.99], [1.5, 1.49, 1.51])
    assert r["p_x_meilleur"] > 0.99
    r = bayes.p_moyenne_inferieure([1.0, 1.2, 0.8], [1.0, 1.1, 0.9])
    assert 0.2 < r["p_x_meilleur"] < 0.8


def test_thompson_converge_vers_le_meilleur_bras():
    ops = ["a", "b", "c"]
    vrai = {"a": 0.05, "b": 0.30, "c": 0.10}
    ts = bayes.ThompsonOperateurs(ops)
    rng = random.Random(0)
    for _ in range(1500):
        o = ts.choisir(rng)
        ts.observer([o], rng.random() < vrai[o])
    assert ts.nb_choix["b"] > 0.7 * 1500
    m = ts.moyennes()
    assert abs(m["b"] - 0.30) < 0.06
    # variante avec oubli : les comptes restent bornés
    ts2 = bayes.ThompsonOperateurs(ops, oubli=0.98)
    for _ in range(500):
        o = ts2.choisir(rng)
        ts2.observer([o], rng.random() < vrai[o])
    assert max(ts2.a[o] + ts2.b[o] for o in ops) < 2 + 1 / (1 - 0.98) + 1


def test_muter_thompson_produit_des_genomes_valides():
    import mutations
    ts = bayes.ThompsonOperateurs(mutations.OPERATEURS)
    rng = random.Random(1)
    src = genome_initial.SOURCE
    nb_ok = 0
    for _ in range(60):
        s, noms, ok, raison = bayes.muter_thompson(src, ts, rng, [src], K=10)
        if ok:
            nb_ok += 1
            assert garde_fous.verifier(s)[0]
            assert all(n in mutations.OPERATEURS for n in noms)
    assert nb_ok >= 30
    assert sum(ts.nb_choix.values()) > 0


def test_modeles_de_reference_reproduisent_les_scores_connus():
    ab = donnees.charger_ab()
    J = {k: v[:, :2000] if v.ndim == 3 else v for k, v in ab["A"].items()}
    J["y"] = ab["A"]["y"][:2000]
    L = bayes.pertes_genome(bayes.source_champion(), J)
    assert L.shape == (2000,) and np.isfinite(L).all()
    # le génome de départ est PPM-C : pires pertes que la référence à la main
    Ls = bayes.pertes_genome(genome_initial.SOURCE, J)
    Lm = bayes.pertes_genome(bayes.source_reference_main(), J)
    assert Lm.mean() < Ls.mean()


def test_fixed_share_et_bma_sur_modeles_reels_ne_depassent_pas_la_borne():
    ab = donnees.charger_ab()
    J = {k: v[:, :3000] if v.ndim == 3 else v for k, v in ab["A"].items()}
    J["y"] = ab["A"]["y"][:3000]
    L = np.array([bayes.pertes_genome(genome_initial.SOURCE, J), bayes.pertes_genome(bayes.source_champion(), J),
                  bayes.pertes_genome(bayes.source_reference_main(), J)])
    P = 2.0 ** -L
    r = bayes.bma_en_ligne(P)
    assert r["pertes"].sum() <= L.sum(axis=1).min() + math.log2(3) + 1e-6


def test_tranche_fraiche_D_est_disjointe_et_comptages_exacts():
    td, debut, nval = bayes.texte_frais()
    ta, tb, tc = donnees.decouper(donnees.lire_textes()[1])
    assert debut == len(ta) + len(tb) + len(tc) == 90000
    assert len(td) == nval - debut > 40000
    bayes.construire_d(verbeux=False)
    D = bayes.charger_d()
    assert len(D["y"]) == len(td)
    lib, val = donnees.lire_textes()
    alpha = donnees.alphabet_de(lib, val)
    assert "".join(alpha[i] for i in D["y"][:50]) == td[:50]
    for t, k in ((200, 3), (1500, 4), (30000, 2)):
        ctx = td[t - k:t]
        att = [donnees.comptage_brut(lib, ctx, alpha[s]) for s in (D["y"][t],)]
        assert D["n"][k][t][D["y"][t]] == att[0]
    assert abs(D["N"][2][500, 0] - D["n"][2][500].sum()) < 1e-3


def test_d_n_a_pas_ete_ouverte_par_ce_module_et_c_reste_intacte():
    src = open(os.path.join(ICI, "bayes.py"), encoding="utf-8").read()
    assert "c.npz" not in src.replace("jamais cache/c.npz", "").replace("cache/c.npz (réservé", "")
    import re
    assert not re.search(r"\bCHEMIN_C\b", src) and "charger_c" not in src
    assert garde_fous.verifier_empreintes() == []     # juge, garde_fous, données, caches : octets identiques


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    echecs = 0
    for nom, f in tests:
        try:
            f()
            print("OK    ", nom)
        except Exception as e:  # noqa
            echecs += 1
            import traceback
            traceback.print_exc()
            print("ECHEC ", nom, type(e).__name__, e)
    print("%d tests, %d échec(s)" % (len(tests), echecs))
    sys.exit(1 if echecs else 0)
