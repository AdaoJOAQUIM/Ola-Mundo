"""Tests du paradigme bayésien : causalité, normalisation, comptage, bornes de regret, discipline de lecture.

Lancer : python3 -m unittest test_bayes_langue -v   (ou pytest). Aucun test ne lit E ni le français.
"""
import json
import math
import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import donnees_bayes as db  # noqa: E402
import experts as ex  # noqa: E402
import melange as ml  # noqa: E402
from evolution_comptage import brut  # noqa: E402

ICI = os.path.dirname(os.path.abspath(__file__))
_LIB = None


def petite_lib():
    """Bibliothèque jouet : 120 000 premiers caractères de la bibliothèque réduite ; texte : suite."""
    global _LIB
    if _LIB is None:
        lib = db.bibliotheque_reduite()
        _LIB = (lib[:120000], lib[5000000:5003000])
    return _LIB


def construire(texte, lib=None):
    lib0, _ = petite_lib()
    lib = lib or lib0
    alphabet = "".join(sorted(set(petite_lib()[0]) | set(petite_lib()[1]) | set(texte)))
    return db.calculer_tableaux(lib, {"T": texte}, alphabet, verbeux=False)["T"], alphabet


class TestBibliothequeReduite(unittest.TestCase):
    def test_partition_deterministe(self):
        lignes = db._lignes_biblio()
        ret = db._indices_E(len(lignes))
        self.assertEqual(ret, db._indices_E(len(lignes)))
        self.assertTrue(all(i % db.PAS_E == db.PAS_E - 1 for i in ret))
        taille_E = sum(len(lignes[i]) for i in ret) + len(ret) - 1
        self.assertTrue(40000 <= taille_E <= 60000, taille_E)
        reduite = db.bibliotheque_reduite().split("\n")
        self.assertEqual(len(reduite) + len(ret), len(lignes))
        ens = set(reduite)
        # aucun paragraphe de E ne reste dans la bibliothèque réduite (sauf doublon exact d'une autre ligne)
        restes = [i for i in ret if lignes[i] in ens and lignes.count(lignes[i]) == 1]
        self.assertEqual(restes, [])


class TestComptage(unittest.TestCase):
    def test_comptes_contre_brut(self):
        lib, texte = petite_lib()
        jeu, alphabet = construire(texte[:400])
        rng = np.random.default_rng(1)
        for _ in range(40):
            t = int(rng.integers(12, 400))
            k = int(rng.integers(0, 11))
            ctx = texte[t - k:t]
            s = int(rng.integers(0, len(alphabet)))
            attendu = brut(lib, ctx, alphabet[s])
            self.assertEqual(int(jeu["n"][k, t, s]), attendu)

    def test_causalite_comptes_biblio(self):
        lib, texte = petite_lib()
        a, alphabet = construire(texte[:300])
        t0 = 150
        autre = texte[:t0] + texte[400:550]
        b, _ = construire(autre)
        # alphabet identique ? sinon on le reconstruit sur le même jeu
        if a["n"].shape[2] == b["n"].shape[2]:
            np.testing.assert_array_equal(a["n"][:, :t0 + 1], b["n"][:, :t0 + 1])


class TestCoherenceAvecOriginal(unittest.TestCase):
    def test_A_reduit_moins_que_A_original(self):
        """Retirer ~0,4 % des paragraphes fait baisser les comptes de moins de 1 % (A recompté)."""
        import donnees
        if not os.path.exists(donnees.CHEMIN_AB) or not os.path.exists(db.CHEMIN_AB):
            self.skipTest("caches absents")
        ancien = donnees.charger_ab()["A"]
        nouveau = db.tableaux_dev(verbeux=False)["A"]
        self.assertEqual(ancien["n"].shape, nouveau["n"].shape)
        np.testing.assert_array_equal(ancien["y"], nouveau["y"])
        # seules les jonctions entre lignes voisines du retrait créent de nouveaux n-grammes : très rares
        augmente = (nouveau["n"] > ancien["n"]).sum() / max(1, (nouveau["n"] > 0).sum())
        self.assertLess(float(augmente), 1e-3)
        self.assertGreater(float(ancien["n"].sum()), float(nouveau["n"].sum()))
        self.assertLess(float((ancien["n"].sum() - nouveau["n"].sum()) / ancien["n"].sum()), 0.01)


class TestFlux(unittest.TestCase):
    def test_flux_contre_force_brute(self):
        _, texte = petite_lib()
        t = texte[:260]
        alphabet = sorted(set(t))
        y = np.array([alphabet.index(c) for c in t])
        f = ex.comptes_flux(y, len(alphabet), 5)["n"]
        rng = np.random.default_rng(2)
        for _ in range(60):
            pos = int(rng.integers(0, 260))
            k = int(rng.integers(0, 6))
            if pos < k:
                self.assertEqual(f[k, pos].sum(), 0)
                continue
            ctx = t[pos - k:pos]
            attendu = np.zeros(len(alphabet))
            for j in range(k, pos):                      # occurrences j < pos : t[j-k:j] == ctx, symbole t[j]
                if t[j - k:j] == ctx:
                    attendu[y[j]] += 1
            np.testing.assert_array_equal(f[k, pos], attendu)

    def test_causalite_flux(self):
        _, texte = petite_lib()
        t = texte[:300]
        alphabet = sorted(set(texte))
        y = np.array([alphabet.index(c) for c in t])
        y2 = y.copy()
        t0 = 140
        y2[t0:] = (y2[t0:] + 7) % len(alphabet)          # on change le futur, y compris y[t0]
        a = ex.comptes_flux(y, len(alphabet), 10)["n"]
        b = ex.comptes_flux(y2, len(alphabet), 10)["n"]
        np.testing.assert_array_equal(a[:, :t0 + 1], b[:, :t0 + 1])
        self.assertTrue((a[:, t0 + 1:] != b[:, t0 + 1:]).any())


def jeu_de(texte):
    jeu, alphabet = construire(texte)
    return jeu


class TestExperts(unittest.TestCase):
    def setUp(self):
        _, texte = petite_lib()
        self.t = texte[:500]
        self.jeu = jeu_de(self.t)

    def test_normalisation_statique(self):
        src = ex.Statique(self.jeu)
        for u1 in (False, True):
            snaps = ex.chaine(src, [8.0] * 11, u1, {2, 6, 10})
            for k, p in snaps.items():
                np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
                for g in (0.5, 0.8, 1.0):
                    np.testing.assert_allclose(ex.distribution(p, g).sum(axis=1), 1.0, atol=1e-9)

    def test_normalisation_mise_a_jour(self):
        fd = ex.FluxDense(self.jeu, self.jeu["y"])
        for beta in (1.0, 0.01, 0.001):
            snaps = ex.chaine(ex.source_maj(self.jeu, fd, beta), [16.0] * 11, False, {4, 10})
            for p in snaps.values():
                np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
                self.assertTrue((p > 0).all())

    def test_normalisation_pitman_yor_et_log2_vrai(self):
        d = ex.distribution_py(self.jeu, [0.5] * 11, [10.0] * 11)
        np.testing.assert_allclose(d.sum(axis=1), 1.0, atol=1e-9)
        p = ex.chaine(ex.Statique(self.jeu), [16.0] * 11, False, {10})[10]
        l2 = ex.log2_vrai(p, self.jeu["y"], 0.8)
        dist = ex.distribution(p, 0.8)
        np.testing.assert_allclose(2 ** l2, dist[np.arange(len(l2)), self.jeu["y"]], rtol=1e-9)

    def test_causalite_distributions(self):
        """Modifier un caractère futur ne change jamais la distribution aux positions antérieures (statique et mise à jour)."""
        lib, texte = petite_lib()
        t0 = 200
        t1 = texte[:500]
        t2 = t1[:t0] + "".join(chr(97 + (ord(c) * 7) % 26) for c in t1[t0:])
        alphabet = "".join(sorted(set(lib) | set(t1) | set(t2)))
        ja = db.calculer_tableaux(lib, {"T": t1}, alphabet, verbeux=False)["T"]
        jb = db.calculer_tableaux(lib, {"T": t2}, alphabet, verbeux=False)["T"]
        self.assertTrue((ja["y"][t0:] != jb["y"][t0:]).any())
        np.testing.assert_array_equal(ja["y"][:t0], jb["y"][:t0])
        for cas in ("statique", "maj"):
            ds = []
            for jeu in (ja, jb):
                if cas == "statique":
                    src = ex.Statique(jeu)
                else:
                    src = ex.source_maj(jeu, ex.FluxDense(jeu, jeu["y"]), 0.1)
                p = ex.chaine(src, [16.0] * 11, False, {10})[10]
                ds.append(ex.distribution(p, 0.65))
            np.testing.assert_allclose(ds[0][:t0 + 1], ds[1][:t0 + 1], rtol=1e-12, atol=0)
            self.assertFalse(np.allclose(ds[0][t0 + 5:], ds[1][t0 + 5:]))

    def test_main_egale_grille(self):
        """e = 16·d, ordre 10, γ = 1 : la référence à la main du juge (source épinglée) = notre chaîne."""
        be = ex.bayes_evolution()
        src_main = be.source_reference_main()
        l_juge = ex.log2_genome(src_main, self.jeu)
        p = ex.chaine(ex.Statique(self.jeu), [16.0] * 11, False, {10})[10]
        l_nous = ex.log2_vrai(p, self.jeu["y"], 1.0)
        np.testing.assert_allclose(l_juge, l_nous, rtol=1e-4, atol=1e-5)

    def test_flux_vide_retrouve_statique(self):
        """β = 1 et aucun texte lu (flux vide) : l'expert à mise à jour est l'expert statique."""
        fd = ex.FluxDense(self.jeu, self.jeu["y"])
        fd.nf[:] = 0
        fd.Nf[:] = 0
        fd.df = self.jeu["d"].astype(np.float64)
        p1 = ex.chaine(ex.source_maj(self.jeu, fd, 1.0), [16.0] * 11, False, {10})[10]
        p0 = ex.chaine(ex.Statique(self.jeu), [16.0] * 11, False, {10})[10]
        np.testing.assert_allclose(p1, p0, rtol=1e-12)


class TestMelange(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.P = rng.uniform(0.02, 0.9, size=(800, 7))
        self.P[:, 0] = np.clip(self.P[:, 0] + 0.2, 0, 0.95)

    def test_regret_bma(self):
        r = ml.melanger(self.P)
        perte_mel = r["pertes"].sum()
        pertes_i = -np.log2(self.P).sum(axis=0)
        self.assertTrue((perte_mel <= pertes_i + math.log2(7) + 1e-9).all())

    def test_causalite_melange(self):
        t0 = 300
        P2 = self.P.copy()
        P2[t0 + 1:] = np.random.default_rng(4).uniform(0.02, 0.9, size=P2[t0 + 1:].shape)
        for alpha in (0.0, 1e-3):
            a = ml.melanger(self.P, alpha)["pertes"]
            b = ml.melanger(P2, alpha)["pertes"]
            np.testing.assert_array_equal(a[:t0 + 1], b[:t0 + 1])

    def test_egal_bma_evolution(self):
        be = ex.bayes_evolution()
        for alpha in (0.0, 3e-3):
            r1 = ml.melanger(self.P, alpha)["pertes"]
            r2 = be.bma_en_ligne(self.P.T.copy(), None, alpha)["pertes"]
            np.testing.assert_allclose(r1, r2, rtol=1e-9)

    def test_masse_somme_un_et_entropie(self):
        g = np.array([0, 0, 1, 1, 2, 2, 2])
        r = ml.melanger(self.P, 1e-3, None, g, 100)
        np.testing.assert_allclose(r["masses"].sum(axis=1), 1.0, atol=1e-9)
        self.assertAlmostEqual(r["entropie"][0], math.log2(7))
        self.assertLess(r["entropie"][-1], r["entropie"][0])      # le postérieur se concentre


class TestDisciplineDeLecture(unittest.TestCase):
    """E et le français ne sont lus que par final.py."""

    def sources(self):
        out = {}
        for f in os.listdir(ICI):
            if f.endswith(".py") and not f.startswith("test_"):
                out[f] = open(os.path.join(ICI, f), encoding="utf-8").read()
        return out

    def appels_interdits(self):
        return ["texte_E" + "(", "texte_francais" + "(", "tableaux_final" + "("]

    def test_seul_final_appelle(self):
        for nom, src in self.sources().items():
            for motif in self.appels_interdits():
                for m in re.finditer(re.escape(motif), src):
                    ligne = src[:m.start()].count("\n") + 1
                    debut = src[src.rfind("\n", 0, m.start()) + 1:m.start()]
                    if nom == "donnees_bayes.py" and (debut.lstrip().startswith("def ") or
                                                      not motif.startswith("tableaux_final")):
                        continue      # définition, ou usage interne à tableaux_final (lui-même réservé à final.py)
                    self.assertEqual(nom, "final.py", "%s ligne %d appelle %s" % (nom, ligne, motif))

    def test_pas_de_chemin_francais_ailleurs(self):
        for nom, src in self.sources().items():
            if nom in ("donnees_bayes.py",):
                continue
            self.assertNotIn("francais.txt", src, nom)
            self.assertNotIn("memoire_fr", src, nom)

    def test_rien_n_est_ecrit_pour_E_ou_francais(self):
        """Aucun module n'écrit de tableau ou de texte E/français sur disque (le français n'est pas redistribuable)."""
        for nom, src in self.sources().items():
            for motif in ("e.npz", "fr.npz", "francais_cache"):
                self.assertNotIn(motif, src, nom)
        # le seul fichier écrit par final.py : témoin + résultats chiffrés
        final = self.sources()["final.py"]
        ecritures = re.findall(r'open\(([^)]*), "w"\)', final)
        self.assertTrue(all(("TEMOIN" in e) or ("SORTIE_JSON" in e) or ("final_essai" in e) for e in ecritures), ecritures)

    def test_temoin_refuse_deuxieme_execution(self):
        import final
        ancien = final.TEMOIN
        final.TEMOIN = os.path.join(ICI, "cache", "temoin_test.json")
        try:
            with open(final.TEMOIN, "w") as f:
                f.write("{}")
            sys.argv = ["final.py"]
            with self.assertRaises(SystemExit):
                final.main()
        finally:
            os.remove(final.TEMOIN)
            final.TEMOIN = ancien

    def test_json_sans_texte_francais(self):
        """Le JSON de résultats ne contient que des nombres et des noms d'experts (pas de phrase du corpus)."""
        chemin = os.path.join(ICI, "resultats_bayes_langue.json")
        if not os.path.exists(chemin):
            self.skipTest("pas encore généré")
        txt = open(chemin, encoding="utf-8").read()
        self.assertLess(len(txt), 400000)
        self.assertNotRegex(txt, r"[a-zéèêàç]{4,} [a-zéèêàç]{4,} [a-zéèêàç]{4,} [a-zéèêàç]{4,}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
