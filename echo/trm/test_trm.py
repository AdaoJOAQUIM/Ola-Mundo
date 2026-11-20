"""Tests unitaires : solveur, augmentation, formes du modèle, taille d'export."""
import os
import random
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import donnees
from modele import TRM, nb_params, quantifier_int8, taille_export

# puzzle classique à solution unique
PUZ = [int(c) for c in
       "530070000600195000098000060800060003400803001700020006060000280000419005000080079"]
SOL = [int(c) for c in
       "534678912672195348198342567859761423426853791713924856961537284287419635345286179"]


class TestSolveur(unittest.TestCase):
    def test_resout_classique(self):
        self.assertEqual(donnees.resoudre(PUZ), SOL)
        self.assertEqual(donnees.compter_solutions(PUZ), 1)

    def test_detecte_non_unicite(self):
        self.assertEqual(donnees.compter_solutions([0] * 81, 2), 2)

    def test_detecte_contradiction(self):
        p = PUZ[:]
        p[2] = 5  # doublon de 5 dans la ligne 0
        self.assertEqual(donnees.compter_solutions(p), 0)
        self.assertIsNone(donnees.resoudre(p))

    def test_generateur(self):
        rng = random.Random(7)
        for _ in range(5):
            puz, sol = donnees.generer_puzzle(rng)
            self.assertTrue(donnees.grille_valide(sol))
            self.assertEqual(donnees.compter_solutions(puz, 2), 1)
            self.assertEqual(donnees.resoudre(puz), sol)
            self.assertTrue(all(p in (0, s) for p, s in zip(puz, sol)))
            self.assertLessEqual(sum(1 for v in puz if v), 30)


class TestAugmentation(unittest.TestCase):
    def test_preserve_validite(self):
        rng = np.random.default_rng(0)
        p0, s0 = np.array(PUZ, dtype=np.uint8), np.array(SOL, dtype=np.uint8)
        for _ in range(200):
            p, s = donnees.augmenter(p0, s0, rng)
            self.assertTrue(donnees.grille_valide(s.tolist()))
            self.assertEqual(int((p > 0).sum()), int((p0 > 0).sum()))
            self.assertTrue(((p == 0) | (p == s)).all())
            self.assertEqual(donnees.compter_solutions(p.tolist(), 2), 1)
            self.assertEqual(donnees.resoudre(p.tolist()), s.tolist())


class TestModele(unittest.TestCase):
    def test_formes(self):
        m = TRM(d=32, n_blocs=2, n=2, T=2, h_tok=16, h_can=48)
        x = torch.randint(0, 10, (3, 81))
        y, z = m.init_etat(3)
        y2, z2, lg, h = m.etape(x, y, z)
        self.assertEqual(lg.shape, (3, 81, 9))
        self.assertEqual(h.shape, (3,))
        self.assertEqual(y2.shape, (3, 81, 32))
        self.assertFalse(y2.requires_grad)
        lg.sum().backward()
        self.assertTrue(all(p.grad is not None for p in m.net.parameters()))

    def test_gradient_seulement_dernier_processus(self):
        m = TRM(d=16, n_blocs=1, n=1, T=3, h_tok=8, h_can=16)
        x = torch.randint(0, 10, (2, 81))
        y, z = m.init_etat(2)
        _, _, lg, _ = m.etape(x, y, z)
        self.assertTrue(lg.requires_grad)

    def test_export(self):
        for blocs in (2, 8):
            m = TRM(d=128, n_blocs=blocs)
            t = taille_export(m)
            self.assertEqual(t["params"], nb_params(m))
            self.assertLessEqual(t["int8_octets"], 10_000_000)
            self.assertEqual(t["fp16_octets"], 2 * t["params"])
            _, octets = quantifier_int8(m)
            self.assertEqual(octets, t["int8_octets"])


if __name__ == "__main__":
    unittest.main()
