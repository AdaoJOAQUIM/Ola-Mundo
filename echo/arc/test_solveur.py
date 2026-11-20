# -*- coding: utf-8 -*-
"""Tests unitaires : primitives et quelques tâches faites à la main."""
import unittest
import solveur as s

G = ((1, 2, 3), (4, 5, 6))


class TestPrimitives(unittest.TestCase):
    def test_geometrie(self):
        self.assertEqual(s.rot90(G), ((4, 1), (5, 2), (6, 3)))
        self.assertEqual(s.rot180(s.rot180(G)), G)
        self.assertEqual(s.fliph(G), ((3, 2, 1), (6, 5, 4)))
        self.assertEqual(s.flipv(G), ((4, 5, 6), (1, 2, 3)))
        self.assertEqual(s.transp(G), ((1, 4), (2, 5), (3, 6)))
        self.assertEqual(s.rot90(s.rot270(G)), G)

    def test_echelle(self):
        self.assertEqual(s.upscale(2)(((1, 2),)), ((1, 1, 2, 2), (1, 1, 2, 2)))
        self.assertEqual(s.downscale(2)(s.upscale(2)(G)), G)
        self.assertEqual(s.tile(1, 2)(((1,),)), ((1, 1),))
        self.assertEqual(s.miroir_h(((1, 2),)), ((1, 2, 2, 1),))

    def test_crop_et_objets(self):
        g = ((0, 0, 0, 0), (0, 5, 5, 0), (0, 5, 0, 0), (0, 0, 0, 7))
        self.assertEqual(s.crop_bbox(g), ((5, 5, 0), (5, 0, 0), (0, 0, 7)))
        self.assertEqual(len(s.objets(g, 'c4')), 2)
        self.assertEqual(s.prim_crop_objet('c4', 'plus_grand', False)(g), ((5, 5), (5, 0)))

    def test_trous(self):
        g = ((0, 0, 0, 0, 0), (0, 1, 1, 1, 0), (0, 1, 0, 1, 0), (0, 1, 1, 1, 0), (0, 0, 0, 0, 0))
        self.assertEqual(s.remplir_trous(g)[2][2], s.M)
        self.assertEqual(s.nb_trous(g, s.objets(g, 'c4')[0]), 1)

    def test_logique(self):
        g = ((1, 0, 5, 1, 1), (0, 1, 5, 0, 1))  # moitiés ((1,0),(0,1)) et ((1,1),(0,1)) : ET -> diag
        r = s.op_logique('sep', 'et')(g)
        self.assertEqual(r, ((s.M, 0), (0, s.M)))

    def test_gravite(self):
        g = ((1, 0), (0, 0), (0, 2))
        self.assertEqual(s.gravite('bas')(g), ((0, 0), (0, 0), (1, 2)))

    def test_symetrie(self):
        g = ((1, 2, 2, 1), (3, 4, 9, 3))
        r = s.completer_symetrie(g, 9, grand=1)
        self.assertEqual(r[1], (3, 4, 4, 3))


def tache(paires, test):
    return {'train': [{'input': a, 'output': b} for a, b in paires],
            'test': [{'input': t} for t in test]}


class TestTaches(unittest.TestCase):
    def test_flip(self):
        t = tache([([[1, 2], [3, 4]], [[2, 1], [4, 3]]), ([[5, 6, 7]], [[7, 6, 5]]),
                   ([[1, 2, 3, 4], [5, 6, 7, 8]], [[4, 3, 2, 1], [8, 7, 6, 5]])], [[[1, 2, 3], [4, 5, 6]]])
        r = s.resoudre(t, 5)
        self.assertIn(((3, 2, 1), (6, 5, 4)), [c[1][0] for c in r['candidats']])

    def test_couleurs(self):
        t = tache([([[1, 0], [0, 1]], [[2, 0], [0, 2]]), ([[1, 1]], [[2, 2]])], [[[0, 1, 1]]])
        r = s.resoudre(t, 5)
        self.assertEqual(r['candidats'][0][1][0], ((0, 2, 2),))

    def test_remplir_trous(self):
        a = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        b = [[1, 1, 1], [1, 4, 1], [1, 1, 1]]
        a2 = [[0, 0, 0, 0], [0, 2, 2, 2], [0, 2, 0, 2], [0, 2, 2, 2]]
        b2 = [[0, 0, 0, 0], [0, 2, 2, 2], [0, 2, 4, 2], [0, 2, 2, 2]]
        r = s.resoudre(tache([(a, b), (a2, b2)], [a2]), 5)
        self.assertEqual(r['candidats'][0][1][0], s.en_grille(b2))


if __name__ == '__main__':
    unittest.main()
