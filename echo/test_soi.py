"""Tests unitaires du « soi » d'Écho (unittest, bibliothèque standard)."""
import json
import math
import os
import random
import tempfile
import unittest

import soi
from predicteur import Memoire


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.m = Memoire("the cat sat on the mat. the cat ate the rat.\n" * 30)
        self.texte = "the cat sat on the rat.\nthe mat"
        self.recs = soi.collecter(self.m, self.texte)

    def test_collecter_equivaut_a_distribution(self):
        """Mêmes probabilités que Memoire.distribution (le départ d'ordre n'y change rien)."""
        for i in range(len(self.texte)):
            d = self.m.distribution(self.texte[max(0, i - 16):i])
            r = self.recs[i]
            self.assertAlmostEqual(r["lptrue"], math.log(d[self.texte[i]]), places=9)
            self.assertAlmostEqual(r["lptop"], math.log(max(d.values())), places=9)

    def test_masse_totale_un(self):
        for r in self.recs:
            tot = sum(math.exp(x) for x in r["lps"]) + r["nl"] * math.exp(r["lpart"])
            self.assertAlmostEqual(tot, 1.0, places=9)

    def test_temperature_un_ne_change_rien(self):
        for (c, _), r in zip(soi.confiances(self.recs, 1.0), self.recs):
            self.assertAlmostEqual(c, math.exp(r["lptop"]), places=9)

    def test_temperature_grande_aplatit(self):
        c1 = [c for c, _ in soi.confiances(self.recs, 1.0)]
        c5 = [c for c, _ in soi.confiances(self.recs, 5.0)]
        self.assertLess(sum(c5), sum(c1))

    def test_ece_parfait_et_pire(self):
        parfait = [(0.75, True)] * 3 + [(0.75, False)]
        self.assertAlmostEqual(soi.ece(parfait), 0.0)
        faux = [(0.95, False)] * 10
        self.assertAlmostEqual(soi.ece(faux), 0.95)
        self.assertGreater(soi.biais_confiance(faux), 0)

    def test_brier(self):
        self.assertAlmostEqual(soi.brier_top1([(1.0, True), (0.0, False)]), 0.0)
        self.assertAlmostEqual(soi.brier_top1([(0.5, True), (0.5, False)]), 0.25)

    def test_fiabilite_comptes(self):
        paires = [(0.05, False), (0.95, True), (1.0, True)]
        tab = soi.fiabilite(paires)
        self.assertEqual(sum(s["n"] for s in tab), 3)
        self.assertEqual(tab[9]["n"], 2)

    def test_section_doree_trouve_un_minimum(self):
        """Sur des enregistrements synthétiques surconfiants, T optimal > 1 et le NLL baisse."""
        rng = random.Random(1)
        recs = []
        for _ in range(400):
            # 3 candidats ; le meilleur annonce 0,9 mais n'a raison que 60 % du temps
            juste = rng.random() < 0.6
            lp = [math.log(0.9), math.log(0.05), math.log(0.05)]
            recs.append({"lps": lp, "lpart": -50.0, "nl": 0,
                         "lptrue": lp[0] if juste else lp[1], "lptop": lp[0], "juste": juste})
        T = soi.ajuster_temperature(recs)
        self.assertGreater(T, 1.2)
        self.assertLess(soi.nll_bits(recs, T), soi.nll_bits(recs, 1.0))
        self.assertLess(soi.ece(soi.confiances(recs, T)), soi.ece(soi.confiances(recs, 1.0)))

    def test_distribution_calibree_somme_un(self):
        d = soi.distribution_calibree(self.m, "the c", 1.7)
        self.assertAlmostEqual(sum(d.values()), 1.0, places=9)
        c, p = soi.prediction_confiante(self.m, "the c", 1.0)
        self.assertEqual(c, "a")
        self.assertGreater(p, 0.5)


class TestAutoPrediction(unittest.TestCase):
    CTX = ["Marie va dans la cuisine.", "Marie prend la clé.", "Rex est un chien.",
           "Tout chien est un animal."]

    def test_non_compris(self):
        pred, _ = soi.estimer_competence([], "Quelle est la capitale de la France ?")
        self.assertEqual(pred, soi.PRED_NON)
        self.assertEqual(soi.executer([], "Quelle est la capitale de la France ?")[0], "incompris")

    def test_comprend_question_connue(self):
        pred, _ = soi.estimer_competence(self.CTX, "Rex est-il un animal ?")
        self.assertEqual(pred, soi.PRED_COMPRENDRE)
        genre, val = soi.executer(self.CTX, "Rex est-il un animal ?")
        self.assertEqual((genre, val), ("reponse", "oui"))

    def test_abstention_entite_inconnue(self):
        pred, _ = soi.estimer_competence(self.CTX, "Où est Paul ?")
        self.assertEqual(pred, soi.PRED_ABSTENIR)
        self.assertEqual(soi.executer(self.CTX, "Où est Paul ?")[0], "abstention")

    def test_calcul(self):
        pred, _ = soi.estimer_competence([], "Combien font 7 plus 5 ?")
        self.assertEqual(pred, soi.PRED_COMPRENDRE)
        self.assertEqual(soi.executer([], "Combien font 7 plus 5 ?"), ("reponse", 12))

    def test_affirmation(self):
        self.assertEqual(soi.estimer_competence([], "Socrate est mortel.")[0], soi.PRED_COMPRENDRE)
        self.assertEqual(soi.executer([], "Socrate est mortel.")[0], "accepte")
        self.assertEqual(soi.estimer_competence([], "Le chat dort sur le canapé.")[0], soi.PRED_NON)


class TestDescription(unittest.TestCase):
    def test_nombres_lus_dans_les_fichiers(self):
        """Les nombres de decrire_soi viennent des fichiers : on les remplace par des valeurs étranges."""
        avant = soi.ICI
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "resultats_soi.json"), "w", encoding="utf-8") as f:
                json.dump({"memoire": {"en": {"paragraphes": 1234567, "caracteres": 7654321},
                                       "fr": {"paragraphes": 111, "caracteres": 222},
                                       "paragraphes_total": 1234678, "caracteres_total": 7654543,
                                       "langues": ["klingon"]}}, f)
            with open(os.path.join(d, "memoire.xz"), "wb") as f:
                f.write(b"x" * 4242)
            soi.ICI = d
            try:
                t = soi.decrire_soi()
                r = soi.repondre_soi("Combien de textes connais-tu ?")
            finally:
                soi.ICI = avant
        self.assertIn("1 234 678", t)
        self.assertIn("klingon", t)
        self.assertIn("4 242", t)
        self.assertIn("1 234 678", r)
        self.assertIn("absent", t)          # modules absents du dossier temporaire

    def test_pas_conscient(self):
        r = soi.repondre_soi("Es-tu conscient ?")
        self.assertTrue(r.startswith("Non"))
        self.assertIn("métacognition", r)

    def test_questions_soi(self):
        for q in ["Qui es-tu ?", "Que sais-tu faire ?", "Sur quoi te trompes-tu ?"]:
            self.assertTrue(len(soi.repondre_soi(q)) > 20)

    def test_aucun_fichier_ne_plante(self):
        avant = soi.ICI
        with tempfile.TemporaryDirectory() as d:
            soi.ICI = d
            try:
                self.assertIn("Paramètres appris : 0", soi.decrire_soi())
            finally:
                soi.ICI = avant


class TestBanc(unittest.TestCase):
    def test_demandes_deterministes_et_taille(self):
        import banc_soi
        a = banc_soi.construire_demandes(7)
        b = banc_soi.construire_demandes(7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 300)


if __name__ == "__main__":
    unittest.main()
