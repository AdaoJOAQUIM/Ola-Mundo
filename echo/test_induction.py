"""Tests unitaires de l'induction (unittest, bibliothèque standard)."""

import unittest

import raisonneur as R
import induction as I


def _chiens(ind, noms, aboyeur=True):
    for nom in noms:
        ind.ajouter(f"{nom} est un chien.")
        ind.ajouter(f"{nom} est un aboyeur." if aboyeur else f"{nom} n'est pas un aboyeur.")


class TestUncertainty(unittest.TestCase):
    def test_laplace(self):
        self.assertAlmostEqual(I.laplace(3), 0.8)
        self.assertAlmostEqual(I.laplace(0), 0.5)
        self.assertAlmostEqual(I.laplace(8, 1), 9 / 11)

    def test_beta(self):
        b = I.resume_beta(3)
        self.assertAlmostEqual(b["moyenne"], 0.8)
        self.assertAlmostEqual(b["borne_inf_5pct"], 0.05 ** 0.25)
        # la quantile numérique est cohérente avec la forme close
        self.assertAlmostEqual(I.quantile_beta(0.05, 4, 1), 0.05 ** 0.25, places=4)


class TestHypotheses(unittest.TestCase):
    def test_seuil_smin(self):
        ind = I.Inducteur(s_min=3)
        _chiens(ind, ["Médor", "Rex"])
        self.assertNotIn(("tout", "chien", "aboyeur"), ind.hyps)
        _chiens(ind, ["Fido"])
        h = ind.hyps[("tout", "chien", "aboyeur")]
        self.assertEqual((h.s, h.c), (3, 0))
        self.assertAlmostEqual(h.confiance, 0.8)

    def test_jamais_traitee_comme_certaine(self):
        ind = I.Inducteur()
        _chiens(ind, ["Médor", "Rex", "Fido"])
        ind.ajouter("Pluto est un chien.")
        n_avant = len(ind.base.phrases)
        regles_avant = {k: list(v) for k, v in ind.base.regles.items()}
        r = ind.repondre("Pluto est-il un aboyeur ?")
        self.assertEqual(r.statut, "probable")
        self.assertFalse(r.certain)
        self.assertTrue(r.texte.startswith("Probablement oui (confiance 80 %)"))
        self.assertIn("tout chien est un aboyeur", r.texte)
        self.assertIn("3 exemples", r.texte)
        # la base du raisonneur est intacte, et le raisonneur seul ne sait pas
        self.assertEqual(len(ind.base.phrases), n_avant)
        self.assertEqual({k: list(v) for k, v in ind.base.regles.items()}, regles_avant)
        seul = ind.rais.repondre("Pluto est-il un aboyeur ?")
        self.assertFalse(seul.verifie)
        self.assertEqual(seul.valeur, "inconnu")

    def test_certain_prioritaire(self):
        ind = I.Inducteur()
        _chiens(ind, ["Médor", "Rex", "Fido"])
        ind.ajouter("Tout chien est un animal.")
        r = ind.repondre("Médor est-il un animal ?")
        self.assertTrue(r.certain)
        self.assertTrue(r.reponse_certaine.verifie)
        self.assertEqual(r.valeur, "oui")
        # une règle certaine n'est jamais proposée comme hypothèse
        self.assertNotIn(("tout", "chien", "animal"), ind.hyps)

    def test_exemples_cites(self):
        ind = I.Inducteur()
        _chiens(ind, ["Médor", "Rex", "Fido"])
        h = ind.hyps[("tout", "chien", "aboyeur")]
        self.assertEqual(h.phrases, [1, 2, 3, 4, 5, 6])

    def test_aucun_et_non(self):
        ind = I.Inducteur()
        for nom in ["Mistigri", "Felix", "Tom"]:
            ind.ajouter(f"{nom} est un chat.")
            ind.ajouter(f"{nom} n'est pas un aboyeur.")
        self.assertIn(("aucun", "chat", "aboyeur"), ind.hyps)
        ind.ajouter("Minou est un chat.")
        r = ind.repondre("Minou est-il un aboyeur ?")
        self.assertEqual((r.statut, r.valeur), ("probable", "non"))
        self.assertIn("aucun chat n'est un aboyeur", r.texte)

    def test_chaine_produit_des_confiances(self):
        ind = I.Inducteur()
        for nom in ["a", "b", "c"]:
            n = nom.upper() + "ko"
            ind.ajouter(f"{n} est un dogue.")
            ind.ajouter(f"{n} est un chien.")
            ind.ajouter(f"{n} est un aboyeur.")
        ind.ajouter("Zuko est un dogue.")
        r = ind.repondre("Zuko est-il un aboyeur ?")
        self.assertEqual(r.statut, "probable")
        # la confiance d'un chemin à une hypothèse est celle de cette hypothèse
        self.assertLessEqual(r.confiance, 0.8 + 1e-9)


class TestRevision(unittest.TestCase):
    def test_contre_exemple_retracte(self):
        ind = I.Inducteur()
        _chiens(ind, ["Médor", "Rex", "Fido"])
        self.assertIn(("tout", "chien", "aboyeur"), ind.hyps)
        ind.ajouter("Rintintin est un chien.")
        rap = ind.ajouter("Rintintin n'est pas un aboyeur.")
        self.assertNotIn(("tout", "chien", "aboyeur"), ind.hyps)
        cles = [h.cle for h, _ in rap.retractees]
        self.assertIn(("tout", "chien", "aboyeur"), cles)
        self.assertIn("Hypothèse retirée", rap.message)
        self.assertIn("Rintintin", rap.message)
        # et la réponse revient à « je ne sais pas »
        ind.ajouter("Pluto est un chien.")
        self.assertEqual(ind.repondre("Pluto est-il un aboyeur ?").statut, "inconnu")

    def test_pas_de_retour(self):
        ind = I.Inducteur()
        _chiens(ind, ["Médor", "Rex", "Fido"])
        ind.ajouter("Rintintin est un chien.")
        ind.ajouter("Rintintin n'est pas un aboyeur.")
        _chiens(ind, ["Idéfix", "Milou", "Snoopy", "Beethoven"])
        self.assertNotIn(("tout", "chien", "aboyeur"), ind.hyps)
        self.assertIn(("tout", "chien", "aboyeur"), ind.refutees)

    def test_promotion_en_certitude(self):
        ind = I.Inducteur()
        _chiens(ind, ["Médor", "Rex", "Fido"])
        rap = ind.ajouter("Tout chien est un aboyeur.")
        self.assertEqual([h.cle for h in rap.promues], [("tout", "chien", "aboyeur")])
        self.assertEqual(rap.retractees, [])


class TestBoucle(unittest.TestCase):
    def _monde(self, **kw):
        ind = I.Inducteur(**kw)
        _chiens(ind, ["Médor", "Rex", "Fido"])
        # trois individus « chien » sans information sur aboyeur
        for nom in ["Pluto", "Droopy", "Idéfix"]:
            ind.ajouter(f"{nom} est un chien.")
        return ind

    def test_faits_provisoires_ne_soutiennent_pas(self):
        ind = self._monde()
        avant = {k: (h.s, h.c) for k, h in ind.hyps.items()}
        rb = ind.boucle()
        self.assertGreater(rb.nb_faits_provisoires, 0)    # Pluto etc. sont « aboyeur »
        self.assertIn(("pluto", "+", "aboyeur"), rb.faits_provisoires)
        apres = {k: (h.s, h.c) for k, h in rb.hypotheses.items()}
        self.assertEqual(avant, apres)                     # soutien inchangé
        self.assertEqual(rb.hypotheses[("tout", "chien", "aboyeur")].s, 3)
        # et l'induction sur la base donnée est identique après la boucle
        self.assertEqual({k: (h.s, h.c) for k, h in ind.induire().items()}, avant)

    def test_contraste_auto_renforcement(self):
        sain = self._monde().boucle()
        naif = self._monde(soutien_provisoire=True).boucle()
        s_sain = sain.hypotheses[("tout", "chien", "aboyeur")].s
        s_naif = naif.hypotheses[("tout", "chien", "aboyeur")].s
        self.assertEqual(s_sain, 3)
        self.assertGreater(s_naif, 3)    # le cercle vicieux gonfle le soutien

    def test_terminaison(self):
        ind = self._monde()
        rb = ind.boucle()
        self.assertTrue(rb.converge)
        self.assertFalse(rb.plafond_atteint)
        f = ind.fermeture()
        borne = 2 * len(f.individus) * len(f.classes) + 2
        self.assertLessEqual(rb.iterations, borne)
        self.assertGreaterEqual(rb.iterations, 2)
        # F est croissante le long des itérations
        tailles = [nf for _, _, nf in rb.historique]
        self.assertEqual(tailles, sorted(tailles))

    def test_plafond_signale(self):
        ind = self._monde(iter_max=1)
        rb = ind.boucle()
        self.assertTrue(rb.plafond_atteint)
        self.assertFalse(rb.converge)
        self.assertEqual(rb.iterations, 1)

    def test_base_vide(self):
        rb = I.Inducteur().boucle()
        self.assertTrue(rb.converge)
        self.assertEqual((rb.nb_hypotheses, rb.nb_faits_provisoires), (0, 0))

    def test_chaine_necessite_plusieurs_iterations(self):
        ind = I.Inducteur()
        for nom in ["Aku", "Bilo", "Cemu"]:          # dogue -> chien (exemples)
            ind.ajouter(f"{nom} est un dogue.")
            ind.ajouter(f"{nom} est un chien.")
        for nom in ["Dodo", "Eko", "Fali"]:          # chien -> aboyeur (exemples)
            ind.ajouter(f"{nom} est un chien.")
            ind.ajouter(f"{nom} est un aboyeur.")
        ind.ajouter("Zuko est un dogue.")
        rb = ind.boucle()
        self.assertIn(("zuko", "+", "chien"), rb.faits_provisoires)
        self.assertIn(("zuko", "+", "aboyeur"), rb.faits_provisoires)   # 2 étapes
        self.assertEqual(rb.iterations, 3)
        self.assertTrue(rb.converge)


class TestCouverture(unittest.TestCase):
    def test_garde_optionnelle(self):
        def monde(**kw):
            ind = I.Inducteur(**kw)
            for nom in ["Aku", "Bilo", "Cemu"]:
                ind.ajouter(f"{nom} est un chat.")
                ind.ajouter(f"{nom} est un animal.")
            for nom in ["Dodo", "Eko", "Fali", "Gumo"]:
                ind.ajouter(f"{nom} est un animal.")
            return ind
        self.assertIn(("tout", "chat", "animal"), monde().hyps)
        self.assertIn(("tout", "animal", "chat"), monde().hyps)
        self.assertNotIn(("tout", "animal", "chat"), monde(couverture_min=0.5).hyps)


if __name__ == "__main__":
    unittest.main()
