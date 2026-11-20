"""Tests du planificateur et de l'aiguilleur (unittest, bibliothèque standard)."""
import lzma
import os
import random
import tempfile
import unittest

import aiguilleur as ag
import banc_aiguilleur as banc
import planificateur as pf
import raisonneur as rz


def monde(*phrases):
    r = rz.Raisonneur()
    for p in phrases:
        assert r.ajouter(p).statut in ("ajoutee", "doublon"), p
    return r


BASE = ["Marie va dans la cuisine.", "Jean va au jardin.", "Paul va dans le salon.",
        "Marie prend la pomme.", "Marie pose la pomme.", "Jean prend le livre."]


class TestObjectifs(unittest.TestCase):
    def test_analyse_articles_accents(self):
        for t in ("Objectif : la pomme doit être au jardin.",
                  "objectif: La pomme doit etre au jardin",
                  "Objectif : l'écharpe doit être dans le salon."):
            self.assertIsNotNone(pf.analyser_objectif(t), t)
        b = pf.analyser_objectif("Objectif : Marie doit avoir le livre.")
        self.assertEqual((b["sujet"], b["verbe"]), ("marie", "avoir"))
        self.assertIsNone(pf.analyser_objectif("Marie va au jardin."))

    def test_trois_genres(self):
        r = monde(*BASE)
        pl = pf.Planificateur()
        a = pl.planifier(r, "Objectif : la pomme doit être au jardin.")
        b = pl.planifier(r, "Objectif : Marie doit avoir le livre.")
        c = pl.planifier(r, "Objectif : Jean doit être dans la cuisine.")
        for x in (a, b, c):
            self.assertEqual(x.statut, "plan", x.message)
            self.assertTrue(x.verifie)
        self.assertEqual(a.cout, 2)
        self.assertEqual(c.phrases, ["Jean va dans la cuisine."])

    def test_deja_atteint(self):
        r = monde(*BASE)
        x = pf.Planificateur().planifier(r, "Objectif : la pomme doit être dans la cuisine.")
        self.assertEqual((x.statut, x.cout, x.phrases), ("plan", 0, []))
        self.assertTrue(x.verifie)

    def test_cout_par_personne_change_le_plan(self):
        r = monde(*BASE)
        obj = "Objectif : la pomme doit être au jardin."
        normal = pf.Planificateur().planifier(r, obj)
        self.assertTrue(all(p.startswith("Marie") for p in normal.phrases))
        lent = pf.Planificateur(multiplicateurs={"Marie": 5}).planifier(r, obj)
        self.assertEqual(lent.cout, 3)
        self.assertTrue(all(p.startswith("Jean") for p in lent.phrases), lent.phrases)

    def test_impossible_avec_raison(self):
        r = monde(*BASE)
        pl = pf.Planificateur()
        for obj, mot in (("Objectif : la pomme doit être dans la cave.", "lieu inconnu"),
                         ("Objectif : le stylo doit être au jardin.", "inconnue"),
                         ("Objectif : Pierre doit être au jardin.", "inconnue"),
                         ("Objectif : Pierre doit avoir la pomme.", "inconnue"),
                         ("Objectif : Marie doit avoir le stylo.", "inconnu")):
            x = pl.planifier(r, obj)
            self.assertEqual(x.statut, "impossible", obj)
            self.assertIn("Objectif impossible", x.message)
            self.assertIn(mot, x.message)

    def test_objet_pose_en_lieu_inconnu_impossible(self):
        r = monde("Léa prend la clé.", "Léa pose la clé.", "Jean va au jardin.")
        x = pf.Planificateur().planifier(r, "Objectif : la clé doit être au jardin.")
        self.assertEqual(x.statut, "impossible")

    def test_budget_distinct_de_impossible(self):
        r = monde(*BASE)
        x = pf.Planificateur(budget=1).planifier(r, "Objectif : la pomme doit être au jardin.")
        self.assertEqual(x.statut, "budget")
        self.assertIn("Budget", x.message)
        self.assertNotIn("Objectif impossible", x.message)

    def test_incompris(self):
        r = monde(*BASE)
        self.assertEqual(pf.Planificateur().planifier(r, "Fais-moi un café.").statut, "incompris")

    def test_lieu_declare(self):
        r = monde(*BASE, "Le grenier est un lieu.")
        x = pf.Planificateur().planifier(r, "Objectif : la pomme doit être dans le grenier.")
        self.assertEqual(x.statut, "plan", x.message)
        self.assertTrue(x.verifie)

    def test_plan_non_verifie_est_refuse(self):
        class Menteur(pf.Planificateur):
            def phrase(self, action, mo):          # produit une phrase qui ne fait pas le travail
                return "Marie va dans la cuisine."
        r = monde(*BASE)
        x = Menteur().planifier(r, "Objectif : la pomme doit être au jardin.")
        self.assertEqual(x.statut, "non_verifie")
        self.assertEqual(x.phrases, [])
        self.assertFalse(x.verifie)

    def test_rejeu_dans_un_raisonneur_neuf(self):
        r = monde(*BASE)
        n = len(r.base.phrases)
        x = pf.Planificateur().planifier(r, "Objectif : la pomme doit être au jardin.")
        self.assertEqual(len(r.base.phrases), n)       # le monde d'origine n'est pas modifié
        self.assertTrue(x.verifie)

    def test_couts_invalides(self):
        with self.assertRaises(ValueError):
            pf.Planificateur(couts={"aller": 0})
        with self.assertRaises(ValueError):
            pf.Planificateur(multiplicateurs={"Marie": -1})


class TestOptimalite(unittest.TestCase):
    def test_contre_oracle_exhaustif(self):
        """A* = recherche à coût uniforme indépendante (état complet), 40 mondes aléatoires."""
        rng = random.Random(7)
        compares = 0
        for _ in range(40):
            P = banc.PERSONNES[:rng.randint(2, 3)]
            lieux = ["cuisine", "jardin", "salon", "cave"][:rng.randint(2, 4)]
            sim = banc.Sim(P, lieux, ["pomme", "livre"][:rng.randint(1, 2)])
            for p in P:
                sim.appliquer(("va", p, rng.choice(lieux)))
            for _ in range(rng.randint(2, 8)):
                sim.appliquer(sim.tirer(rng))
            mult = {p: rng.choice([1, 2, 3]) for p in P}
            couts = dict(pf.COUTS_DEFAUT, prendre=2) if rng.random() < .5 else dict(pf.COUTS_DEFAUT)
            if not sim.vus_objets():
                continue
            o = rng.choice(sorted(sim.vus_objets()))
            l = rng.choice(sorted(sim.lieux_vus))
            p = rng.choice(sorted(sim.pers_vues))
            for but, txt in ((("objet_lieu", o, l), f"Objectif : {banc.OBJETS[o]} doit être {banc.LIEUX[l]}."),
                             (("personne_lieu", p, l), f"Objectif : {p} doit être {banc.LIEUX[l]}."),
                             (("possede", o, p), f"Objectif : {p} doit avoir {banc.OBJETS[o]}.")):
                etat, c = banc.oracle(sim, but, couts, mult)
                rr = rz.Raisonneur()
                for ph in sim.histoire:
                    rr.ajouter(ph)
                x = pf.Planificateur(couts=couts, multiplicateurs=mult).planifier(rr, txt)
                if etat == "plan":
                    self.assertEqual(x.statut, "plan", txt)
                    self.assertTrue(x.verifie)
                    self.assertAlmostEqual(x.cout, c, msg=f"{txt} {sim.histoire}")
                    ok, atteint, cp = banc.rejouer_plan(sim, x.phrases, but, couts, mult)
                    self.assertTrue(ok and atteint)
                    compares += 1
                elif etat == "impossible":
                    self.assertEqual(x.statut, "impossible", txt)
        self.assertGreater(compares, 60)


class TestAiguilleur(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = tempfile.mkdtemp()
        cls.memoire = os.path.join(cls.rep, "petite.xz")
        texte = ("Le chat dort sur le canapé. Le chat mange la souris. Le chien dort sur le tapis. "
                 "Le chien mange le chat. La souris dort sur le tapis. " * 40)
        with lzma.open(cls.memoire, "wt", encoding="utf-8") as f:
            f.write(texte)

    def nouveau(self, **kw):
        return ag.Aiguilleur(chemin_memoire=self.memoire, **kw)

    def test_module_et_cache(self):
        a = self.nouveau()
        for p in BASE:
            self.assertEqual(a.traiter(p).module, 1)
        r1 = a.traiter("Où est la pomme ?")
        r2 = a.traiter("où est la pomme ?  ")
        self.assertEqual((r1.module, r1.statut, r1.verifie), (1, "verifie", True))
        self.assertEqual((r2.module, r2.statut), (0, "verifie"))
        self.assertEqual(r1.reponse, r2.reponse)
        self.assertAlmostEqual(r1.energie_j, r1.cpu_s * 10.0)

    def test_cache_invalide_par_nouvelle_phrase(self):
        a = self.nouveau()
        for p in BASE:
            a.traiter(p)
        a.traiter("Où est Jean ?")
        a.traiter("Jean va dans la cuisine.")
        r = a.traiter("Où est Jean ?")
        self.assertNotEqual(r.module, 0)
        self.assertIn("cuisine", r.reponse)

    def test_calcul_survit_aux_insertions(self):
        a = self.nouveau()
        a.traiter("Combien font 12 fois 7 ?")
        a.traiter("Marie va dans la cuisine.")
        r = a.traiter("Combien font 12 fois 7 ?")
        self.assertEqual((r.module, r.valeur), (0, 84))

    def test_abstention_non_escaladee(self):
        a = self.nouveau()
        r = a.traiter("Qui a le stylo ?")
        self.assertEqual((r.statut, r.module), ("abstention", 1))
        self.assertIsNone(a._T)                       # la génération n'a pas été chargée
        r2 = a.traiter("Quelle est la capitale de la France ?")
        self.assertEqual(r2.statut, "incompris")
        self.assertIsNone(a._T)

    def test_objectif_puis_cache(self):
        a = self.nouveau()
        for p in BASE:
            a.traiter(p)
        r = a.traiter("Objectif : la pomme doit être au jardin.")
        self.assertEqual((r.module, r.statut, r.verifie), (2, "verifie", True))
        self.assertEqual(a.traiter("Objectif : la pomme doit être au jardin.").module, 0)
        imp = a.traiter("Objectif : la pomme doit être dans la cave.")
        self.assertEqual((imp.module, imp.statut), (2, "impossible"))

    def test_generation_paresseuse_et_etiquetee(self):
        a = self.nouveau(n_gen=60)
        self.assertIsNone(a._T)
        a.traiter("Marie va dans la cuisine.")
        self.assertIsNone(a._T)
        r = a.traiter("Écris : Le chat")
        self.assertEqual((r.module, r.statut, r.verifie), (3, "non_verifie", False))
        self.assertTrue(r.reponse.startswith("[texte généré, non vérifié]"))
        self.assertGreater(len(r.reponse), 40)
        self.assertIsNotNone(a._T)
        r2 = a.traiter("Continue : Le chien")
        self.assertEqual(r2.module, 3)

    def test_budget_depasse(self):
        a = self.nouveau(budget_cpu=0.0)
        r = a.traiter("Où est la pomme ?")
        self.assertEqual(r.statut, "budget_depasse")
        self.assertIn("budget dépassé", r.reponse.lower())
        g = self.nouveau(budget_cpu=1e-9).traiter("Écris : Le chat")
        self.assertEqual(g.statut, "budget_depasse")

    def test_budget_pendant_la_generation(self):
        a = self.nouveau(n_gen=100000, budget_cpu=0.05)
        a._bibliotheque()                         # chargement hors budget
        r = a.traiter("Écris : Le chat")
        self.assertEqual(r.statut, "budget_depasse")

    def test_toujours_tout_meme_reponses(self):
        reqs = BASE + ["Où est la pomme ?", "Qui a le livre ?", "Combien font 2 plus 3 ?",
                       "Objectif : Jean doit être dans la cuisine.", "Qui a le stylo ?",
                       "Marie prend le livre."]
        a, b = self.nouveau(), self.nouveau(strategie="toujours_tout", n_gen_toujours_tout=10)
        for q in reqs:
            ra, rb = a.traiter(q), b.traiter(q)
            self.assertEqual((ra.statut, ra.valeur), (rb.statut, rb.valeur), q)
        self.assertIsNotNone(b._T)                    # « toujours tout » a bien chargé Écho
        self.assertIsNone(a._T)

    def test_banc_charge_petite(self):
        charge = banc.construire_charge(3, 60)
        self.assertEqual(len(charge), 60)
        a = self.nouveau(strategie="sans_cache")
        for rq in charge:
            if rq["categorie"] == "generation":
                continue
            res = a.traiter(rq["texte"])
            ok, halluc, note = banc.evaluer(rq, res)
            self.assertTrue(ok, (rq["texte"], note, res.reponse))
            self.assertFalse(halluc)


if __name__ == "__main__":
    unittest.main()
