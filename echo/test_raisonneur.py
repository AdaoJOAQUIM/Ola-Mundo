#!/usr/bin/env python3
"""Tests unitaires du raisonneur (unittest, bibliothèque standard)."""

import copy
import unittest

import raisonneur as rz
from raisonneur import Raisonneur, verifier, SANS_REPONSE


def base_socrate():
    r = Raisonneur()
    for p in ["Socrate est un homme.", "Tout homme est mortel.",
              "Les mortels sont des êtres vivants.", "Médor est un chien.",
              "Tous les chiens sont des animaux.", "Socrate n'est pas un oiseau."]:
        assert r.ajouter(p).statut == "ajoutee", p
    return r


class TestNormalisation(unittest.TestCase):
    def test_accents_et_casse(self):
        self.assertEqual(rz.normaliser("Médor ÉTAIT là"), "medor etait la")

    def test_longueur_conservee(self):
        t = "L’été à Genève : œuf, ça va !"
        self.assertEqual(len(rz.normaliser(t)), len(t))

    def test_singulier(self):
        self.assertEqual(rz.singulier("chiens"), "chien")
        self.assertEqual(rz.singulier("animaux"), "animal")
        self.assertEqual(rz.singulier("oiseaux"), "oiseau")
        self.assertEqual(rz.singulier("hiboux"), "hibou")
        self.assertEqual(rz.singulier("souris"), "souris")
        self.assertEqual(rz.singulier("chevaux"), "cheval")

    def test_variantes_de_surface(self):
        # casse, ponctuation, accents, article élidé
        for t in ["socrate est un homme", "SOCRATE EST UN HOMME !",
                  "Socrate est un homme.", "  Socrate   est un homme  "]:
            a = rz.analyser(t)
            self.assertEqual(a.forme, ("est", "socrate", "homme", True), t)
        self.assertEqual(rz.analyser("L'homme est mortel.").forme,
                         ("regle", "homme", "mortel"))
        self.assertEqual(rz.analyser("Les hommes sont mortels").forme,
                         ("regle", "homme", "mortel"))
        self.assertEqual(rz.analyser("Tout homme est mortel").forme,
                         ("regle", "homme", "mortel"))
        self.assertEqual(rz.analyser("Tous les chiens sont des animaux").forme,
                         ("regle", "chien", "animal"))
        self.assertEqual(rz.analyser("Socrate n'est pas un oiseau.").forme,
                         ("est", "socrate", "oiseau", False))
        self.assertEqual(rz.analyser("Socrate n’est pas un oiseau.").forme,
                         ("est", "socrate", "oiseau", False))
        self.assertEqual(rz.analyser("Aucun chien n'est un oiseau.").forme,
                         ("disjoint", "chien", "oiseau"))

    def test_evenements(self):
        self.assertEqual(rz.analyser("Marie va dans la cuisine.").forme,
                         ("evt", "va", "marie", "cuisine"))
        self.assertEqual(rz.analyser("Marie est allée au jardin.").forme,
                         ("evt", "va", "marie", "jardin"))
        self.assertEqual(rz.analyser("Jean prend la pomme.").forme,
                         ("evt", "prend", "jean", "pomme"))
        self.assertEqual(rz.analyser("Jean pose la pomme.").forme,
                         ("evt", "pose", "jean", "pomme"))
        self.assertEqual(rz.analyser("Jean donne la pomme à Marie.").forme,
                         ("evt", "donne", "jean", "pomme", "marie"))
        self.assertEqual(rz.analyser("Jean prend l'écharpe.").forme,
                         ("evt", "prend", "jean", "echarpe"))

    def test_questions(self):
        self.assertEqual(rz.analyser("Socrate est-il mortel ?").forme,
                         ("q_est", "ind", "socrate", "mortel", True))
        self.assertEqual(rz.analyser("Est-ce que Médor est un animal ?").forme,
                         ("q_est", "ind", "medor", "animal", True))
        self.assertEqual(rz.analyser("Où est Marie ?").forme, ("q_ou", "marie"))
        self.assertEqual(rz.analyser("Où est la pomme ?").forme, ("q_ou", "pomme"))
        self.assertEqual(rz.analyser("Qui a la pomme ?").forme, ("q_qui", "pomme"))
        self.assertEqual(rz.analyser("Les chiens sont-ils des animaux ?").forme,
                         ("q_est", "cls", "chien", "animal", True))

    def test_incompris(self):
        for t in ["", "   ", "Il pleut beaucoup.", "Bonjour tout le monde",
                  "Pourquoi le ciel est bleu ?", "Marie et Jean vont au parc."]:
            self.assertEqual(rz.analyser(t).genre, "incompris", t)


class TestDeduction(unittest.TestCase):
    def test_chaine(self):
        r = base_socrate()
        rep = r.repondre("Socrate est-il un être vivant ?")
        self.assertEqual(rep.valeur, "oui")
        self.assertTrue(rep.verifie)
        self.assertEqual([e["regle"] for e in rep.preuve],
                         ["donnée", "donnée", "modus ponens", "donnée", "modus ponens"])
        self.assertEqual(rep.reponse, "Oui, Socrate est un être vivant.")

    def test_est_ce_que_et_pluriel(self):
        r = base_socrate()
        self.assertEqual(r.repondre("Est-ce que Médor est un animal ?").valeur, "oui")

    def test_negation_directe(self):
        r = base_socrate()
        rep = r.repondre("Socrate est-il un oiseau ?")
        self.assertEqual(rep.valeur, "non")
        self.assertTrue(rep.verifie)

    def test_modus_tollens(self):
        r = Raisonneur()
        for p in ["Tout chien est un mammifère.", "Tout mammifère est un animal.",
                  "Rex n'est pas un animal."]:
            r.ajouter(p)
        rep = r.repondre("Rex est-il un chien ?")
        self.assertEqual(rep.valeur, "non")
        self.assertIn("modus tollens", [e["regle"] for e in rep.preuve])
        self.assertTrue(rep.verifie)

    def test_disjonction(self):
        r = Raisonneur()
        for p in ["Aucun chien n'est un oiseau.", "Rex est un chien.",
                  "Tout oiseau est un animal."]:
            r.ajouter(p)
        rep = r.repondre("Rex est-il un oiseau ?")
        self.assertEqual(rep.valeur, "non")
        self.assertIn("disjonction", [e["regle"] for e in rep.preuve])

    def test_question_de_classe_transitivite(self):
        r = Raisonneur()
        for p in ["Tout chien est un mammifère.", "Tout mammifère est un animal.",
                  "Tout animal est un être vivant."]:
            r.ajouter(p)
        rep = r.repondre("Tout chien est-il un être vivant ?")
        self.assertEqual(rep.valeur, "oui")
        self.assertIn("transitivité", [e["regle"] for e in rep.preuve])

    def test_chaine_longue(self):
        # chaîne réelle de 40 maillons
        r = Raisonneur()
        noms = [f"classe{'abcdefghij'[i % 10]}{'klmnopqrst'[i // 10]}" for i in range(40)]
        r.ajouter(f"Rex est un {noms[0]}.")
        for a, b in zip(noms, noms[1:]):
            r.ajouter(f"Tout {a} est un {b}.")
        rep = r.repondre(f"Rex est-il un {noms[-1]} ?")
        self.assertEqual(rep.valeur, "oui")
        self.assertTrue(rep.verifie)
        self.assertEqual(len(rep.preuve), 1 + 2 * 39)

    def test_converse_non_deduite(self):
        r = Raisonneur()
        r.ajouter("Tout chien est un animal.")
        r.ajouter("Rex est un animal.")
        self.assertEqual(r.repondre("Rex est-il un chien ?").valeur, "inconnu")

    def test_cycle_de_regles(self):
        r = Raisonneur()
        for p in ["Tout alpha est un beta.", "Tout beta est un alpha.", "Rex est un alpha."]:
            r.ajouter(p)
        self.assertEqual(r.repondre("Rex est-il un beta ?").valeur, "oui")
        self.assertEqual(r.repondre("Rex est-il un zz ?").valeur, "inconnu")


class TestAbstention(unittest.TestCase):
    def test_inconnu(self):
        r = base_socrate()
        for q in ["Platon est-il un homme ?", "Médor est-il mortel ?",
                  "Où est Marie ?", "Où est la pomme ?", "Qui a la pomme ?"]:
            rep = r.repondre(q)
            self.assertEqual(rep.valeur, "inconnu", q)
            self.assertEqual(rep.reponse, SANS_REPONSE)
            self.assertEqual(rep.preuve, [])

    def test_incompris(self):
        r = base_socrate()
        rep = r.repondre("Combien de pattes a Médor ?")
        self.assertEqual(rep.valeur, "incompris")
        self.assertTrue(rep.reponse.startswith("Je ne comprends pas cette phrase"))

    def test_monde_ouvert(self):
        # « Rex n'est pas dit être un chien » ne veut pas dire « Rex n'est pas un chien »
        r = Raisonneur()
        r.ajouter("Rex est un animal.")
        self.assertEqual(r.repondre("Rex est-il un chien ?").valeur, "inconnu")


class TestMonde(unittest.TestCase):
    def test_deplacements(self):
        r = Raisonneur()
        for p in ["Marie va dans la cuisine.", "Marie est allée au jardin.",
                  "Jean va dans le salon."]:
            r.ajouter(p)
        rep = r.repondre("Où est Marie ?")
        self.assertEqual(rep.valeur, "jardin")
        self.assertEqual(rep.reponse, "Marie est au jardin.")
        self.assertTrue(rep.verifie)

    def test_objet_porte(self):
        r = Raisonneur()
        for p in ["Marie va dans la cuisine.", "Marie prend la pomme.",
                  "Marie va au jardin.", "Jean va dans le salon."]:
            r.ajouter(p)
        rep = r.repondre("Où est la pomme ?")
        self.assertEqual(rep.valeur, "jardin")
        self.assertIn("objet porté", [e["regle"] for e in rep.preuve])
        self.assertEqual(r.repondre("Qui a la pomme ?").valeur, "marie")

    def test_pose_et_don(self):
        r = Raisonneur()
        for p in ["Jean va dans le salon.", "Marie va dans la cuisine.",
                  "Jean prend la pomme.", "Jean donne la pomme à Marie.",
                  "Marie va au jardin."]:
            r.ajouter(p)
        self.assertEqual(r.repondre("Où est la pomme ?").valeur, "jardin")
        self.assertEqual(r.repondre("Qui a la pomme ?").valeur, "marie")
        r.ajouter("Marie pose la pomme.")
        r.ajouter("Marie va dans la cave.")
        rep = r.repondre("Où est la pomme ?")
        self.assertEqual(rep.valeur, "jardin")
        self.assertIn("objet posé", [e["regle"] for e in rep.preuve])
        self.assertEqual(r.repondre("Qui a la pomme ?").valeur, "personne")
        self.assertEqual(r.repondre("Où est Marie ?").valeur, "cave")

    def test_lieu_inconnu_du_porteur(self):
        r = Raisonneur()
        r.ajouter("Jean prend la pomme.")
        self.assertEqual(r.repondre("Qui a la pomme ?").valeur, "jean")
        self.assertEqual(r.repondre("Où est la pomme ?").valeur, "inconnu")

    def test_evenement_incoherent(self):
        r = Raisonneur()
        for p in ["Jean va au jardin.", "Jean prend la pomme.", "Marie va au jardin."]:
            r.ajouter(p)
        res = r.ajouter("Marie pose la pomme.")
        self.assertEqual(res.statut, "refusee")
        self.assertTrue(res.contradictions[0]["verifie"])
        # la base n'a pas changé
        self.assertEqual(r.repondre("Qui a la pomme ?").valeur, "jean")
        res = r.ajouter("Marie prend la pomme.")
        self.assertEqual(res.statut, "refusee")

    def test_temps_des_preuves(self):
        # une preuve reste valable pour l'instant où elle a été produite
        r = Raisonneur()
        r.ajouter("Marie va dans la cuisine.")
        ancienne = r.repondre("Où est Marie ?")
        r.ajouter("Marie va au jardin.")
        self.assertTrue(verifier(ancienne.preuve, r.base)[0])  # t=1 est rejoué à t=1
        self.assertEqual(r.repondre("Où est Marie ?").valeur, "jardin")


class TestArithmetique(unittest.TestCase):
    def test_basique(self):
        r = Raisonneur()
        cas = {"Combien font 12 fois 7 ?": 84, "Combien font 100 plus 23 ?": 123,
               "Combien font 5 moins 9 ?": -4, "Combien font 84 divisé par 7 ?": 12,
               "Combien font 6 × 7 ?": 42, "Combien fait 3 x 4 ?": 12,
               "Combien font 2 plus 3 fois 4 ?": 14,
               "Combien font (2 plus 3) fois 4 ?": 20,
               "Combien font 999999999999 fois 999999999999 ?": 999999999999 ** 2,
               "Combien font 1000000000000 moins 1 ?": 10 ** 12 - 1}
        for q, v in cas.items():
            rep = r.repondre(q)
            self.assertEqual(rep.valeur, v, q)
            self.assertTrue(rep.verifie, q)

    def test_abstention(self):
        r = Raisonneur()
        self.assertEqual(r.repondre("Combien font 7 divisé par 2 ?").valeur, "inconnu")
        self.assertEqual(r.repondre("Combien font 7 divisé par 0 ?").valeur, "inconnu")
        self.assertEqual(r.repondre("Combien font 7 fois ?").valeur, "incompris")
        self.assertEqual(r.repondre("Combien font abc plus 2 ?").valeur, "incompris")

    def test_pas_d_eval(self):
        r = Raisonneur()
        rep = r.repondre("Combien font __import__('os').getcwd() ?")
        self.assertEqual(rep.valeur, "incompris")


class TestContradictions(unittest.TestCase):
    def test_directe(self):
        r = base_socrate()
        res = r.ajouter("Socrate est un oiseau.")
        self.assertEqual(res.statut, "refusee")
        c = res.contradictions[0]
        self.assertTrue(c["verifie"])
        self.assertEqual(c["preuve_a"][-1]["conclusion"], ("est", "socrate", "oiseau"))
        self.assertEqual(c["preuve_b"][-1]["conclusion"], ("nest_pas", "socrate", "oiseau"))
        # la phrase refusée n'est pas entrée dans la base
        self.assertEqual(len(r.base.phrases), 6)
        self.assertEqual(r.repondre("Socrate est-il un oiseau ?").valeur, "non")

    def test_derivee(self):
        r = base_socrate()
        res = r.ajouter("Médor n'est pas un animal.")
        self.assertEqual(res.statut, "refusee")
        self.assertTrue(res.contradictions[0]["verifie"])

    def test_par_regle_ajoutee(self):
        r = Raisonneur()
        for p in ["Rex est un chien.", "Rex n'est pas un animal."]:
            self.assertEqual(r.ajouter(p).statut, "ajoutee")
        res = r.ajouter("Tout chien est un animal.")
        self.assertEqual(res.statut, "refusee")
        self.assertTrue(res.contradictions[0]["verifie"])

    def test_par_disjonction(self):
        r = Raisonneur()
        for p in ["Rex est un chien.", "Tout chien est un animal.", "Rex est un oiseau."]:
            self.assertEqual(r.ajouter(p).statut, "ajoutee")
        res = r.ajouter("Aucun animal n'est un oiseau.")
        self.assertEqual(res.statut, "refusee")

    def test_pas_de_fausse_alerte(self):
        r = base_socrate()
        self.assertEqual(r.ajouter("Socrate n'est pas un chien.").statut, "ajoutee")
        self.assertEqual(r.ajouter("Médor n'est pas un oiseau.").statut, "ajoutee")
        self.assertEqual(r.ajouter("Médor est un chien.").statut, "doublon")

    def test_verifier_contradiction_refait(self):
        r = base_socrate()
        res = r.ajouter("Socrate est un oiseau.")
        c = res.contradictions[0]
        self.assertTrue(rz.verifier_contradiction(c, r.base)[0])
        c2 = copy.deepcopy(c)
        c2["preuve_b"][-1]["conclusion"] = ("nest_pas", "socrate", "chat")
        self.assertFalse(rz.verifier_contradiction(c2, r.base)[0])


class TestVerificateur(unittest.TestCase):
    def setUp(self):
        self.r = base_socrate()
        self.rep = self.r.repondre("Socrate est-il un être vivant ?")

    def test_accepte(self):
        self.assertTrue(verifier(self.rep.preuve, self.r.base)[0])

    def test_conclusion_falsifiee(self):
        p = copy.deepcopy(self.rep.preuve)
        p[-1]["conclusion"] = ("est", "socrate", "dieu")
        self.assertFalse(verifier(p, self.r.base)[0])

    def test_donnee_inventee(self):
        p = copy.deepcopy(self.rep.preuve)
        p[0]["conclusion"] = ("est", "socrate", "dieu")
        ok, raison = verifier(p, self.r.base)
        self.assertFalse(ok)
        self.assertIn("ne dit pas", raison)

    def test_phrase_inexistante(self):
        p = copy.deepcopy(self.rep.preuve)
        p[0]["depend"] = (("phrase", 99),)
        self.assertFalse(verifier(p, self.r.base)[0])

    def test_dependance_future(self):
        p = copy.deepcopy(self.rep.preuve)
        p[2]["depend"] = (("etape", 4), ("etape", 2))
        self.assertFalse(verifier(p, self.r.base)[0])

    def test_regle_inconnue_et_vide(self):
        p = copy.deepcopy(self.rep.preuve)
        p[2]["regle"] = "intuition"
        self.assertFalse(verifier(p, self.r.base)[0])
        self.assertFalse(verifier([], self.r.base)[0])

    def test_but_different(self):
        self.assertFalse(verifier(self.rep.preuve, self.r.base, ("est", "socrate", "dieu"))[0])
        self.assertTrue(verifier(self.rep.preuve, self.r.base,
                                 ("est", "socrate", "etre vivant"))[0])

    def test_calcul_faux(self):
        rep = Raisonneur().repondre("Combien font 12 fois 7 ?")
        p = copy.deepcopy(rep.preuve)
        p[0]["conclusion"] = ("calc", 12, "*", 7, 85)
        self.assertFalse(verifier(p, Raisonneur().base)[0])

    def test_calcul_source_incoherente(self):
        # arithmétique juste, mais qui ne répond pas à l'énoncé
        rep = Raisonneur().repondre("Combien font 12 fois 7 ?")
        p = copy.deepcopy(rep.preuve)
        p[0]["conclusion"] = ("calc", 12, "+", 7, 19)
        self.assertFalse(verifier(p, Raisonneur().base)[0])

    def test_monde_falsifie(self):
        r = Raisonneur()
        for ph in ["Marie va dans la cuisine.", "Marie va au jardin."]:
            r.ajouter(ph)
        rep = r.repondre("Où est Marie ?")
        p = copy.deepcopy(rep.preuve)
        p[-1]["conclusion"] = ("lieu", "marie", "cuisine", 2)
        self.assertFalse(verifier(p, r.base)[0])
        # cite un ancien déplacement alors qu'un autre est plus récent
        p = copy.deepcopy(rep.preuve)
        p[0]["depend"] = (("phrase", 1),)
        p[0]["conclusion"] = ("evt", "va", "marie", "cuisine")
        p[1]["conclusion"] = ("lieu", "marie", "cuisine", 2)
        self.assertFalse(verifier(p, r.base)[0])

    def test_systeme_refuse_une_preuve_rejetee(self):
        # on sabote le démonstrateur : sa preuve est fausse, le système s'abstient
        r = base_socrate()
        original = r._prouver_pos

        def faux(cb, x, c):
            k = original(cb, x, c)
            if k is not None:
                cb.etapes[k - 1]["conclusion"] = ("est", x, "dieu")
            return k
        r._prouver_pos = faux
        rep = r.repondre("Socrate est-il mortel ?")
        self.assertEqual(rep.valeur, "inconnu")
        self.assertFalse(rep.verifie)
        self.assertTrue(rep.reponse.startswith("Je ne sais pas"))


class TestApprentissageContinu(unittest.TestCase):
    def test_ajout_sans_oubli(self):
        r = base_socrate()
        avant = {q: r.repondre(q).valeur for q in
                 ["Socrate est-il mortel ?", "Médor est-il un animal ?",
                  "Socrate est-il un oiseau ?"]}
        self.assertEqual(r.repondre("Médor est-il un être vivant ?").valeur, "inconnu")
        r.ajouter("Tout animal est un être vivant.")
        self.assertEqual(r.repondre("Médor est-il un être vivant ?").valeur, "oui")
        apres = {q: r.repondre(q).valeur for q in avant}
        self.assertEqual(avant, apres)

    def test_sauvegarde_rechargement(self):
        r = base_socrate()
        r2 = Raisonneur()
        r2.charger(r.base.textes())
        self.assertEqual(r2.repondre("Socrate est-il mortel ?").valeur, "oui")
        self.assertEqual(len(r2.base.phrases), len(r.base.phrases))

    def test_ordre_indifferent(self):
        phrases = ["Rex est un chien.", "Tout chien est un mammifère.",
                   "Tout mammifère est un animal."]
        for ordre in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
            r = Raisonneur()
            for i in ordre:
                r.ajouter(phrases[i])
            self.assertEqual(r.repondre("Rex est-il un animal ?").valeur, "oui")


if __name__ == "__main__":
    unittest.main(verbosity=2)
