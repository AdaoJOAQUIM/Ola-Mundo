"""Petits tests : garde, indépendance du SDK, absence d'identifiants de jeu.
Lancer :  python -m unittest test_arc3   (avec le python du venv arc3)"""
import ast
import re
import unittest
from pathlib import Path

ICI = Path(__file__).resolve().parent
POLITIQUES = ["cerveau.py", "aleatoire.py"]


class FauxCourtier:
    def dispatch(self, req):
        return {"ok": True, "value": {"terminal": "ACTIVE", "levels_completed": 0}}


class TestGarde(unittest.TestCase):
    def setUp(self):
        import harnais
        self.h = harnais

    def test_garde_refuse_actions_invalides(self):
        g = self.h.Garde(FauxCourtier())
        act = lambda *a: {"op": "act", "actions": list(a)}
        # phase 0 : seul observe est permis
        with self.assertRaises(self.h.RefusGarde):
            g.appeler(act({"name": "ACTION1", "data": {}}))
        g.observe()
        # phase 1 : RESET, lot de 2 actions, champs en trop -> refus
        for mauvais in (act({"name": "RESET", "data": {}}),
                        act({"name": "ACTION1", "data": {}}, {"name": "ACTION2", "data": {}}),
                        act({"name": "ACTION1"}),
                        {"op": "inconnue"}):
            with self.assertRaises(self.h.RefusGarde):
                g.appeler(mauvais)

    def test_courtier_refuse_clic_sans_coordonnees(self):
        import harnais
        c = harnais.CourtierCompat(harnais.JEUX, "ls20")
        for data in ({}, {"x": 70, "y": 3}, {"x": "1", "y": 2}):
            r = c.dispatch({"op": "act", "actions": [{"name": "ACTION6", "data": data}]})
            self.assertFalse(r["ok"])
        r = c.dispatch({"op": "act", "actions": [{"name": "ACTION9", "data": {}}]})
        self.assertFalse(r["ok"])


class TestPolitique(unittest.TestCase):
    def test_pas_d_import_arc_agi(self):
        for f in POLITIQUES:
            arbre = ast.parse((ICI / f).read_text())
            for n in ast.walk(arbre):
                noms = []
                if isinstance(n, ast.Import):
                    noms = [a.name for a in n.names]
                elif isinstance(n, ast.ImportFrom):
                    noms = [n.module or ""]
                for nom in noms:
                    self.assertFalse(nom.split(".")[0] in ("arc_agi", "arcengine"), f"{f} importe {nom}")

    def test_pas_d_identifiant_de_jeu(self):
        jeux = "ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30".split()
        for f in POLITIQUES:
            txt = (ICI / f).read_text().lower()
            for j in jeux:
                self.assertIsNone(re.search(r"(?<![a-z0-9])" + j + r"(?![a-z0-9])", txt), f"{f} contient l'id {j}")

    def test_action_valide(self):
        import numpy as np
        from cerveau import Cerveau
        obs = {"frame": [np.zeros((64, 64), dtype=int).tolist()], "available_actions": [1, 2, 6],
               "levels_completed": 0, "actions_taken": 0, "state": "NOT_FINISHED"}
        a = Cerveau(0).choisir(obs)
        self.assertIn(a["name"], {"ACTION1", "ACTION2", "ACTION6"})


if __name__ == "__main__":
    unittest.main()
