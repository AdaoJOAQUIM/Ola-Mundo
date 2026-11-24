"""Tests : liste blanche, juge (épinglage, règle d'acceptation), taille du bassin, déterminisme, absence de réseau.
Lancer (python du venv arc3) :  python -m unittest test_competences -v"""
import ast
import json
import os
import socket
import sys
import unittest
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
sys.path.insert(0, str(ICI.parent.parent / "arc3"))
import liste_blanche as LB   # noqa: E402
import juge                  # noqa: E402

BON = '''NOM = "t"
BUDGET = 5
COOLDOWN = 1
def activation(h):
    return True
def agir(h):
    return h.A(1)
def terminer(h):
    return False
'''


def variante(corps_agir):
    return BON.replace("    return h.A(1)", corps_agir)


class TestListeBlanche(unittest.TestCase):
    def test_accepte_competence_minimale_et_celles_du_bassin(self):
        self.assertTrue(LB.verifier(BON)[0])
        for c in json.loads((ICI / "bassin.json").read_text())["competences"]:
            src = (ICI / "skills" / (c["fichier"] + ".py")).read_text()
            ok, code, det = LB.verifier(src)
            self.assertTrue(ok, (c, code, det))

    def test_refus(self):
        interdits = {
            "import": "import os\n" + BON,
            "while": variante("    while True:\n        pass\n    return h.A(1)"),
            "dunder": variante("    return h.__class__"),
            "open": variante("    open('x')\n    return h.A(1)"),
            "exec": variante("    exec('1')\n    return h.A(1)"),
            "attribut_hors_liste": variante("    return h.secret"),
            "lambda": variante("    f = lambda x: x\n    return h.A(1)"),
            "fstring": variante("    s = f'{1}'\n    return h.A(1)"),
            "classe": BON + "\nclass A:\n    pass\n",
            "recursion": BON + "\ndef r(h):\n    return r(h)\n",
            "range_non_borne": variante("    for i in range(100000):\n        pass\n    return h.A(1)"),
            "budget_trop_grand": BON.replace("BUDGET = 5", "BUDGET = 9999"),
            "np_hors_liste": variante("    np.load('x')\n    return h.A(1)"),
            "try": variante("    try:\n        pass\n    except Exception:\n        pass\n    return h.A(1)"),
            "global": variante("    global x\n    return h.A(1)"),
        }
        for nom, src in interdits.items():
            self.assertFalse(LB.verifier(src)[0], nom)

    def test_fonction_manquante(self):
        self.assertFalse(LB.verifier(BON.replace("def terminer(h):\n    return False\n", ""))[0])

    def test_charger_execute_dans_espace_restreint(self):
        ns = LB.charger(BON)
        self.assertNotIn("open", ns["activation"].__globals__["__builtins__"])


class TestJuge(unittest.TestCase):
    def test_empreintes_epinglees(self):
        self.assertEqual(juge.verifier_empreintes(), [])
        ref = json.loads((ICI / "empreintes.json").read_text())
        self.assertEqual(set(ref), {"juge.py", "liste_blanche.py"})
        self.assertEqual(ref["juge.py"], juge.sha256(ICI / "juge.py"))

    def test_juge_refuse_si_empreinte_differente(self):
        vieux = juge.FICHIERS_EPINGLES
        try:
            ref = json.loads((ICI / "empreintes.json").read_text())
            ref_modifie = dict(ref, **{"juge.py": "0" * 64})
            orig = Path.read_text
            Path.read_text = lambda self, *a, **k: json.dumps(ref_modifie) if self.name == "empreintes.json" else orig(self, *a, **k)
            with self.assertRaises(RuntimeError):
                juge.juger([], [])
        finally:
            Path.read_text = orig
            juge.FICHIERS_EPINGLES = vieux

    def stats(self, niv, cout=None):
        return {("a", g): (n, [10] * n if cout is None else cout) for g, n in enumerate(niv)}

    def test_regle_borne_inferieure(self):
        base = self.stats([0] * 10)
        cand = self.stats([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        m = juge.comparer(base, cand)
        self.assertGreater(m["borne_inf"], 0)
        self.assertTrue(juge.decider(m)[0])
        un_seul = self.stats([1] + [0] * 9)
        self.assertFalse(juge.decider(juge.comparer(base, un_seul))[0])   # 1 gain sur 10 : borne <= 0

    def test_regle_efficacite_a_niveaux_egaux(self):
        base = {("a", g): (1, [100]) for g in range(10)}
        meilleur = {("a", g): (1, [80]) for g in range(10)}
        pareil = {("a", g): (1, [95]) for g in range(10)}
        self.assertTrue(juge.decider(juge.comparer(base, meilleur))[0])
        self.assertFalse(juge.decider(juge.comparer(base, pareil))[0])

    def test_aucune_perte_de_niveau(self):
        base = {("a", g): (1, [10]) for g in range(10)}
        cand = {("a", g): ((2, [10, 10]) if g < 6 else (0, [])) for g in range(10)}
        cand[("a", 0)] = (2, [10, 10])
        # le candidat gagne beaucoup mais ne résout plus le niveau 0 dans aucune graine à partir de g>=6 : ici, il
        # le résout encore dans les graines < 6 -> pas de perte
        self.assertEqual(juge.comparer(base, cand)["niveaux_perdus"], [])
        cand2 = {("a", g): (0, []) for g in range(10)}
        m = juge.comparer(base, cand2)
        self.assertEqual(m["niveaux_perdus"], [("a", 0)])
        self.assertFalse(juge.decider(m)[0])

    def test_mode_retrait(self):
        base = {("a", g): (1, [10]) for g in range(10)}
        self.assertTrue(juge.decider(juge.comparer(base, dict(base)), "retrait")[0])
        perte = {("a", g): (0, []) for g in range(10)}
        self.assertFalse(juge.decider(juge.comparer(base, perte), "retrait")[0])

    def test_taille_du_bassin_et_nouvelles_competences(self):
        pool = [{"nom": "c%d" % i, "source": BON.replace('"t"', '"c%d"' % i)} for i in range(21)]
        r = juge.juger([], pool)
        self.assertFalse(r["accepte"])
        self.assertIn("bassin", r["raison"])
        quatre = [{"nom": "n%d" % i, "source": BON} for i in range(4)]
        r = juge.juger([], quatre)
        self.assertFalse(r["accepte"])
        self.assertIn("nouvelles", r["raison"])
        r = juge.juger([], [{"nom": "x", "source": "import os"}])
        self.assertFalse(r["accepte"])

    def test_bassin_actuel_au_plus_20(self):
        b = json.loads((ICI / "bassin.json").read_text())["competences"]
        self.assertLessEqual(len(b), 20)


class TestExecution(unittest.TestCase):
    def pool(self):
        import ronde
        return [{"nom": b["nom"], "source": b["source"]} for b in ronde.charger_bassin()]

    def test_determinisme_des_competences(self):
        import jouer
        pool = self.pool()
        a = jouer.jouer_lot([("ka59", 0, pool, jouer.prieurs_geles() or None, False)], 1)[0]
        b = jouer.jouer_lot([("ka59", 0, pool, jouer.prieurs_geles() or None, False)], 1)[0]
        self.assertEqual(a["trace"], b["trace"])
        self.assertEqual(a["niveaux_actions"], b["niveaux_actions"])
        self.assertGreater(a["actions_competences"], 0)

    def test_bassin_vide_identique_a_la_baseline(self):
        import jouer
        ref = {r["prefixe"]: r for r in json.load(open(ICI.parent.parent / "arc3" / "res_dev_gele.json"))}
        r = jouer.jouer_lot([("ls20", 0, [], jouer.prieurs_geles() or None, False)], 1)[0]
        self.assertEqual(r["niveaux_actions"], ref["ls20"]["niveaux_actions"])

    def test_aucun_reseau_ni_llm_a_l_execution(self):
        interdit = {"socket", "urllib", "http", "requests", "anthropic", "openai", "subprocess", "httpx", "aiohttp", "ftplib"}
        for f in ("eleve.py", "outils.py", "liste_blanche.py", "../../arc3/cerveau.py"):
            arbre = ast.parse((ICI / f).read_text())
            for nd in ast.walk(arbre):
                if isinstance(nd, ast.Import):
                    for a in nd.names:
                        self.assertNotIn(a.name.split(".")[0], interdit, f)
                if isinstance(nd, ast.ImportFrom):
                    self.assertNotIn((nd.module or "").split(".")[0], interdit, f)
        for c in json.loads((ICI / "bassin.json").read_text())["competences"]:
            self.assertTrue(LB.verifier((ICI / "skills" / (c["fichier"] + ".py")).read_text())[0])
        # dynamique : un jeu entier joué dans ce processus avec socket.socket remplacé par une fonction qui lève
        os.chdir(ICI / "runs")
        import harnais, eleve, jouer
        import arc_agi, arcengine  # noqa : imports faits avant de couper le réseau (ssl sous-classe socket)
        orig = socket.socket

        def interdit_socket(*a, **k):
            raise AssertionError("appel réseau interdit à l'exécution")
        socket.socket = interdit_socket
        try:
            res, pol = harnais.jouer("ka59", lambda: eleve.Eleve(0, jouer.prieurs_geles() or None, self.pool()))
        finally:
            socket.socket = orig
        self.assertEqual(res["actions"], 500)


if __name__ == "__main__":
    unittest.main()
