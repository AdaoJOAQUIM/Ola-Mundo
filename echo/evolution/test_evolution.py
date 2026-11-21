"""Tests du moteur d'évolution. Lancer : python3 test_evolution.py (ou pytest)."""
import math
import os
import random
import shutil
import sys
import tempfile

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)

import donnees  # noqa: E402
import garde_fous  # noqa: E402
import genome_initial  # noqa: E402
import juge  # noqa: E402
import mutations  # noqa: E402

ENTETE = "def estimer(n, N, d, u1, u2, K):\n"


def _gen(corps):
    return ENTETE + "".join("    %s\n" % l for l in corps.strip().split("\n"))


MALVEILLANTS = {
    "open": "p = open('/etc/passwd')\nreturn p",
    "__import__": "p = __import__('os')\nreturn p",
    "import os": "import os\np = 0 * n[0]\nreturn p",
    "np.load": "p = np.load('cache/c.npz')\nreturn p",
    "n.__class__": "p = n.__class__\nreturn p",
    "exec": "p = exec('x = 1')\nreturn p",
    "lambda": "p = (lambda x: x)(n[0])\nreturn p",
    "while": "p = 0 * n[0]\nwhile True:\n    p = p + 1.0\nreturn p",
    "getattr": "p = getattr(n, 'sum')\nreturn p",
    "eval": "p = eval('1')\nreturn p",
    "compréhension": "p = [x for x in range(3)]\nreturn p",
    "np.save": "p = np.save('x.npy', n)\nreturn p",
    "attribut .sum": "p = n.sum()\nreturn p",
    "puissance géante": "p = 9 ** 9 ** 9 + 0 * n[0]\nreturn p",
    "chaîne": "p = 'abc'\nreturn p",
    "dunder nom": "p = __builtins__\nreturn p",
    "boucle non range": "for k in n:\n    p = k\nreturn p",
}


def test_liste_blanche_graine():
    ok, code, _ = garde_fous.verifier(genome_initial.SOURCE)
    assert ok, code


def test_liste_blanche_rejette_malveillants():
    assert len(MALVEILLANTS) >= 8
    for nom, corps in MALVEILLANTS.items():
        # le corps est écrit à la main : on construit la source directement
        src = _gen(corps)
        ok, code, detail = garde_fous.verifier(src)
        assert not ok, "génome malveillant accepté : %s" % nom
    # et jamais exécutés par le juge
    jeu = {"n": np.zeros((1, 2, 3), np.float32), "N": np.zeros((1, 2, 1), np.float32),
           "d": np.zeros((1, 2, 1), np.float32), "u1": np.zeros((1, 2, 1), np.float32),
           "u2": np.zeros((1, 2, 1), np.float32), "y": np.zeros(2, np.int64)}
    r = juge.juger(_gen(MALVEILLANTS["open"]), jeu)
    assert not r["ok"]


def test_taille_et_syntaxe():
    assert not garde_fous.verifier("def estimer(:\n")[0]
    gros = genome_initial.SOURCE + "# " + "x" * 5000 + "\n"
    assert garde_fous.verifier(gros)[1] == "taille"


def _jeu_petit(T=50, A=7, K=2):
    rng = np.random.default_rng(0)
    n = rng.integers(0, 5, size=(K + 1, T, A)).astype(np.float32)
    return {"n": n, "N": n.sum(2, keepdims=True), "d": (n > 0).sum(2, keepdims=True).astype(np.float32),
            "u1": (n == 1).sum(2, keepdims=True).astype(np.float32),
            "u2": (n == 2).sum(2, keepdims=True).astype(np.float32),
            "y": rng.integers(0, A, size=T)}


def test_juge_constant_donne_log2_A():
    jeu = _jeu_petit()
    r = juge.juger(_gen("p = 0 * n[0] + 3.0\nreturn p"), jeu)
    assert r["ok"] and abs(r["bpc"] - math.log2(7)) < 1e-9, r
    # sur les vraies données A
    ab = donnees.charger_ab()
    A = len(ab["alphabet"])
    r = juge.juger(_gen("p = 0 * n[0] + 1.0\nreturn p"), ab["A"])
    assert abs(r["bpc"] - math.log2(A)) < 1e-9, r


def test_juge_rejette_nan_negatif_forme():
    jeu = _jeu_petit()
    assert juge.juger(_gen("p = 0 * n[0] + np.log1p(-2.0)\nreturn p"), jeu)["raison"] == "non_fini"
    assert juge.juger(_gen("p = 0 * n[0] - 1.0\nreturn p"), jeu)["raison"] == "negatif"
    assert juge.juger(_gen("p = N[0] * 1.0\nreturn p"), jeu)["raison"] == "forme_incorrecte"
    assert juge.juger(_gen("p = n[0] / 0 + 1.0\nreturn p"), jeu)["raison"] in ("non_fini", "forme_incorrecte")
    # lignes nulles : lissage, pas de rejet
    r = juge.juger(_gen("p = 0 * n[0]\nreturn p"), jeu)
    assert r["ok"] and abs(r["bpc"] - math.log2(7)) < 1e-9


def test_evaluateur_en_processus():
    jeu = _jeu_petit()
    ev = garde_fous.Evaluateur({"A": jeu}, nproc=2, delai=10)
    try:
        res = ev.evaluer_lot([genome_initial.SOURCE, _gen(MALVEILLANTS["open"]),
                              _gen("p = 0 * n[0] - 1.0\nreturn p")], "A")
    finally:
        ev.fermer()
    assert res[0]["ok"] and not res[1]["ok"] and res[2]["raison"] == "negatif"
    assert ev.rejets["negatif"] == 1


def test_empreintes_detectent_modification():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "cache"))
        for f in ("juge.py", "garde_fous.py", "donnees.py"):
            shutil.copy(os.path.join(ICI, f), os.path.join(tmp, f))
        for f in ("cache/ab.npz", "cache/c.npz"):
            with open(os.path.join(tmp, f), "wb") as fh:
                fh.write(b"donnees factices " + f.encode())
        ref = os.path.join(tmp, "empreintes.json")
        garde_fous.epingler(tmp, ref)
        assert garde_fous.verifier_empreintes(tmp, ref) == []
        with open(os.path.join(tmp, "juge.py"), "a") as fh:
            fh.write("\n# petite modification\n")
        assert garde_fous.verifier_empreintes(tmp, ref) == ["juge.py"]
        with open(os.path.join(tmp, "cache/ab.npz"), "ab") as fh:
            fh.write(b"x")
        assert set(garde_fous.verifier_empreintes(tmp, ref)) == {"juge.py", "cache/ab.npz"}
    finally:
        shutil.rmtree(tmp)


def test_empreintes_du_depot_intactes():
    assert garde_fous.verifier_empreintes() == []


def test_200_mutations_parsent():
    import ast
    rng = random.Random(7)
    meta = mutations.meta_defaut()
    pop = [genome_initial.SOURCE]
    valides = 0
    for _ in range(200):
        par = rng.choice(pop)
        src, noms, ok, raison = mutations.muter(par, meta, rng, pop)
        ast.parse(src)  # doit toujours être du Python valide
        if ok:
            valides += 1
            pop.append(src)
    assert valides >= 150, valides


def test_meta_heritee_borne():
    rng = random.Random(3)
    m = mutations.meta_defaut()
    for _ in range(500):
        m = mutations.meta_heritee(m, rng)
    assert 1.0 <= m["nmut"] <= 3.0 and 0.02 <= m["sigma"] <= 2.0
    assert all(0.02 <= w <= 20 for w in m["poids"].values())


def test_comptages_contre_force_brute():
    lib, val = donnees.lire_textes()
    alphabet = donnees.alphabet_de(lib, val)
    ta, tb, tc = donnees.decouper(val)
    ab = donnees.charger_ab()
    assert "".join(ab["alphabet"]) == alphabet
    rng = random.Random(11)
    verifies = 0
    for i in range(30):
        nom, texte = (("A", ta), ("B", tb))[i % 2]
        jeu = ab[nom]
        k = rng.choice([0, 1, 2, 3, 4, 6, 8, 10])
        t = rng.randrange(k, len(texte))
        ctx = texte[t - k:t]
        if i % 3 == 0:
            s = int(np.argmax(jeu["n"][k][t]))      # symbole le plus fréquent après ce contexte
        elif i % 3 == 1:
            s = int(jeu["y"][t])                   # symbole réel
        else:
            s = rng.randrange(len(alphabet))        # symbole au hasard
        attendu = donnees.comptage_brut(lib, ctx, alphabet[s])
        assert jeu["n"][k][t][s] == attendu, (nom, t, k, ctx, alphabet[s], jeu["n"][k][t][s], attendu)
        verifies += 1
    # début de tranche : les ordres plus longs que t ont N = 0
    assert ab["A"]["N"][5][2][0] == 0 and ab["A"]["N"][0][0][0] > 0
    assert verifies == 30


def test_moteur_ne_touche_pas_c():
    for f in ("moteur.py", "lancer_ablation.py", "lancer_ablation2.py", "mutations.py", "juge.py", "composants.py"):
        txt = open(os.path.join(ICI, f), encoding="utf-8").read()
        assert "c.npz" not in txt and "CHEMIN_C" not in txt and "import tester" not in txt, f
    # le moteur n'importe que charger_ab (A et B)
    assert "charger_ab" in open(os.path.join(ICI, "moteur.py"), encoding="utf-8").read()


# ---------------------------------------------------------------- composants ajoutés
def test_pareto_jouet():
    import composants as cp
    # (bpc, taille) : 0 et 2 et 3 forment le front ; 1 est dominé par 0 ; 4 par 2
    objs = [(1.0, 5.0), (1.5, 6.0), (2.0, 2.0), (3.0, 1.0), (2.5, 2.5)]
    assert cp.front_pareto(objs) == [0, 2, 3]
    rang = cp.tri_non_domine(objs)
    assert rang[0] == rang[2] == rang[3] == 0 and rang[1] == 1 and rang[4] == 1
    assert cp.domine((1, 1), (2, 2)) and not cp.domine((1, 2), (2, 1)) and not cp.domine((1, 1), (1, 1))
    d = cp.distance_encombrement(objs, [0, 2, 3])
    assert d[0] == d[3] == float("inf") and 0 < d[2] < float("inf")
    # ajouter un point qui domine tout change le front
    assert cp.front_pareto(objs + [(0.5, 0.5)]) == [5]


def test_arret_precoce():
    import composants as cp
    a = cp.ArretPrecoce(delta=0.01, patience=300)
    assert not a.mise_a_jour(0, 1.60)
    assert not a.mise_a_jour(100, 1.50)      # vrai gain : on repart de 100
    assert not a.mise_a_jour(250, 1.495)     # gain < delta : ne compte pas
    assert not a.mise_a_jour(399, 1.495)
    assert a.mise_a_jour(400, 1.495) and "patience" in a.raison
    b = cp.ArretPrecoce(delta=0.01, patience=None, cible=1.5)
    assert not b.mise_a_jour(10, 1.6) and b.mise_a_jour(20, 1.49) and b.raison == "cible_atteinte"


def test_soi_apprend_et_se_calibre():
    import composants as cp
    rng = np.random.default_rng(0)
    s = cp.Soi(min_exemples=100)
    poids = np.zeros(len(s.ops) + 5)
    poids[0] = 2.0          # l'opérateur 0 réussit plus souvent
    poids[-1] = -2.5        # biais : succès rares
    for _ in range(1500):
        x = np.zeros(len(poids))
        x[rng.integers(0, len(s.ops))] = 1.0
        x[-1] = 1.0
        p_vrai = 1 / (1 + np.exp(-(x @ poids)))
        y = rng.random() < p_vrai
        p_pred = s.predire(x)
        s.observer(x, p_pred, [], y)
        s.reentrainer()
    r = s.rapport()
    assert r["ECE_prequentiel"] < 0.05, r
    assert r["Brier_prequentiel"] <= r["Brier_constant_taux_base"] + 1e-3, r
    x0 = np.zeros(len(poids)); x0[0] = 1; x0[-1] = 1
    x1 = np.zeros(len(poids)); x1[1] = 1; x1[-1] = 1
    assert s.predire(x0) > s.predire(x1)
    # saut : p très faible -> on saute, sauf exploration epsilon (~10 %)
    import random
    s.w = np.zeros(len(poids)); s.w[-1] = -8.0
    rr = random.Random(1)
    evalues = sum(s.decider(x1, rr)[0] for _ in range(2000))
    assert 100 < evalues < 320, evalues


def test_cout_mesure():
    import composants as cp
    jeu = _jeu_petit(T=400, A=30)
    ev = cp.EvaluateurMesure({"A": jeu}, nproc=2, delai=10)
    try:
        res = ev.evaluer_lot([genome_initial.SOURCE, _gen(MALVEILLANTS["open"])], "A")
    finally:
        ev.fermer()
    assert res[0]["ok"] and res[0]["cpu_s"] > 0
    assert not res[1]["ok"] and res[1]["cpu_s"] == 0.0


def test_mdl_et_descripteurs_map_elites():
    import composants as cp
    petit = genome_initial.SOURCE
    gros = petit.replace("return p", "p = p + 0.1 * np.sqrt(p) + 0.2 * np.tanh(p) - 0.3 * np.abs(p)\n    return p")
    assert cp.taille_mdl_bits(gros) > cp.taille_mdl_bits(petit)
    assert cp.descripteur(petit)[1:] == (0, 0, 0)
    remise = petit.replace("n[k] + e * p", "np.maximum(n[k] - 0.7, 0.0) + e * p")
    assert "remise" in cp.caracteristiques(remise) and "remise" not in cp.caracteristiques(petit)
    u = petit.replace("e = d[k] * 1.0", "e = d[k] + u1[k]")
    assert "utilise_u1" in cp.caracteristiques(u)
    carte = cp.MapElites()
    assert carte.inserer({"source": petit, "id": 0}, 2.0)
    assert not carte.inserer({"source": petit, "id": 1}, 2.5)   # même case, moins bon
    assert carte.inserer({"source": petit, "id": 2}, 1.9)       # même case, meilleur
    assert carte.inserer({"source": remise, "id": 3}, 2.1)      # autre case
    assert carte.ids() == [2, 3] and abs(carte.couverture() - 2 / cp.NB_CASES_TOTAL) < 1e-12


def test_moteur_complet_court_et_arrets():
    import moteur
    tmp = tempfile.mkdtemp()
    try:
        r = moteur.executer("t_complet", 90, 1, "auto", dossier=tmp, verbeux=False, complet=True)
        assert r["raison_arret"] == "budget" and r["cpu_total_s"] > 0
        for cle in ("soi", "front_pareto", "elites", "cpu_par_evaluation_s", "gain_par_cpu_s"):
            assert cle in r, cle
        assert len(r["front_pareto"]) >= 1
        r2 = moteur.executer("t_cible", 10**5, 1, "auto", dossier=tmp, verbeux=False, cible=10.0)
        assert r2["raison_arret"] == "cible_atteinte"
        r3 = moteur.executer("t_cpu", 10**5, 1, "auto", dossier=tmp, verbeux=False, budget_cpu=8.0)
        assert r3["raison_arret"].startswith("budget_cpu")
        r4 = moteur.executer("t_pat", 10**5, 1, "auto", dossier=tmp, verbeux=False, patience=100, delta=10.0)
        assert r4["raison_arret"].startswith("patience")
    finally:
        shutil.rmtree(tmp)


def lancer():
    tests = [(nom, f) for nom, f in sorted(globals().items()) if nom.startswith("test_") and callable(f)]
    echecs = 0
    for nom, f in tests:
        try:
            f()
            print("OK    ", nom)
        except Exception as e:  # noqa
            echecs += 1
            print("ECHEC ", nom, "->", repr(e)[:300])
    print("%d tests, %d échec(s)" % (len(tests), echecs))
    return echecs


if __name__ == "__main__":
    sys.exit(1 if lancer() else 0)
