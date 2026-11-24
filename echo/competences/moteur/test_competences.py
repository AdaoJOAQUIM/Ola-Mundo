"""Tests : liste blanche, disposition/octets, juge epingle, regle du portail, reserve bayesienne, budgets.
Lancer : python3 -m pytest -q test_competences.py   (rapides : pas de modele)"""
import json, os, shutil, sys, tempfile
import numpy as np
import pytest

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import garde_skills as gs
import layout as lay
import juge
import pool as pl
import donnees

OK = '''NAME = "t"
PARENT = "base4"
MODE = "add"
SEUIL = [0.1, 0.2, 0.3]
def applies(L, e, w, r):
    return w < 0.2 and L >= 1
def plan(L, e, w, r):
    t = SEUIL[min(r, 2)]
    return (4, 3 if w < 0.1 else 4, t, 0.0)
def fallback(L, e, w, r, perte):
    return perte > 0.2
'''


def corps(applies="True", plan="(4, 4, 0.0, 0.0)", fb="False", tete='NAME = "t"\nPARENT = "base4"\nMODE = "add"\n'):
    return tete + "def applies(L, e, w, r):\n    return %s\ndef plan(L, e, w, r):\n    return %s\ndef fallback(L, e, w, r, perte):\n    return %s\n" % (applies, plan, fb)


# ------------------------------------------------------------------ liste blanche
def test_base_et_ok():
    assert gs.verifier(gs.BASE4)[0]
    assert gs.verifier(OK)[0]
    c = gs.Competence(OK)
    assert c.applies(2, 5, 0.05, 1) and not c.applies(0, 5, 0.05, 1)
    assert c.plan(2, 5, 0.05, 1) == (4, 3, 0.2, 0.0)
    assert c.fallback(2, 5, 0.05, 1, 0.5) and not c.fallback(2, 5, 0.05, 1, 0.1)


@pytest.mark.parametrize("src,code", [
    ("import os\n" + corps(), "noeud_interdit"),
    (corps(applies="open('x')"), "appel_interdit"),
    (corps(applies="L.__class__"), "attribut_interdit"),
    (corps(applies="().real"), "attribut_interdit"),
    (corps(applies="eval('1')"), "appel_interdit"),
    (corps(applies="L ** 2 > 1"), "noeud_interdit"),
    (corps(plan="[x for x in (1, 2)]"), "noeud_interdit"),
    (corps(applies="(lambda: 1)()"), "noeud_interdit"),
    (corps(applies="'a' == 'a'"), "constante_interdite"),
    (corps(applies="1e9 > 0"), "constante_interdite"),
    (corps(applies="zz > 1"), "nom_inconnu"),
    (corps(tete='NAME = "t"\nPARENT = "base4"\nMODE = "bizarre"\n'), "entete"),
    (corps(tete='NAME = "t"\nMODE = "add"\n'), "entete_manquant"),
    ('NAME = "t"\nPARENT = "base4"\nMODE = "add"\ndef applies(L, e, w, r):\n    while True:\n        pass\n'
     'def plan(L, e, w, r):\n    return (4, 4, 0, 0)\ndef fallback(L, e, w, r, perte):\n    return False\n', "noeud_interdit"),
    ('NAME = "t"\nPARENT = "base4"\nMODE = "add"\ndef applies(L, e, w, r):\n    for i in (1, 2):\n        pass\n    return True\n'
     'def plan(L, e, w, r):\n    return (4, 4, 0, 0)\ndef fallback(L, e, w, r, perte):\n    return False\n', "noeud_interdit"),
    ('NAME = "t"\nPARENT = "base4"\nMODE = "add"\ndef applies(L, e, w, r):\n    return True\n'
     'def plan(L, e, w, r):\n    return (4, 4, 0, 0)\n', "structure"),
    ('NAME = "t"\nPARENT = "base4"\nMODE = "add"\ndef applies(L, e, w, r, z):\n    return True\n'
     'def plan(L, e, w, r):\n    return (4, 4, 0, 0)\ndef fallback(L, e, w, r, perte):\n    return False\n', "structure"),
    (corps(applies="__import__('os')"), "dunder"),
])
def test_rejets(src, code):
    ok, c, d = gs.verifier(src)
    assert not ok and c == code, (c, d)


def test_taille_et_return_manquant():
    assert gs.verifier("x" * 5000)[1] == "taille"
    src = corps().replace("    return False\n", "    x = 1\n")
    assert not gs.verifier(src)[0]
    assert gs.verifier(corps().replace("def fallback(L, e, w, r, perte):\n    return False\n",
                                       "def fallback(L, e, w, r, perte):\n    if perte > 1:\n        return True\n"))[0] is False


def test_pas_d_acces_systeme():
    c = gs.Competence(corps(applies="w < 0.5"))
    assert c.applies.__globals__["__builtins__"] == {}


def test_indexation_table_seulement():
    assert gs.verifier(corps(applies="w < T[0]", tete='NAME = "t"\nPARENT = "base4"\nMODE = "add"\nT = [1, 2]\n'))[0]
    assert not gs.verifier(corps(applies="w < L[0]"))[0]


# ------------------------------------------------------------------ disposition et octets
def test_tailles_layout():
    assert lay.CHUNK == 3538944 == 864 * 4096
    assert lay.GU_BLOCK == 18432 and lay.DN_BLOCK == 36864


def test_octets_dense_egale_naif():
    m_gu, m_dn = np.ones((1, 64), bool), np.ones((1, 32), bool)
    assert lay.slot_bytes(m_gu, m_dn, 4, 4)[0] == lay.NAIVE_SLOT
    assert lay.slot_bytes(m_gu, m_dn, 4, 4, lay.SECTOR)[0] == lay.NAIVE_SLOT


def test_octets_monotones_et_zero():
    g, d = np.ones((1, 64), bool), np.ones((1, 32), bool)
    b4 = lay.slot_bytes(g, d, 4, 4)[0]
    b3 = lay.slot_bytes(g, d, 3, 3)[0]
    b2 = lay.slot_bytes(g, d, 2, 2)[0]
    assert b2 < b3 < b4
    assert lay.slot_bytes(g, d, 0, 4)[0] == 0
    half = np.zeros((1, 64), bool)
    half[0, :32] = True
    assert lay.slot_bytes(half, d, 4, 4)[0] < b4


def test_extensions_egalent_comptage():
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = rng.random((1, 64)) < 0.4
        d = rng.random((1, 32)) < 0.5
        for kg, kd in ((4, 4), (3, 2), (2, 4)):
            ext = lay.extents_for_slot(g[0], d[0], kg, kd)
            assert lay.merged_bytes(ext) == lay.slot_bytes(g, d, kg, kd)[0]
            assert lay.merged_bytes(ext, lay.SECTOR) == lay.slot_bytes(g, d, kg, kd, lay.SECTOR)[0]


def test_donnees_disjointes():
    assert donnees.disjointes()
    assert set(donnees.STARTS) == {"TRAIN", "GATE", "TEST"}


# ------------------------------------------------------------------ juge epingle
def test_empreintes_epinglees_et_detection():
    ref = json.load(open(juge.EMPREINTES))
    assert "juge.py" in ref["fichiers"] and "sim.py" in ref["fichiers"]
    assert juge.verifier_empreintes(gros=False) == []
    tmp = tempfile.mkdtemp()
    try:
        faux = dict(ref)
        faux["fichiers"] = dict(ref["fichiers"], **{"juge.py": "0" * 64})
        p = os.path.join(tmp, "e.json")
        json.dump(faux, open(p, "w"))
        assert "juge.py" in juge.verifier_empreintes(p, gros=False)
    finally:
        shutil.rmtree(tmp)


def test_constantes_du_juge_epinglees():
    assert (juge.SEUIL_OCTETS_PTS, juge.SEUIL_PPL, juge.SEUIL_ACCORD, juge.MAX_CANDIDATES) == (2.0, 1.02, 0.95, 3)


# ------------------------------------------------------------------ portail (avec evaluateur factice)
class Faux:
    """Moteur factice : le resultat depend d'un nombre ecrit dans la source (BITS_PCT, PPL, ACC)."""
    backend = "cm"


def _fake_eval(monkeypatch, table):
    def evaluer(eng, sources, nom, **k):
        key = tuple(gs.Competence(s).name for s in sources)
        pct, ppl, acc = table[key]
        rng = np.random.default_rng(0)
        base = rng.integers(0, 100, 1000)
        arg = base.copy()
        arg[: int(round((1 - acc) * 1000))] += 1
        return dict(pct_naive=pct, pct_naive_sector=pct, ppl=ppl, argmax=arg, faults=0)
    monkeypatch.setattr(juge, "evaluer", evaluer)
    monkeypatch.setattr(juge, "reference", lambda eng, nom: dict(ppl=10.0))
    monkeypatch.setattr(juge, "verifier_empreintes", lambda *a, **k: [])


def _src(nom, mode="add", parent="base4"):
    return corps(tete='NAME = "%s"\nPARENT = "%s"\nMODE = "%s"\n' % (nom, parent, mode))


def test_portail_regles(monkeypatch):
    base = gs.BASE4
    T = {("base4",): (100.0, 10.0, 1.0),
         ("a", "base4"): (97.9, 10.1, 0.99),     # gain 2.1 : passe
         ("b", "base4"): (98.1, 10.1, 0.99),     # gain 1.9 : octets insuffisants
         ("c", "base4"): (80.0, 10.3, 0.99),     # ppl +3 % : rejet
         ("d", "base4"): (70.0, 10.1, 0.94),     # accord 94 % : rejet
         ("e", "base4"): (90.0, 10.2, 0.95)}     # limites : ppl exactement +2 %, accord 95 % : passe
    _fake_eval(monkeypatch, T)
    v = juge.juger(Faux(), [base], {n: _src(n) for n in "abc"})
    c = v["candidates"]
    assert c["a"]["passe"] and not c["b"]["c_octets"] and not c["c"]["c_ppl"]
    assert v["accepte"] == "a"
    v = juge.juger(Faux(), [base], {n: _src(n) for n in "ade"})
    c = v["candidates"]
    assert not c["d"]["c_accord"] and c["e"]["passe"] and v["accepte"] == "e"   # meilleure parmi celles qui passent
    v2 = juge.juger(Faux(), [base], {n: _src(n) for n in "bcd"})
    assert v2["accepte"] is None


def test_portail_max_trois_et_rejets(monkeypatch):
    _fake_eval(monkeypatch, {("base4",): (100.0, 10.0, 1.0), ("a", "base4"): (90.0, 10.0, 1.0),
                             ("b", "base4"): (91.0, 10.0, 1.0)})
    cands = {"a": _src("a"), "b": _src("b"), "bad": "import os", "z": _src("z")}
    v = juge.juger(Faux(), [gs.BASE4], cands)
    assert v["candidates"]["z"]["rejet"] == "trop_de_candidates"
    assert v["candidates"]["bad"]["rejet"].startswith("liste_blanche")
    assert v["accepte"] == "a"


def test_portail_parent_absent(monkeypatch):
    _fake_eval(monkeypatch, {("base4",): (100.0, 10.0, 1.0)})
    v = juge.juger(Faux(), [gs.BASE4], {"z": _src("z", "replace", "inexistant")})
    assert v["candidates"]["z"]["rejet"].startswith("parent_absent")


def test_remplacement(monkeypatch):
    _fake_eval(monkeypatch, {("base4",): (100.0, 10.0, 1.0), ("a", "base4"): (90.0, 10.0, 1.0),
                             ("a2", "base4"): (80.0, 10.0, 1.0)})
    v = juge.juger(Faux(), [_src("a"), gs.BASE4], {"x": _src("a2", "replace", "a")}) if False else None
    pool2, err = juge.assembler([_src("a"), gs.BASE4], _src("a2", "replace", "a"))
    assert err is None and [gs.Competence(s).name for s in pool2] == ["a2", "base4"]


# ------------------------------------------------------------------ reserve bayesienne
def test_posterieur():
    m, sd, p = pl.posterieur([3.0, 3.2, 2.8, 3.1])
    assert 2.5 < m < 3.2 and p > 0.99
    m0, _, p0 = pl.posterieur([])
    assert m0 == 0 and p0 == 0.5
    m1, _, p1 = pl.posterieur([-1.0, -1.2, -0.9])
    assert m1 < 0 and p1 < 0.1


def test_elagage_et_plafond(tmp_path):
    r = pl.Reserve(str(tmp_path / "p.json"))
    r.ajouter("base4.py", "base4", "add", "base4")
    r.ajouter("faible.py", "faible", "add", "base4")
    r.ajouter("bonne.py", "bonne", "add", "base4")
    r.observer("faible", [0.05, 0.0, 0.1, -0.05])
    r.observer("bonne", [4.0, 4.2, 3.9, 4.1])
    out = r.elaguer()
    assert out == ["faible"] and "faible.py" not in r.fichiers and "base4.py" in r.fichiers
    r2 = pl.Reserve(str(tmp_path / "q.json"))
    r2.ajouter("base4.py", "base4", "add", "base4")
    for i in range(15):
        r2.ajouter("s%d.py" % i, "s%d" % i, "add", "base4")
        r2.observer("s%d" % i, [1.0 + i * 0.1] * 4)
    r2.elaguer()
    assert len(r2.fichiers) <= pl.PLAFOND and "base4.py" in r2.fichiers


def test_remplacement_dans_reserve(tmp_path):
    r = pl.Reserve(str(tmp_path / "p.json"))
    r.ajouter("base4.py", "base4", "add", "base4")
    r.ajouter("a.py", "a", "add", "base4")
    r.ajouter("a2.py", "a2", "replace", "a")
    assert r.fichiers == ["a2.py", "base4.py"]


# ------------------------------------------------------------------ budgets
def test_budget_memoire_du_moteur():
    sys.path.insert(0, os.path.abspath(os.path.join(ICI, "..", "..", "monde")))
    import moteur as mt
    b = mt.Budget(4 * 2 ** 30)
    b.declare("noyau", 2 * 2 ** 30)
    with pytest.raises(AssertionError):
        b.declare("trop", 3 * 2 ** 30)


def test_budget_disque_nouveau():
    tot = 0
    for p in ("/home/user/monde_data/cm",):
        if os.path.isdir(p):
            tot += sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p))
    assert tot <= 8 * 10 ** 9
