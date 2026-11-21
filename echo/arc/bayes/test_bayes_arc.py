"""Petits tests de bayes_arc.py (python3 test_bayes_arc.py). N'exigent pas le nouvel échantillon TRM."""
import json
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bayes_arc as B  # noqa: E402


def cand(votes=0, rang=0, dsl=False, q=0.5, g=None):
    return {"grille": g or [[votes, rang, int(dsl)]], "votes": votes, "q": q, "rang": rang, "dsl": dsl}


def test_disjoints_et_graine():
    anc = json.load(open(os.path.join(B.ARC, "sous_ensemble_fusion.json")))["taches"]
    nou = json.load(open(os.path.join(B.ICI, "sous_ensemble_bayes.json")))["taches"]
    assert len(nou) == 20 and not set(anc) & set(nou)
    ids = B.toutes_taches()
    # reproduit le protocole d'origine (graine 2025) puis le nouveau (graine 2026)
    assert sorted(random.Random(2025).sample(ids, 20)) == anc
    mono = [i for i in ids if i not in set(anc)
            and len(json.load(open(os.path.join(B.ARC, "data", "evaluation", f"{i}.json")))["test"]) == 1]
    assert sorted(random.Random(2026).sample(mono, 20)) == nou


def test_beta_lissage():
    # 1 juste (votes 5+, dsl) et 3 faux (votes 1) -> a priori (1+1)/(4+2) = 1/3, LR de bucket calculés à la main
    v = B.Verite()
    pool_cal = {"00576224": {"a": cand(8, 1, True, g=None), "b": cand(1, 2), "c": cand(1, 3), "d": cand(1, 4)}}
    gold = v.sortie("00576224")[0]
    pool_cal["00576224"]["a"]["grille"] = gold  # force le candidat « a » juste
    dsl = {"x": {"ok": [True, False]}, "y": {"ok": [False]}}
    p = B.ajuster(pool_cal, dsl, ["x", "y"], B.Verite())
    assert abs(p["prior"] - 2 / 6) < 1e-12
    # P(bucket 4 | juste) = (1+1)/(1+5) ; P(bucket 4 | faux) = (0+1)/(3+5)
    attendu = math.log((2 / 6) / (1 / 8))
    assert abs(p["log_lr"]["bucket"][4] - attendu) < 1e-12
    # DSL : golds couverts 1/2 -> (1+1)/(2+2) ; faux DSL = 1 + 1 = 2 ; faux TRM/tâche = 3 -> pool 2 + 2*3 = 8 -> 3/10
    assert abs(p["dsl"]["p_dsl_juste"] - 0.5) < 1e-12 and abs(p["dsl"]["p_dsl_faux"] - 3 / 10) < 1e-12


def test_garde_anti_fuite():
    v = B.Verite(interdits=["00576224"])
    try:
        v.sortie("00576224")
        assert False, "aurait dû lever"
    except PermissionError:
        pass
    # ajuster ne lit jamais une tâche interdite : le pli de calibration est l'autre pli
    pools = {"11e1fe23": {"a": cand(3, 1)}}
    dsl = {"00576224": {"ok": []}}  # tâche de test absente de ids_hors_test
    B.ajuster(pools, dsl, [], B.Verite(interdits=["00576224"]))


def test_posterieur_monotone_et_dans_01():
    p = {"prior": 0.2, "caracteristiques": ["bucket", "dsl"],
         "log_lr": {"bucket": [-1.0, -0.5, 0.0, 0.7, 1.5], "dsl": [0.0, 0.4]}}
    vals = [B.posterieur(cand(v, 1), p) for v in (0, 1, 2, 3, 6)]
    assert vals == sorted(vals) and all(0 < x < 1 for x in vals)
    assert B.posterieur(cand(3, 1, True), p) > B.posterieur(cand(3, 1, False), p)
    # choisir : 2 meilleurs
    pool = {"a": cand(6, 1), "b": cand(1, 2), "c": cand(3, 3)}
    assert [k for k, _ in B.choisir(pool, p)] == ["a", "c"]


def test_gel_et_empreinte():
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "p.json")
        B.geler({"prior": 0.3}, f, {"m": 1})
        assert B.relire(f)[0]["prior"] == 0.3
        c = json.load(open(f))
        c["params"]["prior"] = 0.9
        json.dump(c, open(f, "w"))
        try:
            B.relire(f)
            assert False
        except ValueError:
            pass


def test_brier_ece():
    parfait = [(1.0, True)] * 5 + [(0.0, False)] * 5
    assert B.brier(parfait) == 0 and B.ece(parfait) == 0
    calibre = [(0.7, True)] * 7 + [(0.7, False)] * 3
    assert abs(B.ece(calibre)) < 1e-12 and abs(B.brier(calibre) - 0.21) < 1e-12
    assert B.ece([(0.9, False)] * 4) > 0.89


def test_vieille_regle_reproduit_8_sur_20():
    anc = json.load(open(os.path.join(B.ARC, "sous_ensemble_fusion.json")))["taches"]
    trm = B.charger_trm(os.path.join(B.ARC, "trm_officiel", "res_K8.json"))
    dsl = B.charger_dsl()
    pub = json.load(open(os.path.join(B.ARC, "fusion", "resultats_fusion.json")))
    v, n = B.Verite(), 0
    for t in anc:
        p = B.pool(t, trm[t], dsl[t])
        gold = v.sortie(t)[0]
        ok = any(p[k]["grille"] == gold for k in B.vieille_regle(p))
        n += ok
        assert ok == pub["detail"][t]["fusion"], t
    assert n == pub["resolus_pass2"]["fusion"] == 8


def test_voc_monotone():
    dsl = {f"t{i}": {"sorties": [[[[i]]]] * (i % 3), "nb_candidats": i % 3, "ok": [i % 2 == 0] * (i % 3), "cpu": 3.0}
           for i in range(30)}
    ids = list(dsl)
    ok_trm = {t: True for t in ids[:10]}
    m = B.modele_voc(dsl, ids[10:], ok_trm, ids[:10])
    assert m["gain"][0] >= m["gain"][1] or m["gain"][1] >= 0
    prev = None
    for lam in (0.0, 1e-4, 1e-3, 1e-2, 1.0):
        n = sum(B.politique(ids, dsl, m, 300.0, lam).values())
        assert prev is None or n <= prev
        prev = n
    assert prev == 0
    # CPU total : sans TRM ne dépasse jamais avec TRM
    ok = {t: True for t in ids}
    cpu_trm = {t: 300.0 for t in ids}
    cpu_dsl = {t: 3.0 for t in ids}
    a = B.evaluer_politique(ids, {t: True for t in ids}, ok, ok, cpu_dsl, cpu_trm)
    b = B.evaluer_politique(ids, {t: False for t in ids}, ok, ok, cpu_dsl, cpu_trm)
    assert b[1] < a[1] and b[1] == 90.0


if __name__ == "__main__":
    n = 0
    for nom, f in sorted(globals().items()):
        if nom.startswith("test_") and callable(f):
            f()
            n += 1
            print("ok", nom)
    print(n, "tests passés")
