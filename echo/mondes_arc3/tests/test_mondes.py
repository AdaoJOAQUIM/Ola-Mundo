"""Tests : conformité d'interface (vs harnais/courtier), solvabilité, déterminisme, mécaniques.

Lancer :  cd echo/mondes_arc3 && python -m pytest tests -q
"""
import inspect
import pickle
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ICI = Path(__file__).resolve().parent.parent
ECHO = ICI.parent
sys.path.insert(0, str(ICI))

from actions import EtatJeu, FrameData, GameAction  # noqa: E402
from env import CourtierMonde, MondeEnv, jouer  # noqa: E402
from generateur import MECA_NIVEAU, MECANIQUES, Monde, gen_niveau  # noqa: E402
from moteur import CLIC0, Moteur, Niveau, oracle  # noqa: E402

SEEDS = list(range(0, 36))


def diff(s):
    return 1 + s % 3


@pytest.fixture(scope="module")
def mondes():
    return {s: Monde(s, diff(s)) for s in SEEDS}


# ----------------------------------------------------------------------------- conformité
def test_actions_noms():
    assert [a.name for a in GameAction] == ["RESET"] + [f"ACTION{i}" for i in range(1, 8)]
    for nom in ("RESET", "ACTION1", "ACTION6", "ACTION7"):
        assert GameAction[nom].name == nom
    assert {e.name for e in EtatJeu} == {"NOT_FINISHED", "WIN", "GAME_OVER"}


def test_signatures_env():
    sig = inspect.signature(MondeEnv.step)
    assert list(sig.parameters)[:3] == ["self", "action", "data"]
    assert sig.parameters["data"].default is None
    assert list(inspect.signature(MondeEnv.reset).parameters) == ["self"]
    assert list(inspect.signature(MondeEnv.__init__).parameters)[:3] == ["self", "seed", "difficulte"]
    for m in ("action1", "action2", "action3", "action4", "action5", "action6", "action7"):
        assert hasattr(MondeEnv, m)


def test_frame_et_types():
    e = MondeEnv(3, 2)
    f = e.reset()
    assert isinstance(f, FrameData)
    assert isinstance(f.frame, list) and len(f.frame) >= 1
    g = np.asarray(f.frame[-1])
    assert g.shape == (64, 64) and g.min() >= 0 and g.max() <= 15
    assert f.state.name in ("NOT_FINISHED", "WIN", "GAME_OVER")
    assert isinstance(f.levels_completed, int) and f.levels_completed == 0
    assert all(isinstance(a, int) and 0 <= a <= 7 for a in f.available_actions)
    assert 3 <= f.win_levels <= 8


def _courtier_reel():
    """Importe le vrai courtier du harnais (prime-agent) ; None si indisponible."""
    pa = ECHO / "arc" / "externes" / "prime-agent"
    if not (pa / "arc3_local_broker.py").exists():
        return None
    sys.path.insert(0, str(pa))
    try:
        import arc3_local_broker as b
        return b
    except Exception:
        return None


def _script(rng, n=60, dispo=(1, 2, 3, 4, 5, 6, 7)):
    out = []
    for _ in range(n):
        a = rng.choice(dispo)
        if a == 6:
            out.append({"name": "ACTION6", "data": {"x": rng.randrange(64), "y": rng.randrange(64)}})
        else:
            out.append({"name": f"ACTION{a}", "data": {}})
    return out


def test_courtier_identique_au_courtier_reel():
    """Le vrai Broker (code du dépôt), piloté sur NOTRE env via le même _apply que CourtierCompat,
    produit exactement les mêmes observations que CourtierMonde."""
    b = _courtier_reel()
    if b is None:
        pytest.skip("courtier réel indisponible")
    # faux module arcengine : GameAction = la nôtre (le harnais fait `from arcengine import GameAction`)
    faux = types.ModuleType("arcengine")
    faux.GameAction = GameAction
    sys.modules.setdefault("arcengine", faux)
    sys.path.insert(0, str(ECHO / "arc3"))
    import importlib
    harnais = importlib.import_module("harnais")

    for seed in (3, 6, 10):
        env = MondeEnv(seed, diff(seed))
        reel = harnais.CourtierCompat.__new__(harnais.CourtierCompat)
        reel.handle = b.Handle(game_id=env.game_id, env=env)
        reel.handle.last = env.reset()
        mien = CourtierMonde(seed, diff(seed))
        assert reel._observation() == mien._observation()
        rng = random.Random(seed)
        dispo = [a for a in mien.handle.env.monde.actions_disponibles()]
        for act in _script(rng, 80, dispo):
            r1 = reel.dispatch({"op": "act", "actions": [act]})
            r2 = mien.dispatch({"op": "act", "actions": [act]})
            assert r1 == r2
        # lots, refus de lots trop grands, actions invalides : mêmes réponses
        for q in ({"op": "act", "actions": []}, {"op": "act", "actions": [{"name": "NOPE"}]},
                  {"op": "act", "actions": [{"name": "ACTION6", "data": {"x": 99, "y": 0}}]}, {"op": "zzz"}):
            assert reel.dispatch(q) == mien.dispatch(q)


def test_garde_et_jouer_harnais():
    """Le Garde du harnais pilote CourtierMonde ; le dict de résultat a les clés de harnais.jouer."""
    sys.path.insert(0, str(ECHO / "arc3"))
    try:
        import harnais
    except Exception:
        pytest.skip("harnais indisponible")
    g = harnais.Garde(CourtierMonde(3, 1))
    o = g.observe()
    assert set(o) == {"game_id", "state", "levels_completed", "available_actions", "frame",
                      "actions_taken", "actions_remaining", "terminal"}
    o = g.act([{"name": "ACTION1", "data": {}}])
    assert o["actions_taken"] == 1

    sys.path.insert(0, str(ECHO / "arc3"))
    import aleatoire
    res, _ = jouer(3, 1, lambda: aleatoire.Aleatoire(0), max_actions=100)
    for cle in ("jeu", "etat", "niveaux", "niveaux_total", "actions", "niveaux_actions", "niveaux_reference",
                "cpu_s", "cpu_cerveau_s", "mur_s"):
        assert cle in res
    assert res["actions"] <= 100 and res["etat"] in ("ACTION_CAP", "WIN")


def test_cerveau_tourne_sans_modification():
    sys.path.insert(0, str(ECHO / "arc3"))
    import cerveau
    res, _ = jouer(0, 1, lambda: cerveau.Cerveau(0), max_actions=120)
    assert res["actions"] <= 120


def test_actions_hors_disponibles_sans_effet():
    e = MondeEnv(1, 2)       # monde "panel" : seules RESET et ACTION6
    f0 = e.reset().frame[0].copy()
    for nom in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"):
        f = e.step(GameAction[nom])
        assert (f.frame[0] == f0).all() and f.state is EtatJeu.NOT_FINISHED


# ----------------------------------------------------------------------------- solvabilité
def test_nb_niveaux_et_progression(mondes):
    ouv = []
    for s, W in mondes.items():
        assert 3 <= W.nl <= 8
        opts = W.optimum()
        assert all(o >= 1 for o in opts)
        ouv.append((np.mean(opts[: max(1, W.nl // 2)]), np.mean(opts[W.nl // 2:])))
    deb, fin = np.mean([a for a, b in ouv]), np.mean([b for a, b in ouv])
    assert fin > deb      # difficulté croissante en moyenne


def test_solution_rejouee_par_l_env(mondes):
    """L'optimum de l'oracle, rejoué via l'interface publique, gagne chaque niveau en `opt` actions."""
    for s, W in mondes.items():
        e = MondeEnv(s, diff(s))
        f = e.reset()
        total = 0
        for k in range(W.nl):
            sol = e.actions_solution(k)
            assert len(sol) == W.niveau(k).opt
            for nom, data in sol:
                f = e.step(GameAction[nom], data if nom == "ACTION6" else None)
                total += 1
                assert f.state is not EtatJeu.GAME_OVER, (s, k)
            assert f.levels_completed == k + 1
        assert f.state is EtatJeu.WIN and total == sum(W.optimum())


def test_optimum_confirme_par_bfs_independant(mondes):
    n = 0
    for s, W in mondes.items():
        for k in range(W.nl):
            lv = W.niveau(k)
            if lv.panel:       # l'oracle décompose panneau/navigation : BFS exact trop large
                continue
            b = Moteur(lv).bfs_optimum(cap=60000)
            if b is None:
                continue
            assert b == lv.opt, (s, k, b, lv.opt)
            n += 1
    assert n >= 40


def test_optimum_panneau_decompose_confirme():
    """Pour les niveaux à panneau, l'optimum décomposé = A* conjoint (petits cas)."""
    vus = 0
    for s in range(60):
        W = Monde(s, diff(s))
        if W.goal == "panel" or not ({"lumieres", "motif"} & W.meca):
            continue
        for k in range(min(2, W.nl)):
            lv = W.niveau(k)
            r = Moteur(lv).resoudre(cap=60000)
            if r is not None:
                assert r[0] == lv.opt
                vus += 1
        if vus >= 6:
            break
    assert vus >= 3


# ----------------------------------------------------------------------------- déterminisme
def test_determinisme_generation():
    for s in (0, 7, 11, 20):
        W1, W2 = Monde(s, diff(s)), Monde(s, diff(s))
        assert (W1.nl, W1.goal, W1.meca, W1.dirperm, W1.col, W1.forme) == (W2.nl, W2.goal, W2.meca, W2.dirperm,
                                                                           W2.col, W2.forme)
        k = W1.nl - 1
        a = pickle.dumps(gen_niveau(W1, k))
        b = pickle.dumps(gen_niveau(W2, k))
        assert a == b
        # indépendant de l'ordre de génération
        assert pickle.dumps(gen_niveau(Monde(s, diff(s)), 0)) == pickle.dumps(gen_niveau(W1, 0))


def test_determinisme_env():
    rng = random.Random(5)
    for s in (2, 3, 6):
        script = _script(rng, 120)
        traces = []
        for _ in range(2):
            e = MondeEnv(s, diff(s))
            f = e.reset()
            tr = [f.frame[0].tobytes()]
            for act in script:
                nom = act["name"]
                f = e.step(GameAction[nom], act["data"] if nom == "ACTION6" else None)
                tr.append((f.frame[0].tobytes(), f.state.name, f.levels_completed))
                if f.state is EtatJeu.GAME_OVER:
                    f = e.reset()
            traces.append(tr)
        assert traces[0] == traces[1]


def test_graines_differentes_regles_differentes():
    perms = {tuple(Monde(s, 1).dirperm) for s in range(30)}
    buts = {Monde(s, 1).goal for s in range(30)}
    assert len(perms) >= 6 and buts == {"sortie", "cibles", "panel"}
    assert Monde(1, 1).meca != Monde(2, 1).meca or Monde(1, 1).col != Monde(2, 1).col


# ----------------------------------------------------------------------------- mécaniques
def test_toutes_les_mecaniques_apparaissent():
    vu_monde, vu_niveau = set(), set()
    for s in range(120):
        W = Monde(s, diff(s))
        vu_monde |= W.meca
        if s < 60:
            for k in range(W.nl):
                vu_niveau |= W.niveau(k).presentes()
    manque = set(MECANIQUES) - vu_monde
    assert not manque, manque
    assert len(MECANIQUES) >= 14
    manque_niv = (set(MECANIQUES) - {"annuler"}) - vu_niveau - {"attente"}
    assert not manque_niv, manque_niv


def test_mecaniques_declarees_presentes_a_chaque_niveau(mondes):
    for s, W in mondes.items():
        requis = {m for m in W.meca & MECA_NIVEAU if m not in ("energie", "attente")}
        if W.goal == "panel":
            requis &= {"lumieres", "motif"}
        for k in range(W.nl):
            lv = W.niveau(k)
            assert requis <= lv.presentes(), (s, k, requis - lv.presentes())
            if "energie" in W.meca:
                assert lv.energy and lv.energy > lv.opt
            if "ennemis" in W.meca:
                assert lv.enemies


def test_regles_communes_aux_niveaux(mondes):
    for s, W in mondes.items():
        gr = {W.niveau(k).grav for k in range(W.nl)} - {None}
        assert gr <= {W.grav}
        sel = {W.niveau(k).selection for k in range(W.nl)}
        assert len(sel) == 1


# --- règles unitaires sur de petits niveaux écrits à la main (n=5, pos = y*5+x)
def P(x, y, n=5):
    return y * n + x


def mk(**kw):
    lv = Niveau(5)
    lv.pieces = (P(0, 0),)
    lv.exits = ((P(4, 0), 0),)
    for k, v in kw.items():
        setattr(lv, k, v)
    return lv


HAUT, BAS, GAU, DRO = 0, 1, 2, 3


def jouer_seq(lv, seq):
    M = Moteur(lv)
    s = M.etat0()
    fl = 0
    for a in seq:
        r = M.pas(s, a)
        if r is None:
            continue
        s, fl = r
        if fl:
            break
    return s, fl, M


def test_regle_mur_et_sortie():
    lv = mk(solid=frozenset({P(2, 0)}))
    s, fl, _ = jouer_seq(lv, [DRO, DRO, DRO, DRO])
    assert fl == 0 and s[0] == (P(1, 0),)
    s, fl, _ = jouer_seq(lv, [BAS, DRO, DRO, DRO, HAUT, DRO])
    assert fl == 1


def test_regle_cle_porte():
    lv = mk(doors={P(2, 0): 0}, keys=((P(0, 1), 0),))
    s, fl, _ = jouer_seq(lv, [DRO, DRO, DRO, DRO])
    assert fl == 0 and s[0] == (P(1, 0),)
    _, fl, _ = jouer_seq(lv, [BAS, HAUT, DRO, DRO, DRO, DRO])
    assert fl == 1


def test_regle_bloc_pousse_et_gravite():
    lv = mk(blocks=(P(1, 0),))
    s, fl, M = jouer_seq(lv, [DRO])
    assert s[2] == (P(2, 0),)
    s, fl, M = jouer_seq(lv, [DRO, DRO, DRO, DRO])        # bloc contre le bord : bloqué
    assert s[2] == (P(3, 0),)
    lv = mk(pieces=(P(0, 3),), blocks=(P(1, 3),), solid=frozenset({P(1, 4)}), grav=BAS)
    assert Moteur(lv).etat0()[2] == (P(1, 3),)           # posé sur le mur
    s, fl, M = jouer_seq(lv, [DRO])
    assert s[2] == (P(2, 4),)            # poussé hors du support, le bloc tombe jusqu'au sol


def test_regle_collectibles_verrou_sortie():
    lv = mk(gems=(P(0, 1),))
    s, fl, _ = jouer_seq(lv, [DRO] * 4)
    assert fl == 0                                    # sortie verrouillée
    _, fl, _ = jouer_seq(lv, [BAS, HAUT] + [DRO] * 4)
    assert fl == 1


def test_regle_danger_mort():
    lv = mk(hazards=frozenset({P(1, 0)}))
    _, fl, _ = jouer_seq(lv, [DRO])
    assert fl == 2


def test_regle_teleporteur():
    lv = mk(tele={P(1, 0): P(3, 3), P(3, 3): P(1, 0)})
    s, fl, _ = jouer_seq(lv, [DRO])
    assert s[0] == (P(3, 3),)


def test_regle_interrupteur_bascule_mur():
    lv = mk(swalls={P(2, 0): (0, 0)}, switches={P(0, 1): 0})
    s, fl, _ = jouer_seq(lv, [DRO, DRO])
    assert s[0] == (P(1, 0),)
    _, fl, _ = jouer_seq(lv, [BAS, HAUT, DRO, DRO, DRO, DRO])
    assert fl == 1


def test_regle_couleurs():
    lv = mk(couleurs=True, exits=((P(4, 0), 1),), paint={P(0, 1): 1}, col0=0)
    _, fl, _ = jouer_seq(lv, [DRO] * 4)
    assert fl == 0
    _, fl, _ = jouer_seq(lv, [BAS, HAUT] + [DRO] * 4)
    assert fl == 1


def test_regle_ennemi_patrouille_et_attente():
    tr = (P(2, 0), P(2, 1), P(2, 0), P(2, 0))        # va-et-vient déterministe
    lv = mk(enemies=(tr,))
    M = Moteur(lv)
    s = M.etat0()
    r = M.pas(s, 4)
    assert r is not None and r[1] == 0               # attendre fait avancer le temps
    # même séquence => même résultat
    assert jouer_seq(lv, [4, DRO, 4, DRO])[0] == jouer_seq(lv, [4, DRO, 4, DRO])[0]
    lv2 = mk(enemies=((P(0, 0), P(0, 0)),))
    _, fl, _ = jouer_seq(lv2, [4])
    assert fl == 2                                    # l'ennemi est sur la case du joueur au tic suivant


def test_regle_lumieres_et_motif():
    from generateur import _lmasks
    k = 3
    lm = _lmasks(3)
    pn = dict(kind="lumieres", cells=tuple(P(x, y) for y in range(3) for x in range(3)), k=3, K=2, ref=(),
              init=lm[4], goal=0)
    lv = Niveau(5)
    lv.goal = "panel"
    lv.panel = pn
    s, fl, _ = jouer_seq(lv, [CLIC0 + pn["cells"][4]])
    assert fl == 1                                    # le clic central éteint la croix
    pm = dict(kind="motif", cells=tuple(P(x, y) for y in range(3) for x in range(3)), k=3, K=3, ref=(),
              init=(0,) * 9, goal=(0, 2, 0, 0, 0, 0, 0, 0, 1))
    lv = Niveau(5)
    lv.goal = "panel"
    lv.panel = pm
    r = oracle(lv)
    assert r[0] == 3


def test_regle_selection_puis_deplacement():
    lv = mk(pieces=(P(0, 0), P(0, 4)), selection=True, couleurs=True, piece_cols=(0, 1),
            exits=((P(4, 0), 0), (P(4, 4), 1)),
            panel=None)
    M = Moteur(lv)
    s = M.etat0()
    s, _ = M.pas(s, DRO)
    assert s[0] == (P(1, 0), P(0, 4))
    s, _ = M.pas(s, CLIC0 + P(0, 4))                   # sélection de la 2e pièce
    s, _ = M.pas(s, DRO)
    assert s[0] == (P(1, 0), P(1, 4))
    assert oracle(lv)[0] == 9              # 4+4 pas + 1 clic de sélection


def test_energie_epuisee_game_over():
    for s in range(40):
        W = Monde(s, diff(s))
        if "energie" in W.meca and W.goal == "sortie":
            break
    e = MondeEnv(s, diff(s))
    f = e.reset()
    lv = e.lv
    # oscille entre deux actions de déplacement jusqu'à épuisement (certaines peuvent être sans effet)
    for _ in range(lv.energy * 4 + 10):
        for nom in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            f = e.step(GameAction[nom])
            if f.state is EtatJeu.GAME_OVER or f.levels_completed:
                break
        if f.state is EtatJeu.GAME_OVER or f.levels_completed:
            break
    assert f.state is EtatJeu.GAME_OVER or f.levels_completed >= 1
    if f.state is EtatJeu.GAME_OVER:
        f2 = e.step(GameAction.ACTION1)                 # inerte après game over
        assert f2.state is EtatJeu.GAME_OVER
        f3 = e.step(GameAction.RESET)
        assert f3.state is EtatJeu.NOT_FINISHED and f3.levels_completed == 0


def test_reset_garde_les_niveaux_et_annuler():
    s = next(s for s in range(60) if "annuler" in Monde(s, 1).meca and Monde(s, 1).goal != "panel")
    e = MondeEnv(s, 1)
    f = e.reset()
    for nom, data in e.actions_solution(0):
        f = e.step(GameAction[nom], data if nom == "ACTION6" else None)
    assert f.levels_completed == 1
    g1 = e.step(GameAction.RESET)
    assert g1.levels_completed == 1                     # reset = recommencer le niveau courant
    base = g1.frame[0].copy()
    for nom in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        e.step(GameAction[nom])
    f = e.step(GameAction.ACTION7)
    n = 0
    while n < 6:
        f = e.step(GameAction.ACTION7)
        n += 1
    assert (f.frame[0] == base).all()                   # l'annulation rembobine jusqu'à l'état initial


def test_paliers_durs_solubles_et_plus_longs():
    """Difficultés 4 et 5 : toujours solubles (solution rejouée) et optimum moyen supérieur au palier 1."""
    moy = {}
    for d in (1, 4, 5):
        opts = []
        for s in range(3000, 3003):
            e = MondeEnv(s, d)
            f = e.reset()
            for k in range(e.monde.nl):
                for nom, data in e.actions_solution(k):
                    f = e.step(GameAction[nom], data if nom == "ACTION6" else None)
                assert f.levels_completed == k + 1
                opts.append(e.monde.niveau(k).opt)
        moy[d] = np.mean(opts)
    assert moy[4] > moy[1] and moy[5] > moy[1]
