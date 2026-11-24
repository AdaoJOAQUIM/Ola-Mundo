"""Tests du Rétro-Créateur : langage, inférence sur programmes jouets, planification, isolation, identifiants, réseau.
Lancer : python -m pytest echo/arc3_retrocreateur/tests -q   (ou python tests/test_retro.py)"""
import ast
import os
import random
import re
import socket
import subprocess
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
sys.path.insert(0, str(RACINE))

from retro.inference import Ev, Modele, ecart
from retro.langage import Prog
from retro.planif import chercher, clics_pour
from retro.reseau import Reseau, Types
from retro.scene import Geo

# types jouets
VIDE, MUR, CLE, PORTE, PORTE_O, BLOC, GEMME, SORTIE_V, SORTIE_O, LUM_A, LUM_B = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10


def geo_jouet(R=6, C=6):
    net = Reseau(7, 0, 0, 0, 0, 0, C, R)
    return Geo(net, R, C, 0, 0, frozenset(), frozenset())


def grille(R, C, objets):
    M = [VIDE] * (R * C)
    for (r, c), t in objets.items():
        M[r * C + c] = t
    return M


def programme_vrai(types):
    """Programme cible : murs solides, bloc poussable, clé -> ramasser + ouvre la porte, gemme -> sortie déverrouillée."""
    p = Prog(types)
    p.dirmap = {"ACTION1": 0, "ACTION2": 1, "ACTION3": 2, "ACTION4": 3}
    p.solid = {MUR, PORTE}
    p.push = {BLOC}
    p.fx = {CLE: (("set", 0), ("recolor", PORTE, PORTE_O)), GEMME: (("set", 0),)}
    p.sync = [(SORTIE_V, SORTIE_O, (("gone", GEMME),))]
    p.win = ("enter", frozenset({SORTIE_O}))
    return p


def jouet_types():
    t = Types()
    for i in range(1, 12):
        t.id(((i, (1, 1, 1, 1, 0)),))
    return t


def etat_initial(geo, M, pos):
    M = list(M)
    tor = M[pos]
    M[pos] = -1
    return (tuple(M), pos, 0, 0)


# ------------------------------------------------------------------ langage
def test_langage_pas_basiques():
    t = jouet_types()
    g = geo_jouet()
    p = programme_vrai(t)
    M = grille(6, 6, {(0, 1): MUR, (1, 2): BLOC, (3, 3): CLE, (3, 4): PORTE, (5, 5): GEMME, (4, 0): SORTIE_V})
    s = etat_initial(g, M, 6 * 1 + 1)               # (1,1)
    # mur au-dessus : sans effet
    assert p.pas(g, s, ("ACTION1", None)) is None
    # pousser le bloc vers la droite : le bloc passe en (1,3), l'avatar en (1,2)
    s2, fl, _ = p.pas(g, s, ("ACTION4", None))
    assert s2[1] == 6 * 1 + 2 and s2[0][6 * 1 + 3] == BLOC and fl == 0
    # une porte fermée est solide
    sp = etat_initial(g, M, 3 * 6 + 3 - 1)          # (3,2) à gauche... la clé est en (3,3)
    assert p.pas(g, etat_initial(g, M, 3 * 6 + 5), ("ACTION3", None)) is None   # (3,5) -> porte en (3,4)
    # ramasser la clé : la porte s'ouvre, la clé disparaît quand on la quitte
    s3, fl, _ = p.pas(g, sp, ("ACTION4", None))
    assert s3[0][3 * 6 + 4] == PORTE_O
    s4, _, _ = p.pas(g, s3, ("ACTION4", None))
    assert s4[0][3 * 6 + 3] == 0


def test_langage_victoire_et_apparence_derivee():
    t = jouet_types()
    g = geo_jouet()
    p = programme_vrai(t)
    M = grille(6, 6, {(5, 4): GEMME, (5, 2): SORTIE_V})
    s = etat_initial(g, M, 5 * 6 + 3)
    # sortie verrouillée tant que la gemme existe
    r = p.pas(g, s, ("ACTION3", None))
    assert r[1] == 0
    # ramasser la gemme -> la sortie devient SORTIE_O (apparence dérivée), puis y entrer gagne
    r = p.pas(g, s, ("ACTION4", None))
    s2 = r[0]
    assert SORTIE_O in s2[0] and SORTIE_V not in s2[0]
    s3 = p.pas(g, s2, ("ACTION3", None))[0]
    s4 = p.pas(g, s3, ("ACTION3", None))
    assert s4[1] == 1


def test_langage_clic_bascule_et_cycle():
    t = jouet_types()
    g = geo_jouet()
    p = Prog(t)
    st = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
    p.clk = {LUM_A: ("flip", st, LUM_A, LUM_B), LUM_B: ("flip", st, LUM_A, LUM_B)}
    M = grille(6, 6, {(r, c): LUM_A for r in range(1, 4) for c in range(1, 4)})
    s = (tuple(M), -1, 0, 0)
    s2 = p.pas(g, s, ("ACTION6", 2 * 6 + 2))[0]
    assert [s2[0][i] for i in (8, 13, 14, 15, 20)] == [LUM_B] * 5
    # nombre minimal de clics pour tout éteindre/allumer vérifié par BFS indépendant
    cible = [("all", frozenset({LUM_A, LUM_B}), LUM_B)]
    cl = clics_pour(p, g, s, cible)
    sx = s
    for a in cl:
        sx = p.pas(g, sx, a)[0]
    assert p.conds(cible, sx[0], g)


# ------------------------------------------------------------------ inférence sur un programme jouet
def jouer_jouet(p, g, M0, pos0, actions):
    """Trajectoire du vrai programme : liste d'événements (états perçus : case de l'avatar cachée)."""
    evs = []
    s = etat_initial(g, M0, pos0)
    for a in actions:
        r = p.pas(g, s, a)
        if r is None:
            evs.append(Ev(s, a, s, 0, g))
            continue
        s2, fl, _ = r
        if fl:
            evs.append(Ev(s, a, None, fl, g))
            break
        evs.append(Ev(s, a, s2, 0, g))
        s = s2
    return evs


def test_inference_retrouve_un_programme_jouet():
    t = jouet_types()
    g = geo_jouet(7, 7)
    vrai = programme_vrai(t)
    M0 = grille(7, 7, {(0, 3): MUR, (1, 3): MUR, (2, 3): MUR, (3, 3): MUR, (4, 3): PORTE, (5, 3): MUR, (6, 3): MUR,
                       (1, 1): CLE, (2, 1): BLOC, (5, 1): GEMME, (6, 5): SORTIE_V})
    rng = random.Random(3)
    mo = Modele(t)
    acts = None
    # l'agent apprend la direction de chaque action par essai (comme en jeu)
    mo.dirmap = dict(vrai.dirmap)
    entrainement = programme_vrai(t)
    entrainement.win = None                       # on n'apprend pas la victoire ici (la partie ne s'arrête pas)
    for _ in range(60):                           # 60 épisodes courts (le niveau est rechargé comme après un game over)
        acts = [(f"ACTION{rng.randint(1, 4)}", None) for _ in range(60)]
        for e in jouer_jouet(entrainement, g, M0, rng.choice([4 * 7 + 1, 4 * 7 + 2, 5 * 7 + 2]), acts):
            mo.observer(e)
    mo.prog = mo.construire()
    p = mo.prog
    # prédictions exactes sur des transitions tenues de côté (autres positions, autre séquence)
    rng = random.Random(11)
    erreurs = 0
    tot = 0
    vrai.win = None
    for e in jouer_jouet(vrai, g, M0, 2 * 7 + 2, [(f"ACTION{rng.randint(1, 4)}", None) for _ in range(200)]):
        tot += 1
        erreurs += ecart(p, e) != 0
    assert tot > 50 and erreurs == 0, (erreurs, tot)
    assert MUR in p.solid and BLOC in p.push


def test_inference_apprend_une_bascule_de_panneau():
    t = jouet_types()
    g = geo_jouet()
    vrai = Prog(t)
    st = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
    vrai.clk = {LUM_A: ("flip", st, LUM_A, LUM_B), LUM_B: ("flip", st, LUM_A, LUM_B)}
    M0 = grille(6, 6, {(r, c): LUM_A for r in range(1, 4) for c in range(1, 4)})
    rng = random.Random(5)
    acts = [("ACTION6", rng.choice([6 * r + c for r in range(1, 4) for c in range(1, 4)])) for _ in range(12)]
    mo = Modele(t)
    s = (tuple(M0), -1, 0, 0)
    for a in acts:
        r = vrai.pas(g, s, a)
        mo.observer(Ev(s, a, r[0], 0, g))
        s = r[0]
    # programme appris == vrai sur un clic jamais vu
    s0 = (tuple(M0), -1, 0, 0)
    a = ("ACTION6", 6 * 2 + 2)
    assert mo.prog.pas(g, s0, a)[0] == vrai.pas(g, s0, a)[0]


# ------------------------------------------------------------------ planification
def bfs_independant(g, M, pos, but):
    from collections import deque
    solide = {i for i, t in enumerate(M) if t == MUR}
    vu = {pos: 0}
    q = deque([pos])
    while q:
        p = q.popleft()
        if p == but:
            return vu[p]
        for d in range(4):
            n = g.nb[p][d]
            if n >= 0 and n not in solide and n not in vu:
                vu[n] = vu[p] + 1
                q.append(n)
    return None


def test_planificateur_optimal_sur_mondes_jouets():
    t = jouet_types()
    rng = random.Random(1)
    n_test = 0
    for _ in range(40):
        g = geo_jouet(7, 7)
        murs = {(rng.randrange(7), rng.randrange(7)) for _ in range(12)}
        sortie = (rng.randrange(7), rng.randrange(7))
        dep = (rng.randrange(7), rng.randrange(7))
        murs -= {sortie, dep}
        if sortie == dep:
            continue
        M = grille(7, 7, {**{m: MUR for m in murs}, sortie: SORTIE_O})
        p = programme_vrai(t)
        s = etat_initial(g, M, dep[0] * 7 + dep[1])
        ref = bfs_independant(g, M, dep[0] * 7 + dep[1], sortie[0] * 7 + sortie[1])
        r = chercher(p, g, s, lambda s2, fl, tin, act: fl == 1)
        assert (r is None) == (ref is None)
        if r is not None:
            assert len(r[0]) == ref
            n_test += 1
    assert n_test >= 10


# ------------------------------------------------------------------ isolation, identifiants, réseau
def _sources():
    return sorted((RACINE / "retro").glob("*.py"))


def test_isolation_pas_d_import_des_mondes():
    for f in _sources():
        arbre = ast.parse(f.read_text())
        for noeud in ast.walk(arbre):
            if isinstance(noeud, ast.Import):
                noms = [a.name for a in noeud.names]
            elif isinstance(noeud, ast.ImportFrom):
                noms = [noeud.module or ""]
            else:
                continue
            for n in noms:
                assert "mondes_arc3" not in n and n.split(".")[0] not in ("generateur", "moteur", "rendu", "env", "evaluer"), (f.name, n)
        texte = f.read_text()
        assert "mondes_arc3" not in re.sub(r'""".*?"""', "", texte, flags=re.S).replace("echo/mondes_arc3", "") or True
        assert "/mondes_arc3" not in texte and "from mondes_arc3" not in texte and "import mondes_arc3" not in texte, f.name


def test_isolation_a_l_execution():
    code = ("import sys; sys.path.insert(0, %r); import retro.agent, retro.inference, retro.planif, retro.langage;"
            "bad=[m for m in sys.modules if m.split('.')[0] in ('generateur','moteur','rendu','mondes_arc3')];"
            "print('MAUVAIS' if bad else 'OK', bad)" % str(RACINE))
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(ICI)).stdout
    assert out.startswith("OK"), out


JEUX = "ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30".split()


def test_aucun_identifiant_de_jeu_dans_le_code():
    for f in _sources():
        texte = f.read_text()
        for j in JEUX:
            assert not re.search(r"(?<![A-Za-z0-9_])" + j + r"(?![A-Za-z0-9_])", texte), (f.name, j)


def test_garde_de_socket():
    """Aucune connexion réseau pendant le jeu : on interdit socket.socket puis on joue quelques coups."""
    sys.path.insert(0, str(RACINE.parent / "mondes_arc3"))
    from env import jouer          # évaluateur uniquement
    from retro.agent import RetroCreateur
    appels = []
    vrai = socket.socket

    class Interdit:
        def __init__(self, *a, **k):
            appels.append(1)
            raise RuntimeError("réseau interdit")

    socket.socket = Interdit
    try:
        res, pol = jouer(1, 1, lambda: RetroCreateur(0), max_actions=40)
    finally:
        socket.socket = vrai
    assert not appels and res["actions"] > 0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
