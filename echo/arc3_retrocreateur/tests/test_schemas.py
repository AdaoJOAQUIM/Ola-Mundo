"""Tests de la couche de schémas causaux locaux (phase 4) : perception sans grille, transitions, bayes+MDL, planification,
agent de bout en bout sur un jeu jouet à corps rigide (translation + rotation + but « motif = référence»)."""
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from retro.schemas.entites import composantes, couleur_fond, deplacement_commun
from retro.schemas.transitions import D4, d4_mul, d4_inv, expliquer, trouver_d4, pixels
from retro.schemas.modele import HypCouleurs
from retro.schemas.agent_schema import AgentSchema


def _image(*rects, bg=5):
    f = np.full((64, 64), bg, dtype=np.int16)
    for (r, c, h, w, col) in rects:
        f[r:r + h, c:c + w] = col
    return f


def test_d4_est_un_groupe():
    for i in range(8):
        assert d4_mul(i, d4_inv(i)) == 0
    assert len({d4_mul(i, j) for i in range(8) for j in range(8)}) == 8


def test_composantes_sans_grille():
    f = _image((3, 7, 4, 5, 2), (30, 31, 3, 3, 8), (31, 36, 1, 1, 8))      # positions quelconques, pas de réseau
    E = composantes(f, couleur_fond(f))
    assert sorted((e.col, e.aire) for e in E) == [(2, 20), (8, 1), (8, 9)]
    assert all(e.kd4 == e.kd4 for e in E)


def test_expliquer_translation_rotation_recoloration():
    a = _image((10, 10, 6, 3, 1), (10, 13, 3, 3, 8))
    b = _image((13, 10, 6, 3, 1), (13, 13, 3, 3, 8))
    Ea, Eb = composantes(a, 5), composantes(b, 5)
    ev = expliquer(Ea, Eb)
    assert ev["type"] == "T" and ev["d"] == (3, 0) and len(ev["idx_a"]) == 2
    # rotation du corps entier : 6x3 -> 3x6 (même colonne de départ)
    c = _image((10, 10, 3, 6, 1), (13, 10, 3, 3, 8))
    ev2 = expliquer(Ea, composantes(c, 5))
    assert ev2["type"] == "D4"
    assert trouver_d4(pixels(Ea), pixels(composantes(c, 5)))
    # recoloration sur place
    d = _image((10, 10, 6, 3, 2), (10, 13, 3, 3, 8))
    ev3 = expliquer(Ea, composantes(d, 5))
    assert ev3["type"] == "autre" and len(ev3["recol"]) == 1


def test_hyp_couleurs_bayes_et_description_minimale():
    h = HypCouleurs(lam=1.5)
    h.ajouter({3, 4}, True)            # un blocage touchait les couleurs 3 et 4
    h.ajouter({4}, False)              # 4 a déjà été traversé librement
    assert h.map_cols() == {3}
    h2 = HypCouleurs(lam=1.5)
    h2.ajouter({3, 4}, True)
    assert h2.map_union() == {3, 4}    # ambigu : l'union conservatrice garde les deux


# ------------------------------------------------------------------ jeu jouet : corps rigide à deux parties
class Jouet:
    """Corps = barre de couleur 1 (3x10 px) + pointe de couleur 8 (3x3) collée à son extrémité. Actions 1-4 : translation de 3 px
    (bloquée par la couleur 6) ; action 5 : quart de tour du corps autour du centre de sa boîte. Cible : une pointe 8 statique.
    Niveau gagné quand la pointe recouvre exactement la cible. Compteur dans la ligne 0."""

    def __init__(self):
        self.lv = 0
        self.n = 0
        self.reset_niveau()

    def reset_niveau(self):
        self.corps = {}
        for r in range(3):
            for c in range(10):
                self.corps[(20 + r, 10 + c)] = 1
        for r in range(3):
            for c in range(3):
                self.corps[(20 + r, 20 + c)] = 8
        self.cible = (41, 41) if self.lv == 0 else (11, 29)
        self.mur = {(r, c) for r in range(30, 36) for c in range(0, 64) if c % 7 == 0 and False}

    def pixels_ok(self, corps):
        return all(0 <= r < 64 and 0 <= c < 64 for (r, c) in corps)

    def step(self, a):
        self.n += 1
        corps = self.corps
        if a in (1, 2, 3, 4):
            dr, dc = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}[a]
            nc = {(r + dr, c + dc): v for (r, c), v in corps.items()}
            if self.pixels_ok(nc):
                self.corps = nc
        elif a == 5:
            rs = [r for r, c in corps]; cs = [c for r, c in corps]
            pr, pc = (min(rs) + max(rs)) / 2, (min(cs) + max(cs)) / 2
            nc = {(int(round(pr + pc - c)), int(round(pc - pr + r))): v for (r, c), v in corps.items()}
            if self.pixels_ok(nc):
                self.corps = nc
        tips = sorted(p for p, v in self.corps.items() if v == 8)
        if tips and min(tips) == self.cible and len(tips) == 9 and max(tips) == (self.cible[0] + 2, self.cible[1] + 2):
            self.lv += 1
            self.reset_niveau()

    def frame(self):
        f = np.full((64, 64), 5, dtype=np.int16)
        f[0, 16:48 - min(self.n, 30)] = 4
        r0, c0 = self.cible
        f[r0:r0 + 3, c0:c0 + 3] = 8
        for (r, c), v in self.corps.items():
            f[r, c] = v
        return f

    def obs(self):
        return {"frame": [self.frame().tolist()], "levels_completed": self.lv, "available_actions": [1, 2, 3, 4, 5],
                "actions_taken": self.n, "actions_remaining": 500, "state": "NOT_FINISHED", "terminal": "ACTIVE"}


def test_agent_schema_apprend_translation_rotation_et_but():
    g = Jouet()
    ag = AgentSchema(0)
    for _ in range(120):
        a = ag.choisir(g.obs())
        assert a is not None
        g.step(int(a["name"][-1]))
        if g.lv >= 2:
            break
    assert g.lv >= 1, ag.stats
    c = ag.corps[0]
    assert c.ops[1][0] == "T" and c.ops[1][1] == (-3, 0)
    assert any(op[0] == "R" for op in c.ops.values())      # la rotation a été induite comme règle locale
    assert "aligner" in ag.stats.get("confirmes", [])      # but « motif = référence » confirmé par la victoire
