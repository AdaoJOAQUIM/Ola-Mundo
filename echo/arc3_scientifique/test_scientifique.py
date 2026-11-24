"""Tests : aucun id de jeu dans le code de l'agent, garde-socket, juge épinglé, déterminisme, inférence de
théories sur des mondes jouets synthétiques. Lancer : python -m pytest test_scientifique.py (python du venv arc3)."""
import hashlib
import json
import re
import socket
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
IDS = "ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30".split()
CODE_AGENT = ["agent.py", "percep.py", "theorie.py", "adapter.py", "run.py", "sommeil.py", "juge.py"]


def test_aucun_id_de_jeu_dans_le_code():
    for f in CODE_AGENT:
        txt = (ICI / f).read_text().lower()
        for i in IDS:
            assert not re.search(r"(?<![a-z0-9])" + re.escape(i) + r"(?![a-z0-9])", txt), (f, i)


def test_juge_epingle():
    emp = json.loads((ICI / "empreintes.json").read_text())
    assert emp["juge.py"] == hashlib.sha256((ICI / "juge.py").read_bytes()).hexdigest()


# ---------------------------------------------------------------- monde jouet (même interface que l'adaptateur)
class MondeJouet:
    """Grille 64x64 : sol 3, murs 4 (bloquent), but 7 (gagne), avatar 3x3 (8 et 9) ; actions 1-4 = haut/bas/gauche/droite
    par pas de 3. Deux niveaux ; but à des endroits différents."""
    NIV = [((30, 30), (30, 48)), ((12, 12), (45, 12))]   # (départ, but)

    def __init__(self, jeu_id="toy", budget=200):
        self.budget, self.n, self.lv = budget, 0, 0
        self._charge()

    def _charge(self):
        f = np.full((64, 64), 4, dtype=np.int8)
        f[6:60, 6:60] = 3
        f[24:27, 6:60] = 4 if self.lv == 1 else 3          # une barre-mur au niveau 2
        f[24:27, 9:12] = 3                                  # avec une porte
        (self.py, self.px), (gy, gx) = self.NIV[self.lv]
        f[gy:gy + 3, gx:gx + 3] = 7
        self.fond = f

    def _frame(self):
        f = self.fond.copy()
        f[self.py:self.py + 3, self.px:self.px + 3] = 8
        f[self.py + 1, self.px + 1] = 9
        return f

    def _obs(self, term="ACTIVE"):
        return {"game_id": "toy", "state": "NOT_FINISHED", "levels_completed": self.lv,
                "available_actions": [1, 2, 3, 4], "frame": [self._frame().tolist()],
                "actions_taken": self.n, "actions_remaining": self.budget - self.n, "terminal": term}

    def observe(self):
        return self._obs()

    def act(self, a):
        self.n += 1
        d = {"ACTION1": (-3, 0), "ACTION2": (3, 0), "ACTION3": (0, -3), "ACTION4": (0, 3)}.get(a["name"])
        if d:
            ny, nx = self.py + d[0], self.px + d[1]
            reg = self.fond[ny:ny + 3, nx:nx + 3]
            if not (reg == 4).any():
                self.py, self.px = ny, nx
                if (reg == 7).any():
                    self.lv += 1
                    if self.lv >= len(self.NIV):
                        return self._obs("WIN")
                    self._charge()
        return self._obs("ACTION_CAP" if self.n >= self.budget else "ACTIVE")

    def resume(self):
        return {"niveaux": self.lv, "actions": self.n}


def _jouer(graine=0, options=None):
    import agent
    import adapter
    env = MondeJouet()
    ag = agent.Scientifique(graine, None, options)
    trace = []
    obs = env.observe()
    while obs["terminal"] == "ACTIVE":
        a = ag.choisir(obs)
        trace.append(json.dumps(a, sort_keys=True))
        obs = env.act(a)
    return env, ag, trace


def test_inference_theorie_monde_jouet():
    env, ag, _ = _jouer()
    assert env.lv == 2, "les deux niveaux du monde jouet doivent être gagnés"
    types = ag.croy.map_types()
    assert types[4] == "block"            # le mur est appris comme bloquant
    assert types[7] == "win"              # le but est inféré comme objectif
    assert types[3] == "pass"
    assert ag.vec[1] and ag._vec(1) == (-3, 0) and ag._vec(4) == (0, 3)


def test_explication_par_elimination():
    from theorie import Croyances, BLOCK, PASS
    c = Croyances()
    c.maj_contact([1, 2], "blocked")      # l'un des deux bloque
    c.maj_contact([1], "moved")           # 1 est traversable -> 2 bloque
    c.maj_contact([1], "moved")
    assert c.get(2).argmax() == BLOCK and c.get(1).argmax() == PASS
    assert c.get(2)[BLOCK] > 0.8


def test_plan_ameliore_exploration_aleatoire():
    e1, a1, _ = _jouer(0)
    e2, a2, _ = _jouer(0, {"explo": "alea"})
    assert e1.n <= e2.n or e2.lv < 2


def test_determinisme():
    _, _, t1 = _jouer(3)
    _, _, t2 = _jouer(3)
    assert t1 == t2


def test_garde_socket(monkeypatch):
    def interdit(*a, **k):
        raise AssertionError("accès réseau interdit pendant le jeu")
    monkeypatch.setattr(socket, "socket", interdit)
    monkeypatch.setattr(socket, "create_connection", interdit)
    env, ag, _ = _jouer()
    assert env.lv == 2
    # l'agent n'importe ni le SDK ni de module réseau
    for f in ("agent.py", "percep.py", "theorie.py"):
        txt = (ICI / f).read_text()
        assert not re.search(r"^\s*(import|from)\s+(arc_agi|arcengine|socket|requests|urllib|http|openai|anthropic)", txt, re.M)


def test_mouvement_translation():
    import percep
    a = np.full((64, 64), 3, dtype=np.int8)
    a[10:15, 10:15] = 9
    b = np.full((64, 64), 3, dtype=np.int8)
    b[10:15, 15:20] = 9
    m = percep.extraire_mouvement(a, b)
    assert m and (m[0]["dy"], m[0]["dx"]) == (0, 5)
