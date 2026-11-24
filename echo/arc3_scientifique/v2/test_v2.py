"""Tests v2 : pas d'id de jeu, garde-socket, déterminisme, chemins de retour du graphe sur mondes jouets,
modèle d'effet, modèle de valeur, motif de référence. Lancer : python3 -m pytest test_v2.py"""
import re
import socket
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import chemin  # noqa: E402,F401
from graphe import Graphe, RESET, hacher  # noqa: E402

IDS = "ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30".split()
CODE = ["bibliotheque.py", "campagne.py", "adaptateur_monde.py", "adaptateur_large.py", "agrege.py", "agent2.py", "effet.py", "graphe.py", "buts.py", "run2.py", "chemin.py", "pretrain.py"]


def test_aucun_id_de_jeu_dans_le_code():
    for f in CODE:
        p = ICI / f
        if not p.exists():
            continue
        txt = p.read_text().lower()
        for i in IDS:
            assert not re.search(r"(?<![a-z0-9])" + re.escape(i) + r"(?![a-z0-9])", txt), (f, i)


# ------------------------------------------------------------------ monde jouet avec avatar (interface de l'adaptateur)
class MondeJouet:
    """Sol 3, murs 4, but 7, avatar 3x3 ; actions 1-4 = haut/bas/gauche/droite par pas de 3 ; 2 niveaux."""
    NIV = [((30, 30), (30, 45)), ((12, 12), (45, 12))]

    def __init__(self, jeu_id="toy", budget=200):
        self.budget, self.n, self.lv = budget, 0, 0
        self._charge()

    def _charge(self):
        f = np.full((64, 64), 4, dtype=np.int8)
        f[6:60, 6:60] = 3
        (self.py, self.px), (gy, gx) = self.NIV[self.lv]
        f[gy:gy + 3, gx:gx + 3] = 7
        self.fond = f
        self.but = (gy, gx)

    def _frame(self):
        f = self.fond.copy()
        f[self.py:self.py + 3, self.px:self.px + 3] = 8
        f[self.py + 1, self.px + 1] = 9
        return f

    def _obs(self, term="ACTIVE"):
        return {"game_id": "toy", "state": "NOT_FINISHED", "levels_completed": self.lv, "available_actions": [1, 2, 3, 4],
                "frame": [self._frame().tolist()], "actions_taken": self.n, "actions_remaining": self.budget - self.n,
                "terminal": term}

    def observe(self):
        return self._obs()

    def act(self, a):
        self.n += 1
        name = a["name"]
        if name == "RESET":
            self._charge()
        else:
            k = int(name[-1])
            dy, dx = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}.get(k, (0, 0))
            ny, nx = self.py + dy, self.px + dx
            if 6 <= ny and ny + 3 <= 60 and 6 <= nx and nx + 3 <= 60:
                self.py, self.px = ny, nx
            if (self.py, self.px) == self.but:
                self.lv += 1
                if self.lv >= len(self.NIV):
                    return self._obs("WIN")
                self._charge()
        term = "ACTION_CAP" if self.n >= self.budget else "ACTIVE"
        return self._obs(term)

    def resume(self):
        return {"niveaux": self.lv, "actions": self.n}

    def ferme(self):
        pass


def _jouer(graine, budget=200, options=None):
    from agent2 import Scientifique2
    env = MondeJouet(budget=budget)
    ag = Scientifique2(graine, options or {})
    obs = env.observe()
    seq = []
    while obs["terminal"] == "ACTIVE":
        a = ag.choisir(obs)
        seq.append((a["name"], tuple(sorted(a["data"].items()))))
        obs = env.act(a)
    return env, ag, seq


def test_determinisme_selon_graine():
    _, _, s1 = _jouer(3, 120)
    _, _, s2 = _jouer(3, 120)
    assert s1 == s2
    _, _, s3 = _jouer(4, 120)
    assert s3 != s1 or True      # graines différentes : pas d'exigence, seulement pas de plantage


def test_monde_jouet_resolu_et_garde_socket(monkeypatch):
    def interdit(*a, **k):
        raise AssertionError("accès réseau interdit au moment du jeu")
    monkeypatch.setattr(socket.socket, "connect", interdit)
    monkeypatch.setattr(socket, "create_connection", interdit)
    monkeypatch.setattr(socket, "getaddrinfo", interdit)
    env, ag, _ = _jouer(0, 300)
    assert env.lv >= 1


def test_pas_d_import_reseau():
    for f in CODE:
        p = ICI / f
        if p.exists():
            txt = p.read_text()
            for mot in ("import requests", "import urllib", "import http", "import socket", "anthropic", "openai"):
                assert mot not in txt, (f, mot)


# ------------------------------------------------------------------ graphe : chemins de retour
def _chaine(n, sens_unique=()):
    g = Graphe()
    g.racine = 0
    for i in range(n):
        g.noeuds[i] = __import__("graphe").Noeud(i, None, (1, 2), i)
    for i in range(n - 1):
        g.noeuds[i].arcs[("A", 1)] = i + 1
        if i not in sens_unique:
            g.noeuds[i + 1].arcs[("A", 2)] = i
    return g


def test_retour_par_arcs_connus():
    g = _chaine(10)
    dist, par = g.chemins(9)
    r = Graphe.route(par, 9, 6)
    assert r == [("A", 2)] * 3 and dist[6] == 3


def test_retour_par_reset_quand_moins_cher():
    g = _chaine(8, sens_unique=range(8))      # aucun retour en arrière possible
    dist, par = g.chemins(6)
    r = Graphe.route(par, 6, 2)
    assert r == [RESET, ("A", 1), ("A", 1)] and dist[2] == 3
    # vers la racine : RESET seul
    assert Graphe.route(par, 6, 0) == [RESET]


def test_reset_choisi_seulement_si_plus_court():
    g = _chaine(12)                            # retour possible
    dist, par = g.chemins(10)
    assert Graphe.route(par, 10, 9) == [("A", 2)]
    # nœud proche de la racine mais loin de la position : RESET + rejeu gagne contre 9 retours
    dist, par = g.chemins(10)
    assert Graphe.route(par, 10, 1) == [RESET, ("A", 1)]
    assert dist[1] == 2


def test_arcs_mortels_evites():
    g = _chaine(5)
    g.marquer_mortel(2, ("A", 1))
    dist, _ = g.chemins(0)
    assert 3 not in dist or dist[3] > 3        # 3 n'est atteignable que par RESET-rejeu impossible : coupé
    assert 4 not in dist


def test_hachage_masque_ignore_compteur():
    a = np.zeros((64, 64), dtype=np.int8)
    b = a.copy()
    b[0, :5] = 7
    m = np.zeros((64, 64), dtype=bool)
    m[0, :] = True
    assert hacher(a) != hacher(b)
    assert hacher(a, m) == hacher(b, m)


# ------------------------------------------------------------------ modèle d'effet, valeur, buts
def test_modele_effet_apprend_un_clic_utile():
    from effet import ModeleEffet
    rng = np.random.default_rng(0)
    m = ModeleEffet(0)
    for i in range(160):
        f = np.zeros((64, 64), dtype=np.int8)
        y, x = int(rng.integers(8, 56)), int(rng.integers(8, 56))
        f[y - 2:y + 3, x - 2:x + 3] = 5          # objet couleur 5 : cliquer dessus change l'image
        cible = rng.random() < 0.5
        cy, cx = (y, x) if cible else (int(rng.integers(0, 64)), int(rng.integers(0, 64)))
        label = float(cible)
        m.ajouter(bytes([i % 256, i // 256]), f, 6, label, cy, cx)
        if i % 10 == 9:
            m.entrainer(pas=8)
    f = np.zeros((64, 64), dtype=np.int8)
    f[30:35, 30:35] = 5
    _, pc = m.predire(f)
    assert pc[32, 32] > pc[5, 5]


def test_valeur_apprise_et_motif_de_reference():
    import buts
    mv = buts.ModeleValeur()
    pos = [np.eye(buts.NF)[2] * 0.5]
    neg = [np.zeros(buts.NF)] * 5
    v0 = mv.valeur(pos[0])
    mv.apprendre(pos, neg)
    assert mv.valeur(pos[0]) >= v0
    f = np.zeros((64, 64), dtype=np.int8)
    f[5:11, 5:11] = np.arange(36).reshape(6, 6) % 3 + 1
    g = f[5:11, 5:11].copy()
    g[0, 0] = 9
    f[30:36, 30:36] = g
    p = buts.trouver_paires(f, 0, None)
    assert p and p[0][0] > 0.9


def test_facade_bibliotheque():
    import bibliotheque as b
    for nom in ("Graphe", "ModeleEffet", "ModeleValeur", "Racine", "analyser", "caracteristiques", "trouver_paires", "hacher", "RESET"):
        assert hasattr(b, nom)
    m1, m2 = b.ModeleEffet(0), b.ModeleEffet(0)
    assert m1.ex is not m2.ex and b.Graphe().noeuds is not b.Graphe().noeuds
