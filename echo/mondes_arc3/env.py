"""MondeEnv : environnement avec la même surface que l'env du SDK vu par le harnais
(`env.reset()`, `env.step(GameAction, data)`, et les anciens `env.actionN()`), plus un
courtier (CourtierMonde) au protocole identique à arc3_local_broker.Broker, et `jouer()`
qui reproduit le dictionnaire de résultats de harnais.jouer."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from actions import EtatJeu, FrameData, GameAction
from generateur import Monde
from moteur import ATTENTE, CLIC0, Moteur
from rendu import origine, rendre

MAX_ACTIONS = 500
MAX_BATCH = 20
VALID_ACTIONS = {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"}


class MondeEnv:
    def __init__(self, seed: int, difficulte: int = 1):
        self.seed = int(seed)
        self.difficulte = int(difficulte)
        self.monde = Monde(seed, difficulte)
        self.game_id = self.monde.game_id
        self.win_levels = self.monde.nl
        self.k = 0
        self.finis = 0
        self.etat = EtatJeu.NOT_FINISHED
        self.actions_niveau = [0] * self.monde.nl      # actions (RESET inclus) dépensées par niveau
        self.n_actions = 0
        self._charger(0)
        self.last = self._frame()

    # ------------------------------------------------------------ interne
    def _charger(self, k):
        self.k = k
        self.lv = self.monde.niveau(k)
        self.M = Moteur(self.lv)
        self.s = self.M.etat0()
        self.used = 0
        self.hist = []

    def _grille(self):
        return rendre(self.monde, self.lv, self.M, self.s, self.used, self.k, self.monde.nl, self.finis)

    def _frame(self):
        return FrameData(
            game_id=self.game_id,
            frame=[self._grille().astype(int)],
            state=self.etat,
            levels_completed=self.finis,
            win_levels=self.win_levels,
            available_actions=[0] + self.monde.actions_disponibles(),
        )

    # ------------------------------------------------------------ API SDK
    def reset(self):
        if self.etat is EtatJeu.WIN:
            self.finis = 0
            self.actions_niveau = [0] * self.monde.nl
            self.n_actions = 0
            self._charger(0)
        else:
            self._charger(self.k)
        self.etat = EtatJeu.NOT_FINISHED
        self.last = self._frame()
        return self.last

    def step(self, action, data=None):
        name = getattr(action, "name", None) or str(action)
        if name == "RESET":
            self.actions_niveau[self.k] += 1
            self.n_actions += 1
            return self.reset()
        if self.etat is not EtatJeu.NOT_FINISHED:
            return self.last
        self.actions_niveau[self.k] += 1
        self.n_actions += 1
        a = self._vers_abstrait(name, data)
        if name == "ACTION7":
            if "annuler" in self.monde.meca and self.hist:
                self.s, self.used = self.hist.pop()
        elif a is not None:
            r = self.M.pas(self.s, a)
            if r is not None:
                self.hist.append((self.s, self.used))
                self.s, flag = r
                self.used += 1
                if flag == 1:
                    self._niveau_gagne()
                elif flag == 2:
                    self.etat = EtatJeu.GAME_OVER
                elif self.lv.energy is not None and self.used >= self.lv.energy:
                    self.etat = EtatJeu.GAME_OVER
        self.last = self._frame()
        return self.last

    def _niveau_gagne(self):
        self.finis += 1
        if self.finis >= self.monde.nl:
            self.etat = EtatJeu.WIN
        else:
            self._charger(self.k + 1)

    def _vers_abstrait(self, name, data):
        dispo = self.monde.actions_disponibles()
        if name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            kk = int(name[-1])
            if kk in dispo:
                return self.monde.action_vers_dir(kk)
            return None
        if name == "ACTION5":
            return ATTENTE if 5 in dispo else None
        if name == "ACTION6":
            if 6 not in dispo or not data:
                return None
            x, y = int(data.get("x", -1)), int(data.get("y", -1))
            n = self.lv.n
            ox, oy, c = origine(n)
            cx, cy = (x - ox) // c, (y - oy) // c
            if x < ox or y < oy or not (0 <= cx < n and 0 <= cy < n):
                return None
            return CLIC0 + cy * n + cx
        return None

    # anciens accesseurs du SDK
    def action1(self): return self.step(GameAction.ACTION1)
    def action2(self): return self.step(GameAction.ACTION2)
    def action3(self): return self.step(GameAction.ACTION3)
    def action4(self): return self.step(GameAction.ACTION4)
    def action5(self): return self.step(GameAction.ACTION5)
    def action7(self): return self.step(GameAction.ACTION7)
    def action6(self, x, y): return self.step(GameAction.ACTION6, {"x": x, "y": y})

    # ------------------------------------------------------------ oracle (pas visible à l'agent)
    def actions_solution(self, k=None):
        """Suite d'appels (nom, data) qui résout le niveau k depuis son état initial (optimum)."""
        k = self.k if k is None else k
        lv = self.monde.niveau(k)
        inv = {self.monde.action_vers_dir(i): i for i in (1, 2, 3, 4)}
        ox, oy, c = origine(lv.n)
        out = []
        for a in lv.sol:
            if a < 4:
                out.append(("ACTION%d" % inv[a], {}))
            elif a == ATTENTE:
                out.append(("ACTION5", {}))
            else:
                p = a - CLIC0
                y, x = divmod(p, lv.n)
                out.append(("ACTION6", {"x": ox + x * c + c // 2, "y": oy + y * c + c // 2}))
        return out


# ------------------------------------------------------------------------------------------------
@dataclass
class Handle:
    game_id: str
    env: Any
    actions: int = 0
    terminal: str = "ACTIVE"
    last: Any = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class CourtierMonde:
    """Même protocole que arc3_local_broker.Broker (observe/status/act, lots <=20, 500 actions,
    RESET automatique et compté après un GAME_OVER)."""

    def __init__(self, seed, difficulte=1, max_actions=MAX_ACTIONS):
        env = MondeEnv(seed, difficulte)
        self.max_actions = max_actions
        self.handle = Handle(game_id=env.game_id, env=env)
        self.handle.last = env.reset()

    def _observation(self):
        h = self.handle
        o = h.last
        state = getattr(getattr(o, "state", None), "name", getattr(o, "state", "NOT_FINISHED"))
        frame = getattr(o, "frame", []) or []
        return {
            "game_id": h.game_id,
            "state": str(state),
            "levels_completed": getattr(o, "levels_completed", None),
            "available_actions": list(getattr(o, "available_actions", []) or []),
            "frame": [z.tolist() if hasattr(z, "tolist") else z for z in frame],
            "actions_taken": h.actions,
            "actions_remaining": max(0, self.max_actions - h.actions),
            "terminal": h.terminal,
        }

    def _apply(self, action):
        h = self.handle
        name = action.get("name")
        data = action.get("data") or {}
        if name == "RESET":
            h.last = h.env.reset()
        else:
            h.last = h.env.step(GameAction[name], data if name == "ACTION6" else None)
        h.actions += 1

    def dispatch(self, query):
        op = query.get("op")
        if op in {"observe", "status"}:
            return {"ok": True, "value": self._observation()}
        if op != "act":
            return {"ok": False, "error": "OperationError", "message": f"unknown op {op!r}"}
        actions = query.get("actions") or []
        if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_BATCH:
            return {"ok": False, "error": "ProtocolError", "message": f"act takes 1..{MAX_BATCH} actions"}
        h = self.handle
        with h.lock:
            if h.terminal != "ACTIVE":
                return {"ok": True, "value": self._observation()}
            before = getattr(h.last, "levels_completed", None)
            for action in actions:
                name = action.get("name")
                if name not in VALID_ACTIONS:
                    return {"ok": False, "error": "ProtocolError", "message": f"bad action {name!r}"}
                if name == "ACTION6":
                    data = action.get("data") or {}
                    if not all(isinstance(data.get(k), int) and 0 <= data[k] <= 63 for k in ("x", "y")):
                        return {"ok": False, "error": "ProtocolError", "message": "ACTION6 needs integer x,y in 0..63"}
                self._apply(action)
                state = str(getattr(getattr(h.last, "state", None), "name", ""))
                if state == "GAME_OVER" and h.actions < self.max_actions:
                    h.last = h.env.reset()
                    h.actions += 1
                    break
                if state == "WIN":
                    h.terminal = "WIN"
                    break
                if getattr(h.last, "levels_completed", None) != before:
                    break
                if h.actions >= self.max_actions:
                    break
            if h.actions >= self.max_actions and h.terminal == "ACTIVE":
                h.terminal = "ACTION_CAP"
        return {"ok": True, "value": self._observation()}


class _Garde:
    """Repli si le garde du harnais n'est pas importable : même contrat observe/act."""

    def __init__(self, courtier):
        self.courtier = courtier

    def observe(self):
        return self.courtier.dispatch({"op": "observe"})["value"]

    def act(self, actions):
        rep = self.courtier.dispatch({"op": "act", "actions": actions})
        if not rep.get("ok"):
            raise RuntimeError(rep.get("message"))
        return rep["value"]


def _garde(courtier):
    try:
        import sys
        from pathlib import Path
        ar = str(Path(__file__).resolve().parent.parent / "arc3")
        if ar not in sys.path:
            sys.path.insert(0, ar)
        import harnais
        return harnais.Garde(courtier)
    except Exception:
        return _Garde(courtier)


def jouer(seed, difficulte, fabrique, max_actions=MAX_ACTIONS):
    """Joue UN monde (run neuf). `fabrique()` renvoie un objet avec .choisir(obs).
    Renvoie (resultat, politique) ; clés identiques à harnais.jouer + optimum."""
    t_cpu = time.process_time()
    t_mur = time.time()
    courtier = CourtierMonde(seed, difficulte, max_actions)
    garde = _garde(courtier)
    pol = fabrique()
    obs = garde.observe()
    t_cerveau = 0.0
    while obs["terminal"] == "ACTIVE":
        t0 = time.process_time()
        action = pol.choisir(obs)
        t_cerveau += time.process_time() - t0
        obs = garde.act([action])
    if hasattr(pol, "observer"):
        pol.observer(obs)
    env = courtier.handle.env
    mo = env.monde
    ref = mo.optimum()
    acts = list(env.actions_niveau)
    eff = [min(1.0, r / a) for r, a, _ in zip(ref, acts, range(env.finis))] if env.finis else []
    res = {
        "jeu": courtier.handle.game_id,
        "seed": seed,
        "difficulte": difficulte,
        "etat": obs["terminal"],
        "niveaux": obs["levels_completed"],
        "niveaux_total": mo.nl,
        "actions": obs["actions_taken"],
        "niveaux_actions": acts,
        "niveaux_reference": ref,
        "efficacite": eff,
        "mecaniques": sorted(mo.meca),
        "but": mo.goal,
        "cpu_s": round(time.process_time() - t_cpu, 2),
        "cpu_cerveau_s": round(t_cerveau, 2),
        "mur_s": round(time.time() - t_mur, 2),
    }
    return res, pol
