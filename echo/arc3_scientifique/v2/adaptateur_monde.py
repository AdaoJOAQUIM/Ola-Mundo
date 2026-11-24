"""Adaptateur des mondes procéduraux (echo/mondes_arc3, lecture seule) vers l'interface de v1/adapter.py.

jeu_id = "graine:difficulte" (ex. "100003:2"). Mêmes 4 méthodes : observe, act, resume, ferme.
Un game-over coûte une action (RESET automatique) comme dans l'adaptateur du SDK.
"""
from __future__ import annotations

import sys
from pathlib import Path

ECHO = Path(__file__).resolve().parent.parent.parent
MONDES = ECHO / "mondes_arc3"
if str(MONDES) not in sys.path:
    sys.path.append(str(MONDES))


class AdaptateurMonde:
    def __init__(self, jeu_id: str, budget: int = 500):
        from env import MondeEnv
        from actions import GameAction
        self._GA = GameAction
        seed, _, d = str(jeu_id).partition(":")
        self.env = MondeEnv(int(seed), int(d or 1))
        self.budget = budget
        self.n = 0
        self.term = "ACTIVE"
        self.last = self.env.reset()
        self.n_reset0 = 0

    def _obs(self):
        o = self.last
        etat = getattr(o.state, "name", str(o.state))
        return {"game_id": o.game_id, "state": etat, "levels_completed": o.levels_completed,
                "available_actions": list(o.available_actions),
                "frame": [(z.tolist() if hasattr(z, "tolist") else z) for z in o.frame],
                "actions_taken": self.n, "actions_remaining": max(0, self.budget - self.n), "terminal": self.term}

    def observe(self):
        return self._obs()

    def act(self, action):
        if self.term != "ACTIVE":
            return self._obs()
        name = action["name"]
        data = action.get("data") or {}
        self.last = self.env.step(self._GA[name], data if name == "ACTION6" else None)
        self.n += 1
        etat = getattr(self.last.state, "name", str(self.last.state))
        if etat == "WIN":
            self.term = "WIN"
        elif etat == "GAME_OVER" and self.n < self.budget:
            self.last = self.env.step(self._GA["RESET"], None)
            self.n += 1
        if self.term == "ACTIVE" and self.n >= self.budget:
            self.term = "ACTION_CAP"
        return self._obs()

    def resume(self):
        e = self.env
        return {"jeu": e.game_id, "etat": self.term, "niveaux": e.finis, "niveaux_total": e.win_levels, "actions": self.n,
                "niveaux_actions": list(e.actions_niveau), "niveaux_reference": None, "score_sdk": None}

    def ferme(self):
        pass
