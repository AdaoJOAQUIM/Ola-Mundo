"""Adaptateur d'environnement : SEUL point d'accès de l'agent au jeu.

Interface (identique pour le SDK ARC-AGI-3 via courtier+garde, et pour tout générateur procédural) :

    env = AdaptateurARC3(jeu_id, budget=500)      # jeu_id : préfixe ou id complet
    obs = env.observe()                           # 1er appel obligatoire (protocole du garde)
    obs = env.act(action)                         # action = {"name": "ACTION1".."ACTION7"|"RESET", "data": {} | {"x":int,"y":int}}
    env.resume() -> dict                          # {"jeu","etat","niveaux","niveaux_total","actions",
                                                  #  "niveaux_actions","niveaux_reference","score_sdk"}
    env.ferme()

`obs` (dict) : game_id:str, state:str, levels_completed:int, available_actions:list[int] (1..7),
frame:list[grille 64x64 d'entiers 0..15] (la dernière est l'état stable), actions_taken:int,
actions_remaining:int, terminal:"ACTIVE"|"WIN"|"ACTION_CAP".
Sémantique : un game-over consomme une action (RESET automatique) ; passer au niveau suivant renvoie
immédiatement la première image du niveau ; le jeu s'arrête à `budget` actions.
Un autre environnement (echo/mondes_arc3) n'a qu'à fournir cette même classe (mêmes 4 méthodes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent / "arc3"))


class AdaptateurARC3:
    def __init__(self, jeu_id: str, budget: int = 500):
        import harnais
        self.budget = budget
        self.courtier = harnais.CourtierCompat(harnais.JEUX, jeu_id)
        self.garde = harnais.Garde(self.courtier)
        self.obs = None

    def observe(self):
        self.obs = self.garde.observe()
        return self._borne(self.obs)

    def act(self, action):
        if self.obs is not None and self.obs["actions_taken"] >= self.budget and self.obs["terminal"] == "ACTIVE":
            self.obs = dict(self.obs, terminal="ACTION_CAP")
            return self.obs
        self.obs = self.garde.act([action])
        return self._borne(self.obs)

    def _borne(self, o):
        if o["actions_taken"] >= self.budget and o["terminal"] == "ACTIVE":
            o = dict(o, terminal="ACTION_CAP")
        return o

    def resume(self):
        o = self.obs
        sc = self.courtier.arcade.get_scorecard()
        scj = json.loads(sc.model_dump_json()) if hasattr(sc, "model_dump_json") else {}
        env = (scj.get("environments") or [{}])[0]
        run = (env.get("runs") or [{}])[-1]
        return {"jeu": self.courtier.handle.game_id, "etat": o["terminal"], "niveaux": o["levels_completed"],
                "niveaux_total": env.get("level_count"), "actions": o["actions_taken"],
                "niveaux_actions": run.get("level_actions"), "niveaux_reference": run.get("level_baseline_actions"),
                "score_sdk": env.get("score")}

    def ferme(self):
        pass


def jouer(env, agent):
    """Boucle de jeu générique sur n'importe quel adaptateur."""
    import time
    t0 = time.process_time()
    obs = env.observe()
    while obs["terminal"] == "ACTIVE":
        obs = env.act(agent.choisir(obs))
    res = env.resume()
    res["cpu_s"] = round(time.process_time() - t0, 2)
    return res
