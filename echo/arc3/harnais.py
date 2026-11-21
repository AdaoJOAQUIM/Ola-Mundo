"""Harnais : relie une politique au jeu, via le courtier et le garde de Prime Agent.

La politique (cerveau.py / baseline aléatoire) ne voit que des dicts
d'observation. Ici, on réutilise tel quel :
  - arc3_local_broker.Broker (courtier) — avec un adaptateur minimal car le SDK
    installé n'expose plus env.action1()...action7() mais env.step(GameAction, data) ;
  - game_protocol_guard._allowed / règles de phase (garde).
Rien du dépôt prime-agent n'est copié ni modifié.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
PRIME = RACINE / "arc" / "externes" / "prime-agent"
JEUX = RACINE / "arc" / "externes" / "arc3-jeux"
sys.path.insert(0, str(PRIME))

import arc3_local_broker as courtier_mod          # noqa: E402
import game_protocol_guard as garde_mod           # noqa: E402


class CourtierCompat(courtier_mod.Broker):
    """Même courtier ; seul _apply change (API step() du SDK actuel)."""

    def _apply(self, action):
        from arcengine import GameAction
        h = self.handle
        name = action.get("name")
        data = action.get("data") or {}
        if name == "RESET":
            h.last = h.env.reset()
        else:
            h.last = h.env.step(GameAction[name], data if name == "ACTION6" else None)
        h.actions += 1


class RefusGarde(RuntimeError):
    pass


class Garde:
    """Applique les règles de phase du garde root de Prime Agent (observe d'abord,
    puis une seule action authentique non-RESET deux fois), puis délègue au courtier."""

    def __init__(self, courtier):
        self.courtier = courtier
        self.phase = 0

    def appeler(self, requete):
        if not garde_mod._allowed(self.phase, requete):
            raise RefusGarde("requête refusée par le garde (phase %d)" % self.phase)
        rep = self.courtier.dispatch(requete)
        if not rep.get("ok"):
            raise RefusGarde(f"{rep.get('error')}: {rep.get('message')}")
        v = rep["value"]
        if self.phase == 0:
            self.phase = 1
        elif self.phase == 1:
            self.phase = 2 if v.get("terminal") == "ACTIVE" else 3
        elif self.phase == 2:
            self.phase = 3
        return v

    def observe(self):
        return self.appeler({"op": "observe"})

    def status(self):
        return self.appeler({"op": "status"})

    def act(self, actions):
        return self.appeler({"op": "act", "actions": actions})


def jouer(prefixe, fabrique):
    """Joue UN jeu (run neuf, graine 0). `fabrique()` renvoie un objet avec .choisir(obs)."""
    t_cpu = time.process_time()
    t_mur = time.time()
    courtier = CourtierCompat(JEUX, prefixe)
    garde = Garde(courtier)
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
    sc = courtier.arcade.get_scorecard()
    scj = json.loads(sc.model_dump_json()) if hasattr(sc, "model_dump_json") else {}
    env = (scj.get("environments") or [{}])[0]
    run = (env.get("runs") or [{}])[-1]
    res = {
        "jeu": courtier.handle.game_id,
        "etat": obs["terminal"],
        "niveaux": obs["levels_completed"],
        "niveaux_total": env.get("level_count"),
        "actions": obs["actions_taken"],
        "resets_compteur": env.get("resets"),
        "score_sdk": env.get("score"),
        "niveaux_actions": run.get("level_actions"),
        "niveaux_reference": run.get("level_baseline_actions"),
        "cpu_s": round(time.process_time() - t_cpu, 2),
        "cpu_cerveau_s": round(t_cerveau, 2),
        "mur_s": round(time.time() - t_mur, 2),
    }
    return res, pol
