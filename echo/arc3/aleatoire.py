"""Baseline aléatoire : même protocole, action uniforme parmi les disponibles,
clic à (x, y) uniforme. Ne joue jamais RESET (le courtier le fait au game-over)."""
import random


class Aleatoire:
    def __init__(self, graine=0):
        self.rng = random.Random(graine)

    def choisir(self, obs):
        dispo = [a for a in obs["available_actions"] if 1 <= int(a) <= 7] or [1]
        a = self.rng.choice(dispo)
        if a == 6:
            return {"name": "ACTION6", "data": {"x": self.rng.randrange(64), "y": self.rng.randrange(64)}}
        return {"name": f"ACTION{a}", "data": {}}
