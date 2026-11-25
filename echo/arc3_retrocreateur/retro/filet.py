"""Filet de sécurité : quand aucun programme ne prédit bien, repli sur la bibliothèque v2 (non modifiée).

    sys.path += echo/arc3_scientifique/v2 ; from bibliotheque import Graphe, ModeleEffet, ModeleValeur, ...
Le filet est l'assemblage de la v2 (graphe d'états + modèle d'effet + valeur + théorie v1), chargé paresseusement.
"""
from __future__ import annotations

import sys
from pathlib import Path

V2 = Path(__file__).resolve().parent.parent.parent / "arc3_scientifique" / "v2"


class Filet:
    def __init__(self, graine=0):
        if str(V2) not in sys.path:
            sys.path.append(str(V2))
        from bibliotheque import Graphe, ModeleEffet, ModeleValeur  # noqa: F401  (composants de la bibliothèque v2)
        import agent2
        self.ag = agent2.Scientifique2(graine, {})
        self.n = 0

    def choisir(self, obs):
        self.n += 1
        a = self.ag.choisir(obs)
        if a.get("name") == "RESET" and obs.get("levels_completed", 0) > 0:
            # un RESET en cours de partie peut renvoyer tout le jeu au niveau 0 : le filet n'en a pas le droit
            dispo = [x for x in obs.get("available_actions", []) if x in (1, 2, 3, 4, 5)]
            return {"name": f"ACTION{dispo[self.n % len(dispo)] if dispo else 1}", "data": {}}
        return a
