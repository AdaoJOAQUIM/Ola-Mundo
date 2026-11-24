"""D2 : même adaptateur que v1, mais le plafond du courtier (500 actions) est relevé pour CE processus seulement
(constante de module changée en mémoire ; aucun fichier modifié). Sert à l'évaluation à budget généreux."""
from __future__ import annotations

import chemin  # noqa: F401
import adapter


class AdaptateurLarge(adapter.AdaptateurARC3):
    def __init__(self, jeu_id, budget=5000):
        import harnais  # noqa: F401  (ajoute prime-agent à sys.path)
        import arc3_local_broker as b
        b.MAX_ACTIONS = max(b.MAX_ACTIONS, int(budget))
        super().__init__(jeu_id, budget)
