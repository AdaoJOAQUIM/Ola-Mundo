"""Choix déterministe des 300 mondes J1 (sans aucune exécution d'agent).
Règle : pour d in (1,2,3), les 100 premières graines >= 9_000_000 dont le monde (graine, d)
n'a AUCUNE des mécaniques scellées (scelle.json) ni... (seul le tirage du monde est lu, pas les niveaux).
Développement : graines 1..2999 uniquement (tout essai hors de cette plage est interdit)."""
import json, sys
from pathlib import Path
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent.parent / "mondes_arc3"))
from generateur import Monde  # évaluateur seulement : l'agent n'importe jamais ceci

SCELLE = set(json.loads((ICI.parent / "scelle.json").read_text())["mecaniques_secretes"])

def choisir(depart, d, n):
    out, s = [], depart
    while len(out) < n:
        if not (Monde(s, d).meca & SCELLE):
            out.append(s)
        s += 1
    return out

def tout(depart=9_000_000, n=100):
    return {d: choisir(depart, d, n) for d in (1, 2, 3)}

if __name__ == "__main__":
    t = tout()
    print({d: (v[0], v[-1], len(v)) for d, v in t.items()})
    dev = {d: choisir(1, d, 40) for d in (1, 2, 3)}
    print({d: (v[0], v[-1]) for d, v in dev.items()})
