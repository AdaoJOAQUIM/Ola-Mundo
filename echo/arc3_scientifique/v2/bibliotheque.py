"""Façade bibliothèque des composants de sécurité de v2 (sans état global, un objet par composant).

    import sys; sys.path.insert(0, "<repo>/echo/arc3_scientifique/v2")
    from bibliotheque import Graphe, ModeleEffet, ModeleValeur, Racine, analyser, caracteristiques, trouver_paires, hacher, RESET

Graphe (graphe.py) : exploration par graphe d'états.
    hacher(frame: ndarray[64,64], masque: ndarray[bool]|None=None) -> bytes          # clé de nœud, HUD masqué
    g = Graphe(); g.initialiser(cle, frame, avail) -> Noeud; g.ajouter(cle, frame, avail, parent=None) -> Noeud
    g.relier(a, arc, b); g.marquer_mortel(a, arc)          # arc = ("A", n) | ("C", x, y) ; RESET = ("R",)
    g.chemins(source, rayon=1e9, avec_reset=True) -> (dist: dict, parent: dict)      # Dijkstra ; RESET compté 1 vers la racine
    Graphe.route(parent, source, cible) -> list[arc] | None                          # chemin le plus court, RESET+rejeu inclus
    Noeud.non_essayes() -> list[arc]   (après avoir rempli Noeud.cands)
ModeleEffet (effet.py) : modèle d'effet d'action (CNN petit, en ligne).
    m = ModeleEffet(graine=0, c=16, lr=2e-3, threads=None)
    m.ajouter(cle, frame, action:int(1..7), label:float, y=-1, x=-1) -> bool         # action 6 + (y,x) pour un clic
    m.entrainer(pas=2, lot=32, force=False) -> perte|None
    m.predire(frame) -> (p_simples[7], p_clic[64,64]) ; m.echantillonner(probas, plancher=0.02) -> indice
    m.etat() -> state_dict ; m.charger(state_dict) ; m.version
ModeleValeur / buts.py : hypothèses de but génériques + valeur transférable d'un niveau au suivant.
    r = Racine(frame, masque)                             # image de départ d'un niveau
    an = analyser(frame, masque) -> dict ; f = caracteristiques(frame, an, r, masque) -> ndarray[NF]
    v = ModeleValeur(actif=True) ; v.valeur(f) -> float[0,1] ; v.apprendre(positifs: list[ndarray], negatifs: list[ndarray])
    trouver_paires(frame, bg:int, masque) -> list[(eq, boite_a, boite_b)]               # motif de référence
Exemple d'assemblage complet : agent2.Scientifique2(graine, options).choisir(obs) -> action dict.
"""
import chemin  # noqa: F401
from buts import ModeleValeur, Racine, analyser, caracteristiques, trouver_paires  # noqa: F401
from effet import ModeleEffet  # noqa: F401
from graphe import Graphe, Noeud, RESET, hacher  # noqa: F401
