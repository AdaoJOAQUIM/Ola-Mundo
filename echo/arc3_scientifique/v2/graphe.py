"""Graphe d'états (idée Blind Squirrel / Explore It Till You Solve It, réimplémentée).

Nœud = hachage de l'image stable (zones HUD masquées). Arc = action (("A",a), ("C",x,y)). RESET = ("R",) ramène à la racine
du niveau (coût 1 comme une action). Le graphe est pur (aucune dépendance au jeu) : `chemin` donne la route la plus
courte vers chaque nœud, soit par arcs connus, soit par RESET puis rejeu depuis la racine, selon ce qui coûte le moins.
"""
from __future__ import annotations

import hashlib
import heapq

import numpy as np

RESET = ("R",)


def hacher(frame, masque=None) -> bytes:
    f = frame
    if masque is not None and masque.any():
        f = frame.copy()
        f[masque] = 0
    return hashlib.blake2b(np.ascontiguousarray(f).tobytes(), digest_size=12).digest()


class Noeud:
    __slots__ = ("cle", "frame", "avail", "arcs", "cands", "probs", "ver", "profondeur", "info", "valeur", "visites")

    def __init__(self, cle, frame, avail, profondeur=0):
        self.cle = cle
        self.frame = frame
        self.avail = avail
        self.arcs = {}          # arc -> cle du nœud atteint ; None = arc mortel (game over)
        self.cands = None       # liste d'arcs candidats (calculée à la demande)
        self.probs = {}         # arc -> proba d'effet
        self.ver = -1
        self.profondeur = profondeur
        self.info = None
        self.valeur = 0.0
        self.visites = 0

    def non_essayes(self):
        return [a for a in (self.cands or []) if a not in self.arcs]


class Graphe:
    def __init__(self):
        self.noeuds = {}
        self.racine = None

    def initialiser(self, cle, frame, avail):
        self.noeuds = {}
        self.racine = cle
        self.noeuds[cle] = Noeud(cle, frame, avail, 0)
        return self.noeuds[cle]

    def ajouter(self, cle, frame, avail, parent=None):
        n = self.noeuds.get(cle)
        if n is None:
            d = (self.noeuds[parent].profondeur + 1) if parent in self.noeuds else 0
            n = self.noeuds[cle] = Noeud(cle, frame, avail, d)
        return n

    def relier(self, a, arc, b):
        self.noeuds[a].arcs[arc] = b

    def marquer_mortel(self, a, arc):
        self.noeuds[a].arcs[arc] = None

    def chemins(self, source, rayon=10 ** 9, avec_reset=True):
        """Dijkstra à coûts unitaires depuis `source`. Option : RESET (coût 1) vers la racine.
        Renvoie (dist, parent) ; parent[k] = (nœud précédent, arc) ; l'arc RESET a pour précédent `source`."""
        dist = {source: 0}
        parent = {}
        tas = [(0, 0, source)]
        cpt = 1
        if avec_reset and self.racine is not None and self.racine != source:
            dist[self.racine] = 1
            parent[self.racine] = (source, RESET)
            heapq.heappush(tas, (1, cpt, self.racine))
            cpt += 1
        fait = set()
        while tas:
            d, _, k = heapq.heappop(tas)
            if k in fait:
                continue
            fait.add(k)
            if d >= rayon:
                continue
            for arc, kn in self.noeuds[k].arcs.items():
                if kn is None or kn == k or kn not in self.noeuds:
                    continue
                if d + 1 < dist.get(kn, 10 ** 9):
                    dist[kn] = d + 1
                    parent[kn] = (k, arc)
                    heapq.heappush(tas, (d + 1, cpt, kn))
                    cpt += 1
        return dist, parent

    @staticmethod
    def route(parent, source, cible):
        """Liste d'arcs de source à cible (RESET compris), ou None si inatteignable."""
        res = []
        k = cible
        garde = 0
        while k != source:
            if k not in parent or garde > 100000:
                return None
            p, arc = parent[k]
            res.append(arc)
            k = p
            garde += 1
        res.reverse()
        return res
