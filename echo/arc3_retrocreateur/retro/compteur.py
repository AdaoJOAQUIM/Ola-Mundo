"""Compteur d'énergie : bande du bandeau qui perd des pixels à chaque action effective, sans jamais en regagner.
Estime la capacité E (nombre d'actions effectives permises par vie) d'après la longueur de la barre.
Variable cachée « minuterie » du langage : mort quand le nombre d'actions effectives atteint E sans victoire."""
from __future__ import annotations

import numpy as np


class Compteur:
    SEUIL = 0.3
    MIN_OBS = 8

    def __init__(self):
        self.n = 0
        self.lig = np.zeros(64, dtype=np.int32)
        self.rev = np.zeros(64, dtype=np.int32)
        self.paires = {}                 # y -> {(a, b)}
        self.bande = None                # tableau d'indices de lignes
        self.ce = None                   # couleur de la partie pleine de la barre
        self.vie = []                    # longueurs successives (k = 0, 1, ...) de la vie courante

    def maj(self, fa, fb):
        """Transition effective (fa -> fb)."""
        self.n += 1
        d = fa != fb
        lignes = np.nonzero(d.any(axis=1))[0]
        for y in lignes.tolist():
            xs = np.nonzero(d[y])[0]
            ps = self.paires.setdefault(y, set())
            rev = False
            for x in xs.tolist():
                a, b = int(fa[y, x]), int(fb[y, x])
                if (b, a) in ps:
                    rev = True
                ps.add((a, b))
            self.lig[y] += 1
            if rev:
                self.rev[y] += 1
        if self.bande is None and self.n >= self.MIN_OBS:
            ok = [y for y in range(64) if self.lig[y] >= self.SEUIL * self.n and self.rev[y] == 0]
            # une bande fine (<= 4 lignes contiguës)
            if ok and len(ok) <= 4 and ok[-1] - ok[0] + 1 == len(ok):
                cnt = {}
                for y in ok:
                    for (a, b) in self.paires[y]:
                        cnt[a] = cnt.get(a, 0) + 1
                self.ce = max(cnt, key=cnt.get)
                self.bande = np.array(ok)

    def longueur(self, frame):
        if self.bande is None:
            return None
        return int((frame[self.bande] == self.ce).sum())

    @staticmethod
    def estimer(longueurs):
        """E tel que L_k = floor(L_0 * (E - k) / E) pour tout k observé ; None si indéterminé."""
        if len(longueurs) < 3:
            return None
        v0 = longueurs[0]
        if v0 <= 0:
            return None
        k = len(longueurs) - 1
        best, be = None, 10 ** 9
        for E in range(max(2, k + 1), 400):
            err = sum(abs(L - (v0 * (E - i)) // E) for i, L in enumerate(longueurs))
            if err < be:
                best, be = E, err
                if err == 0 and E > k + 1 and False:
                    break
        return best if be <= max(1, k // 6) else None
