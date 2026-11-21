"""Cerveau bayésien sans LLM pour ARC-AGI-3 (CPU, agnostique du jeu).

Contrat : ce module ne parle JAMAIS au jeu directement. Il reçoit des
observations (dict du courtier : frame, levels_completed, available_actions,
actions_taken, ...) et renvoie une action {"name":..., "data":{...}}.
Il n'importe ni arc_agi ni arcengine, et ne contient aucun identifiant de jeu.

Méthodes (toutes standard) :
 1. Perception : composantes connexes par couleur, hachage d'état, masque des
    cellules « volatiles » (compteurs/minuteries qui changent à chaque pas).
 2. Modèle bayésien action-effet : postérieurs Beta pour « l'action change la
    frame », « mène à un état nouveau », « fait gagner un niveau », « tue ».
    Les clics (ACTION6) sont des bras au centroïde des objets, notés par leurs
    caractéristiques (couleur, taille) avec rétrécissement hiérarchique.
 3. Exploration : graphe des états observés, échantillonnage de Thompson sur la
    nouveauté/progrès attendus, navigation BFS vers les états-frontière.
 4. Rejeu : chemin-solution mémorisé après un niveau gagné ; après un game-over
    le graphe connu sert de rejeu (BFS depuis l'état initial).
 5. Harnais continu : les « prieurs » (comptes Beta excédentaires) se transfèrent
    d'un jeu à l'autre ; la décision de les garder est prise dans lancer.py.
"""
from __future__ import annotations

import hashlib
import math
from collections import deque

import numpy as np

SIMPLES = (1, 2, 3, 4, 5, 7)           # ACTION1-5 et ACTION7 (annuler)
QUANTITES = ("chg", "new", "up", "mort")
# Prieurs de base (a, b) par quantité : non informatifs pour chg/new,
# pessimistes (événements rares) pour up/mort.
PRIEUR_BASE = {"chg": (1.0, 1.0), "new": (1.0, 1.0), "up": (0.3, 3.0), "mort": (0.3, 3.0)}
POIDS_UP = 3.0
POIDS_MORT = 0.3
GAMMA = 0.9                             # décote par pas de navigation
MAX_OBJETS = 40                         # clics candidats par état
SEUIL_VOLATILE = 0.7                    # fréquence de changement d'une cellule
TOL_REV = 0.1                          # part max. de transitions avec retour en arrière
POINTS_MASQUE = (8, 16, 32, 64, 128, 256, 512)  # nombre de transitions où recalculer le masque


def _hash(frame: np.ndarray, masque) -> bytes:
    f = frame
    if masque is not None and masque.any():
        f = frame.copy()
        f[masque] = 0
    return hashlib.blake2b(f.tobytes(), digest_size=12).digest()


def composantes(frame: np.ndarray, masque=None):
    """Objets = composantes connexes (4-voisinage) par couleur, hors fond.

    Renvoie une liste de dicts : couleur, taille, centroïde entier (une cellule
    de l'objet), boîte englobante.
    """
    h, w = frame.shape
    couleurs, comptes = np.unique(frame, return_counts=True)
    fond = int(couleurs[np.argmax(comptes)])
    vu = np.zeros((h, w), dtype=bool)
    if masque is not None:
        vu |= masque
    objets = []
    for y0 in range(h):
        for x0 in range(w):
            if vu[y0, x0]:
                continue
            c = int(frame[y0, x0])
            if c == fond:
                vu[y0, x0] = True
                continue
            pile = [(y0, x0)]
            vu[y0, x0] = True
            cells = []
            while pile:
                y, x = pile.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not vu[ny, nx] and frame[ny, nx] == c:
                        vu[ny, nx] = True
                        pile.append((ny, nx))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            my, mx = sum(ys) / len(ys), sum(xs) / len(xs)
            # cellule de l'objet la plus proche du centroïde (le clic doit toucher l'objet)
            cy, cx = min(cells, key=lambda p: (p[0] - my) ** 2 + (p[1] - mx) ** 2)
            objets.append({"couleur": c, "taille": len(cells), "x": cx, "y": cy,
                           "bbox": (min(xs), min(ys), max(xs), max(ys))})
    return objets, fond


def _classe_taille(n: int) -> int:
    return int(math.log2(n)) if n > 0 else 0


class Noeud:
    __slots__ = ("frame", "bras", "sortant", "avail")

    def __init__(self, frame, avail):
        self.frame = frame
        self.avail = avail
        self.bras = None            # construits paresseusement
        self.sortant = {}           # bras -> (clé suivante, mort)


class Niveau:
    def __init__(self):
        self.noeuds = {}
        self.debut = None
        self.trans = []             # (frame_a, avail_a, bras, frame_b, mort) pour reconstruire


class Cerveau:
    def __init__(self, graine: int = 0, prieurs: dict | None = None):
        self.rng = np.random.default_rng(graine)
        self.prieurs = prieurs or {}
        # postérieurs : quantité -> groupe -> [a, b]
        self.post = {q: {} for q in QUANTITES}
        for q, gs in self.prieurs.items():
            for g, (a, b) in gs.items():
                self.post[q][g] = [PRIEUR_BASE[q][0] + a, PRIEUR_BASE[q][1] + b]
        self.niveaux = {}
        self.niv = None
        self.lv = None
        self.cle = None
        self.attente = None         # (bras, frame, actions_taken, clé)
        self.hist = []
        self.macro = deque()
        self.sol_precedente = []
        self.solutions = {}
        self.cnt = None
        self.paires = {}
        self.ntr = 0
        self.masque = None
        self.derniere_reset = -99
        self.stats = {"resets_cerveau": 0, "morts": 0, "niveaux": 0, "volatiles": 0}

    # ------------------------------------------------------------ postérieurs
    def _ab(self, q, g):
        d = self.post[q]
        if g not in d:
            d[g] = list(PRIEUR_BASE[q])
        return d[g]

    def _maj(self, q, g, succes):
        ab = self._ab(q, g)
        ab[0 if succes else 1] += 1.0

    @staticmethod
    def _groupe(bras):
        if bras[0] == "A":
            return f"A{bras[1]}"
        return f"C|{bras[3]}|{bras[4]}"

    def _tirer(self, q, g):
        a, b = self._ab(q, g)
        if g.startswith("C|"):          # rétrécissement vers la moyenne globale des clics
            ag, bg = self._ab(q, "C")
            m = ag / (ag + bg)
            a, b = a + 2 * m, b + 2 * (1 - m)
        return float(self.rng.beta(a, b))

    def _score(self, bras, cache):
        g = self._groupe(bras)
        if g not in cache:
            cache[g] = tuple(self._tirer(q, g) for q in QUANTITES)
        chg, new, up, mort = cache[g]
        return chg * new + POIDS_UP * up - POIDS_MORT * mort + 1e-6 * self.rng.random()

    def exporter_comptes(self):
        """Comptes excédentaires (postérieur - prieur de base), pour le transfert."""
        out = {}
        for q in QUANTITES:
            out[q] = {}
            for g, (a, b) in self.post[q].items():
                pa, pb = self.prieurs.get(q, {}).get(g, (0.0, 0.0))
                out[q][g] = [a - PRIEUR_BASE[q][0] - pa, b - PRIEUR_BASE[q][1] - pb]
        return out

    # ------------------------------------------------------------ perception
    def _cle(self, frame):
        return _hash(frame, self.masque)

    def _noeud(self, niv, frame, avail):
        k = self._cle(frame)
        if k not in niv.noeuds:
            niv.noeuds[k] = Noeud(frame, avail)
        return k

    def _bras(self, noeud):
        if noeud.bras is not None:
            return noeud.bras
        bras = [("A", n) for n in SIMPLES if n in noeud.avail]
        if 6 in noeud.avail:
            objets, _ = composantes(noeud.frame, self.masque)
            comptes = {}
            for o in objets:
                comptes[o["couleur"]] = comptes.get(o["couleur"], 0) + o["taille"]
            # objets rares d'abord (couleurs peu fréquentes), puis petits
            objets.sort(key=lambda o: (comptes[o["couleur"]], o["taille"]))
            vus = set()
            for o in objets:
                if len(vus) >= MAX_OBJETS:
                    break
                if (o["x"], o["y"]) in vus:
                    continue
                vus.add((o["x"], o["y"]))
                bras.append(("C", o["x"], o["y"], o["couleur"], _classe_taille(o["taille"])))
            if not vus:     # écran vide : grille grossière de repli
                for y in range(8, 64, 16):
                    for x in range(8, 64, 16):
                        bras.append(("C", x, y, -1, 0))
        noeud.bras = bras
        return bras

    @staticmethod
    def _cle_bras(b):
        return b if b[0] == "A" else ("C", b[1], b[2])

    def _maj_masque(self, fa, fb):
        """Détecte les bandes HUD (barres de pas/minuterie) : une ligne (ou colonne)
        où quelque chose change à presque chaque transition mais où aucune cellule
        ne revient jamais à sa valeur précédente (changement monotone) n'est pas
        de l'état de jeu : on la masque du hachage d'état."""
        h, w = fa.shape
        if self.cnt is None or self.cnt.shape != (h, w):
            self.cnt = np.zeros((h, w), dtype=np.int32)       # inutilisé (compat.)
            self.lig = np.zeros(h, dtype=np.int32)
            self.col = np.zeros(w, dtype=np.int32)
            self.rev_l = np.zeros(h, dtype=np.int32)
            self.rev_c = np.zeros(w, dtype=np.int32)
            self.ntr = 0
        ys, xs = np.nonzero(fa != fb)
        if len(ys) and (np.bincount(ys).max() >= 10 or np.bincount(xs).max() >= 10):
            self.paires = {}        # remplissage massif d'une barre (nouvelle « vie ») : on n'apprend rien
            return
        rl, rc = set(), set()
        for y, x in zip(ys.tolist(), xs.tolist()):
            a, b = int(fa[y, x]), int(fb[y, x])
            p = self.paires.setdefault((y, x), set())
            if (b, a) in p:
                rl.add(y)
                rc.add(x)
            p.add((a, b))
        for y in rl:
            self.rev_l[y] += 1
        for x in rc:
            self.rev_c[x] += 1
        self.lig[np.unique(ys)] += 1
        self.col[np.unique(xs)] += 1
        self.ntr += 1
        if self.ntr in POINTS_MASQUE:
            m = np.zeros((h, w), dtype=bool)
            lignes = (self.lig >= SEUIL_VOLATILE * self.ntr) & (self.rev_l <= TOL_REV * self.ntr)
            colonnes = (self.col >= SEUIL_VOLATILE * self.ntr) & (self.rev_c <= TOL_REV * self.ntr)
            m[lignes, :] = True
            m[:, colonnes] = True
            ancien = self.masque
            if (ancien is None and m.any()) or (ancien is not None and not np.array_equal(ancien, m)):
                self.masque = m
                self.stats["volatiles"] = int(m.sum())
                self._reconstruire()

    def _reconstruire(self):
        """Recalcule les graphes avec le nouveau masque d'état."""
        for lv, niv in self.niveaux.items():
            vieux = niv.trans
            deb = niv.noeuds[niv.debut].frame if niv.debut in niv.noeuds else None
            avd = niv.noeuds[niv.debut].avail if deb is not None else ()
            niv.noeuds = {}
            niv.trans = []
            if deb is not None:
                niv.debut = self._noeud(niv, deb, avd)
            for (fa, aa, br, fb, ab, mort) in vieux:
                ka = self._noeud(niv, fa, aa)
                kb = self._noeud(niv, fb, ab)
                niv.noeuds[ka].sortant[self._cle_bras(br)] = (kb, mort, br)
                niv.trans.append((fa, aa, br, fb, ab, mort))
        if self.niv is not None and self.attente is not None:
            pass
        # clé courante recalculée sur la dernière frame connue
        if self.cle is not None and self.niv is not None and self.derniere_frame is not None:
            self.cle = self._noeud(self.niv, self.derniere_frame, self.dernier_avail)

    derniere_frame = None
    dernier_avail = ()

    # ------------------------------------------------------------ boucle
    def _chemin_vers(self, niv, source):
        """BFS depuis source : renvoie {clé: (distance, premier_bras)} (arêtes non mortelles, hors boucles)."""
        res = {source: (0, None)}
        file = deque([source])
        while file:
            k = file.popleft()
            d, premier = res[k]
            for ck, (kn, mort, br) in niv.noeuds[k].sortant.items():
                if mort or kn == k or kn in res:
                    continue
                res[kn] = (d + 1, br if premier is None else premier)
                file.append(kn)
        return res

    def _solution(self, niv, bras_final):
        """Chemin le plus court début -> état courant + bras final (repli : historique)."""
        try:
            chem = {niv.debut: []}
            file = deque([niv.debut])
            while file:
                k = file.popleft()
                for ck, (kn, mort, br) in niv.noeuds[k].sortant.items():
                    if mort or kn in chem:
                        continue
                    chem[kn] = chem[k] + [br]
                    file.append(kn)
            if self.cle in chem:
                return chem[self.cle] + [bras_final]
        except Exception:
            pass
        return list(self.hist) + [bras_final]

    def _marche_aleatoire(self, noeud):
        ok = [br for (kn, mort, br) in noeud.sortant.values() if not mort and kn != self.cle]
        if not ok:
            return None
        return ok[int(self.rng.integers(len(ok)))]

    def observer(self, obs):
        """Intègre l'observation (apprentissage) ; met à jour l'état courant."""
        frame = np.array(obs["frame"][-1], dtype=np.int8)
        avail = tuple(int(a) for a in obs.get("available_actions", []) if int(a) != 0)
        lv = obs.get("levels_completed") or 0
        if self.attente is None:        # première observation
            self.lv = lv
            self.niv = self.niveaux.setdefault(lv, Niveau())
            self.cle = self._noeud(self.niv, frame, avail)
            self.niv.debut = self.cle
            self.derniere_frame, self.dernier_avail = frame, avail
            return
        bras, fa, n_avant, cle_a = self.attente
        self.attente = None
        delta = obs["actions_taken"] - n_avant
        if bras == "RESET":
            self.derniere_reset = obs["actions_taken"]
            self.paires = {}
            self.hist = []
            self.macro.clear()
            self.derniere_frame, self.dernier_avail = frame, avail
            self.cle = self._noeud(self.niv, frame, avail)
            return
        mort = delta >= 2 and obs.get("state") != "GAME_OVER" and lv == self.lv
        g = self._groupe(bras)
        if lv != self.lv:
            if lv > self.lv:            # niveau gagné
                self.stats["niveaux"] += 1
                chemin = self._solution(self.niv, bras)
                self.solutions[self.lv] = chemin
                for b in chemin:
                    self._maj("up", self._groupe(b), True)
                self.sol_precedente = [b for b in chemin if b[0] == "A"]
                self.macro = deque(self.sol_precedente)
            self.lv = lv
            self.paires = {}
            self.niv = self.niveaux.setdefault(lv, Niveau())
            self.hist = []
            self.derniere_frame, self.dernier_avail = frame, avail
            self.cle = self._noeud(self.niv, frame, avail)
            self.niv.debut = self.cle
            return
        # transition ordinaire (dans le même niveau)
        if not mort:
            self._maj_masque(fa, frame)
        cle_a = self._cle(fa)
        existait = self._cle(frame) in self.niv.noeuds
        kb = self._noeud(self.niv, frame, avail)
        ka = self._noeud(self.niv, fa, self.niv.noeuds[cle_a].avail if cle_a in self.niv.noeuds else avail)
        change = (ka != kb) or mort
        nouveau = (not existait) and not mort
        self.niv.noeuds[ka].sortant[self._cle_bras(bras)] = (kb, mort, bras)
        self.niv.trans.append((fa, self.niv.noeuds[ka].avail, bras, frame, avail, mort))
        self._maj("chg", g, change)
        if g.startswith("C|"):
            self._maj("chg", "C", change)
        if change:
            self._maj("new", g, nouveau)
            if g.startswith("C|"):
                self._maj("new", "C", nouveau)
        self._maj("up", g, False)
        self._maj("mort", g, mort)
        if mort:
            self.paires = {}        # le HUD se remplit à nouveau : nouvelle « vie »
            self.stats["morts"] += 1
            self.hist = []
            self.macro.clear()
        else:
            self.hist.append(bras)
            if self.macro and not (change and nouveau):
                self.macro.clear()
        self.derniere_frame, self.dernier_avail = frame, avail
        self.cle = kb

    def choisir(self, obs):
        self.observer(obs)
        niv, cle = self.niv, self.cle
        noeud = niv.noeuds[cle]
        bras_dispo = self._bras(noeud)
        non_essayes = [b for b in bras_dispo if self._cle_bras(b) not in noeud.sortant]
        choix = None
        # 1. macro : rejeu du chemin-solution du niveau précédent
        while self.macro and choix is None:
            b = self.macro.popleft()
            if b in non_essayes:
                choix = b
        if choix is None:
            cache = {}
            meilleur_local = max(non_essayes, key=lambda b: self._score(b, cache)) if non_essayes else None
            val_local = self._score(meilleur_local, cache) if meilleur_local else -1e9
            # 2. navigation BFS vers la frontière
            cible_val, cible_bras = -1e9, None
            for k, (d, premier) in self._chemin_vers(niv, cle).items():
                if d == 0 or d > 60:
                    continue
                nd = niv.noeuds[k]
                libres = [b for b in self._bras(nd) if self._cle_bras(b) not in nd.sortant]
                if not libres:
                    continue
                v = (GAMMA ** d) * max(self._score(b, cache) for b in libres)
                if v > cible_val:
                    cible_val, cible_bras = v, premier
            if meilleur_local is not None and val_local >= cible_val:
                choix = meilleur_local
            elif cible_bras is not None:
                choix = cible_bras
            elif self._marche_aleatoire(noeud) is not None:
                # frontière épuisée : l'état caché (minuterie, vies) peut rendre les transitions
                # non déterministes -> marche aléatoire sur les arêtes connues non mortelles
                choix = self._marche_aleatoire(noeud)
            elif obs["actions_taken"] >= 3 and obs["actions_taken"] - self.derniere_reset > 2:
                self.stats["resets_cerveau"] += 1
                self.attente = ("RESET", None, obs["actions_taken"], cle)
                return {"name": "RESET", "data": {}}
            else:   # tout est exploré : action aléatoire parmi les disponibles
                choix = bras_dispo[int(self.rng.integers(len(bras_dispo)))] if bras_dispo else ("A", 1)
        self.attente = (choix, self.derniere_frame, obs["actions_taken"], cle)
        if choix[0] == "A":
            return {"name": f"ACTION{choix[1]}", "data": {}}
        return {"name": "ACTION6", "data": {"x": int(choix[1]), "y": int(choix[2])}}
