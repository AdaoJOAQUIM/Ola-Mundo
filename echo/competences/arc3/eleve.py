"""L'élève : un cerveau SANS modèle de langage qui exécute des compétences (Python déterministe, liste
blanche AST) à pleine vitesse, avec le cerveau bayésien de echo/arc3/cerveau.py comme comportement par défaut.

Aucun appel réseau, aucun import de client de modèle. Les budgets sont en nombres d'actions.
Protocole d'une compétence : activation(h) -> bool ; agir(h) -> action ; terminer(h) -> bool.
"""
from __future__ import annotations

import signal
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
sys.path.insert(0, str(ICI.parent.parent / "arc3"))

import cerveau as cerveau_mod      # noqa: E402  (baseline gelée, echo/arc3/cerveau.py)
import liste_blanche               # noqa: E402
from outils import H               # noqa: E402

FENETRE_SUCCES = 25     # actions après la fin d'une activation où un gain de niveau est encore attribué
DELAI_APPEL = 0.5       # secondes (garde-fou ; la borne structurelle est la liste blanche)


class _Delai(Exception):
    pass


def _alarme(signum, frame):
    raise _Delai()


class Competence:
    def __init__(self, nom, source):
        self.nom = nom
        self.source = source
        ns = liste_blanche.charger(source, nom)
        import builtins
        # les imports paresseux de numpy (niveau C) cherchent __import__ dans les builtins du cadre courant ;
        # la liste blanche interdit tout nom contenant « __ », donc une compétence ne peut pas l'appeler.
        ns["activation"].__globals__["__builtins__"]["__import__"] = builtins.__import__
        self.activation = ns["activation"]
        self.agir = ns["agir"]
        self.terminer = ns["terminer"]
        self.budget = int(ns.get("BUDGET", 30))
        self.cooldown = int(ns.get("COOLDOWN", 20))
        self.mem = {}
        self.glob = {}
        self.stats = {"activations": 0, "succes": 0, "actions": 0, "erreurs": 0, "delais": 0}
        self.libre_a_partir_de = 0


def _appel(fn, h):
    ok_sig = hasattr(signal, "setitimer")
    if ok_sig:
        try:
            signal.signal(signal.SIGALRM, _alarme)
            signal.setitimer(signal.ITIMER_REAL, DELAI_APPEL)
        except ValueError:
            ok_sig = False
    try:
        return fn(h)
    finally:
        if ok_sig:
            signal.setitimer(signal.ITIMER_REAL, 0)


class Eleve:
    def __init__(self, graine=0, prieurs=None, bassin=None):
        """bassin : liste de dicts {"nom":..., "source":...}, par ordre de priorité."""
        self.cerv = cerveau_mod.Cerveau(graine, prieurs)
        self.h = H()
        self.pool = [Competence(b["nom"], b["source"]) for b in (bassin or [])]
        self.actif = None            # (Competence, nb_actions_executees)
        self.fenetres = []           # activations en attente d'attribution du succès
        self.derniere = None
        self.trace = []              # (n, nom, action) pour l'analyse de l'enseignant
        self.stats = {"actions_competences": 0, "actions_cerveau": 0}
        self._n_vu = -1

    # ------------------------------------------------------------------ comptabilité
    def _maj(self, obs):
        n = int(obs["actions_taken"])
        if n == self._n_vu and self.h.f is not None:
            return
        self._n_vu = n
        lv_avant = self.h.niv
        nouveau_niveau = self.h.maj(obs, self.cerv.masque, self.derniere)
        for w in self.fenetres:
            if w["ouvert"] and self.h.niv > w["niv0"]:
                w["succes"] = True
        if nouveau_niveau and self.h.niv > lv_avant:
            if self.actif is not None:
                self._cloturer(succes=True)
        if nouveau_niveau:
            for c in self.pool:
                c.mem = {}
        for w in list(self.fenetres):
            if w["ferme_n"] is not None and n > w["ferme_n"] + FENETRE_SUCCES and w["ouvert"]:
                self._finaliser(w)
            elif w["succes"] and w["ferme_n"] is not None:
                self._finaliser(w)

    def _finaliser(self, w):
        w["ouvert"] = False
        if w["succes"]:
            w["comp"].stats["succes"] += 1
        if w in self.fenetres:
            self.fenetres.remove(w)

    def _cloturer(self, succes=False):
        c, nex = self.actif
        for w in self.fenetres:
            if w["comp"] is c and w["ferme_n"] is None:
                w["ferme_n"] = self.h.n
                if succes or self.h.niv > w["niv0"]:
                    w["succes"] = True
        c.libre_a_partir_de = self.h.n + max(1, c.cooldown)
        self.actif = None

    def bilan(self):
        """Statistiques par compétence (activations, succès) + fin des fenêtres encore ouvertes."""
        for w in list(self.fenetres):
            self._finaliser(w)
        return {c.nom: dict(c.stats) for c in self.pool}

    # ------------------------------------------------------------------ cerveau
    def _vers_cerveau(self, act, obs):
        cv = self.cerv
        cv.observer(obs)
        n = obs["actions_taken"]
        if act[0] == "A":
            bras = ("A", act[1])
        elif act[0] == "R":
            cv.attente = ("RESET", None, n, cv.cle)
            return {"name": "RESET", "data": {}}
        else:
            x, y = act[1], act[2]
            fr = cv.derniere_frame
            col = int(fr[y, x])
            taille = int((fr == col).sum()) if col >= 0 else 1
            bras = ("C", x, y, col, cerveau_mod._classe_taille(min(taille, 64)))
        cv.attente = (bras, cv.derniere_frame, n, cv.cle)
        if act[0] == "A":
            return {"name": f"ACTION{act[1]}", "data": {}}
        return {"name": "ACTION6", "data": {"x": int(act[1]), "y": int(act[2])}}

    @staticmethod
    def _valide(act, dispo):
        try:
            if not isinstance(act, tuple) or not act:
                return False
            if act[0] == "A":
                return len(act) == 2 and type(act[1]) is int and act[1] in dispo and act[1] != 6
            if act[0] == "C":
                return (len(act) == 3 and 6 in dispo and type(act[1]) is int and type(act[2]) is int
                        and 0 <= act[1] < 64 and 0 <= act[2] < 64)
            return act == ("R",)
        except Exception:
            return False

    def _essayer(self, c, fn):
        try:
            return True, _appel(fn, self.h)
        except _Delai:
            c.stats["delais"] += 1
        except Exception:
            c.stats["erreurs"] += 1
        return False, None

    # ------------------------------------------------------------------ boucle
    def choisir(self, obs):
        self._maj(obs)
        h = self.h
        act = None
        if self.pool:
            if self.actif is None:
                for c in self.pool:
                    if h.n < c.libre_a_partir_de:
                        continue
                    h.mem, h.glob = c.mem, c.glob
                    ok, val = self._essayer(c, c.activation)
                    if ok and val is True:
                        self.actif = [c, 0]
                        c.stats["activations"] += 1
                        self.fenetres.append({"comp": c, "niv0": h.niv, "ferme_n": None, "succes": False,
                                              "ouvert": True})
                        break
                    if not ok:
                        c.libre_a_partir_de = h.n + 200
            if self.actif is not None:
                c, nex = self.actif
                h.mem, h.glob = c.mem, c.glob
                fini = nex >= c.budget
                if not fini:
                    ok, val = self._essayer(c, c.terminer)
                    fini = (not ok) or (val is True)
                if not fini:
                    ok, val = self._essayer(c, c.agir)
                    if ok and self._valide(val, h.dispo):
                        act = val
                        self.actif[1] += 1
                        c.stats["actions"] += 1
                    else:
                        fini = True
                if fini:
                    self._cloturer()
        if act is None:
            self.stats["actions_cerveau"] += 1
            self.derniere = None
            r = self.cerv.choisir(obs)
            self.derniere = self._action_de(r)
            return r
        self.stats["actions_competences"] += 1
        self.trace.append((h.n, self.actif[0].nom, act))
        self.derniere = act
        return self._vers_cerveau(act, obs)

    @staticmethod
    def _action_de(r):
        if r["name"] == "RESET":
            return ("R",)
        if r["name"] == "ACTION6":
            return ("C", r["data"]["x"], r["data"]["y"])
        return ("A", int(r["name"][-1]))

    def observer(self, obs):
        self._maj(obs)
        self.cerv.observer(obs)
