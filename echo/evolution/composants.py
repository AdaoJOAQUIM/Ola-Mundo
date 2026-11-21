"""Cinq composants ajoutés au moteur : soi, objectif, coût, convergence, émergence.

Méthodes classiques réutilisées (rien d'inventé ici) :
  * SOI : régression logistique + calibration par température (idée de echo/soi.py,
    dont on importe directement ece() et brier_top1()) ;
  * OBJECTIF : longueur de description (MDL, taille compressée du génome) et front
    de Pareto par tri non dominé (NSGA-II, Deb et al. 2002) ;
  * COÛT : temps CPU mesuré par évaluation (idée de echo/aiguilleur.py : cpu_s mesuré) ;
  * CONVERGENCE : arrêt précoce (« early stopping ») avec patience et seuil delta ;
  * ÉMERGENCE : archive MAP-Elites (Mouret et Clune 2015), simple grille (pyribs non utilisé).

Les fichiers épinglés (juge, garde_fous, données) ne sont PAS modifiés : la mesure du
CPU est faite ici, autour de la tâche d'évaluation d'origine.
"""
import ast
import math
import os
import sys
import time
import zlib

import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
sys.path.insert(0, os.path.dirname(ICI))   # echo/ : pour importer soi.py

import garde_fous  # noqa: E402
import mutations  # noqa: E402

try:
    import soi as soi_echo  # calibration d'Écho : ece(), brier_top1()
except Exception:  # import impossible : copie minimale des deux fonctions
    soi_echo = None


def ece(paires, nb=10):
    if soi_echo is not None:
        return soi_echo.ece(paires, nb)
    N = len(paires)
    tot = 0.0
    for k in range(nb):
        s = [(c, o) for c, o in paires if min(nb - 1, int(c * nb)) == k]
        if s:
            tot += len(s) / N * abs(sum(o for _, o in s) / len(s) - sum(c for c, _ in s) / len(s))
    return tot


def brier(paires):
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in paires) / max(1, len(paires))


# ---------------------------------------------------------------------------
# 3. COÛT : évaluateur qui mesure le CPU de chaque évaluation
# ---------------------------------------------------------------------------
def _tache_mesuree(args):
    t0 = time.process_time()
    r = garde_fous._tache(args)          # tâche d'origine : délai, mémoire, juge épinglé
    r["cpu_s"] = time.process_time() - t0
    return r


class EvaluateurMesure(garde_fous.Evaluateur):
    """Même garde-fous ; en plus, chaque résultat porte `cpu_s`."""

    def evaluer_lot(self, sources, nom_jeu="A", pertes=False):
        resultats = [None] * len(sources)
        a_evaluer = []
        for i, s in enumerate(sources):
            ok, code, detail = garde_fous.verifier(s)
            if not ok:
                resultats[i] = {"ok": False, "raison": code, "detail": detail, "cpu_s": 0.0}
                self.noter_rejet(code, detail, s)
            else:
                a_evaluer.append(i)
        if a_evaluer:
            asyncs = [(i, self.pool.apply_async(_tache_mesuree, ((sources[i], nom_jeu, self.delai, pertes),)))
                      for i in a_evaluer]
            limite = time.time() + self.delai * (len(a_evaluer) // self.nproc + 1) + 15
            relancer = False
            for i, ar in asyncs:
                try:
                    r = ar.get(timeout=max(0.1, limite - time.time()))
                except Exception as e:
                    r = {"ok": False, "raison": "delai" if "Timeout" in type(e).__name__ else "travailleur_mort",
                         "detail": type(e).__name__, "cpu_s": self.delai}
                    relancer = True
                resultats[i] = r
                self.nb_eval += 1
                if not r["ok"]:
                    self.noter_rejet(r["raison"], r.get("detail", ""), sources[i])
            if relancer:
                self.fermer()
                self._demarrer()
        return resultats


# ---------------------------------------------------------------------------
# 2. OBJECTIF : MDL + front de Pareto (NSGA-II)
# ---------------------------------------------------------------------------
def taille_mdl_bits(source):
    """Longueur de description du génome : octets compressés (zlib) x 8."""
    return 8 * len(zlib.compress(source.encode("utf-8"), 9))


def domine(a, b):
    """a domine b (minimisation) : pas pire partout, strictement mieux quelque part."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def tri_non_domine(objs):
    """Tri non dominé rapide de NSGA-II. Retourne la liste des rangs (0 = front de Pareto)."""
    n = len(objs)
    dom_par = [[] for _ in range(n)]
    cpt = [0] * n
    rang = [0] * n
    front = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if domine(objs[i], objs[j]):
                dom_par[i].append(j)
            elif domine(objs[j], objs[i]):
                cpt[i] += 1
        if cpt[i] == 0:
            front.append(i)
    r = 0
    while front:
        suivant = []
        for i in front:
            rang[i] = r
            for j in dom_par[i]:
                cpt[j] -= 1
                if cpt[j] == 0:
                    suivant.append(j)
        r += 1
        front = suivant
    return rang


def front_pareto(objs):
    """Indices des points non dominés. Version rapide pour les grandes archives (2 objectifs ou plus)."""
    objs = list(objs)
    res = []
    ordre = sorted(range(len(objs)), key=lambda i: objs[i])
    for i in ordre:
        if not any(domine(objs[j], objs[i]) for j in res):
            res.append(i)
    return sorted(res)


def distance_encombrement(objs, idx):
    """Distance d'encombrement (crowding distance) de NSGA-II pour les points `idx`."""
    d = {i: 0.0 for i in idx}
    for m in range(len(objs[0])):
        tri = sorted(idx, key=lambda i: objs[i][m])
        d[tri[0]] = d[tri[-1]] = float("inf")
        ext = objs[tri[-1]][m] - objs[tri[0]][m]
        if ext <= 0:
            continue
        for a in range(1, len(tri) - 1):
            d[tri[a]] += (objs[tri[a + 1]][m] - objs[tri[a - 1]][m]) / ext
    return d


# ---------------------------------------------------------------------------
# 4. CONVERGENCE : arrêt précoce
# ---------------------------------------------------------------------------
class ArretPrecoce:
    """Arrête quand la meilleure valeur sur B n'a pas baissé de plus de `delta`
    pendant `patience` évaluations. Option : cible (arrêt dès que B <= cible)."""

    def __init__(self, delta=0.002, patience=1500, cible=None):
        self.delta, self.patience, self.cible = delta, patience, cible
        self.meilleur = None
        self.derniere_amelioration = 0
        self.raison = None

    def mise_a_jour(self, evals, valeur_B):
        if self.meilleur is None or valeur_B < self.meilleur - self.delta:
            self.meilleur = valeur_B
            self.derniere_amelioration = evals
        if self.cible is not None and valeur_B <= self.cible:
            self.raison = "cible_atteinte"
            return True
        if self.patience is not None and evals - self.derniere_amelioration >= self.patience:
            self.raison = "patience (aucun gain > %.4f sur B depuis %d évaluations)" % (self.delta, self.patience)
            return True
        return False


# ---------------------------------------------------------------------------
# 5. ÉMERGENCE : descripteurs et MAP-Elites
# ---------------------------------------------------------------------------
BORNES_TAILLE = (0, 45, 70, 100, 140, 10**9)   # en nœuds de l'AST


def _contient_remise(arbre):
    """Dans la boucle sur les ordres : on soustrait un terme DIRECTEMENT au compte brut n[k]
    (n[k] - D, comme la remise absolue de Kneser-Ney). Les autres soustractions ne comptent pas."""
    for boucle in (x for x in ast.walk(arbre) if isinstance(x, ast.For)):
        for nd in ast.walk(boucle):
            if isinstance(nd, ast.BinOp) and isinstance(nd.op, ast.Sub):
                g = nd.left
                if isinstance(g, ast.Subscript) and isinstance(g.value, ast.Name) and g.value.id == "n" \
                        and not isinstance(g.slice, ast.Constant):
                    return True
    return False


def caracteristiques(source):
    """Ensemble d'étiquettes décrivant la structure d'un génome."""
    arbre = ast.parse(source)
    noms = {x.id for x in ast.walk(arbre) if isinstance(x, ast.Name)}
    cars = set()
    for nm in ("u1", "u2"):   # seulement dans la boucle (un u2[0] hors boucle est écrasé : bruit)
        if any(isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name) and x.value.id == nm
               for b in ast.walk(arbre) if isinstance(b, ast.For) for x in ast.walk(b)):
            cars.add("utilise_" + nm)
    if _contient_remise(arbre):
        cars.add("remise")
    for x in ast.walk(arbre):
        if isinstance(x, ast.Subscript) and isinstance(x.value, (ast.List, ast.Tuple)):
            cars.add("constantes_par_ordre")
        if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Pow):
            cars.add("puissance")
        if isinstance(x, ast.Attribute) and x.attr in garde_fous.FONCTIONS_NP and x.attr != "where":
            cars.add("np." + x.attr)
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "div":
            cars.add("div")
    if noms & {"q", "r", "a", "b", "c", "m"}:
        cars.add("variable_intermediaire")
    return cars


def descripteur(source):
    """Case MAP-Elites : (classe de taille, u1/u2 utilisés, remise, constantes par ordre)."""
    arbre = ast.parse(source)
    nb = sum(1 for _ in ast.walk(arbre))
    taille = max(i for i, b in enumerate(BORNES_TAILLE[:-1]) if nb >= b)
    c = caracteristiques(source)
    return (taille, int(bool(c & {"utilise_u1", "utilise_u2"})), int("remise" in c),
            int("constantes_par_ordre" in c))


NB_CASES_TOTAL = (len(BORNES_TAILLE) - 1) * 2 * 2 * 2


class MapElites:
    """Grille : une case = le meilleur génome (objectif le plus bas) de cette structure."""

    def __init__(self):
        self.cases = {}

    def inserer(self, g, objectif):
        cle = descripteur(g["source"])
        cur = self.cases.get(cle)
        if cur is None or objectif < cur[0]:
            self.cases[cle] = (objectif, g["id"])
            return True
        return False

    def couverture(self):
        return len(self.cases) / NB_CASES_TOTAL

    def ids(self):
        return [v[1] for v in self.cases.values()]


# ---------------------------------------------------------------------------
# 1. SOI : prédire si un enfant bat son parent
# ---------------------------------------------------------------------------
def _sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def ajuster_logistique(X, y, l2=1.0, iters=25):
    """Régression logistique par Newton (IRLS) avec pénalité L2."""
    d = X.shape[1]
    w = np.zeros(d)
    for _ in range(iters):
        p = _sigmoide(X @ w)
        g = X.T @ (p - y) + l2 * w
        H = (X.T * (p * (1 - p))) @ X + l2 * np.eye(d)
        w = w - np.linalg.solve(H, g)
    return w


def ajuster_temperature(z, y, bas=0.3, haut=5.0, tol=1e-3):
    """Température T (section dorée) minimisant le Brier de sigmoïde(z / T), comme soi.py."""
    def f(T):
        return float(np.mean((_sigmoide(z / T) - y) ** 2))
    phi = (math.sqrt(5) - 1) / 2
    a, b = bas, haut
    c, d = b - phi * (b - a), a + phi * (b - a)
    while b - a > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - phi * (b - a), a + phi * (b - a)
    return (a + b) / 2


class Soi:
    """Auto-modèle : estime P(l'enfant bat son parent) avant de l'évaluer."""

    def __init__(self, ops=mutations.OPERATEURS, min_exemples=200, epsilon=0.1, facteur_seuil=0.35):
        self.ops = list(ops)
        self.min_exemples = min_exemples
        self.epsilon = epsilon
        self.facteur_seuil = facteur_seuil
        self.essais = {op: 0 for op in self.ops}
        self.succes = {op: 0 for op in self.ops}
        self.X, self.y = [], []
        self.w = None
        self.T = 1.0
        self.preq = []          # prédictions faites AVANT l'évaluation : (p, issue)
        self.nb_saute = 0
        self.nb_explore_bas = 0
        self.succes_explore_bas = 0
        self.nb_pas_saute_haut = 0
        self.succes_haut = 0

    def vecteur(self, mutations_noms, rang_parent, delta_taille, parent_est_front=0):
        x = [0.0] * len(self.ops)
        for m in mutations_noms:
            if m in self.ops:
                x[self.ops.index(m)] += 1.0
        taux = np.mean([(self.succes[m] + 1.0) / (self.essais[m] + 2.0) for m in mutations_noms]) \
            if mutations_noms else 0.0
        return np.array(x + [rang_parent, delta_taille / 100.0, 10 * taux, len(mutations_noms) / 3.0, 1.0])

    def predire(self, x):
        if self.w is None:
            return None
        return float(_sigmoide(x @ self.w / self.T))

    def base(self):
        return (sum(self.y) / len(self.y)) if self.y else 0.05

    def decider(self, x, rng):
        """Retourne (evaluer ?, p_pred). On saute si p << taux de base, sauf exploration epsilon."""
        p = self.predire(x)
        if p is None:
            return True, None
        if p < self.facteur_seuil * self.base():
            if rng.random() < self.epsilon:
                return True, p      # exploration : on évalue quand même
            self.nb_saute += 1
            return False, p
        return True, p

    def observer(self, x, p_pred, mutations_noms, issue):
        issue = 1.0 if issue else 0.0
        self.X.append(x)
        self.y.append(issue)
        for m in mutations_noms:
            if m in self.ops:
                self.essais[m] += 1
                self.succes[m] += int(issue)
        if p_pred is not None:
            self.preq.append((p_pred, issue))
            if p_pred < self.facteur_seuil * self.base():
                self.nb_explore_bas += 1
                self.succes_explore_bas += int(issue)
            else:
                self.nb_pas_saute_haut += 1
                self.succes_haut += int(issue)

    def reentrainer(self):
        n = len(self.y)
        if n < self.min_exemples or sum(self.y) < 5:
            return False
        X, y = np.array(self.X), np.array(self.y)
        c = int(n * 0.75)       # 75 % pour les poids, 25 % (les plus récents) pour la température
        self.w = ajuster_logistique(X[:c], y[:c])
        if sum(y[c:]) >= 1:
            self.T = ajuster_temperature(X[c:] @ self.w, y[c:])
        return True

    def rapport(self):
        paires = [(p, bool(o)) for p, o in self.preq]
        out = {"exemples": len(self.y), "taux_succes_observe": self.base(), "enfants_sautes": self.nb_saute,
               "temperature": self.T, "predictions_avant_evaluation": len(paires)}
        if paires:
            out["ECE_prequentiel"] = ece(paires)
            out["Brier_prequentiel"] = brier(paires)
            out["Brier_constant_taux_base"] = brier([(self.base(), o) for _, o in paires])
        out["exploration_bas_p"] = {"n": self.nb_explore_bas, "taux_succes": (
            self.succes_explore_bas / self.nb_explore_bas if self.nb_explore_bas else None)}
        out["autres_evalues"] = {"n": self.nb_pas_saute_haut, "taux_succes": (
            self.succes_haut / self.nb_pas_saute_haut if self.nb_pas_saute_haut else None)}
        out["succes_par_operateur"] = {op: (self.succes[op] / self.essais[op] if self.essais[op] else None)
                                       for op in self.ops}
        return out
