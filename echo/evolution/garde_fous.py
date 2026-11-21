"""Garde-fous : liste blanche AST, évaluation isolée, empreintes SHA-256.

Principe : le système qui se modifie lui-même ne doit jamais pouvoir modifier
son juge. Donc :
  1. tout génome passe par une liste blanche stricte avant d'être exécuté ;
  2. l'exécution se fait dans des processus séparés, avec délai et limite mémoire ;
  3. les fichiers du juge et des données sont épinglés par SHA-256 (empreintes.json) ;
     le moteur vérifie au départ et périodiquement, et s'arrête à la moindre différence.
"""
import ast
import hashlib
import json
import math
import multiprocessing as mp
import os
import resource
import signal
import time
from collections import Counter

ICI = os.path.dirname(os.path.abspath(__file__))
CHEMIN_EMPREINTES = os.path.join(ICI, "empreintes.json")

# ----------------------------------------------------------------------------
# 1. Liste blanche
# ----------------------------------------------------------------------------
TAILLE_MAX_SOURCE = 2500      # caractères
NOEUDS_MAX = 450
PROFONDEUR_MAX = 30
POW_MAX = 3                   # nombre maximal de puissances (**)
FONCTIONS_NP = {"maximum", "minimum", "sqrt", "log1p", "exp", "abs", "where", "clip", "tanh"}
ARGS_FONCTION = {"maximum": (2,), "minimum": (2,), "sqrt": (1,), "log1p": (1,), "exp": (1,),
                 "abs": (1,), "where": (3,), "clip": (3,), "tanh": (1,), "div": (2,)}
DONNEES = ("n", "N", "d", "u1", "u2")
ARGUMENTS = DONNEES + ("K",)
VARIABLES_LOCALES = ("p", "q", "r", "e", "a", "b", "c", "m")
OPS_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
OPS_CMP = (ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq)
NOEUDS_OK = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.For,
             ast.Return, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Name, ast.Constant,
             ast.List, ast.Tuple, ast.Subscript, ast.Call, ast.Attribute, ast.Load,
             ast.Store, ast.USub, ast.UAdd) + OPS_BIN + OPS_CMP


class Rejet(Exception):
    def __init__(self, code, detail=""):
        super().__init__("%s %s" % (code, detail))
        self.code = code
        self.detail = detail


def _constante_num(nd):
    return (isinstance(nd, ast.Constant) and type(nd.value) in (int, float)
            and math.isfinite(nd.value) and abs(nd.value) <= 1e6)


def _constante_signee(nd):
    if isinstance(nd, ast.UnaryOp) and isinstance(nd.op, (ast.USub, ast.UAdd)):
        return _constante_num(nd.operand)
    return _constante_num(nd)


def _est_k(nd):
    return isinstance(nd, ast.Name) and nd.id == "k"


def _indice_ok(sl, dans_boucle):
    """Indice autorisé : k, K, entier constant, ou k±entier."""
    if isinstance(sl, ast.Name):
        return (sl.id == "k" and dans_boucle) or sl.id == "K"
    if isinstance(sl, ast.Constant):
        return type(sl.value) is int and -1 <= sl.value <= 16
    if isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub):
        return isinstance(sl.operand, ast.Constant) and sl.operand.value == 1 and type(sl.operand.value) is int
    if isinstance(sl, ast.BinOp) and isinstance(sl.op, (ast.Add, ast.Sub)):
        return (dans_boucle and _est_k(sl.left) and isinstance(sl.right, ast.Constant)
                and type(sl.right.value) is int and 0 <= sl.right.value <= 3)
    return False


class _Verif:
    def __init__(self):
        self.pow = 0
        self.noeuds = 0

    # -- expressions --
    def expr(self, nd, dans_boucle, definis, prof=0):
        self.noeuds += 1
        if prof > PROFONDEUR_MAX:
            raise Rejet("trop_profond")
        if isinstance(nd, ast.Name):
            if "__" in nd.id:
                raise Rejet("dunder", nd.id)
            if nd.id == "k":
                if not dans_boucle:
                    raise Rejet("nom_non_defini", "k hors boucle")
                return
            if nd.id in ARGUMENTS or nd.id in definis:
                return
            if nd.id in VARIABLES_LOCALES:
                raise Rejet("nom_non_defini", nd.id)
            raise Rejet("nom_interdit", nd.id)
        if isinstance(nd, ast.Constant):
            if not _constante_num(nd):
                raise Rejet("constante_interdite", repr(nd.value)[:30])
            return
        if isinstance(nd, ast.BinOp):
            if not isinstance(nd.op, OPS_BIN):
                raise Rejet("operateur_interdit", type(nd.op).__name__)
            if isinstance(nd.op, ast.Pow):
                self.pow += 1
                if self.pow > POW_MAX:
                    raise Rejet("trop_de_puissances")
                ex = nd.right
                neg = isinstance(ex, ast.UnaryOp) and isinstance(ex.op, ast.USub)
                v = ex.operand if neg else ex
                if not (_constante_num(v) and abs(v.value) <= 8):
                    raise Rejet("puissance_non_bornee")
            self.expr(nd.left, dans_boucle, definis, prof + 1)
            self.expr(nd.right, dans_boucle, definis, prof + 1)
            return
        if isinstance(nd, ast.UnaryOp):
            if not isinstance(nd.op, (ast.USub, ast.UAdd)):
                raise Rejet("operateur_interdit", type(nd.op).__name__)
            self.expr(nd.operand, dans_boucle, definis, prof + 1)
            return
        if isinstance(nd, ast.Compare):
            if len(nd.ops) != 1 or not isinstance(nd.ops[0], OPS_CMP):
                raise Rejet("comparaison_interdite")
            self.expr(nd.left, dans_boucle, definis, prof + 1)
            self.expr(nd.comparators[0], dans_boucle, definis, prof + 1)
            return
        if isinstance(nd, ast.Call):
            if nd.keywords:
                raise Rejet("appel_interdit", "mots-cles")
            f = nd.func
            if isinstance(f, ast.Name):
                nom = f.id
                if nom != "div":
                    raise Rejet("appel_interdit", nom)
            elif isinstance(f, ast.Attribute):
                if "__" in f.attr:
                    raise Rejet("dunder", f.attr)
                if not (isinstance(f.value, ast.Name) and f.value.id == "np"):
                    raise Rejet("attribut_interdit", f.attr)
                if f.attr not in FONCTIONS_NP:
                    raise Rejet("attribut_interdit", "np." + f.attr)
                nom = f.attr
            else:
                raise Rejet("appel_interdit", type(f).__name__)
            if len(nd.args) not in ARGS_FONCTION[nom]:
                raise Rejet("appel_interdit", "nombre d'arguments de " + nom)
            self.noeuds += 1
            for a in nd.args:
                self.expr(a, dans_boucle, definis, prof + 1)
            return
        if isinstance(nd, ast.Subscript):
            v, sl = nd.value, nd.slice
            if isinstance(v, ast.Name) and v.id in DONNEES:
                if not _indice_ok(sl, dans_boucle):
                    raise Rejet("indice_interdit")
                self.noeuds += 2
                return
            if isinstance(v, ast.Attribute) and v.attr == "shape" and isinstance(v.value, ast.Name) \
                    and v.value.id in DONNEES:
                if not (isinstance(sl, ast.Constant) and type(sl.value) is int and -3 <= sl.value <= 2) \
                        and not (isinstance(sl, ast.UnaryOp) and isinstance(sl.operand, ast.Constant)):
                    raise Rejet("indice_interdit")
                return
            if isinstance(v, (ast.List, ast.Tuple)):
                if not (1 <= len(v.elts) <= 16 and all(_constante_signee(e) for e in v.elts)):
                    raise Rejet("liste_interdite")
                if not (isinstance(sl, ast.Name) and sl.id == "k" and dans_boucle):
                    raise Rejet("indice_interdit")
                return
            raise Rejet("indexation_interdite")
        if isinstance(nd, ast.Attribute):
            raise Rejet("attribut_interdit", nd.attr)
        raise Rejet("noeud_interdit", type(nd).__name__)

    # -- instructions --
    def corps(self, stmts, dans_boucle, definis):
        for s in stmts:
            self.noeuds += 1
            if isinstance(s, ast.Assign):
                if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                    raise Rejet("affectation_interdite")
                nom = s.targets[0].id
                if nom not in VARIABLES_LOCALES:
                    raise Rejet("nom_interdit", nom)
                self.expr(s.value, dans_boucle, definis)
                definis.add(nom)
            elif isinstance(s, ast.For):
                if dans_boucle:
                    raise Rejet("boucle_imbriquee")
                if not (isinstance(s.target, ast.Name) and s.target.id == "k") or s.orelse:
                    raise Rejet("boucle_interdite")
                it = s.iter
                if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"
                        and 1 <= len(it.args) <= 2 and not it.keywords):
                    raise Rejet("boucle_interdite", "pas range")
                for a in it.args:
                    if not self._borne_ok(a):
                        raise Rejet("boucle_interdite", "bornes")
                if not (1 <= len(s.body) <= 10):
                    raise Rejet("boucle_interdite", "corps")
                self.corps(s.body, True, definis)
            elif isinstance(s, ast.Return):
                raise Rejet("return_mal_place")
            else:
                raise Rejet("noeud_interdit", type(s).__name__)

    @staticmethod
    def _borne_ok(a):
        if isinstance(a, ast.Constant):
            return type(a.value) is int and 0 <= a.value <= 16
        if isinstance(a, ast.Name):
            return a.id == "K"
        if isinstance(a, ast.BinOp) and isinstance(a.op, (ast.Add, ast.Sub)):
            return (isinstance(a.left, ast.Name) and a.left.id == "K" and isinstance(a.right, ast.Constant)
                    and type(a.right.value) is int and 0 <= a.right.value <= 3)
        return False


def verifier(source):
    """Liste blanche. Retourne (True, '', '') ou (False, code_de_rejet, detail)."""
    try:
        if not isinstance(source, str):
            raise Rejet("type_source")
        if len(source) > TAILLE_MAX_SOURCE:
            raise Rejet("taille", str(len(source)))
        try:
            arbre = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
            raise Rejet("syntaxe", type(e).__name__)
        # premier passage : types de nœuds et noms dangereux (message précis)
        total = 0
        for nd in ast.walk(arbre):
            total += 1
            if isinstance(nd, ast.Name) and "__" in nd.id:
                raise Rejet("dunder", nd.id)
            if isinstance(nd, ast.Attribute) and "__" in nd.attr:
                raise Rejet("dunder", nd.attr)
            if not isinstance(nd, NOEUDS_OK):
                raise Rejet("noeud_interdit", type(nd).__name__)
        if total > NOEUDS_MAX:
            raise Rejet("taille", "%d noeuds" % total)
        if len(arbre.body) != 1 or not isinstance(arbre.body[0], ast.FunctionDef):
            raise Rejet("structure", "une seule fonction attendue")
        f = arbre.body[0]
        a = f.args
        if (f.name != "estimer" or f.decorator_list or f.returns or a.vararg or a.kwarg or a.kwonlyargs
                or a.defaults or a.posonlyargs or [x.arg for x in a.args] != list(ARGUMENTS)
                or any(x.annotation for x in a.args)):
            raise Rejet("structure", "signature")
        if len(f.body) < 2 or not isinstance(f.body[-1], ast.Return) or f.body[-1].value is None:
            raise Rejet("structure", "return final attendu")
        v = _Verif()
        definis = set()
        v.corps(f.body[:-1], False, definis)
        v.expr(f.body[-1].value, False, definis)
        return True, "", ""
    except Rejet as r:
        return False, r.code, r.detail
    except RecursionError:
        return False, "trop_profond", ""


# ----------------------------------------------------------------------------
# 2. Évaluation isolée
# ----------------------------------------------------------------------------
class DelaiDepasse(Exception):
    pass


def _alarme(signum, frame):
    raise DelaiDepasse()


def _initialiser_travailleur(mem_mo, jeux):
    global _JEUX
    _JEUX = jeux
    os.environ["OMP_NUM_THREADS"] = "1"
    lim = int(mem_mo) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except (ValueError, OSError):
        pass
    signal.signal(signal.SIGALRM, _alarme)


def _tache(args):
    source, nom_jeu, delai, pertes = args
    import juge  # import local : le travailleur n'utilise que le juge épinglé
    signal.alarm(int(delai))
    try:
        res = juge.juger(source, _JEUX[nom_jeu], pertes=pertes)
    except DelaiDepasse:
        res = {"ok": False, "raison": "delai", "detail": ""}
    except MemoryError:
        res = {"ok": False, "raison": "memoire", "detail": ""}
    except BaseException as e:  # noqa : le travailleur ne doit jamais mourir sur un génome
        res = {"ok": False, "raison": "exception", "detail": type(e).__name__}
    finally:
        signal.alarm(0)
    return res


class Evaluateur:
    """Pool de processus (au plus 4) qui évalue des sources avec délai et mémoire bornés."""

    def __init__(self, jeux, nproc=4, delai=20, mem_mo=3500):
        self.jeux = jeux
        self.nproc = min(4, max(1, nproc))
        self.delai = delai
        self.mem_mo = mem_mo
        self.pool = None
        self.rejets = Counter()       # par code de rejet
        self.journal = []             # (raison, detail, source) tronqué
        self.nb_eval = 0
        self._demarrer()

    def _demarrer(self):
        ctx = mp.get_context("fork")
        self.pool = ctx.Pool(self.nproc, initializer=_initialiser_travailleur,
                             initargs=(self.mem_mo, self.jeux))

    def fermer(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def noter_rejet(self, raison, detail, source):
        self.rejets[raison] += 1
        if len(self.journal) < 2000:
            self.journal.append({"raison": raison, "detail": detail, "source": source[:300]})

    def evaluer_lot(self, sources, nom_jeu="A", pertes=False):
        """Liste de résultats {'ok', 'bpc' | 'raison'} ; les rejets sont journalisés."""
        resultats = [None] * len(sources)
        a_evaluer = []
        for i, s in enumerate(sources):
            ok, code, detail = verifier(s)
            if not ok:
                resultats[i] = {"ok": False, "raison": code, "detail": detail}
                self.noter_rejet(code, detail, s)
            else:
                a_evaluer.append(i)
        if a_evaluer:
            asyncs = [(i, self.pool.apply_async(_tache, ((sources[i], nom_jeu, self.delai, pertes),)))
                      for i in a_evaluer]
            limite = time.time() + self.delai * (len(a_evaluer) // self.nproc + 1) + 15
            relancer = False
            for i, ar in asyncs:
                try:
                    r = ar.get(timeout=max(0.1, limite - time.time()))
                except mp.TimeoutError:
                    r = {"ok": False, "raison": "delai", "detail": "pool"}
                    relancer = True
                except Exception as e:  # travailleur tué (mémoire, etc.)
                    r = {"ok": False, "raison": "travailleur_mort", "detail": type(e).__name__}
                    relancer = True
                resultats[i] = r
                self.nb_eval += 1
                if not r["ok"]:
                    self.noter_rejet(r["raison"], r.get("detail", ""), sources[i])
            if relancer:
                self.fermer()
                self._demarrer()
        return resultats


# ----------------------------------------------------------------------------
# 3. Empreintes SHA-256
# ----------------------------------------------------------------------------
def sha256_fichier(chemin):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def fichiers_epingles():
    return ["juge.py", "garde_fous.py", "donnees.py", "cache/ab.npz", "cache/c.npz"]


def epingler(dossier=ICI, chemin=None):
    """Écrit empreintes.json (appelé une seule fois, à la main, après le précalcul)."""
    emp = {f: sha256_fichier(os.path.join(dossier, f)) for f in fichiers_epingles()}
    with open(chemin or CHEMIN_EMPREINTES, "w") as f:
        json.dump(emp, f, indent=2)
    return emp


def verifier_empreintes(dossier=ICI, chemin=None):
    """Retourne la liste des fichiers modifiés ou manquants (vide si tout est intact)."""
    with open(chemin or CHEMIN_EMPREINTES) as f:
        ref = json.load(f)
    mauvais = []
    for nom, h in ref.items():
        p = os.path.join(dossier, nom)
        if not os.path.exists(p) or sha256_fichier(p) != h:
            mauvais.append(nom)
    return mauvais


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "epingler":
        for k, v in epingler().items():
            print(k, v[:16])
