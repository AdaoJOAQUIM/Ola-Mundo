"""Liste blanche AST pour les compétences (même esprit que echo/evolution/garde_fous.py,
qui reste intact : ici on ne l'importe pas, on reprend le schéma Rejet / verifier).

Une compétence est un module Python minuscule :
    NOM = "..."           (str)       BUDGET = 30 (actions max par activation)    COOLDOWN = 20
    def activation(h): -> bool        def agir(h): -> action ("A",n) | ("C",x,y) | ("R",)
    def terminer(h): -> bool
plus d'éventuelles fonctions auxiliaires NON récursives (un appel n'est permis que vers
une fonction définie plus haut dans le fichier).

Interdits : import, while, classes, lambda, try, with, global, f-strings, dunder, exec/eval/open,
tout attribut hors liste blanche, boucles for non bornées (range constant <= 256, ou itération
sur un objet de l'API), imbrication de boucles > 3, source > 7000 caractères.
Le temps est borné deux fois : structurellement (pas de while) et par une alarme à l'exécution.
"""
import ast

TAILLE_MAX = 7000
NOEUDS_MAX = 2600
PROFONDEUR_MAX = 40
IMBRICATION_MAX = 3
RANGE_MAX = 256
BUDGET_MAX = 120

REQUIS = ("activation", "agir", "terminer")
CONSTANTES = {"NOM": str, "BUDGET": int, "COOLDOWN": int}
BUILTINS_OK = {"len", "min", "max", "abs", "range", "int", "float", "bool", "sorted", "sum", "any",
               "all", "enumerate", "zip", "list", "tuple", "set", "dict", "round", "reversed", "divmod"}
ATTRS_API = {  # attributs / méthodes de l'objet h (voir outils.py)
    "f", "p", "f0", "dispo", "niv", "n", "nn", "reste", "etat", "der", "dchg", "mem", "glob", "masque",
    "log", "fx", "vus", "nouveau", "niv_actions", "A", "C", "R", "objets", "chgmask", "boites",
    "mouvements", "chemin", "hach", "dans", "couleurs", "rect", "essais", "dernier_gain"}
ATTRS_NP = {"array", "zeros", "ones", "argwhere", "abs", "sign", "where", "unique", "nonzero", "any", "all",
            "sum", "max", "min", "argmin", "argmax", "mean", "int8", "int32", "bool_", "count_nonzero",
            "clip", "minimum", "maximum", "arange", "bincount", "flatnonzero", "roll", "equal"}
ATTRS_OBJ = {"shape", "sum", "any", "all", "copy", "tolist", "astype", "max", "min", "argmax", "argmin",
             "nonzero", "flatten", "reshape", "mean", "append", "extend", "pop", "get", "items", "keys",
             "values", "index", "count", "add", "discard", "update", "setdefault", "sort", "insert",
             "T", "ravel", "size"}
ATTRS_OK = ATTRS_API | ATTRS_NP | ATTRS_OBJ
BASES_NP = {"np"}

NOEUDS_OK = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.AugAssign, ast.If, ast.For,
             ast.Return, ast.Expr, ast.Pass, ast.Break, ast.Continue, ast.BinOp, ast.BoolOp, ast.UnaryOp,
             ast.Compare, ast.IfExp, ast.Call, ast.Attribute, ast.Subscript, ast.Slice, ast.Name, ast.Constant,
             ast.Tuple, ast.List, ast.Dict, ast.Set, ast.ListComp, ast.GeneratorExp, ast.SetComp,
             ast.DictComp, ast.comprehension, ast.Load, ast.Store, ast.And, ast.Or, ast.Not, ast.USub,
             ast.UAdd, ast.Invert, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
             ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
             ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Starred)


class Rejet(Exception):
    def __init__(self, code, detail=""):
        super().__init__("%s %s" % (code, detail))
        self.code, self.detail = code, detail


def _profondeur(nd, d=0):
    m = d
    for c in ast.iter_child_nodes(nd):
        m = max(m, _profondeur(c, d + 1))
    return m


class _Verif:
    def __init__(self):
        self.helpers = set()      # fonctions auxiliaires déjà définies
        self.locaux = set()

    def nom(self, nd):
        i = nd.id
        if "__" in i:
            raise Rejet("dunder", i)
        ok = (i in BUILTINS_OK or i in BASES_NP or i in self.locaux or i in self.helpers
              or i in CONSTANTES or i in ("True", "False", "None", "h"))
        if not ok:
            raise Rejet("nom_interdit", i)

    def expr(self, nd, nest):
        for c in ast.walk(nd):          # 1er passage : variables liées localement (compréhensions)
            if isinstance(c, ast.comprehension):
                for t in ast.walk(c.target):
                    if isinstance(t, ast.Name):
                        self.locaux.add(t.id)
        for c in ast.walk(nd):
            if isinstance(c, ast.Attribute):
                if "__" in c.attr or c.attr not in ATTRS_OK:
                    raise Rejet("attribut_interdit", c.attr)
                if isinstance(c.value, ast.Name) and c.value.id in BASES_NP and c.attr not in ATTRS_NP:
                    raise Rejet("attribut_interdit", "np." + c.attr)
            elif isinstance(c, ast.Name):
                self.nom_ou_comp(c)
            elif isinstance(c, ast.Constant):
                if isinstance(c.value, (bytes, complex)) or (isinstance(c.value, int) and abs(c.value) > 10 ** 9):
                    raise Rejet("constante_interdite", repr(c.value)[:20])
            elif isinstance(c, ast.Call):
                f = c.func
                if isinstance(f, ast.Name):
                    if f.id not in BUILTINS_OK and f.id not in self.helpers:
                        raise Rejet("appel_interdit", f.id)
                elif not isinstance(f, ast.Attribute):
                    raise Rejet("appel_interdit", type(f).__name__)
            elif isinstance(c, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
                for g in c.generators:
                    self.iter_ok(g.iter)
                    for t in ast.walk(g.target):
                        if isinstance(t, ast.Name):
                            self.locaux.add(t.id)

    def nom_ou_comp(self, c):
        if isinstance(c.ctx, ast.Store):
            if "__" in c.id:
                raise Rejet("dunder", c.id)
            return
        self.nom(c)

    def iter_ok(self, it):
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
            if it.keywords or not (1 <= len(it.args) <= 3):
                raise Rejet("boucle_interdite", "range")
            for a in it.args:
                if isinstance(a, ast.Constant):
                    if not (type(a.value) is int and -RANGE_MAX <= a.value <= RANGE_MAX):
                        raise Rejet("boucle_interdite", "borne")
                elif not self._borne_dyn(a):
                    raise Rejet("boucle_interdite", "borne non constante")
        # sinon : itération sur une liste/dict/array déjà borné(e) par l'API

    def _borne_dyn(self, a):
        """range(len(x)), range(min(CONST, ...)) : bornes dynamiques admises mais plafonnées au runtime."""
        return isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id in ("len", "min")

    def stmts(self, body, nest):
        for s in body:
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                            if "__" in x.id or x.id in CONSTANTES or x.id == "h":
                                raise Rejet("affectation_interdite", x.id)
                            self.locaux.add(x.id)
                        elif isinstance(x, ast.Attribute):
                            raise Rejet("affectation_attribut")
                self.expr(s.value, nest)
                for t in s.targets:
                    self.expr(t, nest)
            elif isinstance(s, ast.AugAssign):
                if isinstance(s.target, ast.Name):
                    self.locaux.add(s.target.id)
                self.expr(s.value, nest)
                self.expr(s.target, nest)
            elif isinstance(s, ast.If):
                self.expr(s.test, nest)
                self.stmts(s.body, nest)
                self.stmts(s.orelse, nest)
            elif isinstance(s, ast.For):
                if nest + 1 > IMBRICATION_MAX:
                    raise Rejet("imbrication", str(nest + 1))
                if s.orelse:
                    raise Rejet("boucle_interdite", "else")
                self.expr(s.iter, nest)
                self.iter_ok(s.iter)
                for x in ast.walk(s.target):
                    if isinstance(x, ast.Name):
                        self.locaux.add(x.id)
                self.stmts(s.body, nest + 1)
            elif isinstance(s, ast.Return):
                if s.value is not None:
                    self.expr(s.value, nest)
            elif isinstance(s, ast.Expr):
                if not isinstance(s.value, ast.Call):
                    raise Rejet("expression_seule")
                self.expr(s.value, nest)
            elif isinstance(s, (ast.Pass, ast.Break, ast.Continue)):
                pass
            else:
                raise Rejet("noeud_interdit", type(s).__name__)


def verifier(source):
    """(True, '', '') ou (False, code, detail)."""
    try:
        if not isinstance(source, str):
            raise Rejet("type_source")
        if len(source) > TAILLE_MAX:
            raise Rejet("taille", str(len(source)))
        try:
            arbre = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
            raise Rejet("syntaxe", type(e).__name__)
        n = 0
        for nd in ast.walk(arbre):
            n += 1
            if isinstance(nd, ast.Name) and "__" in nd.id:
                raise Rejet("dunder", nd.id)
            if isinstance(nd, ast.Attribute) and "__" in nd.attr:
                raise Rejet("dunder", nd.attr)
            if not isinstance(nd, NOEUDS_OK):
                raise Rejet("noeud_interdit", type(nd).__name__)
        if n > NOEUDS_MAX:
            raise Rejet("taille", "%d noeuds" % n)
        if _profondeur(arbre) > PROFONDEUR_MAX:
            raise Rejet("trop_profond")
        v = _Verif()
        vus = set()
        fonctions = {}
        for s in arbre.body:
            if isinstance(s, ast.Assign):
                if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                    raise Rejet("constante", "cible")
                nm = s.targets[0].id
                if nm not in CONSTANTES or not (isinstance(s.value, ast.Constant)
                                                 and type(s.value.value) is CONSTANTES[nm]):
                    raise Rejet("constante", nm)
                if nm == "BUDGET" and not (1 <= s.value.value <= BUDGET_MAX):
                    raise Rejet("constante", "BUDGET hors bornes")
                if nm == "COOLDOWN" and not (0 <= s.value.value <= 500):
                    raise Rejet("constante", "COOLDOWN hors bornes")
                vus.add(nm)
            elif isinstance(s, ast.FunctionDef):
                a = s.args
                if s.decorator_list or s.returns or a.vararg or a.kwarg or a.kwonlyargs or a.defaults or a.posonlyargs:
                    raise Rejet("signature", s.name)
                if "__" in s.name or s.name in v.helpers or s.name in fonctions:
                    raise Rejet("signature", "nom " + s.name)
                if s.name in REQUIS and [x.arg for x in a.args] != ["h"]:
                    raise Rejet("signature", s.name)
                if any(x.annotation for x in a.args):
                    raise Rejet("signature", "annotation")
                fonctions[s.name] = s
                v.locaux = {x.arg for x in a.args}
                v.stmts(s.body, 0)
                # un appel récursif (ou vers une fonction non encore définie) aurait été refusé par expr()
                v.helpers.add(s.name)
            elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                pass   # docstring de module
            else:
                raise Rejet("noeud_interdit", "niveau module : " + type(s).__name__)
        for r in REQUIS:
            if r not in fonctions:
                raise Rejet("structure", "fonction manquante " + r)
        if "NOM" not in vus:
            raise Rejet("structure", "NOM manquant")
        return True, "", ""
    except Rejet as r:
        return False, r.code, r.detail
    except RecursionError:
        return False, "trop_profond", ""


def charger(source, nom_fichier="<competence>"):
    """Vérifie puis exécute la source dans un espace de noms restreint ; renvoie un dict de fonctions/constantes."""
    ok, code, detail = verifier(source)
    if not ok:
        raise Rejet(code, detail)
    import numpy as np
    sûrs = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
            for k in BUILTINS_OK}
    ns = {"__builtins__": sûrs, "np": np}
    exec(compile(source, nom_fichier, "exec"), ns)   # noqa : source déjà passée par la liste blanche
    return {k: ns[k] for k in ns if k not in ("__builtins__", "np")}
