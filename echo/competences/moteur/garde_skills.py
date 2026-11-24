"""Liste blanche AST des competences de chargement + chargeur isole.

Meme esprit que echo/evolution/garde_fous.py (non modifie) : un module de competence n'est execute qu'apres une
verification stricte de son arbre syntaxique. Une competence est un module deterministe :

    NAME   = "nom"            (chaine)
    PARENT = "base4" | nom    (chaine : competence dont elle derive)
    MODE   = "add" | "replace"(chaine : ajouter, ou remplacer PARENT dans la reserve)
    CONSTANTES = nombres ou listes/tuples de nombres (tables)
    def applies(L, e, w, r): -> bool     condition d'activation (couche, expert, poids normalise, rang)
    def plan(L, e, w, r):    -> (bg, bd, tg, td)   execution :
          bg = bits gate/up (0 = ecarter l'expert, 2, 3, 4), bd = bits down (2, 3, 4),
          tg = seuil relatif de bloc (rms du bloc / rms du vecteur) sous lequel un bloc de 32 canaux d'entree de
               gate/up n'est pas lu (0 = tout lire), td = idem pour les canaux intermediaires de down
    def fallback(L, e, w, r, perte): -> bool   terminaison : si vrai (perte = energie d'entree ignoree par tg,
          fraction de 0 a 1), la competence se retire et l'emplacement passe a la competence suivante

Interdits : import, attribut (point), boucle, comprehension, lambda, appel autre que min/max/abs/int/float, noms
a double souligne, puissance, global, with, try, classe, texte hors NAME/PARENT/MODE.
"""
import ast
import hashlib
import math

TAILLE_MAX = 4000
NOEUDS_MAX = 900
PROFONDEUR_MAX = 24
TABLE_MAX = 2048
APPELS_OK = {"min": (2, 3), "max": (2, 3), "abs": (1,), "int": (1,), "float": (1,)}
SIGNATURES = {"applies": ["L", "e", "w", "r"], "plan": ["L", "e", "w", "r"],
              "fallback": ["L", "e", "w", "r", "perte"]}
ENTETES_TEXTE = ("NAME", "PARENT", "MODE")
OPS_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv)
OPS_CMP = (ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq)
NOEUDS_OK = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.Return, ast.If, ast.IfExp,
             ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.BinOp, ast.Compare,
             ast.Name, ast.Constant, ast.Tuple, ast.List, ast.Subscript, ast.Call, ast.Load, ast.Store,
             ast.Expr) + OPS_BIN + OPS_CMP


class Rejet(Exception):
    def __init__(self, code, detail=""):
        super().__init__("%s %s" % (code, detail))
        self.code, self.detail = code, detail


def _num(nd):
    return (isinstance(nd, ast.Constant) and type(nd.value) in (int, float, bool)
            and (type(nd.value) is bool or math.isfinite(nd.value)) and abs(nd.value) <= 1e6)


def _num_signe(nd):
    if isinstance(nd, ast.UnaryOp) and isinstance(nd.op, (ast.USub, ast.UAdd)):
        return _num(nd.operand)
    return _num(nd)


class _V:
    def __init__(self):
        self.noeuds = 0
        self.tables = {}
        self.consts = set()

    def expr(self, nd, loc, prof=0):
        self.noeuds += 1
        if prof > PROFONDEUR_MAX:
            raise Rejet("trop_profond")
        if isinstance(nd, ast.Name):
            if "__" in nd.id:
                raise Rejet("dunder", nd.id)
            if nd.id in loc or nd.id in self.consts or nd.id in self.tables:
                return
            raise Rejet("nom_inconnu", nd.id)
        if isinstance(nd, ast.Constant):
            if not _num(nd):
                raise Rejet("constante_interdite", repr(nd.value)[:20])
            return
        if isinstance(nd, ast.BinOp):
            if not isinstance(nd.op, OPS_BIN):
                raise Rejet("operateur_interdit", type(nd.op).__name__)
            self.expr(nd.left, loc, prof + 1)
            self.expr(nd.right, loc, prof + 1)
            return
        if isinstance(nd, ast.UnaryOp):
            if not isinstance(nd.op, (ast.USub, ast.UAdd, ast.Not)):
                raise Rejet("operateur_interdit", type(nd.op).__name__)
            self.expr(nd.operand, loc, prof + 1)
            return
        if isinstance(nd, ast.BoolOp):
            for v in nd.values:
                self.expr(v, loc, prof + 1)
            return
        if isinstance(nd, ast.Compare):
            if len(nd.ops) != 1 or not isinstance(nd.ops[0], OPS_CMP):
                raise Rejet("comparaison_interdite")
            self.expr(nd.left, loc, prof + 1)
            self.expr(nd.comparators[0], loc, prof + 1)
            return
        if isinstance(nd, ast.IfExp):
            for v in (nd.test, nd.body, nd.orelse):
                self.expr(v, loc, prof + 1)
            return
        if isinstance(nd, ast.Tuple):
            if not 1 <= len(nd.elts) <= 8:
                raise Rejet("tuple_interdit")
            for v in nd.elts:
                self.expr(v, loc, prof + 1)
            return
        if isinstance(nd, ast.Call):
            if nd.keywords or not isinstance(nd.func, ast.Name) or nd.func.id not in APPELS_OK:
                raise Rejet("appel_interdit", getattr(nd.func, "id", type(nd.func).__name__))
            if len(nd.args) not in APPELS_OK[nd.func.id]:
                raise Rejet("appel_interdit", "arguments de " + nd.func.id)
            self.noeuds += 1
            for a in nd.args:
                self.expr(a, loc, prof + 1)
            return
        if isinstance(nd, ast.Subscript):
            if not (isinstance(nd.value, ast.Name) and nd.value.id in self.tables):
                raise Rejet("indexation_interdite")
            if isinstance(nd.slice, (ast.Slice, ast.Tuple)):
                raise Rejet("indexation_interdite", "tranche")
            self.expr(nd.slice, loc, prof + 1)
            return
        if isinstance(nd, ast.Attribute):
            raise Rejet("attribut_interdit", nd.attr)
        raise Rejet("noeud_interdit", type(nd).__name__)

    def instr(self, stmts, loc, retour_ok=True):
        for s in stmts:
            self.noeuds += 1
            if isinstance(s, ast.Assign):
                if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                    raise Rejet("affectation_interdite")
                n = s.targets[0].id
                if "__" in n or n in self.tables or n in self.consts or n in ("L", "e", "w", "r", "perte"):
                    raise Rejet("nom_interdit", n)
                self.expr(s.value, loc)
                loc.add(n)
            elif isinstance(s, ast.If):
                self.expr(s.test, loc)
                self.instr(s.body, set(loc))
                self.instr(s.orelse, set(loc))
            elif isinstance(s, ast.Return):
                if s.value is None:
                    raise Rejet("return_vide")
                self.expr(s.value, loc)
            else:
                raise Rejet("instruction_interdite", type(s).__name__)


def _termine(stmts):
    """Toutes les branches se terminent par return (pas de None implicite)."""
    if not stmts:
        return False
    d = stmts[-1]
    if isinstance(d, ast.Return):
        return True
    if isinstance(d, ast.If):
        return _termine(d.body) and _termine(d.orelse)
    return False


def verifier(source):
    """Retourne (True, '', '') ou (False, code, detail)."""
    try:
        if not isinstance(source, str):
            raise Rejet("type_source")
        if len(source) > TAILLE_MAX:
            raise Rejet("taille", str(len(source)))
        try:
            arbre = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
            raise Rejet("syntaxe", type(e).__name__)
        total = 0
        for nd in ast.walk(arbre):
            total += 1
            if isinstance(nd, ast.Name) and "__" in nd.id:
                raise Rejet("dunder", nd.id)
            if isinstance(nd, ast.Attribute):
                raise Rejet("attribut_interdit", nd.attr)
            if isinstance(nd, ast.Constant) and isinstance(nd.value, str) and "__" in nd.value:
                raise Rejet("dunder", "chaine")
            if not isinstance(nd, NOEUDS_OK):
                raise Rejet("noeud_interdit", type(nd).__name__)
        if total > NOEUDS_MAX:
            raise Rejet("taille", "%d noeuds" % total)
        v = _V()
        textes, fonctions = {}, {}
        for s in arbre.body:
            if isinstance(s, ast.Expr):
                if not (isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)):
                    raise Rejet("module", "expression")
                continue                                   # docstring
            if isinstance(s, ast.Assign):
                if len(s.targets) != 1 or not isinstance(s.targets[0], ast.Name):
                    raise Rejet("module", "affectation")
                n, val = s.targets[0].id, s.value
                if n in ENTETES_TEXTE:
                    if not (isinstance(val, ast.Constant) and isinstance(val.value, str) and len(val.value) <= 60):
                        raise Rejet("entete", n)
                    textes[n] = val.value
                elif n in ("applies", "plan", "fallback"):
                    raise Rejet("module", "nom reserve")
                elif isinstance(val, (ast.List, ast.Tuple)):
                    if not (1 <= len(val.elts) <= TABLE_MAX and all(_num_signe(x) for x in val.elts)):
                        raise Rejet("table_interdite", n)
                    v.tables[n] = len(val.elts)
                elif _num_signe(val):
                    v.consts.add(n)
                else:
                    raise Rejet("module", "valeur de " + n)
            elif isinstance(s, ast.FunctionDef):
                fonctions[s.name] = s
            else:
                raise Rejet("module", type(s).__name__)
        for k in ENTETES_TEXTE:
            if k not in textes:
                raise Rejet("entete_manquant", k)
        if textes["MODE"] not in ("add", "replace"):
            raise Rejet("entete", "MODE")
        if set(fonctions) != set(SIGNATURES):
            raise Rejet("structure", "fonctions attendues : applies, plan, fallback")
        for nom, f in fonctions.items():
            a = f.args
            if (f.decorator_list or f.returns or a.vararg or a.kwarg or a.kwonlyargs or a.defaults
                    or a.posonlyargs or [x.arg for x in a.args] != SIGNATURES[nom]
                    or any(x.annotation for x in a.args)):
                raise Rejet("structure", "signature de " + nom)
            if not (2 <= len(f.body) <= 40) and len(f.body) != 1:
                raise Rejet("structure", "corps de " + nom)
            if not _termine(f.body):
                raise Rejet("structure", "return manquant dans " + nom)
            v.instr(f.body, set(SIGNATURES[nom]))
        if v.noeuds > NOEUDS_MAX:
            raise Rejet("taille", "noeuds")
        return True, "", ""
    except Rejet as r:
        return False, r.code, r.detail
    except RecursionError:
        return False, "trop_profond", ""


class Competence:
    """Competence chargee : appels deterministes, sans acces au systeme."""

    def __init__(self, source):
        ok, code, detail = verifier(source)
        if not ok:
            raise ValueError("competence rejetee : %s %s" % (code, detail))
        self.source = source
        self.sha = hashlib.sha256(source.encode()).hexdigest()
        g = {"__builtins__": {}, "min": min, "max": max, "abs": abs, "int": int, "float": float}
        exec(compile(source, "<competence>", "exec"), g)
        self.name, self.parent, self.mode = g["NAME"], g["PARENT"], g["MODE"]
        self.applies, self.plan, self.fallback = g["applies"], g["plan"], g["fallback"]


BASE4 = '''NAME = "base4"
PARENT = "base4"
MODE = "add"
def applies(L, e, w, r):
    return True
def plan(L, e, w, r):
    return (4, 4, 0.0, 0.0)
def fallback(L, e, w, r, perte):
    return False
'''
