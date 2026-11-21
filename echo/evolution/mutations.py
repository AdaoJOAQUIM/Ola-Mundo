"""Mutations sur l'arbre syntaxique (AST), puis ast.unparse.

Chaque opérateur modifie l'arbre sur place et retourne son nom, ou None s'il
n'est pas applicable. Le résultat passe toujours par la liste blanche avant
d'être évalué.
"""
import ast
import copy
import math

from garde_fous import DONNEES, VARIABLES_LOCALES, verifier

OPERATEURS = ["perturbe", "feuille", "enveloppe", "insere", "supprime",
              "par_ordre", "croise", "post", "instruction"]

META_DEFAUT = {
    "poids": {"perturbe": 2.0, "feuille": 1.0, "enveloppe": 1.0, "insere": 1.0, "supprime": 1.0,
              "par_ordre": 0.7, "croise": 0.5, "post": 0.3, "instruction": 0.5},
    "sigma": 0.3,   # pas des constantes
    "nmut": 2.0,    # nombre moyen de mutations par enfant (1 à 3)
}
TAU = 0.2


def meta_defaut():
    return {"poids": dict(META_DEFAUT["poids"]), "sigma": META_DEFAUT["sigma"], "nmut": META_DEFAUT["nmut"]}


def meta_heritee(meta, rng):
    """Méta-génome de l'enfant : celui du parent, bruité (log-normal, tau = 0.2)."""
    out = {"poids": {}, "sigma": meta["sigma"], "nmut": meta["nmut"]}
    for op, w in meta["poids"].items():
        out["poids"][op] = min(20.0, max(0.02, w * math.exp(TAU * rng.gauss(0, 1))))
    out["sigma"] = min(2.0, max(0.02, meta["sigma"] * math.exp(TAU * rng.gauss(0, 1))))
    out["nmut"] = min(3.0, max(1.0, meta["nmut"] * math.exp(TAU * rng.gauss(0, 1))))
    return out


def _E(txt):
    return ast.parse(txt, mode="eval").body


class Emplacement:
    """Un nœud d'expression et la façon de le remplacer."""
    __slots__ = ("parent", "champ", "idx", "noeud", "boucle", "defs")

    def __init__(self, parent, champ, idx, noeud, boucle, defs):
        self.parent, self.champ, self.idx = parent, champ, idx
        self.noeud, self.boucle, self.defs = noeud, boucle, defs

    def remplacer(self, nouveau):
        if self.idx is None:
            setattr(self.parent, self.champ, nouveau)
        else:
            getattr(self.parent, self.champ)[self.idx] = nouveau
        self.noeud = nouveau


def _utilise_k(nd):
    return any(isinstance(x, ast.Name) and x.id == "k" for x in ast.walk(nd))


def _noms(nd):
    return {x.id for x in ast.walk(nd) if isinstance(x, ast.Name)}


def collecter(f):
    """Retourne (emplacements d'expressions, constantes de listes, positions d'instructions)."""
    slots, listes = [], []

    def visite(nd, parent, champ, idx, boucle, defs):
        slots.append(Emplacement(parent, champ, idx, nd, boucle, frozenset(defs)))
        if isinstance(nd, ast.BinOp):
            visite(nd.left, nd, "left", None, boucle, defs)
            visite(nd.right, nd, "right", None, boucle, defs)
        elif isinstance(nd, ast.UnaryOp):
            visite(nd.operand, nd, "operand", None, boucle, defs)
        elif isinstance(nd, ast.Compare):
            visite(nd.left, nd, "left", None, boucle, defs)
            visite(nd.comparators[0], nd, "comparators", 0, boucle, defs)
        elif isinstance(nd, ast.Call):
            for i in range(len(nd.args)):
                visite(nd.args[i], nd, "args", i, boucle, defs)
        elif isinstance(nd, ast.Subscript) and isinstance(nd.value, (ast.List, ast.Tuple)):
            for i, el in enumerate(nd.value.elts):
                listes.append((nd.value, i))

    def corps(stmts, boucle, defs):
        for s in stmts:
            if isinstance(s, ast.Assign):
                visite(s.value, s, "value", None, boucle, defs)
                defs.add(s.targets[0].id)
            elif isinstance(s, ast.For):
                corps(s.body, True, defs)

    defs = set()
    corps(f.body[:-1], False, defs)
    visite(f.body[-1].value, f.body[-1], "value", None, False, defs)
    return slots, listes


def _constante_alea(rng):
    r = rng.random()
    if r < 0.4:
        return round(rng.choice([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]), 3)
    return round(math.exp(rng.uniform(math.log(0.02), math.log(8))), 4)


def feuille_alea(rng, boucle, defs):
    cand = []
    if boucle:
        for nm in DONNEES:
            cand.append((_E("%s[k]" % nm), 2.0))
        cand.append((_E("k"), 0.3))
    else:
        for nm in DONNEES:
            cand.append((_E("%s[K]" % nm), 0.5))
            cand.append((_E("%s[0]" % nm), 0.5))
    for v in sorted(defs):
        cand.append((_E(v), 3.0 if v == "p" else 1.5))
    cand.append((ast.Constant(_constante_alea(rng)), 3.0))
    tot = sum(w for _, w in cand)
    r = rng.random() * tot
    for nd, w in cand:
        r -= w
        if r <= 0:
            return copy.deepcopy(nd)
    return copy.deepcopy(cand[-1][0])


def _est_feuille(nd):
    return isinstance(nd, (ast.Name, ast.Constant)) or (
        isinstance(nd, ast.Subscript) and isinstance(nd.value, ast.Name) and nd.value.id in DONNEES)


def _constantes(slots):
    return [s for s in slots if isinstance(s.noeud, ast.Constant) and type(s.noeud.value) in (int, float)]


# ---------------------------------------------------------------- opérateurs
def op_perturbe(f, ctx):
    rng, sigma = ctx["rng"], ctx["meta"]["sigma"]
    slots, listes = collecter(f)
    cands = [("s", s) for s in _constantes(slots)] + [("l", x) for x in listes]
    if not cands:
        return None
    typ, c = rng.choice(cands)
    v = c.noeud.value if typ == "s" else ast.literal_eval(c[0].elts[c[1]])
    if v != 0 and rng.random() < 0.7:
        w = v * math.exp(sigma * rng.gauss(0, 1))
    else:
        w = v + sigma * rng.gauss(0, 1)
    w = max(-1000.0, min(1000.0, round(w, 5)))
    if typ == "s":
        c.noeud.value = w
    else:
        c[0].elts[c[1]] = ast.Constant(w)
    return "perturbe"


def op_feuille(f, ctx):
    rng = ctx["rng"]
    slots, _ = collecter(f)
    cands = [s for s in slots if _est_feuille(s.noeud)]
    if not cands:
        return None
    s = rng.choice(cands)
    for _ in range(5):
        nv = feuille_alea(rng, s.boucle, s.defs)
        if ast.dump(nv) != ast.dump(s.noeud):
            s.remplacer(nv)
            return "feuille"
    return None


def op_enveloppe(f, ctx):
    rng = ctx["rng"]
    slots, _ = collecter(f)
    cands = [s for s in slots if not isinstance(s.noeud, ast.Compare)]
    s = rng.choice(cands)
    x = s.noeud
    r = rng.random()
    if r < 0.12:
        nv = ast.UnaryOp(ast.USub(), x)
    else:
        nom = rng.choice(["sqrt", "abs", "tanh", "exp", "log1p", "maximum", "maximum", "minimum", "clip"])
        if nom in ("sqrt", "abs", "tanh", "exp", "log1p"):
            args = [x]
        elif nom == "clip":
            args = [x, ast.Constant(0.0), ast.Constant(_constante_alea(rng))]
        else:
            seuil = 0.0 if (nom == "maximum" and rng.random() < 0.5) else _constante_alea(rng)
            args = [x, ast.Constant(seuil)]
        nv = ast.Call(_E("np." + nom), args, [])
    s.remplacer(nv)
    return "enveloppe"


def op_insere(f, ctx):
    rng = ctx["rng"]
    slots, _ = collecter(f)
    s = rng.choice(slots)
    x = s.noeud
    leaf = feuille_alea(rng, s.boucle, s.defs)
    op = rng.choice(["+", "-", "*", "div"])
    gauche = rng.random() < 0.3
    a, b = (leaf, x) if gauche else (x, leaf)
    if op == "div":
        nv = ast.Call(ast.Name("div", ast.Load()), [a, b], [])
    else:
        nv = ast.BinOp(a, {"+": ast.Add(), "-": ast.Sub(), "*": ast.Mult()}[op], b)
    s.remplacer(nv)
    return "insere"


def op_supprime(f, ctx):
    rng = ctx["rng"]
    slots, _ = collecter(f)
    cands = [s for s in slots if isinstance(s.noeud, (ast.BinOp, ast.UnaryOp, ast.Call))]
    if not cands:
        return None
    s = rng.choice(cands)
    nd = s.noeud
    if isinstance(nd, ast.BinOp):
        fils = [nd.left, nd.right]
    elif isinstance(nd, ast.UnaryOp):
        fils = [nd.operand]
    else:
        fils = list(nd.args)
    s.remplacer(rng.choice(fils))
    return "supprime"


def op_par_ordre(f, ctx):
    rng = ctx["rng"]
    slots, _ = collecter(f)
    cands = [s for s in _constantes(slots) if s.boucle]
    if not cands:
        return None
    s = rng.choice(cands)
    v = s.noeud.value
    K1 = ctx["K"] + 1
    elts = [ast.Constant(round(v * math.exp(0.05 * rng.gauss(0, 1)), 5) if v != 0 else round(0.05 * rng.gauss(0, 1), 5))
            for _ in range(K1)]
    s.remplacer(ast.Subscript(ast.List(elts, ast.Load()), ast.Name("k", ast.Load()), ast.Load()))
    return "par_ordre"


def op_croise(f, ctx):
    rng = ctx["rng"]
    autres = ctx.get("archive_sources") or []
    if not autres:
        return None
    try:
        donneur = ast.parse(rng.choice(autres)).body[0]
    except SyntaxError:
        return None
    ds, _ = collecter(donneur)
    slots, _ = collecter(f)
    for _ in range(6):
        t = rng.choice(slots)
        d = rng.choice(ds)
        if isinstance(d.noeud, ast.Constant):
            continue
        sous = copy.deepcopy(d.noeud)
        if not t.boucle and _utilise_k(sous):
            continue
        if not (_noms(sous) - {"k", "K", "np", "div"} - set(DONNEES)) <= set(t.defs):
            continue
        t.remplacer(sous)
        return "croise"
    return None


def op_post(f, ctx):
    rng = ctx["rng"]
    modele = rng.choice(["p = p ** %s", "p = np.maximum(p, %s)", "p = p + %s * np.sqrt(p)", "p = p * %s + 1e-4"])
    if modele.startswith("p = p **"):
        c = round(math.exp(rng.uniform(-0.3, 0.3)), 4)
    elif "maximum" in modele:
        c = round(10 ** rng.uniform(-6, -3), 7)
    else:
        c = round(rng.uniform(0.2, 1.5), 4)
    stmt = ast.parse(modele % c).body[0]
    f.body.insert(len(f.body) - 1, stmt)
    return "post"


def op_instruction(f, ctx):
    rng = ctx["rng"]
    boucles = [s for s in f.body if isinstance(s, ast.For)]
    if not boucles:
        return None
    b = rng.choice(boucles)
    libres = [v for v in VARIABLES_LOCALES if v not in {t.targets[0].id for t in ast.walk(f) if isinstance(t, ast.Assign)}]
    if not libres:
        return None
    v = rng.choice(libres)
    pos = rng.randint(0, len(b.body) - 1)
    # noms définis à cet endroit : p + variables des instructions précédentes
    defs = {"p"} | {t.targets[0].id for t in f.body[:f.body.index(b)] if isinstance(t, ast.Assign)} \
        | {t.targets[0].id for t in b.body[:pos] if isinstance(t, ast.Assign)}
    a, c = feuille_alea(rng, True, defs), feuille_alea(rng, True, defs)
    op = rng.choice(["+", "-", "*", "div"])
    if op == "div":
        expr = ast.Call(ast.Name("div", ast.Load()), [a, c], [])
    else:
        expr = ast.BinOp(a, {"+": ast.Add(), "-": ast.Sub(), "*": ast.Mult()}[op], c)
    nv = ast.Assign([ast.Name(v, ast.Store())], expr)
    b.body.insert(pos, nv)
    # on branche souvent la nouvelle variable sur une feuille plus loin dans la boucle
    if rng.random() < 0.8:
        slots, _ = collecter(f)
        pos_ids = {id(s) for s in b.body[pos + 1:]}
        cands = [s for s in slots if s.boucle and _est_feuille(s.noeud) and v in s.defs
                 and any(s.parent is x or any(s.parent is y for y in ast.walk(x)) for x in b.body[pos + 1:])]
        if cands:
            rng.choice(cands).remplacer(ast.Name(v, ast.Load()))
    return "instruction"


FONCTIONS = {"perturbe": op_perturbe, "feuille": op_feuille, "enveloppe": op_enveloppe,
             "insere": op_insere, "supprime": op_supprime, "par_ordre": op_par_ordre,
             "croise": op_croise, "post": op_post, "instruction": op_instruction}


def muter(source, meta, rng, archive_sources=None, K=10, essais=6, deja_vus=None):
    """Applique 1 à 3 mutations. Retourne (nouvelle_source, [noms], ok_liste_blanche, raison).

    On refait jusqu'à `essais` tentatives pour obtenir une source valide et nouvelle.
    """
    dernier = (source, [], False, "aucune_mutation")
    for _ in range(essais):
        arbre = ast.parse(source)
        f = arbre.body[0]
        nb = int(meta["nmut"])
        if rng.random() < meta["nmut"] - nb:
            nb += 1
        nb = max(1, min(3, nb))
        ctx = {"rng": rng, "meta": meta, "archive_sources": archive_sources, "K": K}
        noms = []
        ops, poids = list(meta["poids"].keys()), list(meta["poids"].values())
        for _m in range(nb):
            for _t in range(6):
                op = rng.choices(ops, poids)[0]
                try:
                    r = FONCTIONS[op](f, ctx)
                except (IndexError, ValueError, KeyError):
                    r = None
                if r:
                    noms.append(r)
                    break
        if not noms:
            continue
        try:
            ast.fix_missing_locations(arbre)
            nouvelle = ast.unparse(arbre) + "\n"
        except Exception:
            continue
        ok, code, _ = verifier(nouvelle)
        dernier = (nouvelle, noms, ok, code)
        if not ok:
            continue
        if nouvelle == source or (deja_vus is not None and nouvelle in deja_vus):
            dernier = (nouvelle, noms, False, "doublon")
            continue
        return nouvelle, noms, True, ""
    return dernier
