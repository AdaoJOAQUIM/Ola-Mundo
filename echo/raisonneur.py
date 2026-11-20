#!/usr/bin/env python3
"""Écho — module de raisonnement symbolique et d'état du monde.

Python standard uniquement. Aucun réseau de neurones, aucun poids appris,
aucune dépendance externe, aucun ``eval``.

Ce module montre, à petite échelle et de façon mesurable, trois propriétés :

1. un modèle du monde explicite (qui est où, qui porte quoi) ;
2. un raisonnement vérifiable : chaque réponse est accompagnée d'une preuve,
   re-contrôlée par un vérificateur INDÉPENDANT du démonstrateur ;
3. l'absence d'hallucination : on ne répond que si une preuve existe et est
   acceptée ; sinon « Je ne sais pas ». Les contradictions sont détectées
   à l'insertion.

Apprentissage continu : on ajoute des faits ou des règles à tout moment,
sans réentraînement et sans rien perdre (la base est la liste des phrases).

Langage d'entrée : un français CONTRÔLÉ (voir la section « limites » de
resultats_raisonneur.md). Tout le reste est refusé honnêtement.

Organisation du fichier :
    1. normalisation et analyse (regex)           -> ``analyser``
    2. la base de phrases                          -> ``Base``
    3. le démonstrateur                            -> ``Raisonneur``
    4. le vérificateur (code séparé)               -> ``verifier``
    5. interface en ligne de commande et démo
"""

import argparse
import re
import sys
import unicodedata
from collections import deque, namedtuple, defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace

SANS_REPONSE = "Je ne sais pas (aucun fait connu ne permet de répondre)"
INCOMPRIS = "Je ne comprends pas cette phrase"


# ======================================================================
# 1. Normalisation et analyse
# ======================================================================

_REMPLACEMENTS = {"’": "'", "‘": "'", "`": "'", " ": " ",
                  " ": " ", "…": ".", "«": '"', "»": '"'}


def normaliser(texte):
    """Minuscules sans accents, apostrophes droites.

    La longueur est conservée caractère par caractère : on peut donc
    retrouver, dans le texte d'origine, ce que la regex a reconnu
    (affichage fidèle, accents compris).
    """
    sortie = []
    for c in texte:
        c = _REMPLACEMENTS.get(c, c)
        d = unicodedata.normalize("NFD", c)
        base = d[0] if d else c
        bas = base.lower()
        sortie.append(bas if len(bas) == 1 else base)
    return "".join(sortie)


_EXCEPTIONS_SINGULIER = {
    "souris", "tapis", "pays", "corps", "bras", "gaz", "bus", "virus",
    "cactus", "fois", "mois", "temps", "repas", "riz", "nez", "gros",
    "pas", "heros", "dos", "bois", "palais", "jamais", "puis", "sans",
    "plus", "moins", "tous", "concours", "cours", "discours", "poids",
    "tuyaux", "noyaux", "boyaux", "joyaux", "as", "ananas", "obus",
}


def singulier(mot):
    """Pluriel -> singulier, règle simple (retire s/x final ; -aux -> -al).

    Appliquée de la même façon à toutes les occurrences d'un mot, donc
    cohérente même quand elle se trompe (« heureux » -> « heureu » des deux
    côtés). Limite : les pluriels irréguliers hors -aux ne sont pas gérés.
    """
    if mot in _EXCEPTIONS_SINGULIER or len(mot) < 3:
        return mot
    if mot.endswith("aux") and not mot.endswith("eaux") and len(mot) > 4:
        return mot[:-3] + "al"
    if mot.endswith("s") and not mot.endswith("ss"):
        return mot[:-1]
    if mot.endswith("x"):
        return mot[:-1]
    return mot


_ARTICLE = re.compile(r"^(?:(le|la|les|un|une|des|du)\s+|(l'))")
_PRONOMS = {"il", "elle", "ils", "elles", "on", "ce", "ceci", "cela", "ca",
            "je", "tu", "nous", "vous", "qui", "que", "quoi", "moi", "toi",
            "lui", "eux", "se", "me", "te", "y", "en"}
_MOTS_INTERDITS = {"dans", "sur", "au", "aux", "chez", "avec", "et", "puis",
                   "ou", "qui", "que", "mais", "donc", "car", "quand", "si",
                   "parce", "pas", "ne", "n'", "pourquoi", "comment", "quel",
                   "quelle", "quels", "quelles", "combien", "lequel", "a",
                   "plupart", "certain", "certains", "certaines", "quelques",
                   "plusieurs", "beaucoup", "souvent", "parfois", "rarement",
                   "jamais", "toujours", "presque", "environ", "seulement",
                   "seul", "seule", "meme"}
_PREMIERS_PRED_INTERDITS = {"dans", "au", "aux", "en", "chez", "sur", "sous",
                            "avec", "de", "du", "par", "pour", "vers", "comme",
                            "tres", "trop", "plus", "moins", "aussi", "assez",
                            "tout", "tous", "toute", "toutes", "pas", "ne",
                            "a", "bien", "mal", "tellement"}
_CLE_VALIDE = re.compile(r"^[a-z][a-z'\- ]*[a-z]$")


def _cle_ok(cle):
    """Une clé (nom, classe, objet, lieu) est-elle acceptable ?"""
    if not cle or not _CLE_VALIDE.match(cle):
        return False
    mots = cle.split()
    if len(mots) > 4 or cle in _PRONOMS:
        return False
    return not any(m in _MOTS_INTERDITS or m in _PRONOMS for m in mots)


def _decouper_article(seg_n, seg_o):
    """Sépare l'article éventuel. Retourne (article, reste_norm, reste_orig)."""
    m = _ARTICLE.match(seg_n)
    if not m:
        return "", seg_n.strip(), seg_o.strip()
    art = m.group(1) or m.group(2)
    return art, seg_n[m.end():].strip(), seg_o[m.end():].strip()


def _cle_classe(reste_n):
    return " ".join(singulier(m) for m in reste_n.split())


def _maj(s):
    return s[:1].upper() + s[1:]


Analyse = namedtuple("Analyse", "genre forme aff")


def _aff_vide():
    return {"noms": {}, "cls": {}, "det": {}, "lieu": {}, "obj": {}}


_N = r"[a-z][a-z\-]*"
_PREP = r"(?:dans|au|aux|a|en|chez|vers|sur)"
_VERBES_VA = (r"(?:va|est alle|est allee|est retourne|est retournee|retourne|"
              r"se rend|s'est rendu|s'est rendue|entre|est entre|est entree|"
              r"part|est parti|est partie|court|est couru|marche|monte|"
              r"est monte|est montee|descend|est descendu|est descendue|"
              r"se deplace|s'est deplace|s'est deplacee|revient|est revenu|"
              r"est revenue|se trouve|est)")
_VERBES_PREND = (r"(?:prend|ramasse|saisit|attrape|recupere|a pris|"
                 r"a ramasse|a saisi|a attrape|a recupere)")
_VERBES_POSE = (r"(?:pose|depose|lache|laisse|jette|abandonne|a pose|"
                r"a depose|a lache|a laisse|a jete|a abandonne)")
_VERBES_DONNE = (r"(?:donne|passe|remet|tend|offre|a donne|a passe|a remis|"
                 r"a tendu|a offert)")

_RE_VA = re.compile(rf"^(?P<p>{_N})\s+{_VERBES_VA}\s+(?P<prep>{_PREP})\s+(?P<lieu>.+)$")
_RE_DONNE = re.compile(rf"^(?P<p>{_N})\s+{_VERBES_DONNE}\s+(?P<o>.+?)\s+a\s+(?P<q>{_N})$")
_RE_PREND = re.compile(rf"^(?P<p>{_N})\s+{_VERBES_PREND}\s+(?P<o>.+)$")
_RE_POSE = re.compile(rf"^(?P<p>{_N})\s+{_VERBES_POSE}\s+(?P<o>.+)$")
_RE_AUCUN = re.compile(r"^aucune?\s+(?P<a>.+?)\s+n'?\s*est\s+(?P<b>.+)$")
_RE_TOUT = re.compile(r"^(?:tout|toute|tous|toutes)\s+(?P<a>.+?)\s+(?:est|sont)\s+(?P<b>.+)$")
_RE_NEG = re.compile(r"^(?P<s>.+?)\s+(?:n'?\s*est|ne\s+sont)\s+pas\s+(?P<b>.+)$")
_RE_POS = re.compile(r"^(?P<s>.+?)\s+(?:est|sont)\s+(?P<b>.+)$")

_RE_CALC = re.compile(r"^combien\s+(?:font|fait|vaut|valent|donnent|donne)\s+(?P<e>.+)$")
_RE_OU = re.compile(r"^ou\s+(?:est|sont|se trouve|se trouvent)\s+(?P<x>.+)$")
_RE_QUI = re.compile(r"^qui\s+(?:a|possede|porte|tient|detient)\s+(?P<o>.+)$")
_RE_ESTCE = re.compile(r"^est-ce\s+(?:que|qu')\s*(?P<r>.+)$")
_RE_INVERSION = re.compile(
    r"^(?P<s>.+?)\s+(?:est|sont)-(?:t-)?(?:il|elle|ils|elles)\s+(?P<b>.+)$")


def _pred(seg_n, seg_o):
    """Prédicat : (clé, surface, article) ou None."""
    art, rn, ro = _decouper_article(seg_n, seg_o)
    premier = rn.split()[0] if rn.split() else ""
    if premier in _PREMIERS_PRED_INTERDITS:
        return None
    cle = _cle_classe(rn)
    if not _cle_ok(cle):
        return None
    return cle, ro, art


def _sujet(seg_n, seg_o, universel=False):
    """Sujet : ('ind', clé, surface, '') ou ('cls', clé, surface, article)."""
    art, rn, ro = _decouper_article(seg_n, seg_o)
    if art or universel:
        cle = _cle_classe(rn)
        return ("cls", cle, ro, art) if _cle_ok(cle) else None
    if rn in _PRONOMS or not _cle_ok(rn):
        return None
    return ("ind", rn, ro, "")


def _surface_singuliere(surface):
    """Met au singulier la forme d'affichage (accents conservés)."""
    mots = []
    for w in surface.split():
        nw = normaliser(w)
        sw = singulier(nw)
        if sw == nw:
            mots.append(w)
        elif nw.endswith("aux"):
            mots.append(w[:-3] + "al")
        else:
            mots.append(w[:-1])
    return " ".join(mots)


def _noter_classe(aff, cle, surface, art):
    aff["cls"].setdefault(cle, _surface_singuliere(surface))
    if art in ("un", "une"):
        aff["det"][cle] = art
    elif art == "le":
        aff["det"].setdefault(cle, "un")
    elif art == "la":
        aff["det"].setdefault(cle, "une")
    elif art == "des":
        aff["det"].setdefault(cle, "un(e)")


def _objet(seg_n, seg_o, aff):
    art, rn, ro = _decouper_article(seg_n, seg_o)
    if not _cle_ok(rn):
        return None
    aff["obj"][rn] = seg_o.strip()
    return rn


def _assertion(ns, orig):
    """Reconnaît une phrase déclarative. Retourne (forme, aff) ou None."""
    aff = _aff_vide()

    def seg(m, g):
        return ns[m.start(g):m.end(g)], orig[m.start(g):m.end(g)]

    def personne(m, g):
        cle, surf = seg(m, g)
        if not _cle_ok(cle):
            return None
        aff["noms"][cle] = _maj(surf)
        return cle

    # --- événements ---
    m = _RE_VA.match(ns)
    if m:
        p = personne(m, "p")
        ln, lo = seg(m, "lieu")
        _, rn, _ = _decouper_article(ln, lo)
        if p and _cle_ok(rn):
            aff["lieu"][rn] = orig[m.start("prep"):m.end("lieu")]
            return ("evt", "va", p, rn), aff
        return None
    m = _RE_DONNE.match(ns)
    if m:
        p, q = personne(m, "p"), personne(m, "q")
        o = _objet(*seg(m, "o"), aff)
        if p and q and o:
            return ("evt", "donne", p, o, q), aff
        return None
    for regex, typ in ((_RE_PREND, "prend"), (_RE_POSE, "pose")):
        m = regex.match(ns)
        if m:
            p = personne(m, "p")
            o = _objet(*seg(m, "o"), aff)
            if p and o:
                return ("evt", typ, p, o), aff
            return None

    # --- règles et faits ---
    m = _RE_AUCUN.match(ns)
    if m:
        a = _cle_classe(_decouper_article(*seg(m, "a"))[1])
        pb = _pred(*seg(m, "b"))
        if _cle_ok(a) and pb:
            _noter_classe(aff, a, _decouper_article(*seg(m, "a"))[2], "")
            _noter_classe(aff, pb[0], pb[1], pb[2])
            return ("disjoint", a, pb[0]), aff
        return None
    m = _RE_TOUT.match(ns)
    if m:
        sa = _sujet(*seg(m, "a"), universel=True)
        pb = _pred(*seg(m, "b"))
        if sa and pb:
            _noter_classe(aff, sa[1], sa[2], "")
            _noter_classe(aff, pb[0], pb[1], pb[2])
            return ("regle", sa[1], pb[0]), aff
        return None
    m = _RE_NEG.match(ns)
    if m:
        s = _sujet(*seg(m, "s"))
        pb = _pred(*seg(m, "b"))
        if not (s and pb):
            return None
        _noter_classe(aff, pb[0], pb[1], pb[2])
        if s[0] == "ind":
            aff["noms"][s[1]] = _maj(s[2])
            return ("est", s[1], pb[0], False), aff
        _noter_classe(aff, s[1], s[2], s[3])
        return ("disjoint", s[1], pb[0]), aff
    m = _RE_POS.match(ns)
    if m:
        s = _sujet(*seg(m, "s"))
        pb = _pred(*seg(m, "b"))
        if not (s and pb):
            return None
        _noter_classe(aff, pb[0], pb[1], pb[2])
        if s[0] == "ind":
            aff["noms"][s[1]] = _maj(s[2])
            return ("est", s[1], pb[0], True), aff
        _noter_classe(aff, s[1], s[2], s[3])
        return ("regle", s[1], pb[0]), aff
    return None


def _nettoyer(texte):
    """Retire espaces et ponctuation finale. Retourne (orig, norm, interro)."""
    n = normaliser(texte)
    deb = len(n) - len(n.lstrip())
    fin = len(n.rstrip(" .!?;:\t\r\n"))
    interro = "?" in n[fin:]
    if fin <= deb:
        return "", "", interro
    return texte[deb:fin], n[deb:fin], interro


# --- opérations arithmétiques (analyse côté démonstrateur) ---------
_RE_JETON = re.compile(
    r"\s*(\d+|\(|\)|multiplie\s+par|divise\s+par|plus|moins|fois|[x×*/÷+\-])")
_OPS = {"plus": "+", "moins": "-", "fois": "*", "x": "*", "×": "*",
        "*": "*", "/": "/", "÷": "/", "+": "+", "-": "-"}


def _tokeniser(expr):
    """Découpe une expression en jetons ; None si un résidu est illisible."""
    jetons, pos = [], 0
    while pos < len(expr):
        m = _RE_JETON.match(expr, pos)
        if not m:
            if expr[pos:].strip() == "":
                break
            return None
        j = re.sub(r"\s+", " ", m.group(1))
        if j.startswith("multiplie"):
            j = "*"
        elif j.startswith("divise"):
            j = "/"
        jetons.append(("n", int(j)) if j.isdigit() else ("o", _OPS.get(j, j)))
        pos = m.end()
    return jetons or None


def analyser(texte):
    """Analyse une phrase. Retourne Analyse(genre, forme, aff).

    genre : "assertion", "question" ou "incompris".
    """
    orig, ns, interro = _nettoyer(texte)
    if not ns:
        return Analyse("incompris", None, {})

    m = _RE_CALC.match(ns)
    if m:
        expr = m.group("e")
        jetons = _tokeniser(expr)
        if jetons and any(t[0] == "o" for t in jetons):
            return Analyse("question", ("q_calc", expr, orig), {})
        return Analyse("incompris", None, {})

    m = _RE_OU.match(ns)
    if m:
        _, rn, _ = _decouper_article(m.group("x"), orig[m.start("x"):m.end("x")])
        return (Analyse("question", ("q_ou", rn), {}) if _cle_ok(rn)
                else Analyse("incompris", None, {}))

    m = _RE_QUI.match(ns)
    if m:
        art, rn, _ = _decouper_article(m.group("o"), orig[m.start("o"):m.end("o")])
        if art and _cle_ok(rn):
            return Analyse("question", ("q_qui", rn), {})
        return Analyse("incompris", None, {})

    m = _RE_ESTCE.match(ns)
    if m:
        a = _assertion(ns[m.start("r"):], orig[m.start("r"):])
        return _en_question(a)

    m = _RE_INVERSION.match(ns)
    if m:
        universel = bool(re.match(r"^(?:tout|toute|tous|toutes)\s", m.group("s")))
        sn, so = ns[m.start("s"):m.end("s")], orig[m.start("s"):m.end("s")]
        if universel:
            sn, so = sn.split(None, 1)[1], so.split(None, 1)[1]
        s = _sujet(sn, so, universel=universel)
        pb = _pred(ns[m.start("b"):m.end("b")], orig[m.start("b"):m.end("b")])
        if s and pb:
            aff = _aff_vide()
            _noter_classe(aff, pb[0], pb[1], pb[2])
            if s[0] == "ind":
                aff["noms"][s[1]] = _maj(s[2])
            else:
                _noter_classe(aff, s[1], s[2], s[3])
            return Analyse("question", ("q_est", s[0], s[1], pb[0], True), aff)
        return Analyse("incompris", None, {})

    a = _assertion(ns, orig)
    if a is None:
        return Analyse("incompris", None, {})
    if interro:
        return _en_question(a)
    return Analyse("assertion", a[0], a[1])


def _en_question(a):
    """Transforme une déclarative en question oui/non (si possible)."""
    if a is None:
        return Analyse("incompris", None, {})
    forme, aff = a
    if forme[0] == "est":
        return Analyse("question", ("q_est", "ind", forme[1], forme[2], forme[3]), aff)
    if forme[0] == "regle":
        return Analyse("question", ("q_est", "cls", forme[1], forme[2], True), aff)
    return Analyse("incompris", None, {})


# ======================================================================
# 2. La base de phrases
# ======================================================================

class Base:
    """Base de connaissances : la liste ordonnée des phrases acceptées.

    Les index (faits, règles, événements) sont reconstruits de façon
    incrémentale. Ajouter une phrase ne modifie jamais les précédentes
    (apprentissage continu sans oubli).
    """

    def __init__(self):
        self.phrases = []          # {"n", "texte", "forme"}
        self.evenements = []       # (n_phrase, forme)
        self.est_pos = defaultdict(list)   # individu -> [(classe, n)]
        self.est_neg = defaultdict(list)
        self.regles = defaultdict(list)    # classe -> [(classe_sup, n)]
        self.disj = []                     # (a, b, n)
        self.noms, self.cls, self.det = {}, {}, {}
        self.lieux, self.objets = {}, {}

    # -- affichage ------------------------------------------------
    def maj_affichage(self, aff):
        for k, v in aff["noms"].items():
            self.noms.setdefault(k, v)
        for k, v in aff["cls"].items():
            self.cls.setdefault(k, v)
        for k, v in aff["det"].items():
            if k not in self.det or (self.det[k] == "un(e)" and v != "un(e)"):
                self.det[k] = v
        for k, v in aff["lieu"].items():
            self.lieux.setdefault(k, v)
        for k, v in aff["obj"].items():
            self.objets.setdefault(k, v)

    def nom(self, cle):
        return self.noms.get(cle) or _maj(self.cls.get(cle, cle))

    def attribut(self, cle):
        """« un homme » / « mortel » : forme d'affichage d'un prédicat."""
        nom = self.cls.get(cle, cle)
        det = self.det.get(cle)
        return f"{det} {nom}" if det else nom

    # -- indexation -----------------------------------------------
    def indexer(self, forme, n):
        t = forme[0]
        if t == "est":
            (self.est_pos if forme[3] else self.est_neg)[forme[1]].append((forme[2], n))
        elif t == "regle":
            self.regles[forme[1]].append((forme[2], n))
        elif t == "disjoint":
            self.disj.append((forme[1], forme[2], n))
        elif t == "evt":
            self.evenements.append((n, forme))

    def desindexer(self, forme):
        """Annule l'indexation de la DERNIÈRE phrase ajoutée."""
        t = forme[0]
        if t == "est":
            d = self.est_pos if forme[3] else self.est_neg
            d[forme[1]].pop()
            if not d[forme[1]]:
                del d[forme[1]]
        elif t == "regle":
            self.regles[forme[1]].pop()
            if not self.regles[forme[1]]:
                del self.regles[forme[1]]
        elif t == "disjoint":
            self.disj.pop()
        elif t == "evt":
            self.evenements.pop()

    def connue(self, forme):
        t = forme[0]
        if t == "est":
            d = self.est_pos if forme[3] else self.est_neg
            return any(c == forme[2] for c, _ in d.get(forme[1], ()))
        if t == "regle":
            return any(c == forme[2] for c, _ in self.regles.get(forme[1], ()))
        if t == "disjoint":
            return any({a, b} == {forme[1], forme[2]} for a, b, _ in self.disj)
        return False

    def individus(self):
        return sorted(set(self.est_pos) | set(self.est_neg))

    def textes(self):
        return [p["texte"] for p in self.phrases]


# ======================================================================
# Objets de résultat
# ======================================================================

@dataclass
class Reponse:
    """Réponse du système : texte + preuve vérifiée (ou abstention)."""
    reponse: str
    preuve: list = field(default_factory=list)
    valeur: object = "inconnu"   # "oui", "non", "inconnu", "incompris", lieu, nombre...
    verifie: bool = False
    raison: str = ""             # verdict du vérificateur ou cause de l'abstention

    def __str__(self):
        return self.reponse

    def to_dict(self):
        return {"reponse": self.reponse, "valeur": self.valeur,
                "verifie": self.verifie, "raison": self.raison,
                "preuve": [dict(e, conclusion=list(e["conclusion"]),
                                depend=[list(d) for d in e["depend"]])
                           for e in self.preuve]}


@dataclass
class Insertion:
    """Résultat de l'ajout d'une phrase à la base."""
    statut: str                  # ajoutee | doublon | refusee | incomprise
    message: str
    numero: int = 0
    forme: tuple = None
    contradictions: list = field(default_factory=list)

    def __str__(self):
        return self.message


class _Constructeur:
    """Construit une preuve étape par étape (avec dédoublonnage)."""

    def __init__(self):
        self.etapes = []
        self._index = {}

    def ajouter(self, regle, concl, depend, texte):
        cle = (regle, concl, tuple(depend))
        if cle in self._index:
            return self._index[cle]
        k = len(self.etapes) + 1
        self.etapes.append({"n": k, "regle": regle, "conclusion": concl,
                            "depend": tuple(depend), "texte": texte})
        self._index[cle] = k
        return k

    def concl(self, k):
        return self.etapes[k - 1]["conclusion"]

    def preuve(self, k):
        """Extrait la preuve de l'étape k (étapes utiles seulement), renumérotée."""
        garde, pile = set(), [k]
        while pile:
            j = pile.pop()
            if j in garde:
                continue
            garde.add(j)
            pile.extend(d[1] for d in self.etapes[j - 1]["depend"] if d[0] == "etape")
        ordre = sorted(garde)
        renum = {old: i + 1 for i, old in enumerate(ordre)}
        res = []
        for old in ordre:
            e = self.etapes[old - 1]
            dep = tuple(("etape", renum[d[1]]) if d[0] == "etape" else d
                        for d in e["depend"])
            res.append(dict(e, n=renum[old], depend=dep))
        return res


# ======================================================================
# 3. Le démonstrateur
# ======================================================================

class Raisonneur:
    """Démonstrateur : chaînage sur les classes + rejeu de l'état du monde."""

    def __init__(self):
        self.base = Base()

    # ------------------------------------------------------------------
    # Entrée principale
    # ------------------------------------------------------------------
    def traiter(self, texte):
        """Question -> Reponse ; assertion -> Insertion."""
        a = analyser(texte)
        if a.genre == "question":
            return self._repondre(a)
        return self.ajouter(texte)

    def repondre(self, texte):
        a = analyser(texte)
        if a.genre != "question":
            return Reponse(INCOMPRIS + " (ce n'est pas une question)",
                           valeur="incompris", raison="non question")
        return self._repondre(a)

    # ------------------------------------------------------------------
    # Ajout d'une phrase (apprentissage continu + contradictions)
    # ------------------------------------------------------------------
    def ajouter(self, texte):
        a = analyser(texte)
        if a.genre == "question":
            return Insertion("incomprise", "C'est une question, pas une affirmation.")
        if a.genre != "assertion":
            return Insertion("incomprise", INCOMPRIS + ".")
        forme, aff = a.forme, a.aff
        if self.base.connue(forme):
            return Insertion("doublon", "Je le savais déjà.", forme=forme)
        n = len(self.base.phrases) + 1
        phrase = {"n": n, "texte": texte.strip(), "forme": forme}
        self.base.phrases.append(phrase)
        self.base.indexer(forme, n)
        contradictions = self._chercher_contradictions(forme, n)
        if contradictions:
            # on annule l'insertion : la base reste cohérente
            self.base.phrases.pop()
            self.base.desindexer(forme)
            for c in contradictions:
                c["phrase_refusee"] = phrase
                c["verifie"], c["raison"] = verifier_contradiction(c, self.base)
            msg = "REFUSÉE, contradiction : " + " ; ".join(
                c["message"] for c in contradictions)
            return Insertion("refusee", msg, forme=forme,
                             contradictions=contradictions)
        self.base.maj_affichage(aff)
        return Insertion("ajoutee", f"Noté (phrase n°{n}).", numero=n, forme=forme)

    def charger(self, textes):
        """Rejoue une liste de phrases (apprentissage incrémental)."""
        return [self.ajouter(t) for t in textes]

    # ------------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------------
    def _chercher_contradictions(self, forme, n):
        t = forme[0]
        if t == "est":
            return self._contradictions_individus([forme[1]])
        if t in ("regle", "disjoint"):
            return self._contradictions_individus(self.base.individus())
        if t == "evt":
            return self._contradiction_evenement(forme, n)
        return []

    def _contradictions_individus(self, individus):
        trouvees = []
        for x in individus:
            if not self._graines_neg(x):
                continue
            for c in sorted(self._bfs_pos(x)):
                cn = _Constructeur()
                kn = self._prouver_neg(cn, x, c)
                if kn is None:
                    continue
                cp = _Constructeur()
                kp = self._prouver_pos(cp, x, c)
                msg = (f"« {self.base.nom(x)} est {self.base.attribut(c)} » est "
                       f"démontrable (preuve A) alors que « {self.base.nom(x)} "
                       f"n'est pas {self.base.attribut(c)} » l'est aussi (preuve B)")
                trouvees.append({"genre": "est", "sujet": x, "classe": c,
                                 "message": msg,
                                 "preuve_a": cp.preuve(kp),
                                 "preuve_b": cn.preuve(kn)})
                break
        return trouvees

    def _contradiction_evenement(self, forme, n):
        t0 = len(self.base.evenements) - 1   # temps juste avant l'événement
        typ, p, o = forme[1], forme[2], forme[3]
        if typ == "va":
            return []
        r = self._dernier_obj(o, t0)
        if r is None:
            return []
        _, _, f = r
        cb = _Constructeur()
        if f[1] in ("prend", "donne"):
            h = f[2] if f[1] == "prend" else f[4]
            if typ == "prend" and h == p:
                return []
            if typ != "prend" and h == p:
                return []
            k = self._porteur(cb, o, t0)[1]
            qui = f"{self.base.nom(h)} porte déjà {self._obj(o)}"
        else:
            if typ == "prend":
                return []
            k = self._porteur(cb, o, t0)[1]
            qui = f"personne ne porte {self._obj(o)} (posé en dernier)"
        cb2 = _Constructeur()
        kb = cb2.ajouter("donnée", forme, [("phrase", n)],
                         self.base.phrases[n - 1]["texte"])
        msg = (f"« {self.base.phrases[n - 1]['texte']} » est incompatible avec "
               f"l'état du monde : {qui} (preuve A, phrase refusée en preuve B)")
        return [{"genre": "evenement", "sujet": p, "classe": o, "message": msg,
                 "preuve_a": cb.preuve(k), "preuve_b": cb2.preuve(kb)}]

    # ------------------------------------------------------------------
    # Chaînage sur les classes
    # ------------------------------------------------------------------
    def _bfs_pos(self, x):
        """Classes de x : {classe: ('fait', n) | ('regle', précédente, n)}."""
        parents, file = {}, deque()
        for c, n in self.base.est_pos.get(x, ()):
            if c not in parents:
                parents[c] = ("fait", n)
                file.append(c)
        while file:
            c = file.popleft()
            for b, n in self.base.regles.get(c, ()):
                if b not in parents:
                    parents[b] = ("regle", c, n)
                    file.append(b)
        return parents

    def _graines_neg(self, x):
        """Classes dont x est exclu directement : faits négatifs, disjonctions."""
        g = {}
        for c, n in self.base.est_neg.get(x, ()):
            g.setdefault(c, ("fait", n))
        if self.base.disj:
            par = self._bfs_pos(x)
            for a, b, n in self.base.disj:
                for p, q in ((a, b), (b, a)):
                    if p in par and q not in g:
                        g[q] = ("disj", p, n)
        return g

    def _phrase(self, n):
        return self.base.phrases[n - 1]

    def _donnee(self, cb, concl, n):
        return cb.ajouter("donnée", concl, [("phrase", n)],
                          self._phrase(n)["texte"])

    def _texte_concl(self, c):
        b = self.base
        if c[0] == "est":
            return f"{b.nom(c[1])} est {b.attribut(c[2])}"
        if c[0] == "nest_pas":
            return f"{b.nom(c[1])} n'est pas {b.attribut(c[2])}"
        if c[0] == "regle":
            return f"tout {b.cls.get(c[1], c[1])} est {b.attribut(c[2])}"
        return str(c)

    def _prouver_pos(self, cb, x, c):
        par = self._bfs_pos(x)
        if c not in par:
            return None
        chaine, cur = [], c
        while True:
            p = par[cur]
            chaine.append((cur, p))
            if p[0] == "fait":
                break
            cur = p[1]
        chaine.reverse()
        k = None
        for cur, p in chaine:
            if p[0] == "fait":
                k = self._donnee(cb, ("est", x, cur), p[1])
            else:
                kr = self._donnee(cb, ("regle", p[1], cur), p[2])
                k = cb.ajouter("modus ponens", ("est", x, cur), [("etape", k), ("etape", kr)],
                               self._texte_concl(("est", x, cur)))
        return k

    def _prouver_neg(self, cb, x, c):
        graines = self._graines_neg(x)
        if not graines:
            return None
        par, file, trouve = {c: None}, deque([c]), None
        while file:
            cur = file.popleft()
            if cur in graines:
                trouve = cur
                break
            for b, n in self.base.regles.get(cur, ()):
                if b not in par:
                    par[b] = (cur, n)
                    file.append(b)
        if trouve is None:
            return None
        chemin, cur = [], trouve
        while par[cur] is not None:
            bas, n = par[cur]
            chemin.append((bas, cur, n))
            cur = bas
        g = graines[trouve]
        if g[0] == "fait":
            k = self._donnee(cb, ("nest_pas", x, trouve), g[1])
        else:
            _, p, n = g
            kp = self._prouver_pos(cb, x, p)
            forme = self._phrase(n)["forme"]
            kd = self._donnee(cb, ("disjoint", forme[1], forme[2]), n)
            k = cb.ajouter("disjonction", ("nest_pas", x, trouve),
                           [("etape", kp), ("etape", kd)],
                           self._texte_concl(("nest_pas", x, trouve)))
        for bas, haut, n in chemin:
            kr = self._donnee(cb, ("regle", bas, haut), n)
            k = cb.ajouter("modus tollens", ("nest_pas", x, bas),
                           [("etape", k), ("etape", kr)],
                           self._texte_concl(("nest_pas", x, bas)))
        return k

    def _prouver_regle(self, cb, a, b):
        """Preuve que tout a est b (transitivité de règles)."""
        if a == b:
            return None
        par, file = {a: None}, deque([a])
        while file:
            cur = file.popleft()
            for s, n in self.base.regles.get(cur, ()):
                if s not in par:
                    par[s] = (cur, n)
                    file.append(s)
        if b not in par:
            return None
        chemin, cur = [], b
        while par[cur] is not None:
            bas, n = par[cur]
            chemin.append((bas, cur, n))
            cur = bas
        chemin.reverse()
        k = None
        for bas, haut, n in chemin:
            kr = self._donnee(cb, ("regle", bas, haut), n)
            if k is None:
                k = kr
            else:
                k = cb.ajouter("transitivité", ("regle", a, haut),
                               [("etape", k), ("etape", kr)],
                               self._texte_concl(("regle", a, haut)))
        return k

    # ------------------------------------------------------------------
    # État du monde (retrouvé par parcours arrière des événements)
    # ------------------------------------------------------------------
    def _dernier_obj(self, o, t):
        ev = self.base.evenements
        for i in range(min(t, len(ev)), 0, -1):
            n, f = ev[i - 1]
            if f[1] in ("prend", "pose", "donne") and f[3] == o:
                return i, n, f
        return None

    def _lieu_personne(self, cb, p, t):
        ev = self.base.evenements
        for i in range(min(t, len(ev)), 0, -1):
            n, f = ev[i - 1]
            if f[1] == "va" and f[2] == p:
                kd = self._donnee(cb, f, n)
                lieu = self.base.lieux.get(f[3], f[3])
                return cb.ajouter("déplacement", ("lieu", p, f[3], t), [("etape", kd)],
                                  f"{self.base.nom(p)} est {lieu} (après {t} événement(s))")
        return None

    def _porteur(self, cb, o, t):
        """Retourne (valeur, étape) : porteur de o ou 'personne' ; (None, None) si inconnu."""
        r = self._dernier_obj(o, t)
        if r is None:
            return None, None
        i, n, f = r
        kd = self._donnee(cb, f, n)
        if f[1] == "pose":
            k = cb.ajouter("objet lâché", ("personne", o, t), [("etape", kd)],
                           f"personne ne porte {self._obj(o)}")
            return "personne", k
        h = f[2] if f[1] == "prend" else f[4]
        k = cb.ajouter("prise", ("porte", h, o, t), [("etape", kd)],
                       f"{self.base.nom(h)} porte {self._obj(o)}")
        return h, k

    def _lieu_objet(self, cb, o, t):
        r = self._dernier_obj(o, t)
        if r is None:
            return None
        i, n, f = r
        if f[1] == "pose":
            kd = self._donnee(cb, f, n)
            kl = self._lieu_personne(cb, f[2], i)
            if kl is None:
                return None
            lieu = cb.concl(kl)[2]
            return cb.ajouter("objet posé", ("lieu", o, lieu, t),
                              [("etape", kd), ("etape", kl)],
                              f"{self._obj(o)} reste {self.base.lieux.get(lieu, lieu)} (posé là)")
        h, kp = self._porteur(cb, o, t)
        kl = self._lieu_personne(cb, h, t)
        if kl is None:
            return None
        lieu = cb.concl(kl)[2]
        return cb.ajouter("objet porté", ("lieu", o, lieu, t),
                          [("etape", kp), ("etape", kl)],
                          f"{self._obj(o)} suit {self.base.nom(h)} : {self.base.lieux.get(lieu, lieu)}")

    def _obj(self, o):
        return self.base.objets.get(o, o)

    # ------------------------------------------------------------------
    # Réponses aux questions
    # ------------------------------------------------------------------
    def _inconnu(self, raison=SANS_REPONSE):
        return Reponse(raison if raison.startswith("Je ne sais pas")
                       else f"Je ne sais pas ({raison})",
                       valeur="inconnu", raison=raison)

    def _finaliser(self, cb, k, but, valeur, texte):
        preuve = cb.preuve(k)
        ok, raison = verifier(preuve, self.base, but)
        if not ok:
            return Reponse(f"Je ne sais pas (preuve rejetée par le vérificateur : {raison})",
                           valeur="inconnu", raison=raison)
        return Reponse(texte, preuve, valeur, True, raison)

    def _repondre(self, a):
        q = a.forme
        t = q[0]
        # les noms rencontrés dans la question servent à l'affichage
        aff = a.aff
        if aff:
            for k, v in aff["noms"].items():
                self.base.noms.setdefault(k, v)
            for k, v in aff["cls"].items():
                self.base.cls.setdefault(k, v)
            for k, v in aff["det"].items():
                if k not in self.base.det or self.base.det[k] == "un(e)":
                    self.base.det[k] = v
        if t == "q_est":
            return self._rep_est(*q[1:])
        if t == "q_ou":
            return self._rep_ou(q[1])
        if t == "q_qui":
            return self._rep_qui(q[1])
        if t == "q_calc":
            return self._rep_calc(q[1], q[2])
        return Reponse(INCOMPRIS, valeur="incompris")

    def _rep_est(self, genre, s, c, pos):
        b = self.base
        if genre == "cls":
            cb = _Constructeur()
            k = self._prouver_regle(cb, s, c)
            if k is None:
                return self._inconnu()
            txt = f"Oui, tout {b.cls.get(s, s)} est {b.attribut(c)}."
            return self._finaliser(cb, k, ("regle", s, c), "oui", txt)
        cp, cn = _Constructeur(), _Constructeur()
        kp = self._prouver_pos(cp, s, c)
        kn = self._prouver_neg(cn, s, c)
        if kp and kn:
            return self._inconnu("la base est incohérente sur ce point")
        if kp:
            valeur = "oui" if pos else "non"
            txt = f"{'Oui' if pos else 'Non'}, {b.nom(s)} est {b.attribut(c)}."
            return self._finaliser(cp, kp, ("est", s, c), valeur, txt)
        if kn:
            valeur = "non" if pos else "oui"
            txt = f"{'Non' if pos else 'Oui'}, {b.nom(s)} n'est pas {b.attribut(c)}."
            return self._finaliser(cn, kn, ("nest_pas", s, c), valeur, txt)
        return self._inconnu()

    def _est_objet(self, cle):
        return any(f[1] in ("prend", "pose", "donne") and f[3] == cle
                   for _, f in self.base.evenements)

    def _est_personne(self, cle):
        for _, f in self.base.evenements:
            if f[2] == cle or (f[1] == "donne" and f[4] == cle):
                return True
        return False

    def _rep_ou(self, x):
        b, t = self.base, len(self.base.evenements)
        cb = _Constructeur()
        if self._est_objet(x):
            k = self._lieu_objet(cb, x, t)
            sujet = _maj(self._obj(x))
        elif self._est_personne(x):
            k = self._lieu_personne(cb, x, t)
            sujet = b.nom(x)
        else:
            return self._inconnu()
        if k is None:
            return self._inconnu()
        lieu = cb.concl(k)[2]
        txt = f"{sujet} est {b.lieux.get(lieu, lieu)}."
        return self._finaliser(cb, k, ("lieu", x, lieu, t), lieu, txt)

    def _rep_qui(self, o):
        b, t = self.base, len(self.base.evenements)
        cb = _Constructeur()
        val, k = self._porteur(cb, o, t)
        if k is None:
            return self._inconnu()
        if val == "personne":
            return self._finaliser(cb, k, ("personne", o, t), "personne",
                                   f"Personne ne porte {self._obj(o)}.")
        return self._finaliser(cb, k, ("porte", val, o, t), val,
                               f"{b.nom(val)} a {self._obj(o)}.")

    # ------------------------------------------------------------------
    # Arithmétique exacte (entiers), sans eval
    # ------------------------------------------------------------------
    def _rep_calc(self, expr, source):
        jetons = _tokeniser(expr)
        pos = [0]
        cb = _Constructeur()
        erreur = []

        def vu():
            return jetons[pos[0]] if pos[0] < len(jetons) else None

        def facteur():
            j = vu()
            if j is None:
                raise ValueError("expression incomplète")
            if j[0] == "n":
                pos[0] += 1
                return j[1], None
            raise ValueError("jeton inattendu")

        def primaire():
            j = vu()
            if j == ("o", "("):
                pos[0] += 1
                v = somme()
                if vu() != ("o", ")"):
                    raise ValueError("parenthèse non fermée")
                pos[0] += 1
                return v
            return facteur()

        def appliquer(g, op, d):
            (a, ka), (b2, kb) = g, d
            if op == "/":
                if b2 == 0:
                    raise ZeroDivisionError("division par zéro")
                if a % b2:
                    raise ArithmeticError("la division n'est pas entière")
                r = a // b2
            elif op == "*":
                r = a * b2
            elif op == "+":
                r = a + b2
            else:
                r = a - b2
            dep = [("etape", x) for x in (ka, kb) if x is not None]
            sym = {"*": "×", "/": "÷"}.get(op, op)
            k = cb.ajouter("calcul exact", ("calc", a, op, b2, r), dep,
                           f"{a} {sym} {b2} = {r}")
            cb.etapes[k - 1]["source"] = source
            return r, k

        def produit():
            g = primaire()
            while vu() in (("o", "*"), ("o", "/")):
                op = vu()[1]
                pos[0] += 1
                g = appliquer(g, op, primaire())
            return g

        def somme():
            g = produit()
            while vu() in (("o", "+"), ("o", "-")):
                op = vu()[1]
                pos[0] += 1
                g = appliquer(g, op, produit())
            return g

        try:
            val, k = somme()
            if pos[0] != len(jetons) or k is None:
                raise ValueError("expression mal formée")
        except ZeroDivisionError as e:
            return self._inconnu(str(e))
        except ArithmeticError as e:
            return self._inconnu(str(e))
        except ValueError:
            return Reponse(INCOMPRIS, valeur="incompris", raison="expression illisible")
        return self._finaliser(cb, k, ("resultat", val), val, f"Le résultat est {val}.")


# ======================================================================
# 4. Le vérificateur INDÉPENDANT
# ======================================================================
# Ce code n'utilise AUCUN index ni AUCUNE fonction du démonstrateur : il ne
# lit que les phrases stockées (base.phrases) et la liste d'événements
# (base.evenements), et refait ses propres calculs. Seul l'analyseur de
# phrases (la traduction texte -> forme) est partagé : il sert à vérifier que
# la forme stockée correspond bien au texte stocké.

def _v_concl_de_forme(f):
    """Conclusions équivalentes à une phrase stockée."""
    if f[0] == "est":
        return [("est", f[1], f[2])] if f[3] else [("nest_pas", f[1], f[2])]
    if f[0] == "regle":
        return [("regle", f[1], f[2])]
    if f[0] == "disjoint":
        return [("disjoint", f[1], f[2]), ("disjoint", f[2], f[1])]
    if f[0] == "evt":
        return [f]
    return []


def _v_rejeu(evts, t):
    """Rejoue les t premiers événements depuis zéro (code propre au vérificateur).

    Retourne (lieu, porteur, pose, dernier_va, dernier_obj) :
      lieu[p] = lieu courant de p ; porteur[o] = qui porte o ;
      pose[o] = (lieu_de_la_pose, indice) ; dernier_va[p] = (indice, n_phrase) ;
      dernier_obj[o] = (indice, n_phrase, forme) du dernier événement sur o.
    """
    lieu, porteur, pose, dva, dobj = {}, {}, {}, {}, {}
    for i, (n, f) in enumerate(evts[:t], 1):
        typ = f[1]
        if typ == "va":
            lieu[f[2]] = f[3]
            dva[f[2]] = (i, n)
        else:
            o = f[3]
            dobj[o] = (i, n, f)
            if typ == "prend":
                porteur[o] = f[2]
                pose.pop(o, None)
            elif typ == "donne":
                porteur[o] = f[4]
                pose.pop(o, None)
            elif typ == "pose":
                porteur.pop(o, None)
                pose[o] = (lieu.get(f[2]), i)
    return lieu, porteur, pose, dva, dobj


def _v_norm(s):
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return s.lower()


def _v_evaluer(source):
    """Évalue un énoncé « Combien font … » par descente récursive (indépendant)."""
    s = _v_norm(source)
    s = re.sub(r"^.*?\b(?:font|fait|vaut|valent|donnent|donne)\s+", "", s, count=1)
    s = s.replace("×", "*").replace("÷", "/")
    s = re.sub(r"multiplie\s+par|\bfois\b", "*", s)
    s = re.sub(r"divise\s+par", "/", s)
    s = re.sub(r"\bplus\b", "+", s)
    s = re.sub(r"\bmoins\b", "-", s)
    s = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", s)
    s = s.strip(" .?!")
    toks = re.findall(r"\d+|[()+\-*/]", s)
    if re.sub(r"\d+|[()+\-*/]|\s+", "", s) != "":
        raise ValueError("énoncé illisible")
    i = [0]

    def peek():
        return toks[i[0]] if i[0] < len(toks) else None

    def facteur():
        t = peek()
        i[0] += 1
        if t == "(":
            v = expr()
            if peek() != ")":
                raise ValueError("parenthèse")
            i[0] += 1
            return v
        if t is None or not t.isdigit():
            raise ValueError("opérande attendu")
        return int(t)

    def terme():
        v = facteur()
        while peek() in ("*", "/"):
            op = peek()
            i[0] += 1
            w = facteur()
            if op == "*":
                v = v * w
            else:
                if w == 0 or v % w:
                    raise ValueError("division non entière")
                v = v // w
        return v

    def expr():
        v = terme()
        while peek() in ("+", "-"):
            op = peek()
            i[0] += 1
            w = terme()
            v = v + w if op == "+" else v - w
        return v

    v = expr()
    if i[0] != len(toks):
        raise ValueError("résidu")
    return v


def verifier(preuve, base, but=None):
    """Re-valide chaque étape d'une preuve contre les phrases stockées.

    ``base`` : tout objet avec ``.phrases`` et ``.evenements``.
    ``but`` : conclusion attendue de la dernière étape (optionnel).
    Retourne (True/False, raison).
    """
    if not preuve:
        return False, "preuve vide"
    concl = {}
    phr_de = {}    # étape de type « donnée » -> numéro de phrase
    evts = base.evenements
    for k, e in enumerate(preuve, 1):
        try:
            if e.get("n") != k:
                return False, f"étape {k} : mauvais numéro"
            regle, c, dep = e["regle"], e["conclusion"], e["depend"]
            prem, ids, phrases = [], [], []
            for d in dep:
                if d[0] == "etape":
                    if not (isinstance(d[1], int) and 1 <= d[1] < k):
                        return False, f"étape {k} : dépendance invalide {d}"
                    prem.append(concl[d[1]])
                    ids.append(d[1])
                elif d[0] == "phrase":
                    if not (isinstance(d[1], int) and 1 <= d[1] <= len(base.phrases)):
                        return False, f"étape {k} : phrase {d[1]} inexistante"
                    phrases.append(d[1])
                else:
                    return False, f"étape {k} : dépendance inconnue {d}"
            ok, raison = _v_etape(regle, c, prem, ids, phrases, base, evts, phr_de, e, preuve)
        except (KeyError, IndexError, TypeError, ValueError) as ex:
            return False, f"étape {k} : preuve mal formée ({ex.__class__.__name__}: {ex})"
        if not ok:
            return False, f"étape {k} ({e.get('regle')}) : {raison}"
        concl[k] = c
        if regle == "donnée":
            phr_de[k] = phrases[0]
    if but is not None:
        c = concl[len(preuve)]
        if but[0] == "resultat":
            if not (c[0] == "calc" and c[4] == but[1]):
                return False, "la dernière étape n'établit pas le résultat annoncé"
        elif c != but:
            return False, f"la dernière étape établit {c} et non {but}"
    return True, "toutes les étapes sont valides"


def _v_etape(regle, c, prem, ids, phrases, base, evts, phr_de, e, preuve):
    """Valide une étape. Retourne (ok, raison)."""
    if regle == "donnée":
        if len(phrases) != 1 or prem:
            return False, "une donnée cite exactement une phrase"
        ph = base.phrases[phrases[0] - 1]
        if c not in _v_concl_de_forme(ph["forme"]):
            return False, f"la phrase {phrases[0]} ne dit pas {c}"
        re_analyse = analyser(ph["texte"])
        if re_analyse.genre != "assertion" or re_analyse.forme != ph["forme"]:
            return False, "le texte de la phrase ne redonne pas la forme stockée"
        return True, ""

    if regle == "modus ponens":
        if c[0] != "est" or len(prem) != 2:
            return False, "forme invalide"
        _, x, b = c
        for p1, p2 in (prem, prem[::-1]):
            if p1[0] == "est" and p1[1] == x and p2 == ("regle", p1[2], b):
                return True, ""
        return False, "prémisses (x est A) et (tout A est B) absentes"

    if regle == "transitivité":
        if c[0] != "regle" or len(prem) != 2:
            return False, "forme invalide"
        _, a, cc = c
        for p1, p2 in (prem, prem[::-1]):
            if p1[0] == "regle" and p1[1] == a and p2 == ("regle", p1[2], cc):
                return True, ""
        return False, "prémisses (A⊂B) et (B⊂C) absentes"

    if regle == "modus tollens":
        if c[0] != "nest_pas" or len(prem) != 2:
            return False, "forme invalide"
        _, x, a = c
        for p1, p2 in (prem, prem[::-1]):
            if p1[0] == "nest_pas" and p1[1] == x and p2 == ("regle", a, p1[2]):
                return True, ""
        return False, "prémisses (x n'est pas B) et (tout A est B) absentes"

    if regle == "disjonction":
        if c[0] != "nest_pas" or len(prem) != 2:
            return False, "forme invalide"
        _, x, q = c
        for p1, p2 in (prem, prem[::-1]):
            if p1[0] == "est" and p1[1] == x and p2[0] == "disjoint" and \
                    {p2[1], p2[2]} == {p1[2], q} and p1[2] != q:
                return True, ""
        return False, "prémisses (x est P) et (aucun P n'est Q) absentes"

    if regle == "calcul exact":
        if c[0] != "calc" or len(c) != 5:
            return False, "forme invalide"
        _, a, op, b, r = c
        if op == "+":
            att = a + b
        elif op == "-":
            att = a - b
        elif op == "*":
            att = a * b
        elif op == "/":
            if b == 0 or a % b:
                return False, "division interdite"
            att = a // b
        else:
            return False, "opérateur inconnu"
        if att != r:
            return False, f"{a} {op} {b} vaut {att} et non {r}"
        src = e.get("source")
        if not src:
            return False, "énoncé source absent"
        nombres = {int(x) for x in re.findall(r"\d+", src)}
        issus = {concl_r[4] for concl_r in prem if concl_r[0] == "calc"}
        for v in (a, b):
            if v not in nombres and v not in issus:
                return False, f"l'opérande {v} ne vient ni de l'énoncé ni d'une étape citée"
        if e is preuve[-1]:
            try:
                tot = _v_evaluer(src)
            except ValueError as ex:
                return False, f"énoncé illisible pour le vérificateur ({ex})"
            if tot != r:
                return False, f"l'énoncé complet vaut {tot} et non {r}"
        return True, ""

    # --- état du monde ---
    if regle in ("déplacement", "prise", "objet lâché", "objet posé", "objet porté"):
        t = c[-1]
        if not (isinstance(t, int) and 0 <= t <= len(evts)):
            return False, "temps invalide"
        lieu, porteur, pose, dva, dobj = _v_rejeu(evts, t)

    if regle == "déplacement":
        _, p, L, t = c
        if len(ids) != 1 or prem[0] != ("evt", "va", p, L):
            return False, "il faut citer l'événement (p va en L)"
        if lieu.get(p) != L or dva.get(p, (0, 0))[1] != phr_de[ids[0]]:
            return False, f"après {t} événement(s), {p} n'est pas en {L} d'après le dernier déplacement"
        return True, ""

    if regle == "prise":
        _, p, o, t = c
        if len(ids) != 1:
            return False, "un seul événement à citer"
        cite = phr_de[ids[0]]
        f = prem[0]
        if o not in dobj or dobj[o][1] != cite:
            return False, "l'événement cité n'est pas le dernier concernant l'objet"
        if porteur.get(o) != p or not (
                (f[1] == "prend" and f[2] == p) or (f[1] == "donne" and f[4] == p)):
            return False, f"d'après le rejeu, {o} n'est pas porté par {p}"
        return True, ""

    if regle == "objet lâché":
        _, o, t = c
        if len(ids) != 1:
            return False, "un seul événement à citer"
        if o in porteur or o not in pose or dobj[o][1] != phr_de[ids[0]]:
            return False, "d'après le rejeu, l'objet n'a pas été posé en dernier"
        return True, ""

    if regle == "objet porté":
        _, o, L, t = c
        if len(prem) != 2:
            return False, "deux prémisses attendues"
        for p1, p2 in (prem, prem[::-1]):
            if p1[0] == "porte" and p1[2] == o and p1[3] == t and \
                    p2 == ("lieu", p1[1], L, t):
                return True, ""
        return False, "prémisses (p porte o) et (p est en L) absentes"

    if regle == "objet posé":
        _, o, L, t = c
        if len(prem) != 2:
            return False, "deux prémisses attendues"
        evt = next((p for p in prem if p[0] == "evt"), None)
        lp = next((p for p in prem if p[0] == "lieu"), None)
        if evt is None or lp is None or evt[1] != "pose" or evt[3] != o:
            return False, "prémisses (p pose o) et (p est en L) absentes"
        if o in porteur or o not in pose:
            return False, "d'après le rejeu, l'objet est porté ou jamais posé"
        lieu_pose, i = pose[o]
        if lieu_pose != L or lp != ("lieu", evt[2], L, i):
            return False, "le lieu de la pose ne correspond pas"
        if evts[i - 1][1] != evt:
            return False, "événement de pose incohérent"
        return True, ""

    return False, f"règle inconnue « {regle} »"


def verifier_contradiction(c, base):
    """Vérifie les deux preuves d'une contradiction détectée à l'insertion.

    La phrase refusée n'est plus dans la base : on la réintroduit dans une vue
    temporaire, uniquement pour ce contrôle.
    """
    ph = c["phrase_refusee"]
    vue = SimpleNamespace(
        phrases=list(base.phrases) + [ph],
        evenements=list(base.evenements) + ([(ph["n"], ph["forme"])]
                                            if ph["forme"][0] == "evt" else []))
    ok_a, ra = verifier(c["preuve_a"], vue)
    ok_b, rb = verifier(c["preuve_b"], vue)
    if not (ok_a and ok_b):
        return False, f"preuve A : {ra} ; preuve B : {rb}"
    ca, cb = c["preuve_a"][-1]["conclusion"], c["preuve_b"][-1]["conclusion"]
    if c["genre"] == "est":
        if ca[0] == "est" and cb[0] == "nest_pas" and ca[1:] == cb[1:]:
            return True, "A démontre « x est C », B démontre « x n'est pas C »"
        return False, "les conclusions ne sont pas contradictoires"
    # événement : A établit l'état, B est la phrase refusée qui le contredit
    f = ph["forme"]
    t0 = len(base.evenements)
    if cb != f or f[0] != "evt":
        return False, "la preuve B doit établir la phrase refusée"
    if ca[0] == "porte" and ca[3] == t0 and ca[2] == f[3] and ca[1] != f[2] \
            and f[1] in ("prend", "pose", "donne"):
        return True, "l'objet est déjà porté par une autre personne"
    if ca[0] == "personne" and ca[2] == t0 and ca[1] == f[3] and f[1] in ("pose", "donne"):
        return True, "l'objet a été posé : personne ne peut le poser ou le donner"
    return False, "l'état établi ne contredit pas l'événement"


# ======================================================================
# Affichage
# ======================================================================

def formater_preuve(preuve, indent="    "):
    lignes = []
    for e in preuve:
        refs = []
        for d in e["depend"]:
            refs.append(f"phrase {d[1]}" if d[0] == "phrase" else f"étape {d[1]}")
        cite = f" [{', '.join(refs)}]" if refs else ""
        lignes.append(f"{indent}({e['n']}) {e['texte']}  -- {e['regle']}{cite}")
    return "\n".join(lignes)


def formater_reponse(r):
    s = r.reponse
    if r.preuve:
        s += "\n  Preuve :\n" + formater_preuve(r.preuve)
        s += f"\n  Vérificateur : {'ACCEPTÉE' if r.verifie else 'REJETÉE'} ({r.raison})"
    return s


def formater_insertion(i):
    s = i.message
    for c in i.contradictions:
        s += "\n  Preuve A :\n" + formater_preuve(c["preuve_a"], "      ")
        s += "\n  Preuve B :\n" + formater_preuve(c["preuve_b"], "      ")
        s += f"\n  Vérificateur : {'OK' if c['verifie'] else 'ÉCHEC'} ({c['raison']})"
    return s


def decouper_phrases(texte):
    """Découpe un texte en phrases (après . ! ?)."""
    return [p for p in re.split(r"(?<=[.!?])\s+", texte.strip()) if p]


# ======================================================================
# 5. Démo et ligne de commande
# ======================================================================

def _demo():
    r = Raisonneur()

    def titre(t):
        print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)

    def dire(phrase):
        print(f"\n> {phrase}")
        res = r.traiter(phrase)
        print(formater_reponse(res) if isinstance(res, Reponse) else formater_insertion(res))

    titre("1. Déduction avec preuve (chaîne de règles)")
    for p in ["Socrate est un homme.", "Tout homme est mortel.",
              "Les mortels sont des êtres vivants.", "Médor est un chien.",
              "Tous les chiens sont des animaux.", "Socrate n'est pas un oiseau.",
              "Socrate est-il un être vivant ?", "Est-ce que Médor est un animal ?",
              "Socrate est-il un oiseau ?"]:
        dire(p)

    titre("2. Pas d'hallucination : sans preuve, on s'abstient")
    for p in ["Médor est-il mortel ?", "Platon est-il un homme ?",
              "Médor est-il un chien ?", "Combien de pattes a Médor ?"]:
        dire(p)

    titre("3. Contradiction détectée à l'insertion")
    dire("Socrate est un oiseau.")
    dire("Médor n'est pas un animal.")

    titre("4. État du monde (événements rejoués dans l'ordre)")
    for p in ["Marie va dans la cuisine.", "Jean prend la pomme.",
              "Jean va au jardin.", "Où est la pomme ?", "Jean donne la pomme à Marie.",
              "Qui a la pomme ?", "Où est la pomme ?", "Marie pose la pomme.",
              "Marie va dans le salon.", "Où est la pomme ?", "Où est Marie ?",
              "Jean pose la pomme.", "Où est le ballon ?"]:
        dire(p)

    titre("5. Arithmétique exacte (entiers arbitrairement grands)")
    for p in ["Combien font 12 fois 7 ?",
              "Combien font 987654321012 fois 123456789 ?",
              "Combien font 2 plus 3 fois 4 ?",
              "Combien font 100 divisé par 7 ?"]:
        dire(p)

    titre("6. Apprentissage continu : on ajoute un fait, l'ancien reste valable")
    dire("Tout animal est un être vivant.")
    dire("Médor est-il un être vivant ?")
    dire("Socrate est-il mortel ?")

    titre("7. Un vérificateur qui ne se laisse pas tromper")
    rep = r.repondre("Socrate est-il mortel ?")
    faux = [dict(e) for e in rep.preuve]
    faux[-1]["conclusion"] = ("est", "socrate", "oiseau")
    print("Preuve falsifiée (conclusion changée en « Socrate est un oiseau »)")
    print("  ->", verifier(faux, r.base))
    faux2 = [dict(e) for e in rep.preuve]
    faux2[0]["conclusion"] = ("est", "socrate", "dieu")
    print("Preuve falsifiée (donnée inventée)")
    print("  ->", verifier(faux2, r.base))
    print("Preuve authentique")
    print("  ->", verifier(rep.preuve, r.base))

    titre("8. Phrase hors du français contrôlé")
    dire("Il pleut beaucoup depuis hier soir.")


def _interactif(r):
    print("Écho — raisonneur. Tapez des phrases ; les questions se terminent par « ? ».")
    print("Commandes : :aide  :base  :sauver FICHIER  :charger FICHIER  :quitter")
    while True:
        try:
            ligne = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not ligne:
            continue
        if ligne.startswith(":"):
            cmd, _, arg = ligne.partition(" ")
            if cmd in (":quitter", ":q"):
                return
            if cmd == ":aide":
                print(__doc__)
            elif cmd == ":base":
                for p in r.base.phrases:
                    print(f"  {p['n']:3d}. {p['texte']}")
            elif cmd == ":sauver" and arg:
                with open(arg, "w", encoding="utf-8") as f:
                    f.write("\n".join(r.base.textes()) + "\n")
                print(f"{len(r.base.phrases)} phrases sauvegardées dans {arg}")
            elif cmd == ":charger" and arg:
                with open(arg, encoding="utf-8") as f:
                    res = r.charger([l.strip() for l in f if l.strip()])
                print(f"{sum(1 for i in res if i.statut == 'ajoutee')} phrase(s) chargée(s)")
            else:
                print("Commande inconnue (:aide).")
            continue
        for phrase in decouper_phrases(ligne):
            res = r.traiter(phrase)
            print(formater_reponse(res) if isinstance(res, Reponse)
                  else formater_insertion(res))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Écho : raisonneur symbolique vérifiable.")
    ap.add_argument("--demo", action="store_true", help="affiche une démonstration")
    ap.add_argument("--charger", help="fichier de phrases à charger au départ")
    args = ap.parse_args(argv)
    if args.demo:
        _demo()
        return 0
    r = Raisonneur()
    if args.charger:
        with open(args.charger, encoding="utf-8") as f:
            r.charger([l.strip() for l in f if l.strip()])
    _interactif(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
