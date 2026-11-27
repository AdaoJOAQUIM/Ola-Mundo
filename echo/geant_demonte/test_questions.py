"""Tests du generateur de questions et de l'analyseur roff (python3 -m pytest test_questions.py)."""
import json, os, re
import bibliotheque as bib
import questions as Q
import fuite

TABLES = bib.construire()


def test_propre_roff():
    assert bib.propre(r"\fB\-\-echo\-all\fR est \(lqok\(rq\&.") == '--echo-all est "ok".'
    assert bib.propre(r"\f[V]--create\f[R] doesn\[aq]t") == "--create doesn't"


def test_flags():
    assert bib.extraire_flags("-c or --create") == ["-c", "--create"]
    assert bib.extraire_flags("-F format --format=format") == ["-F", "--format"]
    assert bib.extraire_flags("-XX:+PrintCompilation") == ["-XX:+PrintCompilation"]


def test_pandoc_et_docbook_minimaux():
    pandoc = [".TP", r"\f[V]-c\f[R] or \f[V]--create\f[R]", "Creates the archive.", ".TP", r"\f[V]-v\f[R]", "Verbose."]
    r = bib.options_pandoc(pandoc)
    assert r[0] == ("-c or --create", "Creates the archive.")
    docbook = [".PP", r"\fB\-a\fR", ".br", r"\fB\-\-echo\-all\fR", ".RS 4", r"Print all lines\&.", ".RE"]
    r = bib.options_docbook(docbook)
    assert r and r[0][0] == "-a --echo-all" and r[0][1] == "Print all lines."


def test_premiere_phrase():
    assert bib.premiere_phrase("Creates the archive. Then more text.") == "Creates the archive."


def test_cite():
    assert Q.cite("Use --verbose mode now", "--verbose")
    assert Q.cite("Print verbose output to the screen", "--verbose")  # noyau long cite
    assert not Q.cite("Print output to the screen", "--verbose")


def test_determinisme():
    a = Q.construire_questions(TABLES)
    b = Q.construire_questions(TABLES)
    assert a == b
    c = Q.construire_questions(TABLES, graine=Q.GRAINE + 1)
    assert a != c


def test_structure_et_exigences():
    qs = Q.construire_questions(TABLES)
    assert len(qs) == 400
    assert len({q["cmd"] for q in qs}) >= 40
    assert sum(q["split"] == "test" for q in qs) == 300 and sum(q["split"] == "tune" for q in qs) == 100
    ids = [q["id"] for q in qs]
    assert len(set(ids)) == len(ids)
    assert len({q["stem"] for q in qs}) == len(qs)
    for q in qs:
        assert len(q["choices"]) == 4 and len(set(q["choices"])) == 4
        assert q["choices"][q["answer"]] == q["answer_flag"]
        opts = TABLES[q["cmd"]]
        affiches = {Q.affiche(o) for o in opts}
        assert set(q["choices"]) <= affiches                      # distracteurs = drapeaux de la meme commande
        vrai = [o for o in opts if Q.affiche(o) == q["answer_flag"]]
        assert vrai and any(q["stem"] == Q.stem_de(o) for o in vrai)           # la reponse est bien l'option decrite
        for f in q["choices"]:                                     # aucune fuite triviale dans l'enonce
            assert not Q.cite(q["stem"], f), (q["stem"], f)
        sc = [fuite.score_lex(q["stem"], f) for f in q["choices"]]  # aucun recouvrement lexical discriminant
        assert len(set(sc)) == 1 and sc[0] <= Q.SEUIL_LEX, (q["stem"], q["choices"], sc)
    assert abs(fuite.precision_lexicale(qs) - 0.25) < 1e-9


def test_fichier_committe_coherent():
    chemin = Q.SORTIE
    if os.path.exists(chemin):
        d = json.load(open(chemin, encoding="utf-8"))
        assert d["questions"] == Q.construire_questions(TABLES)
        assert d["n"] == len(d["questions"])


if __name__ == "__main__":
    import sys
    fails = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try:
                f(); print("ok  ", n)
            except AssertionError as e:
                fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
