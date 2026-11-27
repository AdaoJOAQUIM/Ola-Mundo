"""Controles de fuite sans modele : recouvrement lexical entre l'enonce et chaque option."""
import re
import numpy as np

STOP = set("the a an of to in on for and or is are be by it its this that with as at from if not do does can will "
           "all any more than when which used use using specified specifies given file files option options "
           "command name".split())


def morceaux(flag):
    f = flag.lstrip("-")
    f = re.sub(r"([a-z])([A-Z])", r"\1 \2", f)
    return [p.lower() for p in re.split(r"[^A-Za-z0-9]+", f) if len(p) >= 3 and p.lower() not in STOP]


def mots(stem):
    return {w for w in re.findall(r"[a-z0-9]+", stem.lower()) if w not in STOP}


def racine(w):
    return re.sub(r"(ing|ed|es|s|e)$", "", w)


def score_lex(stem, flag):
    ms = mots(stem)
    rs = {racine(w) for w in ms}
    s = 0
    for p in morceaux(flag):
        if p in ms or racine(p) in rs:
            s += 1
        elif any(len(p) >= 4 and (w.startswith(p) or p.startswith(w) and len(w) >= 4) for w in ms):
            s += 0.5
    return s


def precision_lexicale(qs):
    """Precision attendue du choix 'option au plus fort recouvrement' (egalites : credit fractionnaire)."""
    tot = 0.0
    for q in qs:
        sc = np.array([score_lex(q["stem"], c) for c in q["choices"]])
        best = sc.max()
        gagnants = np.flatnonzero(sc == best)
        tot += (q["answer"] in gagnants) / len(gagnants)
    return tot / len(qs)


def part_avec_signal(qs):
    """Part des questions ou la bonne option a un recouvrement > 0."""
    return float(np.mean([score_lex(q["stem"], q["choices"][q["answer"]]) > 0 for q in qs]))
