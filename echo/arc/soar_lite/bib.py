# -*- coding: utf-8 -*-
"""Bibliothèque apprise (ordre des primitives + macros) injectée dans solveur."""
import sys, os, re, json, time
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import solveur

BASE = list(solveur.PRIMS)             # primitives d'origine (ordre d'origine)
BASE_FN = dict(BASE)
assert not any('+' in n for n, _ in BASE)


def _macro(noms):
    fs = [BASE_FN[n] for n in noms]

    def f(g):
        for h in fs:
            g = solveur.appliquer(h, g)
            if g is None:
                return None
        return g
    return f


def appliquer_bib(bib):
    """bib = {'ordre': [noms], 'macros': {nom: [noms de base]}} ; None -> bibliothèque d'origine."""
    if not bib:
        P = list(BASE)
    else:
        fn = dict(BASE)
        for m, seq in bib['macros'].items():
            fn[m] = _macro(seq)
        P = [(n, fn[n]) for n in bib['ordre'] if n in fn]
    solveur.PRIMS[:] = P               # modifié en place : recherche_dsl lit solveur.PRIMS
    solveur.PRIM_FN.clear()
    solveur.PRIM_FN.update(dict(P))


def noms_prog(texte, macros=None):
    """Noms de primitives de base d'un texte de programme (macros développées) ; None si hors DSL."""
    out = []
    for p in texte.split(' ; '):
        if p.startswith('couleurs{'):
            continue
        if p in BASE_FN:
            out.append(p)
        elif p.count('+') and all(x in BASE_FN for x in p.split('+')):
            out.extend(p.split('+'))
        else:
            return None
    return out


def developper(texte):
    return texte.replace('+', ' ; ')


def hindsight(train, base_s=1.2, nmax=3):
    """Relabeling a posteriori : pour chaque paire d'entraînement prise isolément, programmes
    (profondeur <= 2, primitives de base) qui produisent EXACTEMENT sa sortie."""
    progs = []
    for gi, go in train[:2]:
        t_fin = time.process_time() + base_s
        trouves, vus, niv = [], {gi}, [((), gi)]
        try:
            for d in (1, 2):
                nxt = []
                for prog, g in niv:
                    for nom, f in BASE:
                        if time.process_time() > t_fin or len(trouves) >= nmax:
                            raise TimeoutError
                        r = solveur.appliquer(f, g)
                        if r is None or not solveur.grille_valide(r) or r in vus:
                            continue
                        if r == go:
                            trouves.append(prog + (nom,))
                        elif d == 1 and len(r) <= 45 and len(r[0]) <= 45:
                            vus.add(r)
                            nxt.append((prog + (nom,), r))
                niv = nxt
        except TimeoutError:
            pass
        progs.extend(trouves)
    return progs


def apprendre(donnees, K, ancien=None):
    """donnees: liste (tache, poids, [noms de base]). Retourne une bibliothèque."""
    usage = defaultdict(float)
    sup = defaultdict(lambda: defaultdict(float))
    for tid, w, seq in donnees:
        for n in set(seq):
            usage[n] = max(usage[n], 0) + 0  # placeholder
    pt = defaultdict(lambda: defaultdict(float))   # clé -> tâche -> poids max
    for tid, w, seq in donnees:
        for n in seq:
            pt[(n,)][tid] = max(pt[(n,)][tid], w)
        for L in (2, 3):
            for i in range(len(seq) - L + 1):
                k = tuple(seq[i:i + L])
                if len(set(k)) > 1 or L == 2 and False:
                    pt[k][tid] = max(pt[k][tid], w)
    score = {k: sum(v.values()) for k, v in pt.items()}
    macros = dict(ancien['macros']) if ancien else {}
    cand = sorted(((s, k) for k, s in score.items() if len(k) > 1 and s >= 1.5),
                  key=lambda x: (-x[0], -len(x[1]), x[1]))
    for s, k in cand:
        if len(macros) >= K:
            break
        macros.setdefault('+'.join(k), list(k))
    sc = {}
    for n, _ in BASE:
        sc[n] = score.get((n,), 0.0)
    for m, seq in macros.items():
        sc[m] = score.get(tuple(seq), 0.0)
    noms = [n for n, _ in BASE] + list(macros)
    ordre = sorted(noms, key=lambda n: -sc[n])   # tri stable : ordre d'origine à égalité
    return {'ordre': ordre, 'macros': macros, 'scores': {n: sc[n] for n in noms if sc[n]}}
