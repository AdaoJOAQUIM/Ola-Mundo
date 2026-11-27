"""Generateur de questions a choix multiple a partir des tables d'options des pages de man.

Aucun modele n'ecrit de question : tout est programmatique, graine fixee.
Question : « Dans la page de man de `cmd`, quelle option correspond a : <1re phrase de la description> ? »
Reponse vraie = le drapeau de l'option ; 3 distracteurs = drapeaux d'autres options de la MEME commande.
"""
import json, os, random, re, sys
import bibliotheque as bib

GRAINE = 20260801
N_CIBLE = 400
N_TUNE = 100
MAX_PAR_CMD = 20
ICI = os.path.dirname(os.path.abspath(__file__))
SORTIE = os.path.join(ICI, "questions.json")


def affiche(opt):
    """Forme affichee : drapeau long s'il existe, sinon le premier drapeau (jamais l'argument)."""
    longs = [f for f in opt["flags"] if f.startswith("--")]
    return longs[0] if longs else opt["flags"][0]


def noyau(flag):
    return flag.lstrip("-").lstrip("X").lstrip(":+") if flag.startswith("-X") else flag.lstrip("-")


def cite(stem, flag):
    """Vrai si le texte de la question contient le drapeau (exact, borne par non-mots) ou son noyau long."""
    s = stem.lower()
    f = flag.lower()
    if re.search(r"(?<![\w-])" + re.escape(f) + r"(?![\w-])", s):
        return True
    c = noyau(flag).lower()
    c = c.split(":")[-1].lstrip("+") if ":" in c else c
    return len(c) >= 5 and re.search(r"(?<![\w])" + re.escape(c) + r"(?![\w])", s) is not None


def stem_de(opt):
    p = bib.premiere_phrase(opt["desc"])
    p = re.sub(r"\s+", " ", p).strip()
    if len(p.split()) < 6 or len(p.split()) > 45:
        return None
    if p.endswith(":") or "following" in p.lower() and p.lower().startswith("the following"):
        return None
    if "available options for" in p.lower() or "can be one of" in p.lower():
        return None
    return p


def candidats(tables):
    """Pour chaque commande : liste de (option, stem) admissibles (stem sans fuite de la reponse)."""
    res = {}
    for cmd, opts in tables.items():
        if len(opts) < 4:
            continue
        shown = [affiche(o) for o in opts]
        out = []
        for o in opts:
            st = stem_de(o)
            if st is None:
                continue
            if any(cite(st, f) for f in o["flags"]):  # fuite triviale
                continue
            out.append((o, st))
        if out:
            res[cmd] = out
    return res


def construire_questions(tables, graine=GRAINE, n_cible=N_CIBLE, n_tune=N_TUNE, max_par_cmd=MAX_PAR_CMD):
    rng = random.Random(graine)
    cands = candidats(tables)
    pool = []
    for cmd in sorted(cands):
        lst = cands[cmd][:]
        rng.shuffle(lst)
        pool.append((cmd, lst[:max_par_cmd]))
    # tour de role entre commandes jusqu'a n_cible
    choisis, i = [], 0
    while len(choisis) < n_cible + 60 and any(lst for _, lst in pool):
        for cmd, lst in pool:
            if lst and len(choisis) < n_cible + 60:
                choisis.append((cmd, lst.pop()))
    qs, vus = [], set()
    for cmd, (opt, stem) in choisis:
        if stem in vus:
            continue
        vus.add(stem)
        autres = [o for o in tables[cmd] if set(o["flags"]).isdisjoint(opt["flags"])]
        bonne = affiche(opt)
        dist, formes = [], {bonne}
        rng.shuffle(autres)
        for o in autres:
            a = affiche(o)
            if a in formes or cite(stem, a) or any(cite(stem, f) for f in o["flags"]):
                continue  # distracteur nomme dans la question : exclu (symetrie)
            dist.append(a)
            formes.add(a)
            if len(dist) == 3:
                break
        if len(dist) < 3:
            continue
        choix = dist + [bonne]
        rng.shuffle(choix)
        qs.append(dict(cmd=cmd, stem=stem, choices=choix, answer=choix.index(bonne), answer_flag=bonne,
                       all_flags=opt["flags"]))
    rng.shuffle(qs)
    qs = qs[:n_cible]
    for k, q in enumerate(qs):
        q["id"] = k
        q["split"] = "tune" if k < n_tune else "test"
    return qs


def main():
    tables = bib.construire()
    qs = construire_questions(tables)
    with open(SORTIE, "w", encoding="utf-8") as f:
        json.dump(dict(graine=GRAINE, n=len(qs), questions=qs), f, ensure_ascii=False, indent=1)
    cmds = {q["cmd"] for q in qs}
    nt = sum(q["split"] == "test" for q in qs)
    print(f"{len(qs)} questions, {len(cmds)} commandes, test={nt}, tune={len(qs) - nt}")
    from collections import Counter
    print("position de la bonne reponse :", Counter(q["answer"] for q in qs))


if __name__ == "__main__":
    main()
