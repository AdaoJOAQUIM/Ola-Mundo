"""Bibliotheque locale : pages de man de /usr/share/man/man1 -> texte brut + tables d'options.

Aucun reseau. Deux formats roff sont geres : DocBook (PostgreSQL : `.PP`, `\\fB-x\\fR`, `.br`,
`.RS 4` ... `.RE`) et Pandoc (JDK : `.TP`, `\\f[V]-x\\f[R] or \\f[V]--long\\f[R]`).
Le texte des pages n'est JAMAIS commite (cache/ est ignore par git).
"""
import gzip, json, os, re

MAN_DIR = "/usr/share/man/man1"
# Complement local (hors /usr/share/man, qui ne contient que PostgreSQL et JDK) : pages de cargo/rustc/rustdoc
# de la chaine Rust installee. Voir resultats.md (ecart au plan).
EXTRA_DIR = "/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/share/man/man1"
ICI = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ICI, "cache")
LIB_DIR = os.path.join(CACHE, "bibliotheque")

_ESC = [
    (r"\\f\[[A-Z0-9]+\]", ""), (r"\\f[BIRPV]", ""), (r"\\f\(..", ""), (r"\\\*\(..", ""),
    (r"\\\(lq|\\\(rq|\\\(Lq|\\\(Rq|\\\[lq\]|\\\[rq\]", '"'), (r"\\\[aq\]|\\\(aq|\\\(cq|\\\(oq", "'"),
    (r"\\\(bu|\\\[bu\]", "*"), (r"\\\(em|\\\(en|\\\[em\]|\\\[en\]", "-"),
    (r"\\\[u2026\]", "..."), (r"\\\[[^\]]*\]", ""), (r"\\\([a-zA-Z]{2}", ""),
    (r"\\-", "-"), (r"\\&", ""), (r"\\\|", ""), (r"\\c\b", ""), (r"\\e", "\\"),
    (r"\\ ", " "), (r"\\~", " "), (r"\\\\", "\\"), (r"\\\^", ""), (r"\\h'[^']*'", ""),
    (r"\\w'[^']*'", ""), (r"\\\"", ""),
]


def propre(s):
    for pat, rep in _ESC:
        s = re.sub(pat, lambda _m, r=rep: r, s)
    return re.sub(r"\s+", " ", s).strip()


def lire_page(chemin):
    ouvre = gzip.open if chemin.endswith(".gz") else open
    with ouvre(chemin, "rt", encoding="utf-8", errors="replace") as f:
        return f.read().split("\n")


def lignes_contenu(brut):
    """Retire commentaires et blocs conditionnels triviaux ; garde macros et texte."""
    out = []
    for l in brut:
        if l.startswith('.\\"') or l.startswith("'\\\""):
            continue
        if re.match(r"^\.(if|ie|el)\b", l) or l.strip() in (r"\}", r".\}"):
            continue
        if l.startswith(r"\h'") or l.startswith(r"\}"):
            continue
        out.append(l)
    return out


FLAG_RE = re.compile(r"(?<![\w-])(--?[A-Za-z][\w:+.\-]*)")


def extraire_flags(entete):
    """Liste de flags (sans =valeur) dans une ligne d'en-tete d'option nettoyee."""
    flags = []
    for m in FLAG_RE.finditer(entete):
        f = m.group(1).rstrip(".,;:")
        f = f.split("=")[0]
        if len(f) >= 2 and f not in flags:
            flags.append(f)
    return flags


def premiere_phrase(desc):
    desc = desc.strip()
    m = re.search(r"(?<=[a-z0-9\)\"'])\.(?=\s+[A-Z]|\s*$)", desc)
    return desc[:m.end()] if m else desc


def options_pandoc(lines):
    """`.TP` + ligne d'en-tete + description (jusqu'au prochain .TP/.SH/.SS ou .PP de niveau 0)."""
    res, i, n = [], 0, len(lines)
    while i < n:
        if lines[i].startswith(".TP"):
            j = i + 1
            while j < n and lines[j].startswith(".PD"):
                j += 1
            if j >= n:
                break
            entete = lines[j]
            k, depth, desc = j + 1, 0, []
            while k < n:
                l = lines[k]
                if l.startswith((".TP", ".SH", ".SS")) and depth == 0:
                    break
                if l.startswith(".PP") and depth == 0:
                    break
                if l.startswith(".RS"):
                    depth += 1
                elif l.startswith(".RE"):
                    depth = max(0, depth - 1)
                elif not l.startswith("."):
                    desc.append(l)
                k += 1
            res.append((propre(entete), propre(" ".join(desc))))
            i = k
        else:
            i += 1
    return res


def options_docbook(lines):
    """Bloc : lignes d'en-tete `\\fB\\-x...` (+ .br) puis `.RS 4` ... `.RE` (profondeur geree)."""
    res, i, n = [], 0, len(lines)
    while i < n:
        if lines[i].startswith(r"\fB\-") or lines[i].startswith(r"\fB\-\-"):
            ent = []
            j = i
            while j < n and (lines[j].startswith(r"\fB\-") or lines[j].startswith(".br")):
                if not lines[j].startswith(".br"):
                    ent.append(lines[j])
                j += 1
            if j < n and lines[j].startswith(".RS") and i > 0 and lines[i - 1].startswith((".PP", ".it", ".sp")):
                k, depth, desc = j + 1, 1, []
                while k < n and depth > 0:
                    l = lines[k]
                    if l.startswith(".RS"):
                        depth += 1
                    elif l.startswith(".RE"):
                        depth -= 1
                        if depth == 0:
                            break
                    elif l.startswith(".sp") or l.startswith(".PP") or l.startswith(".TS"):
                        if depth == 1:
                            desc.append("\x00")  # frontiere de paragraphe
                    elif not l.startswith("."):
                        desc.append(l)
                    k += 1
                texte = " ".join(desc)
                res.append((propre(" ".join(ent)), propre(texte.replace("\x00", " ")) if "\x00" not in texte
                            else propre(texte.split("\x00")[0])))
                i = k
                continue
        i += 1
    return res


def texte_page(lines):
    """Texte brut lisible (pour l'index) : paragraphes nettoyes, macros retirees."""
    out = []
    for l in lines:
        if l.startswith(".SH") or l.startswith(".SS"):
            out.append("\n" + propre(re.sub(r"^\.S[HS]\s*", "", l).strip('"')).upper() + "\n")
        elif l.startswith(".TH") or (l.startswith(".") and not l.startswith(".br")):
            out.append("\n" if l.startswith((".PP", ".TP", ".IP", ".HP", ".sp")) else "")
        elif l.startswith(".br"):
            out.append("\n")
        else:
            out.append(propre(l) + " ")
    t = "".join(out)
    t = re.sub(r"[ \t]+\n", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip() + "\n"


def construire():
    """Parcourt les pages, ecrit cache/bibliotheque/<cmd>.txt, renvoie {cmd: [options]}."""
    os.makedirs(LIB_DIR, exist_ok=True)
    tables = {}
    sources = [(MAN_DIR, ".1.gz")] + ([(EXTRA_DIR, ".1")] if os.path.isdir(EXTRA_DIR) else [])
    paires = [(d, n, suf) for d, suf in sources for n in sorted(os.listdir(d)) if n.endswith(suf)]
    for d, nom, suf in paires:
        chemin = os.path.realpath(os.path.join(d, nom))
        if not os.path.exists(chemin):
            continue
        cmd = nom[:-len(suf)]
        lines = lignes_contenu(lire_page(chemin))
        pandoc = any(l.startswith(".TP") for l in lines) and any("Pandoc" in l for l in lire_page(chemin)[:25])
        brutes = options_pandoc(lines) if pandoc else options_docbook(lines)
        opts, vus = [], set()
        for ent, desc in brutes:
            flags = extraire_flags(ent)
            if not flags or not desc:
                continue
            cle = tuple(flags)
            if cle in vus:
                continue
            vus.add(cle)
            opts.append(dict(cmd=cmd, flags=flags, entete=ent, desc=desc))
        tables[cmd] = opts
        with open(os.path.join(LIB_DIR, cmd + ".txt"), "w", encoding="utf-8") as f:
            f.write(texte_page(lines))
    return tables


def taille_bibliotheque():
    brut = 0
    for d, suf in [(MAN_DIR, ".1.gz"), (EXTRA_DIR, ".1")]:
        if os.path.isdir(d):
            brut += sum(os.path.getsize(os.path.realpath(os.path.join(d, n))) for n in os.listdir(d)
                        if n.endswith(suf) and os.path.exists(os.path.realpath(os.path.join(d, n))))
    txt = sum(os.path.getsize(os.path.join(LIB_DIR, n)) for n in os.listdir(LIB_DIR))
    return dict(man_octets_sur_disque=brut, texte_octets=txt, n_pages=len(os.listdir(LIB_DIR)))


if __name__ == "__main__":
    t = construire()
    for c, o in t.items():
        print(f"{c:20s} {len(o):4d}")
    print(sum(len(o) for o in t.values()), "options,", taille_bibliotheque())
