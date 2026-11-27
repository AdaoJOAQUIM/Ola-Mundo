"""BM25 maison (sans dependance) sur des fenetres de mots des pages de la bibliotheque."""
import math, os, pickle, re
from collections import Counter
import bibliotheque as bib

TOK = re.compile(r"[a-z0-9]+")
FENETRE, PAS = 120, 60
INDEX_PATH = os.path.join(bib.CACHE, "index_bm25.pkl")


def tok(s):
    return TOK.findall(s.lower())


def decouper(cmd, texte):
    mots = texte.split()
    out = []
    for s in range(0, max(1, len(mots) - FENETRE + PAS), PAS):
        w = mots[s:s + FENETRE]
        if len(w) >= 8:
            out.append(" ".join(w))
        if s + FENETRE >= len(mots):
            break
    return out


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b      # chunks : liste de (cmd, texte)
        self.tf, self.dl, df = [], [], Counter()
        for cmd, t in chunks:
            c = Counter(tok(cmd + " " + t))
            self.tf.append(c)
            self.dl.append(sum(c.values()))
            df.update(c.keys())
        self.N = len(chunks)
        self.avg = sum(self.dl) / self.N
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.post = {}
        for i, c in enumerate(self.tf):
            for w in c:
                self.post.setdefault(w, []).append(i)

    def search(self, q, k=3):
        sc = Counter()
        for w in set(tok(q)):
            if w not in self.idf:
                continue
            for i in self.post[w]:
                f = self.tf[i][w]
                sc[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
        return [i for i, _ in sc.most_common(k)]


def construire_index(sauver=True):
    chunks = []
    for n in sorted(os.listdir(bib.LIB_DIR)):
        cmd = n[:-4]
        t = open(os.path.join(bib.LIB_DIR, n), encoding="utf-8").read()
        for c in decouper(cmd, t):
            chunks.append((cmd, c))
    idx = BM25(chunks)
    if sauver:
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(idx, f)
    return idx


def charger_index():
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    bib.construire()
    idx = construire_index()
    print(idx.N, "fenetres ; index", os.path.getsize(INDEX_PATH), "octets")
