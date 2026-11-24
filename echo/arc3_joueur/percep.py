"""Aides de perception compactes (jamais de dump brut 64x64 répété)."""
import numpy as np
from collections import deque

HEX = "0123456789abcdef"

def hist(F):
    u, c = np.unique(F, return_counts=True)
    return {int(a): int(b) for a, b in sorted(zip(u, c), key=lambda t: -t[1])}

def comps(F, bg=None, taille_min=1, conn4=True, couleurs=None):
    """Composantes connexes par couleur (hors fond). -> liste de dicts triée par taille décroissante."""
    F = np.asarray(F)
    if bg is None:
        bg = max(hist(F).items(), key=lambda t: t[1])[0]
    H, W = F.shape
    vu = np.zeros_like(F, bool); out = []
    for r in range(H):
        for c in range(W):
            if vu[r, c] or F[r, c] == bg: continue
            col = F[r, c]
            if couleurs is not None and col not in couleurs: continue
            q = deque([(r, c)]); vu[r, c] = True; cells = []
            while q:
                y, x = q.popleft(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)) if conn4 else [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not vu[ny, nx] and F[ny, nx] == col:
                        vu[ny, nx] = True; q.append((ny, nx))
            if len(cells) >= taille_min:
                ys = [p[0] for p in cells]; xs = [p[1] for p in cells]
                out.append({"col": int(col), "n": len(cells), "bbox": (min(ys), min(xs), max(ys), max(xs)),
                            "ctr": (round(sum(ys) / len(ys), 1), round(sum(xs) / len(xs), 1))})
    return sorted(out, key=lambda d: -d["n"])

def fmt_comps(cs, k=25):
    s = ["%s n%d (%d,%d)-(%d,%d)" % (HEX[d["col"]], d["n"], *d["bbox"]) for d in cs[:k]]
    return "; ".join(s) + (" ...+%d" % (len(cs) - k) if len(cs) > k else "")

def diff(A, B):
    """Cellules changées entre A et B, groupées en composantes (8-connexes)."""
    A = np.asarray(A); B = np.asarray(B)
    m = A != B
    n = int(m.sum())
    if n == 0: return n, []
    M = np.where(m, 1, 0)
    cs = comps(M, bg=0, conn4=False)
    res = []
    for d in cs:
        r0, c0, r1, c1 = d["bbox"]
        sub_a = A[r0:r1 + 1, c0:c1 + 1][m[r0:r1 + 1, c0:c1 + 1]]
        sub_b = B[r0:r1 + 1, c0:c1 + 1][m[r0:r1 + 1, c0:c1 + 1]]
        fa = {int(x) for x in np.unique(sub_a)}; fb = {int(x) for x in np.unique(sub_b)}
        res.append({"n": d["n"], "bbox": d["bbox"], "de": sorted(fa), "vers": sorted(fb)})
    return n, res

def fmt_diff(A, B, k=8):
    n, ds = diff(A, B)
    if n == 0: return "aucun changement"
    s = ["n%d (%d,%d)-(%d,%d) %s>%s" % (d["n"], *d["bbox"], "".join(HEX[x] for x in d["de"]), "".join(HEX[x] for x in d["vers"])) for d in ds[:k]]
    return "%d cellules: " % n + "; ".join(s) + (" ...+%d" % (len(ds) - k) if len(ds) > k else "")

def crop(F, r0, c0, r1, c1):
    F = np.asarray(F)
    print("     " + "".join(HEX[c % 16] for c in range(c0, c1 + 1)))
    for r in range(r0, r1 + 1):
        print("%3d: " % r + "".join(HEX[int(v)] for v in F[r, c0:c1 + 1]))

def rle_rows(F, r0=0, r1=None):
    """Compression par lignes: pour repérer la structure (un caractère par pas de 1)."""
    F = np.asarray(F); r1 = F.shape[0] - 1 if r1 is None else r1
    for r in range(r0, r1 + 1):
        s = ""; prev = None; k = 0
        for v in F[r]:
            if v == prev: k += 1
            else:
                if prev is not None: s += "%s%s" % (HEX[prev], k if k > 1 else "")
                prev = int(v); k = 1
        s += "%s%s" % (HEX[prev], k if k > 1 else "")
        print("%2d %s" % (r, s))
