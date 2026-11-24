"""Visite des objets : apprend quelle région est l'avatar (translation quand on appuie sur une flèche),
construit une grille de déplacement au pas de l'avatar avec les murs découverts (flèche sans effet),
puis planifie (BFS, inconnu = libre) jusqu'à l'objet non visité le plus proche et le touche."""
NOM = "visite_objets"
BUDGET = 60
COOLDOWN = 6


def cellule(h, p):
    o = h.glob["origine"]
    s = h.glob["pas"]
    return (int(round((p[0] - o[0]) / s)), int(round((p[1] - o[1]) / s)))


def suivre(h):
    m = h.mem
    g = h.glob
    if "bloq" not in g:
        g["bloq"] = set()
        g["dep"] = {}
    if m.get("n_suivi") == h.n:
        return
    m["n_suivi"] = h.n
    d = h.der
    if d is None or d[0] != "A" or d[1] > 4 or h.p is None:
        return
    fond = int(np.argmax(h.couleurs()))
    cm = h.chgmask()
    mv = [q for q in h.mouvements() if q[6] >= 0.8]
    mv = sorted([(-int(cm[q[0]:q[2] + 1, q[1]:q[3] + 1].sum()), i, q) for i, q in enumerate(mv)])
    mv = [t[2] for t in mv]
    ancien = m.get("pos")
    if len(mv) == 0 or int(cm[mv[0][0]:mv[0][2] + 1, mv[0][1]:mv[0][3] + 1].sum()) < max(4, 0.5 * h.dchg):
        if ancien is not None and "origine" in g:
            g["bloq"].add((cellule(h, ancien), d[1]))
        return
    q = mv[0]
    if min(abs(q[4]), abs(q[5])) > 0 or abs(q[4]) + abs(q[5]) > 20:
        return
    g["dep"][d[1]] = (q[4], q[5])
    sub = cm[q[0]:q[2] + 1, q[1]:q[3] + 1]
    ys, xs = np.nonzero(sub)
    if len(ys) == 0:
        return
    pos = (q[0] + (float(ys.min()) + float(ys.max())) / 2 + q[4] / 2, q[1] + (float(xs.min()) + float(xs.max())) / 2 + q[5] / 2)
    m["pos"] = pos
    if "origine" not in g:
        g["origine"] = pos
        g["pas"] = max(2, abs(q[4]), abs(q[5]))


def depl(h):
    dep = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}
    for a in h.glob["dep"]:
        dep[a] = h.glob["dep"][a]
    return dep


def graphe(h):
    g = h.glob
    o = g["origine"]
    s = g["pas"]
    dep = depl(h)
    res = {}
    for iy in range(29):
        gy = iy - 14
        for ix in range(29):
            gx = ix - 14
            lst = []
            for a in (1, 2, 3, 4):
                if a in h.dispo and ((gy, gx), a) not in g["bloq"]:
                    ny = gy + int(round(dep[a][0] / s))
                    nx = gx + int(round(dep[a][1] / s))
                    py = o[0] + ny * s
                    px = o[1] + nx * s
                    if 2 <= py <= 61 and 2 <= px <= 61:
                        lst.append((a, (ny, nx)))
            res[(gy, gx)] = lst
    return res


def cibles(h):
    res = []
    for o in h.objets():
        b = o["b"]
        if o["n"] < 2 or o["n"] > 150:
            continue
        if b[0] <= 2 or b[1] <= 2 or b[2] >= 61 or b[3] >= 61:
            continue
        res.append(o)
    return res


def prochaine(h):
    m = h.mem
    p = m["pos"]
    best = None
    for o in cibles(h):
        k = (o["y"], o["x"], o["c"])
        if k in m["vis"]:
            continue
        dist = abs(o["cy"] - p[0]) + abs(o["cx"] - p[1])
        if dist < 4:
            m["vis"].add(k)
            continue
        if best is None or dist < best[0]:
            best = (dist, o, k)
    return best


def prepare(h):
    m = h.mem
    if "vis" not in m:
        m["vis"] = set()
        m["cible"] = None
        m["pas"] = 0
    suivre(h)


def activation(h):
    prepare(h)
    if h.mem.get("pos") is None or "origine" not in h.glob or len(h.glob["dep"]) < 1 or h.nn < 6:
        return False
    return len(cibles(h)) > 0


def chemin_cible(h):
    m = h.mem
    o = m["cible"][1]
    cur = cellule(h, m["pos"])
    but = cellule(h, (o["cy"], o["cx"]))
    return h.chemin(graphe(h), cur, {but})


def terminer(h):
    prepare(h)
    m = h.mem
    if m.get("pos") is None:
        return True
    for _ in range(3):
        if m["cible"] is None:
            b = prochaine(h)
            if b is None:
                return True
            m["cible"] = b
            m["pas"] = 0
        o = m["cible"][1]
        p = m["pos"]
        if abs(o["cy"] - p[0]) + abs(o["cx"] - p[1]) < 4 or m["pas"] > 40:
            m["vis"].add(m["cible"][2])
            m["cible"] = None
            continue
        ch = chemin_cible(h)
        if ch is None or len(ch) == 0:
            m["vis"].add(m["cible"][2])
            m["cible"] = None
            continue
        return False
    return True


def agir(h):
    prepare(h)
    m = h.mem
    ch = chemin_cible(h)
    m["pas"] = m["pas"] + 1
    if ch is None or len(ch) == 0:
        return h.A(1)
    return h.A(ch[0])
