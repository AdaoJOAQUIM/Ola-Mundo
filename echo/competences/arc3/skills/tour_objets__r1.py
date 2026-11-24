"""Tour des objets : repère l'avatar (la région qui se translate quand on appuie sur une flèche),
apprend le déplacement de chaque flèche, puis va toucher les objets non visités, le plus proche d'abord."""
NOM = "tour_objets"
BUDGET = 60
COOLDOWN = 10


def suivre(h):
    m = h.mem
    if m.get("n_suivi") == h.n:
        return
    m["n_suivi"] = h.n
    d = h.der
    dep = h.glob.setdefault("dep", {})
    if d is None or d[0] != "A" or d[1] > 4 or h.p is None:
        return
    fond = int(np.argmax(h.couleurs()))
    cm = h.chgmask()
    mv = [q for q in h.mouvements() if q[6] >= 0.8]
    mv = sorted([(-int(cm[q[0]:q[2] + 1, q[1]:q[3] + 1].sum()), i, q) for i, q in enumerate(mv)])
    mv = [t[2] for t in mv]
    pos = m.get("pos")
    if len(mv) == 0 or int(cm[mv[0][0]:mv[0][2] + 1, mv[0][1]:mv[0][3] + 1].sum()) < max(4, 0.5 * h.dchg):
        if pos is not None:
            m["bloq"].add((int(pos[0]), int(pos[1]), d[1]))
        return
    q = mv[0]
    if min(abs(q[4]), abs(q[5])) > 0 or abs(q[4]) + abs(q[5]) > 20:
        return
    dep[d[1]] = (q[4], q[5])
    sub = cm[q[0]:q[2] + 1, q[1]:q[3] + 1]
    pj = h.p[q[0]:q[2] + 1, q[1]:q[3] + 1]
    ys, xs = np.nonzero(sub & (pj != fond))
    if len(ys) == 0:
        return
    m["pos"] = (q[0] + float(ys.mean()) + q[4], q[1] + float(xs.mean()) + q[5])
    if "bloq" not in m:
        m["bloq"] = set()


def prepare(h):
    m = h.mem
    if "bloq" not in m:
        m["bloq"] = set()
        m["vis"] = set()
        m["cible"] = None
        m["pas"] = 0
    suivre(h)


def cibles(h):
    ob = h.objets()
    res = []
    for o in ob:
        b = o["b"]
        if o["n"] < 2 or o["n"] > 150:
            continue
        if b[0] <= 2 or b[1] <= 2 or b[2] >= 61 or b[3] >= 61:
            continue
        res.append(o)
    return res


def activation(h):
    prepare(h)
    dep = h.glob.get("dep", {})
    if len(dep) < 1 or h.mem.get("pos") is None:
        return False
    return len(cibles(h)) > 0 and h.nn > 6


def choisir_cible(h):
    m = h.mem
    p = m["pos"]
    best = None
    for o in cibles(h):
        k = (o["y"], o["x"], o["c"])
        if k in m["vis"]:
            continue
        dist = abs(o["cy"] - p[0]) + abs(o["cx"] - p[1])
        if dist < 3:
            m["vis"].add(k)
            continue
        if best is None or dist < best[0]:
            best = (dist, o, k)
    return best


def terminer(h):
    prepare(h)
    m = h.mem
    if m.get("pos") is None:
        return True
    if m["cible"] is None or m["pas"] > 30:
        if m["cible"] is not None:
            m["vis"].add(m["cible"][2])
        b = choisir_cible(h)
        if b is None:
            return True
        m["cible"] = b
        m["pas"] = 0
    o = m["cible"][1]
    p = m["pos"]
    nb = [a for a in (1, 2, 3, 4) if (int(p[0]), int(p[1]), a) in m["bloq"]]
    if len(nb) >= 3:
        return True
    if abs(o["cy"] - p[0]) + abs(o["cx"] - p[1]) < 3:
        m["vis"].add(m["cible"][2])
        m["cible"] = None
        return False
    return False


def agir(h):
    prepare(h)
    m = h.mem
    if m["cible"] is None:
        b = choisir_cible(h)
        if b is None:
            return h.A(1)
        m["cible"] = b
    o = m["cible"][1]
    p = m["pos"]
    dy, dx = o["cy"] - p[0], o["cx"] - p[1]
    dep = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}
    for a in h.glob["dep"]:
        dep[a] = h.glob["dep"][a]
    cand = []
    for a in dep:
        if a in h.dispo:
            ay, ax = dep[a]
            gain = abs(dy) + abs(dx) - abs(dy - ay) - abs(dx - ax)
            if (int(p[0]), int(p[1]), a) in m["bloq"]:
                gain = gain - 100
            cand.append((gain, a))
    cand = sorted([(-t[0], t[1]) for t in cand])
    m["pas"] = m["pas"] + 1
    return h.A(cand[0][1])
