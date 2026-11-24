"""Rendu 64x64, 16 couleurs. Le plateau occupe les 56 premières lignes ; HUD en bas
(barre d'énergie, compteur de gemmes, pastilles de niveaux)."""
from __future__ import annotations

import numpy as np

from moteur import I_COL, I_GEM, I_KEY, I_PAN, I_PCS, I_SEL, I_SW, I_BLK

H = W_ = 64
ZONE = 56


def taille_case(n):
    return max(3, ZONE // n)


def origine(n):
    c = taille_case(n)
    return (64 - n * c) // 2, (ZONE - n * c) // 2, c


def masque(forme, c):
    m = np.zeros((c, c), dtype=bool)
    a, b = (1, c - 1) if c >= 5 else (0, c)
    mid = c // 2
    if forme == "plein":
        m[a:b, a:b] = True
    elif forme == "anneau":
        m[a:b, a:b] = True
        if b - a > 2:
            m[a + 1:b - 1, a + 1:b - 1] = False
    elif forme == "croix":
        m[a:b, mid] = True
        m[mid, a:b] = True
        if c >= 7:
            m[a:b, mid - 1] = True
            m[mid - 1, a:b] = True
    elif forme == "diamant":
        for y in range(c):
            for x in range(c):
                if abs(y - (c - 1) / 2) + abs(x - (c - 1) / 2) <= (c - 1) / 2 - 0.2:
                    m[y, x] = True
    elif forme == "point":
        s = max(1, c // 3)
        o = (c - s) // 2
        m[o:o + s, o:o + s] = True
    else:  # damier
        for y in range(a, b):
            for x in range(a, b):
                if (x + y) % 2 == 0:
                    m[y, x] = True
    return m


def rendre(W, lv, M, s, used, niv_idx, nl, finis):
    f = np.full((64, 64), W.bg, dtype=np.int8)
    n = lv.n
    ox, oy, c = origine(n)
    col = W.col
    masques = {}

    def dessine(pos, forme, couleur):
        y, x = divmod(pos, n)
        if forme == "plein_total":
            f[oy + y * c: oy + (y + 1) * c, ox + x * c: ox + (x + 1) * c] = couleur
            return
        mk = masques.get(forme)
        if mk is None:
            mk = masques[forme] = masque(forme, c)
        blk = f[oy + y * c: oy + (y + 1) * c, ox + x * c: ox + (x + 1) * c]
        blk[mk] = couleur

    pcs, sel, blk, gem, key, sw, pcol, pan, t = s
    pn = lv.panel
    pancells = set(pn["cells"]) | set(pn.get("ref", ())) if pn else set()
    for p in lv.solid:
        if p not in pancells:
            dessine(p, "plein_total", col["mur"])
    # panneau et référence
    if pn:
        if pn["kind"] == "lumieres":
            for i, p in enumerate(pn["cells"]):
                on = (pan >> i) & 1
                dessine(p, "plein", col["on"] if on else col["off"])
        else:
            for i, p in enumerate(pn["cells"]):
                dessine(p, "plein", col["c%d" % pan[i]])
            for i, p in enumerate(pn["ref"]):
                dessine(p, "anneau", col["c%d" % pn["goal"][i]])
    for p in lv.targets:
        dessine(p, W.forme["cible"], col["cible"])
    for p, cc in lv.paint.items():
        dessine(p, "anneau", col["c%d" % cc])
    for p in lv.hazards:
        dessine(p, W.forme["danger"], col["danger"])
    for i, (p, q) in enumerate(sorted(lv.tele.items())):
        pass
    vus = set()
    pair = 0
    for p, q in sorted(lv.tele.items()):
        if p in vus:
            continue
        vus.add(p)
        vus.add(q)
        r = "tele%d" % (pair % 2)
        dessine(p, "diamant" if W.forme[r] != "diamant" else "anneau", col[r])
        dessine(q, "diamant" if W.forme[r] != "diamant" else "anneau", col[r])
        pair += 1
    for p in lv.switches:
        dessine(p, W.forme["inter"], col["inter"])
    for p, (g, inv) in lv.swalls.items():
        if ((sw >> g) & 1) == inv:
            dessine(p, "plein", col["swall"])
        else:
            dessine(p, "point", col["swall"])
    verrou_ok = M.verrous_ok(s)
    for p, cc in lv.exits:
        couleur = col["c%d" % cc] if lv.couleurs else col["sortie"]
        dessine(p, "plein" if verrou_ok else "anneau", couleur)
    for i, p in enumerate(lv.gems):
        if not (gem >> i) & 1:
            dessine(p, W.forme["gemme"], col["gemme"])
    for p, kid in lv.keys:
        if not (key >> kid) & 1:
            dessine(p, "croix", col["cle%d" % (kid % 2)])
    for p, kid in lv.doors.items():
        if (key >> kid) & 1:
            dessine(p, "anneau", col["cle%d" % (kid % 2)])
        else:
            dessine(p, "plein_total", col["cle%d" % (kid % 2)])
    for p in blk:
        dessine(p, W.forme["bloc"], col["bloc"])
        if p in lv.targets:
            dessine(p, "point", col["cible"])
    if lv.enemies:
        for p in M.epos[t]:
            dessine(p, W.forme["ennemi"], col["ennemi"])
    if M.has_player:
        for i, p in enumerate(pcs):
            if lv.selection:
                cc = col["c%d" % lv.piece_cols[i]]
            elif lv.couleurs:
                cc = col["c%d" % pcol]
            else:
                cc = col["joueur"]
            dessine(p, W.forme["joueur"], cc)
            if lv.selection and i == sel:
                y, x = divmod(p, n)
                y0, x0 = oy + y * c, ox + x * c
                f[y0, x0:x0 + c] = col["sel"]
                f[y0 + c - 1, x0:x0 + c] = col["sel"]
                f[y0:y0 + c, x0] = col["sel"]
                f[y0:y0 + c, x0 + c - 1] = col["sel"]
    # HUD
    if lv.energy:
        rest = max(0, lv.energy - used)
        f[58:60, 2:62] = col["hud_off"]
        f[58:60, 2:2 + (60 * rest) // lv.energy] = col["energie"]
    for i in range(len(lv.gems)):
        x0 = 2 + 5 * i
        f[61:64, x0:x0 + 3] = col["hud_off"]
        if (gem >> i) & 1:
            f[61:64, x0:x0 + 3] = col["hud_on"]
    for i in range(nl):
        x0 = 62 - 4 * (nl - i)
        v = col["hud_on"] if i < finis else (col["sel"] if i == niv_idx else col["hud_off"])
        f[62:64, x0:x0 + 3] = v
    return f
