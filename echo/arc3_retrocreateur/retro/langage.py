"""Langage de programmes du Rétro-Créateur.

État (snapshot) : (M, pos, avc, tor)
    M   : tuple des types de la couche statique (blocs poussables compris), un entier par case ; 0 = vide ;
          la case de l'avatar vaut -1 (contenu caché)
    pos : case de l'avatar (-1 s'il n'y en a pas)
    avc : couleur courante de l'avatar (variable cachée de mode)
    tor : type d'origine de la case où l'avatar se tient (révélé, éventuellement modifié par un effet « set », quand il la quitte)

Programme :
    dirmap  : action -> direction (0 haut, 1 bas, 2 gauche, 3 droite)
    solid   : types qui bloquent l'avatar            solidreg : types qui bloquent tant que le registre n'est pas sur la cible
    push : types poussables (un seul bloc, destination vide)
    kill    : types mortels au contact                fx   : type -> effets au contact
    clk     : type -> effet d'un clic                 sync : apparences dérivées (A, B, conditions)
    win     : condition de victoire                   grav : direction de la gravité des blocs ou None
    (J3) avc contient aussi le mode du registre : avc = couleur (8 bits bas) | élément de D4 << 8
Effets : ('set', T2) la case devient T2 (supprimer = 0) ; ('recolor', A, B) toute case A devient B ;
         ('toggle', A, B) échange A et B partout ; ('avc', couleur) ; ('mod', g) isométrie g de D4 sur le registre ; ('refill',) énergie pleine ; ('die',) ; ('win',)
Clics  : ('flip', décalages, A, B) bascule A<->B sur les cases décalées ; ('next', décalages, {T: T'}) cycle
Conditions (atomes) : ('gone', T) plus aucune case T ; ('all', Z, S) toute case de type dans Z est de type S ;
         ('pmatch', P, R) motif des cases P égal à la référence R (appariement par ordre de lecture et couleur)
Victoire : ('enter', frozenset(T)) ; ('entermatch', classe) : entrer sur une case de cette forme de la couleur de l'avatar ;
         ('cond', atomes) : tous les atomes vrais après un coup ; ('enterreg', frozenset(T)) : entrer sur un type T quand le
         registre est égal à la cible du niveau (geo.regtarget : ensemble de modes ; None = pas d'exigence).
Exécution rapide : tuples, listes locales, aucune allocation superflue sur les pas sans effet.
"""
from __future__ import annotations

from .registre import composer as _composer

NOM_DIRS = ("haut", "bas", "gauche", "droite")


class Prog:
    def __init__(self, types):
        self.types = types
        self.dirmap = {}
        self.solid = set()
        self.push = set()
        self.kill = set()
        self.fx = {}
        self.clk = {}
        self.sync = []
        self.win = None
        self.grav = None
        self.undo = None
        self.solidreg = set()      # types qui bloquent tant que le registre n'est pas égal à la cible (porte à motif)
        self.select = False        # clic sur une pièce non sélectionnée (M < -1) : elle devient l'avatar
        self.selwin = False        # victoire : chaque case de sortie porte la pièce de sa couleur

    def copie(self):
        p = Prog(self.types)
        p.dirmap = dict(self.dirmap)
        p.solid, p.push, p.kill = set(self.solid), set(self.push), set(self.kill)
        p.fx = dict(self.fx)
        p.clk = dict(self.clk)
        p.sync = list(self.sync)
        p.win, p.grav, p.undo = self.win, self.grav, self.undo
        p.select, p.selwin = self.select, self.selwin
        p.solidreg = set(self.solidreg)
        return p

    def taille(self):
        """Longueur de description (a priori : programmes courts préférés)."""
        n = len(self.solid) + len(self.push) + len(self.kill) + len(self.dirmap)
        for e in self.fx.values():
            n += 2 * len(e)
        n += 3 * len(self.clk)
        for _, _, at in self.sync:
            n += 2 + 2 * len(at)
        n += 3 if self.win else 0
        n += 2 if self.grav is not None else 0
        return n

    # ------------------------------------------------------------ atomes
    def atome(self, at, M, geo):
        k = at[0]
        if k == "gone":
            return at[1] not in M
        if k == "all":
            Z, S = at[1], at[2]
            vu = False
            for t in M:
                if t in Z:
                    if t != S:
                        return False
                    vu = True
            return vu
        if k == "pmatch":
            P, R = at[1], at[2]
            lp = [t for t in M if t in P]
            lr = [t for t in M if t in R]
            if not lp or len(lp) != len(lr):
                return False
            cl = self.types.couleurs
            return all(set(cl(a)) == set(cl(b)) for a, b in zip(lp, lr))
        return False

    def conds(self, ats, M, geo):
        for a in ats:
            if not self.atome(a, M, geo):
                return False
        return True

    def faire_sync(self, M, geo):
        """Applique les apparences dérivées ; renvoie M (liste ou tuple) mis à jour."""
        for A, B, ats in self.sync:
            if A in M or B in M:
                if self.conds(ats, M, geo):
                    if A in M:
                        M = [B if t == A else t for t in M]
                else:
                    if B in M:
                        M = [A if t == B else t for t in M]
        return M

    # ------------------------------------------------------------ gravité
    def tomber(self, M, pos, geo):
        d = self.grav
        M = list(M)
        nb = geo.nb
        blocs = [i for i, t in enumerate(M) if t in self.push]
        if not blocs:
            return M
        dr, dc = geo.dirs[d]
        blocs.sort(key=lambda p: -((p // geo.C) * dr + (p % geo.C) * dc))
        bouge = True
        while bouge:
            bouge = False
            for i, p in enumerate(blocs):
                q = nb[p][d]
                if q >= 0 and M[q] == 0 and q != pos and q not in geo.ignore and q not in geo.outer:
                    M[q] = M[p]
                    M[p] = 0
                    blocs[i] = q
                    bouge = True
        return M

    # ------------------------------------------------------------ transition
    def revele(self, tor):
        for e in self.fx.get(tor, ()):
            if e[0] == "set":
                return e[1]
        return tor

    def pas(self, geo, s, act):
        """(s2, drapeau, type_touché) ou None si l'action est sans effet. drapeau : 0 rien, 1 victoire, 2 mort."""
        M, pos, avc, tor = s
        name, arg = act
        d = self.dirmap.get(name)
        if d is not None:
            if pos < 0:
                return None
            q = geo.nb[pos][d]
            if q < 0 or q in geo.ignore:
                return None
            T = M[q]
            if T in self.solid or T < -1:
                return None
            if self.solidreg and T in self.solidreg:
                ct = getattr(geo, "regtarget", None)
                if ct is not None and (avc >> 8) not in ct:
                    return None
            Ml = list(M)
            Ml[pos] = self.revele(tor)
            if T in self.push:
                q2 = geo.nb[q][d]
                if q2 < 0 or q2 in geo.ignore or q2 in geo.outer or Ml[q2] != 0:
                    return None
                Ml[q2] = T
                Ml[q] = 0
                if self.grav is not None:
                    Ml = self.tomber(Ml, q, geo)
                T0 = 0
            else:
                T0 = T
            flag = 0
            if T0 in self.kill:
                return (tuple(Ml), q, avc, T0), 2, T
            Ml[q] = -1
            fx = self.fx.get(T0)
            if fx:
                for e in fx:
                    k = e[0]
                    if k == "recolor":
                        A, B = e[1], e[2]
                        Ml = [B if t == A else t for t in Ml]
                    elif k == "toggle":
                        A, B = e[1], e[2]
                        Ml = [B if t == A else (A if t == B else t) for t in Ml]
                    elif k == "avc":
                        avc = (avc & ~255) | (e[1] & 255)
                    elif k == "mod":
                        # déclencheur de registre : le motif caché subit l'isométrie g (élément de D4)
                        avc = (avc & 255) | (_composer(e[1], avc >> 8) << 8)
                    elif k == "refill":
                        pass            # énergie remise à pleine : géré par pas_u
                    elif k == "die":
                        flag = 2
                    elif k == "win":
                        flag = 1
            w = self.win
            if flag == 0 and w is not None:
                if w[0] == "enter":
                    if T0 in w[1]:
                        flag = 1
                elif w[0] == "entermatch":
                    if (avc & 255, w[1]) in self.types.inv[T0]:
                        flag = 1
                elif w[0] == "enterreg":
                    ct = getattr(geo, "regtarget", None)
                    if T0 in w[1] and (ct is None or (avc >> 8) in ct):
                        flag = 1
            if self.sync:
                Ml = self.faire_sync(Ml, geo)
            if flag == 0 and w is not None and w[0] == "cond" and self.conds(w[1], Ml, geo):
                flag = 1
            if flag == 0 and self.selwin and self.toutes_sorties(geo, Ml, q, avc):
                flag = 1
            return (tuple(Ml), q, avc, T0), flag, T
        if name == "ACTION6":
            p = arg
            if p is None or p < 0 or p >= geo.N or p in geo.ignore:
                return None
            T = M[p]
            if self.select and T < -1 and pos >= 0:
                Ml = list(M)
                Ml[pos] = -(2 + avc)
                avc2 = -T - 2
                Ml[p] = -1
                tor2 = geo.exit_type.get(p, 0)
                if self.sync:
                    Ml = self.faire_sync(Ml, geo)
                flag = 1 if (self.selwin and self.toutes_sorties(geo, Ml, p, avc2)) else 0
                return (tuple(Ml), p, avc2, tor2), flag, T
            ck = self.clk.get(T)
            if ck is None:
                return None
            Ml = list(M)
            ch = False
            r0, c0 = divmod(p, geo.C)
            if ck[0] == "flip":
                A, B = ck[2], ck[3]
                for dr, dc in ck[1]:
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < geo.R and 0 <= c < geo.C:
                        i = r * geo.C + c
                        if Ml[i] == A:
                            Ml[i] = B
                            ch = True
                        elif Ml[i] == B:
                            Ml[i] = A
                            ch = True
            else:
                nm = ck[2] if geo.nm is None else {**ck[2], **geo.nm}
                for dr, dc in ck[1]:
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < geo.R and 0 <= c < geo.C:
                        i = r * geo.C + c
                        t = Ml[i]
                        if t in nm:
                            Ml[i] = nm[t]
                            ch = True
            if not ch:
                return None
            if self.sync:
                Ml = self.faire_sync(Ml, geo)
            flag = 0
            w = self.win
            if w is not None and w[0] == "cond" and self.conds(w[1], Ml, geo):
                flag = 1
            elif w is not None and w[0] == "enter" and pos >= 0 and self.sync:
                # l'avatar est posé sur une sortie verrouillée que ce clic vient d'ouvrir : victoire
                tt = tor
                for A, B, ats in self.sync:
                    if tt == A and self.conds(ats, Ml, geo):
                        tt = B
                if tt in w[1]:
                    flag = 1
            return (tuple(Ml), pos, avc, tor), flag, T
        return None

    def toutes_sorties(self, geo, Ml, pos, avc):
        """Chaque case de sortie porte la pièce de sa couleur ; sorties déverrouillées (conditions des apparences dérivées)."""
        if not geo.exits:
            return False
        for e, col in geo.exits.items():
            if e == pos:
                if (avc & 255) != col:
                    return False
            elif Ml[e] != -(2 + col):
                return False
        for A, B, ats in self.sync:
            if B in geo.exit_open and not self.conds(ats, Ml, geo):
                return False
        return True

    def pas_u(self, geo, s, u, act):
        """Pas avec minuterie d'énergie : u = nombre d'actions effectives déjà dépensées dans la vie courante."""
        r = self.pas(geo, s, act)
        if r is None:
            return None
        s2, fl, tin = r
        u2 = u + 1
        if self.fx and any(e[0] == "refill" for e in self.fx.get(s2[3], ())):
            u2 = 0               # déclencheur d'énergie (la case vient d'être consommée : plus d'effet ensuite)
        if fl == 0 and geo.cap is not None and u2 >= geo.cap:
            fl = 2
        return s2, fl, tin, u2

    def actions(self, geo, s, avail_click=True):
        """Actions que le programme sait exécuter depuis s."""
        acts = [(n, None) for n in self.dirmap]
        if self.clk:
            M = s[0]
            for p, t in enumerate(M):
                if t in self.clk and p not in geo.ignore:
                    acts.append(("ACTION6", p))
        return acts
