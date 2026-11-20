#!/usr/bin/env python3
"""Banc d'essai de l'induction (émergence, convergence, coût).

Bibliothèque standard uniquement. Graines fixes (SEED). Tout ce qui sert de
VÉRITÉ TERRAIN est calculé par ce banc, à partir du modèle générateur du monde
(taxonomie cachée + probabilités par traits), sans jamais interroger l'inducteur
ni le raisonneur. Le système ne voit qu'un sous-ensemble aléatoire des faits vrais.

Sorties : resultats_induction.md et resultats_induction.json.

    python3 banc_induction.py            # banc complet
    python3 banc_induction.py --rapide   # version réduite (contrôle rapide)
"""

import argparse
import json
import math
import platform
import random
import sys
import time
from collections import Counter, defaultdict

import raisonneur as R
import induction as I

SEED = 2024
WATTS = 10.0                       # ESTIMATION : 10 W par cœur occupé
CONS, VOY = "bdfgklmnprtvz", "aeiou"

# ======================================================================
# 1. Génération des mondes (vérité terrain indépendante)
# ======================================================================

VALEURS_TRAIT = [1.0, 0.0, 0.7, 0.2]      # universelle, jamais, « exception », rare
POIDS_TRAIT = [0.30, 0.30, 0.20, 0.20]


def nom_seq(i, prefixe="zu"):
    """Nom déterministe et unique (syllabes consonne+voyelle) pour l'exemple i."""
    out, n = "", i
    while True:
        n, d = divmod(n, len(CONS) * len(VOY))
        out += CONS[d % len(CONS)] + VOY[d // len(CONS)]
        if n == 0:
            return prefixe + out


def _mot(rng, utilises):
    while True:
        m = "".join(rng.choice(CONS) + rng.choice(VOY) for _ in range(rng.choice((2, 3))))
        if m not in utilises:
            utilises.add(m)
            return m


class Monde:
    """Monde caché : arbre de classes + traits dont la probabilité dépend du nœud."""

    def __init__(self, w, k, r=0.7, r_neg=0.5, p_regles=0.0):
        rs = random.Random(f"struct-{w}")
        self.k, self.r, self.r_neg = k, r, r_neg
        n_arbre, n_traits = rs.randint(4, 8), rs.randint(2, 4)
        mots = set()
        self.arbre = [_mot(rs, mots) for _ in range(n_arbre)]
        self.traits = [_mot(rs, mots) for _ in range(n_traits)]
        self.classes = self.arbre + self.traits
        self.parent = [None] + [rs.randrange(i) for i in range(1, n_arbre)]
        self.anc = []
        for i in range(n_arbre):
            a, j = {i}, i
            while self.parent[j] is not None:
                j = self.parent[j]
                a.add(j)
            self.anc.append(a)
        # P[n][t] : probabilité du trait t pour un individu de base n (hérité à 80 %)
        self.P = [[None] * n_traits for _ in range(n_arbre)]
        for t in range(n_traits):
            for n in range(n_arbre):
                if n == 0 or rs.random() > 0.8:
                    self.P[n][t] = rs.choices(VALEURS_TRAIT, POIDS_TRAIT)[0]
                else:
                    self.P[n][t] = self.P[self.parent[n]][t]
        self._verite()
        # individus et observations (graine dépendant du monde ET de k)
        ro = random.Random(f"obs-{w}-{k}")
        self.individus, self.vrai = [], {}
        for n in range(n_arbre):
            for _ in range(k):
                nom = _mot(ro, mots)
                cl = {self.arbre[a] for a in self.anc[n]}
                for t in range(n_traits):
                    if ro.random() < self.P[n][t]:
                        cl.add(self.traits[t])
                self.individus.append(nom)
                self.vrai[nom] = cl
                self._base_de = getattr(self, "_base_de", {})
                self._base_de[nom] = self.arbre[n]
        self.obs = {}     # (individu, classe) -> True/False (révélé)
        for x in self.individus:
            rev = False
            for c in self.classes:
                if c in self.vrai[x]:
                    if ro.random() < r:
                        self.obs[(x, c)] = True
                        rev = True
                elif ro.random() < r_neg:
                    self.obs[(x, c)] = False
            if not rev:
                self.obs[(x, self._base_de[x])] = True
        self.regles_donnees = []
        for (a, b) in sorted(self.universelles):
            if ro.random() < p_regles:
                self.regles_donnees.append((a, b))
        self.ordre = [("regle", a, b) for a, b in self.regles_donnees]
        faits = [("fait", x, c, v) for (x, c), v in sorted(self.obs.items())]
        ro.shuffle(faits)
        self.ordre += faits

    # -- vérité terrain, calculée depuis le modèle générateur uniquement ----
    def p(self, n, cls):
        if cls in self.arbre:
            return 1.0 if self.arbre.index(cls) in self.anc[n] else 0.0
        return self.P[n][self.traits.index(cls)]

    def _verite(self):
        N = range(len(self.arbre))
        self.universelles, self.disjointes, self.categorie = set(), set(), {}
        for a in self.classes:
            S = [n for n in N if self.p(n, a) > 0]
            for b in self.classes:
                if a == b or not S:
                    continue
                if all(self.p(n, b) == 1.0 for n in S):
                    self.universelles.add((a, b))
                    self.categorie[(a, b)] = "universelle"
                    continue
                conj = sum(self.p(n, a) * self.p(n, b) for n in N)
                masse = sum(self.p(n, a) for n in N)
                if conj == 0:
                    self.disjointes.add((a, b))
                    self.categorie[(a, b)] = "disjointe"
                else:
                    ratio = conj / masse
                    self.categorie[(a, b)] = "tendance" if ratio >= 0.5 else "rare"

    def phrases(self):
        out = []
        for e in self.ordre:
            if e[0] == "regle":
                out.append(f"Tout {e[1]} est un {e[2]}.")
            else:
                _, x, c, v = e
                out.append(f"{x.capitalize()} {'est' if v else 'n' + chr(39) + 'est pas'} un {c}.")
        return out

    def questions(self, q_max, rng):
        """Paires (individu, classe) NON révélées, avec la vérité terrain."""
        cand = [(x, c) for x in self.individus for c in self.classes
                if (x, c) not in self.obs]
        rng.shuffle(cand)
        return [(x, c, c in self.vrai[x]) for x, c in cand[:q_max]]


def construire(monde, s_min=3, **kw):
    """Charge les phrases du monde dans un Raisonneur et crée l'inducteur."""
    r = R.Raisonneur()
    statuts = Counter(ins.statut for ins in r.charger(monde.phrases()))
    assert set(statuts) <= {"ajoutee", "doublon"}, statuts   # le banc ne génère rien d'incompris
    return I.Inducteur(r, s_min=s_min, **kw)


# ======================================================================
# 2. Évaluation d'un inducteur sur un monde
# ======================================================================

def juger_hypotheses(monde, hyps):
    tp_t = fp_t = tp_a = fp_a = 0
    compo = Counter()
    for (genre, a, b) in hyps:
        if genre == "tout":
            cat = monde.categorie.get((a, b), "inconnue")
            compo[cat] += 1
            if cat == "universelle":
                tp_t += 1
            else:
                fp_t += 1
        else:
            if (a, b) in monde.disjointes:
                tp_a += 1
            else:
                fp_a += 1
                compo["aucun-faux"] += 1
    return tp_t, fp_t, tp_a, fp_a, compo


def evaluer(monde, ind, q_max=300, graine=0):
    rq = random.Random(f"q-{graine}")
    qs = monde.questions(q_max, rq)
    f = ind.fermeture()
    n_univ = sum(1 for p in monde.universelles if not f.tout_certain(*p))
    n_disj = sum(1 for p in monde.disjointes if not f.aucun_certain(*p))
    tp_t, fp_t, tp_a, fp_a, compo = juger_hypotheses(monde, ind.hyps)
    res = {"tp_t": tp_t, "fp_t": fp_t, "tp_a": tp_a, "fp_a": fp_a,
           "T_t": n_univ, "T_a": n_disj, "compo": dict(compo),
           "nq": len(qs), "certain": 0, "certain_ok": 0, "prob": 0, "prob_ok": 0,
           "prob_oui": 0, "prob_oui_ok": 0, "prob_non": 0, "prob_non_ok": 0,
           "inconnu": 0, "non_baseline_ok": 0, "conf": [], "conf_oui": [], "t_query": 0.0}
    for x, c, vrai in qs:
        res["non_baseline_ok"] += (not vrai)
        t0 = time.process_time()
        rep = ind.repondre(f"Est-ce que {x.capitalize()} est un {c} ?")
        res["t_query"] += time.process_time() - t0
        if rep.statut == "certain":
            res["certain"] += 1
            res["certain_ok"] += (rep.valeur == "oui") == vrai
        elif rep.statut == "probable":
            ok = (rep.valeur == "oui") == vrai
            res["prob"] += 1
            res["prob_ok"] += ok
            res["prob_" + rep.valeur] += 1
            res["prob_" + rep.valeur + "_ok"] += ok
            res["conf"].append((rep.confiance, bool(ok)))
            if rep.valeur == "oui":
                res["conf_oui"].append((rep.confiance, bool(ok)))
        else:
            res["inconnu"] += 1
    return res


def fusion(rs):
    out = Counter()
    conf, conf_oui = [], []
    compo = Counter()
    for r in rs:
        for k, v in r.items():
            if k == "conf":
                conf.extend(v)
            elif k == "conf_oui":
                conf_oui.extend(v)
            elif k == "compo":
                compo.update(v)
            else:
                out[k] += v
    out = dict(out)
    out["conf"], out["conf_oui"], out["compo"] = conf, conf_oui, dict(compo)
    return out


def frac(a, b):
    return a / b if b else None


# ======================================================================
# 3. Calibration
# ======================================================================

BORNES = [0.0, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0001]


def calibration(conf):
    """Seaux de confiance ; erreur de calibration attendue (ECE) = somme pondérée
    des écarts |confiance moyenne - exactitude observée|."""
    seaux, n_tot, ece = [], len(conf), 0.0
    for lo, hi in zip(BORNES, BORNES[1:]):
        s = [(c, ok) for c, ok in conf if lo <= c < hi]
        if not s:
            continue
        cm = sum(c for c, _ in s) / len(s)
        ex = sum(ok for _, ok in s) / len(s)
        ece += len(s) / n_tot * abs(cm - ex)
        seaux.append({"de": lo, "a": min(hi, 1.0), "n": len(s), "conf_moy": cm, "exactitude": ex})
    return seaux, (ece if n_tot else None)


# ======================================================================
# 4. Convergence : un seul couple de classes, exemples un par un
# ======================================================================

def courbe_confiance(ns):
    """Confiance mesurée dans le vrai système pour une règle vraie, n exemples."""
    out = []
    for n in ns:
        ind = I.Inducteur(s_min=1)
        for i in range(n):
            nom = nom_seq(i)
            ind.ajouter(f"{nom.capitalize()} est un chien.", rafraichir=False)
            ind.ajouter(f"{nom.capitalize()} est un aboyeur.", rafraichir=False)
        ind.hyps = ind.induire()
        h = ind.hyps[("tout", "chien", "aboyeur")]
        b = h.beta()
        out.append({"n": n, "confiance": h.confiance, "erreur": 1 - h.confiance,
                    "beta_moyenne": b["moyenne"], "beta_borne_inf_5": b["borne_inf_5pct"]})
    return out


def simuler_exception(p, essais, s_min, graine, n_max=300):
    """Flux d'exemples (chien ?, aboyeur avec proba p). Chaque exemple = 2 phrases.
    On s'arrête au premier contre-exemple (l'hypothèse, si elle existe, est
    retirée à cet instant par le système lui-même : on le vérifie)."""
    rng = random.Random(graine)
    proposees, delais, longueurs, retirees_ok = 0, [], [], 0
    actif_conf = defaultdict(lambda: [0, 0])      # s -> [observations, succès]
    for _ in range(essais):
        ind = I.Inducteur(s_min=s_min)
        propose = False
        for i in range(1, n_max + 1):
            nom = nom_seq(i).capitalize()
            cle = ("tout", "chien", "aboyeur")
            actif = ind.hyps.get(cle)
            succes = rng.random() < p
            if actif is not None:
                actif_conf[actif.s][0] += 1
                actif_conf[actif.s][1] += succes
            ind.ajouter(f"{nom} est un chien.")
            rap = ind.ajouter(f"{nom} est un aboyeur." if succes else f"{nom} n'est pas un aboyeur.")
            if cle in ind.hyps and not propose:
                propose = True
                proposees += 1
                t_prop = i
            if not succes:
                longueurs.append(i)
                if propose:
                    delais.append(i - t_prop)
                    retirees_ok += cle in ind.refutees and cle not in ind.hyps
                break
        else:
            longueurs.append(n_max)
    moy = lambda v: sum(v) / len(v) if v else None
    return {"p": p, "essais": essais, "s_min": s_min,
            "proposee": proposees / essais, "theorie_proposee": p ** s_min,
            "exemples_jusqu_a_proposition": s_min if proposees else None,
            "exemples_apres_proposition_jusqu_a_retrait": moy(delais),
            "theorie_apres_proposition": 1 / (1 - p) if p < 1 else None,
            "exemples_jusqu_au_premier_contre_exemple": moy(longueurs),
            "retraits_verifies": retirees_ok, "retraits": len(delais),
            "calibration_active": {s: {"n": v[0], "exactitude": frac(v[1], v[0]),
                                       "conf_annoncee": I.laplace(s)}
                                   for s, v in sorted(actif_conf.items()) if s <= 8}}


# ======================================================================
# 5. Banc principal
# ======================================================================

def fmt(x, d=1, pct=False):
    if x is None:
        return "n/a"
    if pct:
        return f"{100 * x:.{d}f} %".replace(".", ",")
    return f"{x:.{d}f}".replace(".", ",")


def lancer(rapide=False):
    W = 8 if rapide else 30
    KS = [1, 2, 3, 5, 10, 20]
    SMINS = [2, 3, 5]
    Q = 150 if rapide else 300
    out = {"graine": SEED, "mondes": W, "k": KS, "s_min": SMINS,
           "questions_par_monde": Q, "r": 0.7, "r_neg": 0.5,
           "python": platform.python_version(), "watts_estimes": WATTS}
    cellules = {}
    boucles = []
    cout_induire = defaultdict(list)
    cout_query = defaultdict(lambda: [0.0, 0])
    naif = defaultdict(list)
    couv = defaultdict(list)
    t_global = time.time()
    for w in range(W):
        for k in KS:
            m = Monde(w, k)
            ind = construire(m)
            for s_min in SMINS:
                ind.s_min = s_min
                t0 = time.process_time()
                ind._ferm = None
                ind.hyps = ind.induire()
                cout_induire[(k, s_min)].append(time.process_time() - t0)
                ev = evaluer(m, ind, Q, graine=f"{w}-{k}")
                cout_query[(k, s_min)][0] += ev["t_query"]
                cout_query[(k, s_min)][1] += ev["nq"]
                cellules.setdefault((k, s_min), []).append(ev)
                rb = ind.boucle()
                boucles.append({"k": k, "s_min": s_min, "it": rb.iterations,
                                "conv": rb.converge, "plafond": rb.plafond_atteint,
                                "H": rb.nb_hypotheses, "F": rb.nb_faits_provisoires,
                                "conflits": rb.nb_conflits,
                                "tp_t": None})
            # contraste auto-renforcement et garde de couverture, à s_min = 3
            ind.s_min = 3
            ind.hyps = ind.induire()
            sain = ind.boucle()
            tp, fp, *_ = juger_hypotheses(m, sain.hypotheses)
            naif["sain"].append((k, tp, fp, sain.iterations, sain.plafond_atteint,
                                 sum(h.s for h in sain.hypotheses.values()),
                                 len(sain.hypotheses),
                                 sum(h.confiance for h in sain.hypotheses.values())))
            ia = I.Inducteur(ind.rais, s_min=3, soutien_provisoire=True)
            nv = ia.boucle()
            tp, fp, *_ = juger_hypotheses(m, nv.hypotheses)
            naif["naif"].append((k, tp, fp, nv.iterations, nv.plafond_atteint,
                                 sum(h.s for h in nv.hypotheses.values()),
                                 len(nv.hypotheses),
                                 sum(h.confiance for h in nv.hypotheses.values())))
            for cm in (0.0, 0.5):
                ic = I.Inducteur(ind.rais, s_min=3, couverture_min=cm)
                couv[(k, cm)].append(evaluer(m, ic, Q, graine=f"{w}-{k}"))
        sys.stderr.write(f"monde {w + 1}/{W} ({time.time() - t_global:.0f} s)\n")

    # ---- variante : la moitié des règles vraies est donnée comme phrases ----
    variante = {}
    for k in (3, 5, 10):
        rs = []
        for w in range(W):
            m = Monde(w, k, p_regles=0.5)
            ind = construire(m, s_min=3)
            rs.append(evaluer(m, ind, Q, graine=f"v{w}-{k}"))
        variante[k] = fusion(rs)

    # ---- agrégation ----
    emergence = []
    for k in KS:
        for s_min in SMINS:
            a = fusion(cellules[(k, s_min)])
            nh_t, nh_a = a["tp_t"] + a["fp_t"], a["tp_a"] + a["fp_a"]
            emergence.append({
                "k": k, "s_min": s_min,
                "hyp_tout": nh_t / W, "hyp_aucun": nh_a / W,
                "precision_tout": frac(a["tp_t"], nh_t), "rappel_tout": frac(a["tp_t"], a["T_t"]),
                "precision_aucun": frac(a["tp_a"], nh_a), "rappel_aucun": frac(a["tp_a"], a["T_a"]),
                "questions": a["nq"], "repondues_probablement": a["prob"],
                "part_repondue": frac(a["prob"], a["nq"]),
                "exactitude_probable": frac(a["prob_ok"], a["prob"]),
                "oui": a["prob_oui"], "exactitude_oui": frac(a["prob_oui_ok"], a["prob_oui"]),
                "non": a["prob_non"], "exactitude_non": frac(a["prob_non_ok"], a["prob_non"]),
                "baseline_toujours_non": frac(a["non_baseline_ok"], a["nq"]),
                "certaines": a["certain"], "certaines_ok": a["certain_ok"],
                "compo_fausses": a["compo"]})
    calib = {}
    for s_min in SMINS:
        conf = [c for k in KS if k >= 2 for ev in cellules[(k, s_min)] for c in ev["conf"]]
        seaux, ece = calibration(conf)
        calib[s_min] = {"n": len(conf), "ece": ece, "seaux": seaux,
                        "exactitude_globale": frac(sum(o for _, o in conf), len(conf)),
                        "confiance_moyenne": frac(sum(c for c, _ in conf), len(conf))}
    for s_min in SMINS:
        conf = [c for k in KS if k >= 2 for ev in cellules[(k, s_min)] for c in ev["conf_oui"]]
        seaux, ece = calibration(conf)
        calib[s_min]["oui"] = {"n": len(conf), "ece": ece, "seaux": seaux,
                               "exactitude_globale": frac(sum(o for _, o in conf), len(conf)),
                               "confiance_moyenne": frac(sum(c for c, _ in conf), len(conf))}
    calib_par_k = {}
    for k in KS:
        conf = [c for ev in cellules[(k, 3)] for c in ev["conf"]]
        _, ece = calibration(conf)
        calib_par_k[k] = {"n": len(conf), "ece": ece,
                          "exactitude": frac(sum(o for _, o in conf), len(conf)),
                          "confiance_moyenne": frac(sum(c for c, _ in conf), len(conf))}

    bo = {}
    for s_min in SMINS:
        its = [b["it"] for b in boucles if b["s_min"] == s_min]
        bo[s_min] = {"n": len(its), "histogramme": dict(sorted(Counter(its).items())),
                     "moyenne": sum(its) / len(its), "max": max(its),
                     "plafond_atteint": sum(b["plafond"] for b in boucles if b["s_min"] == s_min),
                     "non_converge": sum(not b["conv"] for b in boucles if b["s_min"] == s_min),
                     "conflits_moy": sum(b["conflits"] for b in boucles if b["s_min"] == s_min) / len(its),
                     "faits_provisoires_moy": sum(b["F"] for b in boucles if b["s_min"] == s_min) / len(its)}
    naif_tab = {}
    for mode, lst in naif.items():
        naif_tab[mode] = {}
        for k in KS:
            L = [e for e in lst if e[0] == k]
            tp, fp = sum(e[1] for e in L), sum(e[2] for e in L)
            naif_tab[mode][k] = {"hyp_moy": sum(e[6] for e in L) / len(L),
                                 "precision": frac(tp, tp + fp),
                                 "soutien_moyen": frac(sum(e[5] for e in L), sum(e[6] for e in L)),
                                 "confiance_moyenne": frac(sum(e[7] for e in L), sum(e[6] for e in L)),
                                 "iterations_moy": sum(e[3] for e in L) / len(L),
                                 "plafond": sum(e[4] for e in L)}
    couv_tab = {}
    for (k, cm), rs in couv.items():
        a = fusion(rs)
        couv_tab.setdefault(k, {})[cm] = {
            "precision_tout": frac(a["tp_t"], a["tp_t"] + a["fp_t"]),
            "rappel_tout": frac(a["tp_t"], a["T_t"]),
            "repondues": a["prob"] / W, "exactitude": frac(a["prob_ok"], a["prob"])}
    cout = []
    for k in KS:
        for s_min in SMINS:
            ti = sum(cout_induire[(k, s_min)]) / len(cout_induire[(k, s_min)])
            tq, nq = cout_query[(k, s_min)]
            cout.append({"k": k, "s_min": s_min, "cpu_induction_s": ti, "cpu_question_s": tq / nq,
                         "energie_induction_J": ti * WATTS, "energie_question_J": tq / nq * WATTS})
    variante_tab = {}
    for k, a in variante.items():
        variante_tab[k] = {
            "certaines": a["certain"], "certaines_ok": a["certain_ok"],
            "probables": a["prob"], "probables_ok": a["prob_ok"],
            "inconnues": a["inconnu"], "nq": a["nq"],
            "rappel_tout": frac(a["tp_t"], a["T_t"]),
            "precision_tout": frac(a["tp_t"], a["tp_t"] + a["fp_t"])}

    out.update({"emergence": emergence, "calibration": calib, "calibration_par_k": calib_par_k,
                "boucle": bo, "auto_renforcement": naif_tab, "couverture": couv_tab,
                "cout": cout, "variante_regles_donnees": variante_tab})
    # ---- convergence sur un couple de classes ----
    out["courbe_confiance"] = courbe_confiance([1, 2, 3, 4, 5, 8, 10, 15, 20, 50, 100])
    essais = 400 if rapide else 2000
    out["exceptions"] = [simuler_exception(p, essais, 3, f"exc-{p}") for p in (0.5, 0.7, 0.9)]
    out["exceptions_smin5"] = [simuler_exception(0.7, essais, 5, "exc-5")]
    out["duree_banc_s"] = time.time() - t_global
    return out


# ======================================================================
# 6. Rapport
# ======================================================================

def rapport_md(o):
    L = []
    P = L.append
    P("# Résultats de l'induction d'Écho\n")
    P(f"Généré par `banc_induction.py` (graine {o['graine']}, {o['mondes']} mondes synthétiques par "
      f"configuration, Python {o['python']}). Bibliothèque standard seulement : aucun réseau de "
      "neurones, aucun poids appris, aucun gradient. L'induction ne fait que **compter** des exemples "
      "et des contre-exemples.\n")
    P("## Avertissement\n")
    P("Les mondes sont **générés par nous** : une taxonomie cachée (arbre de 4 à 8 classes) plus 2 à 4 "
      "« traits » dont la probabilité dépend du nœud (1 = règle universelle, 0 = jamais, 0,7 = "
      "tendance/exception, 0,2 = rare), soit 6 à 12 classes au total. La vérité terrain (règles "
      "universelles vraies, paires disjointes, réponses attendues) est calculée par le banc à partir du "
      f"modèle générateur. Le système ne voit qu'une fraction des faits vrais : chaque appartenance vraie "
      f"est révélée avec la probabilité r = {fmt(o['r'], 1)}, chaque non-appartenance vraie avec r_neg = "
      f"{fmt(o['r_neg'], 1)}. Aucune règle n'est donnée (sauf dans la variante signalée). Les résultats "
      "disent ce que fait le mécanisme **sur ce type de monde** ; ils ne prouvent rien sur le monde réel.\n")
    P("Définitions : s = exemples, c = contre-exemples ; hypothèse « tout A est B » proposée si s >= s_min "
      "et c = 0 ; confiance = règle de succession de Laplace (s+1)/(s+2) (moyenne a posteriori de "
      "Beta(s+1, c+1), prior uniforme). k = nombre d'individus par classe de l'arbre (échantillon "
      "d'exemples). Les questions portent sur des (individu, classe) **non révélés**, donc inconnus du "
      "raisonneur seul ; une réponse « probable » est un savoir que personne n'a écrit.\n")

    P("## 1. Émergence : règles induites et nouvelles questions\n")
    P("Précision = hypothèses qui sont de vraies règles universelles / hypothèses proposées. Rappel = "
      "vraies règles universelles retrouvées / vraies règles universelles du monde (donc limité par "
      "l'échantillon : une règle sur une classe sans individu observé est invisible).\n")
    P("### Règles « Tout A est B » (moyenne sur les mondes)\n")
    P("| k | s_min | hypothèses / monde | précision | rappel |")
    P("|---|---|---|---|---|")
    for e in o["emergence"]:
        P(f"| {e['k']} | {e['s_min']} | {fmt(e['hyp_tout'])} | {fmt(e['precision_tout'], 1, True)} | "
          f"{fmt(e['rappel_tout'], 1, True)} |")
    P("\n### Règles « Aucun A n'est B »\n")
    P("| k | s_min | hypothèses / monde | précision | rappel |")
    P("|---|---|---|---|---|")
    for e in o["emergence"]:
        P(f"| {e['k']} | {e['s_min']} | {fmt(e['hyp_aucun'])} | {fmt(e['precision_aucun'], 1, True)} | "
          f"{fmt(e['rappel_aucun'], 1, True)} |")
    P("\n### Nouvelles questions rendues répondables par l'induction\n")
    P(f"{o['questions_par_monde']} questions par monde et par configuration, toutes sur des faits non "
      "révélés (aucune règle donnée : le raisonneur seul répond « Je ne sais pas » à toutes). "
      "Colonne « toujours non » : exactitude d'un témoin trivial qui répondrait toujours non (la plupart "
      "des appartenances sont fausses), à comparer à l'exactitude des réponses probables.\n")
    P("| k | s_min | répondues « probablement » | part des questions | exactitude | dont oui | exact. oui | dont non | exact. non | témoin « toujours non » |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for e in o["emergence"]:
        P(f"| {e['k']} | {e['s_min']} | {e['repondues_probablement']} | {fmt(e['part_repondue'], 1, True)} | "
          f"{fmt(e['exactitude_probable'], 1, True)} | {e['oui']} | {fmt(e['exactitude_oui'], 1, True)} | "
          f"{e['non']} | {fmt(e['exactitude_non'], 1, True)} | {fmt(e['baseline_toujours_non'], 1, True)} |")
    P("\nRéponses certaines (preuves vérifiées) dans cette expérience : "
      f"{sum(e['certaines'] for e in o['emergence'])} (attendu 0 sans règle donnée).\n")
    P("### Composition des hypothèses « Tout A est B » fausses (s_min = 3)\n")
    P("« tendance » : B vrai pour au moins 50 % des A mais pas tous (exceptions) ; « rare » : B vrai pour "
      "moins de la moitié des A ; « disjointe » : A et B ne coexistent jamais. Totaux sur tous les mondes.\n")
    P("| k | universelles vraies | tendance | rare | disjointe |")
    P("|---|---|---|---|---|")
    for e in o["emergence"]:
        if e["s_min"] != 3:
            continue
        c = e["compo_fausses"]
        P(f"| {e['k']} | {c.get('universelle', 0)} | {c.get('tendance', 0)} | {c.get('rare', 0)} | "
          f"{c.get('disjointe', 0)} |")

    P("\n### Variante : la moitié des règles vraies est donnée comme phrases (s_min = 3)\n")
    P("Les règles données sont certaines (le raisonneur répond avec preuve vérifiée) ; le rappel est "
      "calculé sur les règles universelles non déjà certaines.\n")
    P("| k | certaines (justes) | probables (justes) | inconnues | rappel des règles restantes | précision |")
    P("|---|---|---|---|---|---|")
    for k, v in o["variante_regles_donnees"].items():
        P(f"| {k} | {v['certaines']} ({v['certaines_ok']}) | {v['probables']} ({v['probables_ok']}) | "
          f"{v['inconnues']} | {fmt(v['rappel_tout'], 1, True)} | {fmt(v['precision_tout'], 1, True)} |")

    P("\n## 2. Calibration de la confiance\n")
    P("Réponses « probablement » regroupées par confiance annoncée (k >= 2, tous mondes). ECE = somme "
      "pondérée des |confiance moyenne − exactitude observée| par seau.\n")
    for s_min, c in o["calibration"].items():
        sens = "sous-confiant (prudent)" if c["exactitude_globale"] > c["confiance_moyenne"] else "sur-confiant"
        o_ = c["oui"]
        P(f"**s_min = {s_min}** : {c['n']} réponses ; confiance moyenne {fmt(c['confiance_moyenne'], 1, True)} ; "
          f"exactitude globale {fmt(c['exactitude_globale'], 1, True)} ; **ECE = {fmt(c['ece'], 3)}** "
          f"(globalement {sens}). Réponses « probablement oui » seules : {o_['n']} réponses, confiance "
          f"moyenne {fmt(o_['confiance_moyenne'], 1, True)}, exactitude {fmt(o_['exactitude_globale'], 1, True)}, "
          f"ECE = {fmt(o_['ece'], 3)}.\n")
        P("| confiance annoncée | n | confiance moyenne | exactitude observée |")
        P("|---|---|---|---|")
        for s in c["seaux"]:
            P(f"| [{fmt(s['de'], 2)} ; {fmt(s['a'], 2)}[ | {s['n']} | {fmt(s['conf_moy'], 1, True)} | "
              f"{fmt(s['exactitude'], 1, True)} |")
        P("")
    P("ECE par k (s_min = 3) :\n")
    P("| k | n réponses | confiance moyenne | exactitude | ECE |")
    P("|---|---|---|---|---|")
    for k, c in o["calibration_par_k"].items():
        P(f"| {k} | {c['n']} | {fmt(c['confiance_moyenne'], 1, True)} | {fmt(c['exactitude'], 1, True)} | {fmt(c['ece'], 3)} |")

    P("\n## 3. Convergence\n")
    P("### 3.1 Une règle vraie : confiance en fonction du nombre d'exemples\n")
    P("Mesuré dans le vrai système (s_min = 1, n individus « chien » et « aboyeur »). `Beta 5 %` : borne "
      "inférieure à 5 % de la loi Beta(n+1, 1).\n")
    P("| n exemples | confiance | erreur \\|confiance − 1\\| | Beta moyenne | Beta 5 % |")
    P("|---|---|---|---|---|")
    for c in o["courbe_confiance"]:
        P(f"| {c['n']} | {fmt(c['confiance'], 1, True)} | {fmt(c['erreur'], 4)} | "
          f"{fmt(c['beta_moyenne'], 4)} | {fmt(c['beta_borne_inf_5'], 3)} |")
    P("\nL'erreur vaut 1/(n+2) : elle décroît, mais lentement et sans jamais atteindre 0 (une "
      "induction n'est jamais une preuve).\n")
    P("### 3.2 Une relation « exception » : proposée, puis retirée\n")
    P("Flux d'individus « chien » dont la propriété « aboyeur » est vraie avec la probabilité p "
      "(chaque exemple = 2 phrases). On arrête au premier contre-exemple. Le retrait est fait par le "
      "système lui-même (vérifié à chaque essai). Théorie : P(proposée) = p^s_min ; nombre moyen "
      "d'exemples entre la proposition et le retrait = 1/(1−p).\n")
    P("| p | s_min | essais | proposée au moins une fois | théorie p^s_min | exemples moy. après proposition jusqu'au retrait | théorie 1/(1−p) | exemples moy. jusqu'au 1er contre-exemple | retraits vérifiés |")
    P("|---|---|---|---|---|---|---|---|---|")
    for e in o["exceptions"] + o["exceptions_smin5"]:
        P(f"| {fmt(e['p'], 1)} | {e['s_min']} | {e['essais']} | {fmt(e['proposee'], 1, True)} | "
          f"{fmt(e['theorie_proposee'], 1, True)} | {fmt(e['exemples_apres_proposition_jusqu_a_retrait'], 2)} | "
          f"{fmt(e['theorie_apres_proposition'], 2)} | {fmt(e['exemples_jusqu_au_premier_contre_exemple'], 2)} | "
          f"{e['retraits_verifies']}/{e['retraits']} |")
    P("\nExactitude de la réponse « probablement oui » tant que l'hypothèse est active, par soutien s "
      "(exemples où l'hypothèse était active juste avant l'exemple suivant) :\n")
    P("| p | s | confiance annoncée | observations | exactitude observée |")
    P("|---|---|---|---|---|")
    for e in o["exceptions"]:
        for s, v in e["calibration_active"].items():
            if int(s) in (3, 4, 5, 6):
                P(f"| {fmt(e['p'], 1)} | {s} | {fmt(v['conf_annoncee'], 1, True)} | {v['n']} | "
                  f"{fmt(v['exactitude'], 1, True)} |")
    P("\nLecture : comparer la colonne « confiance annoncée » à « exactitude observée ». Pour p = 1 la "
      "confiance de Laplace est trop basse (80 % pour s = 3 alors que la règle est vraie) ; pour p < 1 elle "
      "est trop haute, et d'autant plus que p est faible (la règle de Laplace suppose un taux inconnu à prior "
      "uniforme, pas le mélange de relations à 100 %, 70 % ou 50 % que contient un vrai monde). Après un "
      "contre-exemple, la confiance en la règle tombe à zéro : l'hypothèse n'existe plus.\n")
    P("### 3.3 Boucle de point fixe (déduire → induire → appliquer → ré-induire)\n")
    P("Une boucle par monde, par k et par s_min. Les faits provisoires ne comptent pas comme exemples. "
      "Le plafond est de 50 itérations ; la preuve de terminaison est dans le docstring de `induction.py`.\n")
    P("| s_min | boucles | itérations moyennes | max | histogramme (itérations : nombre) | plafond atteint | non convergées | faits provisoires moy. | conflits moy. |")
    P("|---|---|---|---|---|---|---|---|---|")
    for s_min, b in o["boucle"].items():
        hist = ", ".join(f"{i} : {n}" for i, n in b["histogramme"].items())
        P(f"| {s_min} | {b['n']} | {fmt(b['moyenne'], 2)} | {b['max']} | {hist} | {b['plafond_atteint']} | "
          f"{b['non_converge']} | {fmt(b['faits_provisoires_moy'], 1)} | {fmt(b['conflits_moy'], 1)} |")
    P("\n« Conflits » : faits provisoires contradictoires entre eux ou avec un fait déduit (par exemple "
      "deux hypothèses incompatibles) ; ils sont comptés, jamais cachés, et la requête s'abstient dans ce cas.\n")
    P("### 3.4 Pourquoi les faits provisoires ne comptent pas : mesure de l'auto-renforcement (s_min = 3)\n")
    P("Mode sain = faits provisoires exclus du soutien. Mode naïf = ils comptent comme exemples (cercle vicieux, "
      "uniquement pour la mesure). « Hypothèses » compte « tout » et « aucun » ; la précision ne porte que "
      "sur « tout ». Soutien moyen = nombre moyen d'exemples par hypothèse ; la confiance en découle.\n")
    P("| k | mode | hypothèses / monde | précision « tout » | soutien moyen | confiance moyenne annoncée | itérations moy. | plafond atteint |")
    P("|---|---|---|---|---|---|---|---|")
    for k in o["k"]:
        for mode in ("sain", "naif"):
            v = o["auto_renforcement"][mode][k]
            P(f"| {k} | {mode} | {fmt(v['hyp_moy'])} | {fmt(v['precision'], 1, True)} | "
              f"{fmt(v['soutien_moyen'])} | {fmt(v['confiance_moyenne'], 1, True)} | "
              f"{fmt(v['iterations_moy'], 2)} | {v['plafond']} |")
    P("\nLecture : le mode naïf gonfle le soutien (donc la confiance) sans aucune observation nouvelle ; "
      "la colonne précision montre si cette confiance supplémentaire est méritée. Si la précision n'augmente "
      "pas autant que la confiance, c'est de la confiance fabriquée.\n")

    P("\n### 3.5 Option : garde de couverture (s / |A| >= seuil), s_min = 3\n")
    P("L'absence d'information n'est pas un contre-exemple : sans garde, « tout animal est un chat » peut "
      "être proposée si seuls des chats ont été vus avec leur espèce. Cette garde est une option (désactivée "
      "par défaut, conformément à la spécification) ; on mesure son effet.\n")
    P("| k | seuil | précision « tout » | rappel « tout » | réponses probables / monde | exactitude |")
    P("|---|---|---|---|---|---|")
    for k, d in o["couverture"].items():
        for cm, v in sorted(d.items()):
            P(f"| {k} | {fmt(cm, 1)} | {fmt(v['precision_tout'], 1, True)} | {fmt(v['rappel_tout'], 1, True)} | "
              f"{fmt(v['repondues'], 1)} | {fmt(v['exactitude'], 1, True)} |")

    P("\n## 4. Coût de calcul\n")
    P("Temps CPU mesuré avec `time.process_time()` (moyenne sur les mondes). Une « passe d'induction » "
      "inclut la fermeture déductive de la base. **ESTIMATION, pas une mesure** : énergie = temps CPU × "
      f"{fmt(o['watts_estimes'], 0)} W (hypothèse de 10 W par cœur occupé) ; aucun wattmètre n'a été utilisé.\n")
    P("| k | s_min | CPU par passe d'induction (ms) | CPU par question (µs) | énergie estimée par passe (mJ) | énergie estimée par question (µJ) |")
    P("|---|---|---|---|---|---|")
    for c in o["cout"]:
        P(f"| {c['k']} | {c['s_min']} | {fmt(c['cpu_induction_s'] * 1e3, 2)} | {fmt(c['cpu_question_s'] * 1e6, 0)} | "
          f"{fmt(c['energie_induction_J'] * 1e3, 2)} | {fmt(c['energie_question_J'] * 1e6, 0)} |")
    P(f"\nDurée totale du banc : {fmt(o['duree_banc_s'], 0)} s.\n")
    return "\n".join(L)


LIMITES = """## 5. Limites (à lire avant de citer un chiffre)

- **L'induction peut avoir tort.** Une hypothèse soutenue par 3, 10 ou 100 exemples sans contre-exemple
  reste fausse si le contre-exemple n'est pas encore observé (cygne noir). La confiance (Laplace) est une
  probabilité sous une hypothèse d'échangeabilité, pas une garantie. Elle n'est pas bien calibrée sur un
  monde qui mélange des règles universelles et des relations « exception » (voir sections 2 et 3.2).
- **Compromis s_min.** Un s_min bas propose beaucoup plus de règles (rappel, nouvelles réponses) mais
  davantage de fausses règles ; un s_min haut est plus sûr mais répond à moins de questions et exige des
  exemples qui n'existent pas toujours. Les tableaux de la section 1 le montrent chiffres à l'appui.
- **Monde ouvert, pas de contre-exemple par défaut.** Ne pas savoir qu'un chien n'est pas un chat n'est
  pas un contre-exemple de « tout animal est un chat » ; l'inducteur peut donc proposer des règles
  contradictoires entre elles (comptées comme conflits). La garde de couverture est une atténuation
  partielle, pas une solution.
- **Seules des relations d'appartenance de classes** sont induites (« tout A est B », « aucun A n'est B »),
  dans le français contrôlé du raisonneur. Pas de relations à plusieurs arguments, pas de lieux, pas de
  nombres, pas de règles conditionnelles, pas de causalité.
- **Biais d'échantillonnage.** Les exemples sont supposés tirés sans biais. Un biais de sélection (on ne
  parle que des chiens qui aboient) produit des règles fausses et très confiantes.
- **Le banc est écrit par nous.** Mondes en arbre + traits, vérité terrain issue du même générateur,
  révélation aléatoire uniforme des faits : c'est un cas favorable et structuré. Les performances sur
  des données réelles (langage naturel, taxonomies incohérentes, négations rares) ne sont pas mesurées ici.
- **Chaînage et produit de confiances** : multiplier les confiances suppose l'indépendance des
  hypothèses, ce qui est faux quand elles reposent sur les mêmes individus. Le chiffre d'un chemin de
  plusieurs hypothèses est une approximation.
- **Coût énergétique estimé**, non mesuré (10 W par cœur occupé supposés).
- **Les hypothèses ne sont pas des faits** : elles ne sont jamais insérées dans la base du raisonneur ;
  elles disparaissent si on recrée l'inducteur sans recalculer, et se recalculent entièrement (pas
  d'apprentissage incrémental fin) à chaque phrase acceptée.
"""


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rapide", action="store_true")
    p.add_argument("--sortie", default=".", help="dossier de sortie")
    a = p.parse_args(argv)
    o = lancer(a.rapide)
    md = rapport_md(o) + "\n" + LIMITES
    suf = "_rapide" if a.rapide else ""
    with open(f"{a.sortie}/resultats_induction{suf}.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(f"{a.sortie}/resultats_induction{suf}.json", "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=1, default=str)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
