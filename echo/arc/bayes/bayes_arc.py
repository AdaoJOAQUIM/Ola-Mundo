"""Fusion bayésienne TRM + DSL sur ARC, et routeur « valeur du calcul » (Russell & Wefald 1991).

Méthodes standard seulement :
  - calibration bêta–binomiale (a priori Beta(1,1) ; pour une variable à plusieurs
    modalités, son extension de Dirichlet(1,...,1)) ;
  - combinaison naïve de Bayes de l'évidence discrète (rapports de vraisemblance) ;
  - choix à utilité espérée (les 2 candidats de plus forte probabilité a posteriori).

Protocole anti-fuite :
  - les paramètres d'un pli sont ajustés SANS lire la solution d'aucune tâche du pli de test
    (`Verite` lève une exception si on le tente) ;
  - ils sont écrits dans `params_pliN.json` (avec une empreinte SHA-256) AVANT toute notation du
    pli de test ; la notation relit ce fichier figé et vérifie l'empreinte.
"""
import hashlib
import json
import math
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
ARC = os.path.dirname(ICI)
ECHO = os.path.dirname(ARC)
sys.path.insert(0, ECHO)
from soi import ece  # noqa: E402  (ECE officielle du dépôt)

COUT_DSL_MAX_S = 10.0  # budget nominal du DSL (le CPU mesuré est utilisé pour les comptes)


# ----------------------------------------------------------------------------- données
def cle(grille):
    return json.dumps(grille, separators=(",", ":"))


class Verite:
    """Accès aux sorties attendues, avec interdiction explicite (garde anti-fuite)."""

    def __init__(self, interdits=()):
        self.interdits = set(interdits)
        self.lectures = set()

    def sortie(self, t):
        if t in self.interdits:
            raise PermissionError(f"lecture interdite de la solution de {t} (pli de test)")
        self.lectures.add(t)
        sol = json.load(open(os.path.join(ARC, "data", "evaluation", f"{t}.json")))["test"]
        return [x["output"] for x in sol]


def toutes_taches():
    return sorted(f[:-5] for f in os.listdir(os.path.join(ARC, "data", "evaluation")))


def charger_dsl():
    d = json.load(open(os.path.join(ARC, "soar_lite", "eval_finale.json")))["details"]
    return {x["id"]: x for x in d}


def charger_trm(chemin):
    return json.load(open(chemin))


# ----------------------------------------------------------------------------- candidats
def bucket_votes(v):
    return 0 if v <= 0 else 1 if v == 1 else 2 if v == 2 else 3 if v <= 4 else 4


def cat_rang(r):
    return 0 if r <= 0 else min(r, 3)


def pool(t, trm_rec, dsl_rec):
    """Candidats d'une tâche à 1 entrée de test : TRM top-5 (votes, q) ∪ tentatives DSL (cohérentes)."""
    p = {}
    for r, c in enumerate(trm_rec["per_test"][0], start=1):
        p[cle(c["grid"])] = {"grille": c["grid"], "votes": c["votes"], "q": c["q"], "rang": r, "dsl": False}
    for s in dsl_rec["sorties"][: dsl_rec["nb_candidats"]]:
        k = cle(s[0])
        p.setdefault(k, {"grille": s[0], "votes": 0, "q": 0.0, "rang": 0, "dsl": False})["dsl"] = True
    return p


def evidence(c):
    return {"bucket": bucket_votes(c["votes"]), "rang": cat_rang(c["rang"]), "dsl": int(c["dsl"]),
            "accord": int(c["dsl"] and c["votes"] > 0)}


MODALITES = {"bucket": 5, "rang": 4, "accord": 2}


# ----------------------------------------------------------------------------- ajustement
def _vrais(pools, verite):
    """Étiquette chaque candidat (juste / faux) pour les tâches de `pools` : {t: {cle: bool}}."""
    out = {}
    for t, p in pools.items():
        gold = verite.sortie(t)[0]
        out[t] = {k: (c["grille"] == gold) for k, c in p.items()}
    return out


def ajuster(pools_cal, dsl, ids_hors_test, verite, caracteristiques=("bucket", "rang", "accord")):
    """Ajuste les vraisemblances.
    - côté TRM (bucket, rang, accord) : sur le pli de calibration `pools_cal` (l'autre pli) ;
    - côté DSL : sur toutes les tâches d'évaluation hors pli de test (`ids_hors_test`).
    Tout est en comptes + Beta(1,1)/Dirichlet(1) : (n + 1) / (N + K)."""
    lab = _vrais(pools_cal, verite)
    cnt = {True: {f: [0] * MODALITES[f] for f in MODALITES}, False: {f: [0] * MODALITES[f] for f in MODALITES}}
    n = {True: 0, False: 0}
    faux_trm = 0
    for t, p in pools_cal.items():
        for k, c in p.items():
            y = lab[t][k]
            n[y] += 1
            e = evidence(c)
            for f in MODALITES:
                cnt[y][f][e[f]] += 1
            if not y and c["votes"] > 0:
                faux_trm += 1
    N = n[True] + n[False]
    prior = (n[True] + 1) / (N + 2)
    lr = {}
    for f in caracteristiques:
        K = MODALITES[f]
        lr[f] = [math.log(((cnt[True][f][v] + 1) / (n[True] + K)) / ((cnt[False][f][v] + 1) / (n[False] + K)))
                 for v in range(K)]
    # côté DSL, sur les tâches hors test (la solution est lue : ce sont des tâches de calibration)
    couvertes = faux_dsl = n_out = 0
    for t in ids_hors_test:
        d = dsl[t]
        n_out += 1
        couvertes += any(d["ok"])
        faux_dsl += sum(1 for o in d["ok"] if not o)
    m_faux_trm = faux_trm / max(1, len(pools_cal))
    faux_pool = faux_dsl + n_out * m_faux_trm  # approximation : recouvrements TRM/DSL parmi les faux ignorés
    p_dsl_juste = (couvertes + 1) / (n_out + 2)
    p_dsl_faux = (faux_dsl + 1) / (faux_pool + 2)
    lr["dsl"] = [math.log((1 - p_dsl_juste) / (1 - p_dsl_faux)), math.log(p_dsl_juste / p_dsl_faux)]
    return {"prior": prior, "n_calibration": {"justes": n[True], "faux": n[False]},
            "log_lr": lr, "caracteristiques": list(caracteristiques) + ["dsl"],
            "dsl": {"taches_hors_test": n_out, "golds_couverts": couvertes, "candidats_faux": faux_dsl,
                    "faux_trm_par_tache": m_faux_trm, "p_dsl_juste": p_dsl_juste, "p_dsl_faux": p_dsl_faux},
            "comptes_trm": {str(k): v for k, v in cnt.items()}}


def geler(params, chemin, meta):
    corps = {"params": params, "meta": meta}
    s = json.dumps(corps, sort_keys=True, ensure_ascii=False)
    corps["sha256"] = hashlib.sha256(s.encode()).hexdigest()
    json.dump(corps, open(chemin, "w"), indent=1, ensure_ascii=False)
    return corps["sha256"]


def relire(chemin):
    corps = json.load(open(chemin))
    h = corps.pop("sha256")
    if hashlib.sha256(json.dumps(corps, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != h:
        raise ValueError("paramètres figés modifiés après gel : " + chemin)
    return corps["params"], corps["meta"]


# ----------------------------------------------------------------------------- inférence
def posterieur(c, params):
    e = evidence(c)
    lo = math.log(params["prior"] / (1 - params["prior"]))
    for f in params["caracteristiques"]:
        lo += params["log_lr"][f][e[f]]
    return 1 / (1 + math.exp(-lo))


def choisir(p, params, n=2):
    """Utilité espérée : les n candidats de plus forte probabilité a posteriori
    (égalités : votes puis q de TRM). Retourne [(clé, posterior)]."""
    cl = sorted(p.items(), key=lambda kv: (posterieur(kv[1], params), kv[1]["votes"], kv[1]["q"]), reverse=True)
    return [(k, posterieur(c, params)) for k, c in cl[:n]]


def vieille_regle(p, n=2):
    """Règle pré-enregistrée de fusionner.py (sans CompressARC) : DSL vérifié d'abord, puis accord, puis
    vote TRM ; TRM n'y contribue que par ses 2 premières tentatives."""
    cand = {}
    for k, c in p.items():
        meth = set()
        if c["dsl"]:
            meth.add("soar")
        if 1 <= c["rang"] <= 2:
            meth.add("trm")
        if not meth:
            continue
        prio = (0, 0, 0) if "soar" in meth else (2, 1, c["rang"])
        cand[k] = (prio, meth)
    ordre = sorted(cand.items(), key=lambda kv: (0 if kv[1][0][0] == 0 else (1 if len(kv[1][1]) >= 2 else 2),
                                                  kv[1][0][1:], p[kv[0]]["rang"] if p[kv[0]]["rang"] else 9))
    return [k for k, _ in ordre[:n]]


# ----------------------------------------------------------------------------- métriques
def brier(paires):
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in paires) / len(paires)


def noter_pli(taches, pools, dsl, params, verite):
    """Notation du pli de test (APRÈS gel). Retourne le détail et les agrégats."""
    det, paires, paires_top1 = {}, [], []
    for t in taches:
        gold = verite.sortie(t)[0]
        p = pools[t]
        juste = {k: c["grille"] == gold for k, c in p.items()}
        ch = choisir(p, params)
        trm2 = [k for k, c in sorted(p.items(), key=lambda kv: kv[1]["rang"]) if 1 <= c["rang"] <= 2]
        dsl_k = [k for k, c in p.items() if c["dsl"]]
        det[t] = {
            "trm_top1": bool(trm2 and juste[trm2[0]]), "trm": any(juste[k] for k in trm2),
            "dsl": any(juste[k] for k in dsl_k), "vieille_regle": any(juste[k] for k in vieille_regle(p)),
            "bayes": any(juste[k] for k, _ in ch), "bayes_top1": bool(ch and juste[ch[0][0]]),
            "plafond": any(juste.values()),
            "posterieurs_choisis": [round(q, 4) for _, q in ch],
        }
        for k, c in p.items():
            paires.append((posterieur(c, params), juste[k]))
        if ch:
            paires_top1.append((ch[0][1], juste[ch[0][0]]))
    agr = {m: sum(d[m] for d in det.values()) for m in
           ("trm_top1", "trm", "dsl", "vieille_regle", "bayes", "bayes_top1", "plafond")}
    return {"detail": det, "agregats": agr, "paires": paires, "paires_top1": paires_top1}


def calibration(paires, prior):
    return {"n": len(paires), "brier": brier(paires), "ece": ece(paires),
            "brier_prior_seul": brier([(prior, ok) for _, ok in paires])}


# ----------------------------------------------------------------------------- valeur du calcul
def groupe_dsl(d):
    """Groupe du DSL avant de décider de lancer TRM : nombre de candidats distincts (0, 1 ou 2)."""
    return len({cle(s[0]) for s in d["sorties"][: d["nb_candidats"]]})


def modele_voc(dsl, ids_hors_test, trm_ok_cal, taches_cal):
    """Gain espéré de TRM par groupe, depuis le modèle calibré :
      p_maintenant(g) = taux de réussite (Beta(1,1)) du DSL aux tâches hors test de groupe g (0 si g = 0) ;
      p_apres = taux (Beta(1,1)) de réussite pass@2 de TRM sur le pli de calibration ;
      gain(g) = max(0, p_apres - p_maintenant(g))."""
    stat = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    for t in ids_hors_test:
        g = groupe_dsl(dsl[t])
        stat[g][1] += 1
        stat[g][0] += any(dsl[t]["ok"])
    p_now = {g: (0.0 if g == 0 else (s + 1) / (n + 2)) for g, (s, n) in stat.items()}
    s = sum(trm_ok_cal[t] for t in taches_cal)
    p_apres = (s + 1) / (len(taches_cal) + 2)
    return {"p_maintenant": p_now, "p_apres": p_apres,
            "gain": {g: max(0.0, p_apres - p_now[g]) for g in p_now},
            "effectifs": {g: stat[g][1] for g in stat}}


def politique(taches, dsl, modele, cout_trm, lam):
    """Lance TRM ssi gain(g) > lam * cout_trm. Retourne {t: bool}."""
    return {t: modele["gain"][groupe_dsl(dsl[t])] > lam * cout_trm for t in taches}


def evaluer_politique(taches, lance, ok_dsl, ok_avec_trm, cpu_dsl, cpu_trm):
    """pass@2 et CPU total : DSL toujours exécuté ; TRM seulement si `lance`."""
    reussi = sum(ok_avec_trm[t] if lance[t] else ok_dsl[t] for t in taches)
    cpu = sum(cpu_dsl[t] + (cpu_trm[t] if lance[t] else 0.0) for t in taches)
    return reussi, cpu, sum(lance.values())


def choisir_lambda(taches_cal, dsl, modele, cout_trm, ok_dsl, ok_trm, cpu_dsl, cpu_trm, grille, tolerance=1):
    """Plus grand λ de la grille dont le pass@2 sur le pli de calibration reste >= (toujours-TRM) - tolerance
    tâche(s). Règle fixée avant de voir le pli de test."""
    ref = evaluer_politique(taches_cal, {t: True for t in taches_cal}, ok_dsl, ok_trm, cpu_dsl, cpu_trm)[0]
    meilleur = 0.0
    for lam in sorted(grille):
        r = evaluer_politique(taches_cal, politique(taches_cal, dsl, modele, cout_trm, lam), ok_dsl, ok_trm,
                              cpu_dsl, cpu_trm)[0]
        if r >= ref - tolerance:
            meilleur = lam
    return meilleur
