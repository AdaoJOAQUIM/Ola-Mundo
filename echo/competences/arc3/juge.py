"""JUGE ÉPINGLÉ (région de confiance, « PPO Gate »). Ne jamais modifier après son premier commit :
son SHA-256 (et celui de liste_blanche.py) est dans empreintes.json et vérifié à chaque jugement.

Règle (pré-enregistrée dans criteres.md) :
  jeux DEV-gate (tn36, sc25), graines du cerveau 0..4, 500 actions, paires (jeu, graine) : n = 10.
  ACCEPTE si (a) ET (b) :
   (a) borne inférieure unilatérale à 90 % du gain en niveaux  moyenne - 1,383*sd/sqrt(n)  > 0 ;
       OU niveaux totaux égaux (>= 1) et actions cumulées dans les niveaux résolus en baisse d'au moins 10 % ;
   (b) aucun niveau (jeu, indice) résolu par le bassin courant dans au moins une graine n'est perdu.
  Mode "retrait" (élagage) : accepte si (b) et si les niveaux totaux ne baissent pas.
Au plus 3 compétences nouvelles ou révisées par ronde ; bassin <= 20.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))

JEUX_GATE = ["tn36", "sc25"]
GRAINES = [0, 1, 2, 3, 4]
T90_DF9 = 1.383
SEUIL_EFFICACITE = 0.90
MAX_NOUVELLES = 3
MAX_BASSIN = 20
FICHIERS_EPINGLES = ["juge.py", "liste_blanche.py"]


def sha256(chemin):
    return hashlib.sha256(Path(chemin).read_bytes()).hexdigest()


def verifier_empreintes():
    ref = json.loads((ICI / "empreintes.json").read_text())
    return [f for f in FICHIERS_EPINGLES if ref.get(f) != sha256(ICI / f)]


def _cle_bassin(bassin):
    return hashlib.sha256(json.dumps([[b["nom"], b["source"]] for b in bassin]).encode()).hexdigest()[:16]


def _jouer(bassin, prieurs):
    import jouer
    taches = [(j, g, bassin, prieurs, False) for j in JEUX_GATE for g in GRAINES]
    return jouer.jouer_lot(taches, 2)


def _stats(res):
    """{(jeu, graine): (niveaux, [actions par niveau résolu])}"""
    out = {}
    for r in res:
        k = (r["prefixe"], r["graine"])
        out[k] = (int(r["niveaux"]), [int(a) for a in (r["niveaux_actions"] or [])[:int(r["niveaux"])]])
    return out


def nouvelles_competences(courant, candidat):
    src = {b["nom"]: b["source"] for b in courant}
    return [b["nom"] for b in candidat if src.get(b["nom"]) != b["source"]]


def comparer(sb, sc):
    """Comparaison appariée base (sb) / candidat (sc). Renvoie un dict de mesures (sans décision)."""
    cles = sorted(sb)
    d = [sc[k][0] - sb[k][0] for k in cles]
    n = len(d)
    moy = sum(d) / n
    sd = math.sqrt(sum((x - moy) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    lcb = moy - T90_DF9 * sd / math.sqrt(n)
    perdus = []
    for jeu in sorted({k[0] for k in cles}):
        nmax = max(max(sb[k][0], sc[k][0]) for k in cles if k[0] == jeu)
        for i in range(nmax):
            base = any(sb[k][0] > i for k in cles if k[0] == jeu)
            cand = any(sc[k][0] > i for k in cles if k[0] == jeu)
            if base and not cand:
                perdus.append((jeu, i))
    tot_b = sum(sb[k][0] for k in cles)
    tot_c = sum(sc[k][0] for k in cles)
    cout_b = sum(sum(sb[k][1]) for k in cles)
    cout_c = sum(sum(sc[k][1]) for k in cles)
    return {"n_paires": n, "gain_moyen": moy, "sd": sd, "borne_inf": lcb, "niveaux_base": tot_b,
            "niveaux_candidat": tot_c, "cout_actions_base": cout_b, "cout_actions_candidat": cout_c,
            "niveaux_perdus": perdus}


def decider(m, mode="ajout"):
    if mode == "retrait":
        ok = (not m["niveaux_perdus"]) and m["niveaux_candidat"] >= m["niveaux_base"]
        return ok, "retrait : niveaux perdus=%s" % (m["niveaux_perdus"],)
    a1 = m["borne_inf"] > 0
    a2 = (m["niveaux_candidat"] == m["niveaux_base"] and m["niveaux_base"] >= 1
          and m["cout_actions_candidat"] <= SEUIL_EFFICACITE * m["cout_actions_base"])
    b = not m["niveaux_perdus"]
    raison = "a1(borne>0)=%s a2(efficacite)=%s b(aucune perte)=%s" % (a1, a2, b)
    return bool((a1 or a2) and b), raison


def juger(courant, candidat, prieurs=None, mode="ajout", cache=None):
    """courant, candidat : listes de {"nom","source"}. Renvoie le dossier complet de la décision."""
    modifies = verifier_empreintes()
    if modifies:
        raise RuntimeError("juge ou liste blanche modifié : %s" % modifies)
    import liste_blanche
    rejets = []
    for b in candidat:
        ok, code, det = liste_blanche.verifier(b["source"])
        if not ok:
            rejets.append((b["nom"], code, det))
    if rejets:
        return {"accepte": False, "raison": "liste blanche : %s" % rejets}
    if len(candidat) > MAX_BASSIN:
        return {"accepte": False, "raison": "bassin > %d" % MAX_BASSIN}
    nouvelles = nouvelles_competences(courant, candidat)
    if mode == "ajout" and len(nouvelles) > MAX_NOUVELLES:
        return {"accepte": False, "raison": "plus de %d compétences nouvelles/révisées" % MAX_NOUVELLES}
    cache = cache if cache is not None else {}
    out = {}
    for nom, b in (("base", courant), ("candidat", candidat)):
        k = _cle_bassin(b)
        if k not in cache:
            cache[k] = _stats(_jouer(b, prieurs))
        out[nom] = cache[k]
    m = comparer(out["base"], out["candidat"])
    ok, raison = decider(m, mode)
    par_jeu = {}
    for nom in ("base", "candidat"):
        for (jeu, g), (lv, _) in out[nom].items():
            par_jeu.setdefault(jeu, {}).setdefault(nom, []).append(lv)
    return {"accepte": ok, "raison": raison, "mesures": m, "niveaux_par_jeu_graine": par_jeu,
            "nouvelles": nouvelles, "mode": mode,
            "empreintes": {f: sha256(ICI / f) for f in FICHIERS_EPINGLES}}
