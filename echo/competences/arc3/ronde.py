"""Pilote d'une ronde Skill-Pro.
  python ronde.py train  K [nom__rK ...]   : joue DEV-train (graines 0..2) avec bassin courant (+ candidats), écrit gradients/resume_K.json
  python ronde.py juger  K nom__rK ...     : juge les candidats un par un (séquentiel), met à jour bassin.json et lignee.json
  python ronde.py elaguer K                : élagage par posterior Beta (>= 10 activations)
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import jouer          # noqa: E402
import enseignant     # noqa: E402

SKILLS = ICI / "skills"
GRAD = ICI / "gradients"
GRAINES_TRAIN = (0, 1, 2)
SEUIL_ELAGAGE = 0.05
MIN_ACTIVATIONS = 10


def lire(nom, defaut):
    p = ICI / nom
    return json.loads(p.read_text()) if p.exists() else defaut


def ecrire(nom, obj):
    (ICI / nom).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def charger_bassin():
    b = lire("bassin.json", {"competences": []})
    return [{"nom": c["nom"], "fichier": c["fichier"], "source": source_de(c["fichier"])}
            for c in b["competences"]]


def source_de(fichier):
    return (SKILLS / (fichier if fichier.endswith(".py") else fichier + ".py")).read_text()


def nom_de(fichier):
    import re
    m = re.search(r'^NOM\s*=\s*"([^"]+)"', source_de(fichier), re.M)
    return m.group(1)


def train(k, candidats=()):
    bassin = charger_bassin() + [{"nom": nom_de(c), "fichier": c, "source": source_de(c)} for c in candidats]
    pr = jouer.prieurs_geles() or None
    taches = [(j, g, bassin, pr, g == 0) for j in jouer.DEV_TRAIN for g in GRAINES_TRAIN]
    res = jouer.jouer_lot(taches, 2)
    resumes = [enseignant.resumer(r, jouer.TRAJ / f"{r['prefixe']}_s{r['graine']}.npz" if r["graine"] == 0 else None)
               for r in res]
    GRAD.mkdir(exist_ok=True)
    ecrire(f"gradients/resume_{k}.json", {"ronde": k, "bassin": [b["nom"] for b in bassin], "resumes": resumes})
    for r in resumes:
        print(r["jeu"], "s%d" % r["graine"], "niv", r["niveaux"], r["niveaux_actions"], "comp/cerv",
              r["actions_competences"], r["actions_cerveau"], {n: (v["activations"], v["succes"]) for n, v in r["competences"].items()},
              r.get("traj", {}).get("images_distinctes"))
    return res


def juger(k, candidats):
    import juge
    bassin = charger_bassin()
    pr = jouer.prieurs_geles() or None
    lign = lire("lignee.json", {"evenements": []})
    cache = {}
    for c in candidats:
        cand = {"nom": nom_de(c), "fichier": c, "source": source_de(c)}
        courant = [b for b in bassin]
        nouveau = [b for b in courant if b["nom"] != cand["nom"]] + [cand]
        if any(b["nom"] == cand["nom"] for b in courant):     # révision : même place dans l'ordre
            nouveau = [cand if b["nom"] == cand["nom"] else b for b in courant]
        dossier = juge.juger([{"nom": b["nom"], "source": b["source"]} for b in courant],
                             [{"nom": b["nom"], "source": b["source"]} for b in nouveau], pr, "ajout", cache)
        ev = {"date": datetime.now().isoformat(timespec="seconds"), "ronde": k, "candidat": c,
              "sha_source": hashlib.sha256(cand["source"].encode()).hexdigest()[:16],
              "accepte": dossier["accepte"], "raison": dossier.get("raison"), "mesures": dossier.get("mesures"),
              "niveaux_par_jeu_graine": dossier.get("niveaux_par_jeu_graine"),
              "empreintes_juge": dossier.get("empreintes")}
        lign["evenements"].append(ev)
        print(c, "->", "ACCEPTE" if dossier["accepte"] else "REJETE", dossier.get("raison"),
              {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in (dossier.get("mesures") or {}).items()})
        if dossier["accepte"]:
            bassin = nouveau
            ecrire("bassin.json", {"competences": [{"nom": b["nom"], "fichier": b["fichier"]} for b in bassin]})
        ecrire("lignee.json", lign)
    return bassin


def juger_groupe(k, candidats):
    """Un seul jugement pour un groupe (<= 3 nouvelles/révisées) ajouté ensemble au bassin courant."""
    import juge
    bassin = charger_bassin()
    pr = jouer.prieurs_geles() or None
    lign = lire("lignee.json", {"evenements": []})
    nouveau = list(bassin)
    for c in candidats:
        cand = {"nom": nom_de(c), "fichier": c, "source": source_de(c)}
        if any(b["nom"] == cand["nom"] for b in nouveau):
            nouveau = [cand if b["nom"] == cand["nom"] else b for b in nouveau]
        else:
            nouveau.append(cand)
    dossier = juge.juger([{"nom": b["nom"], "source": b["source"]} for b in bassin],
                         [{"nom": b["nom"], "source": b["source"]} for b in nouveau], pr, "ajout", {})
    ev = {"date": datetime.now().isoformat(timespec="seconds"), "ronde": k, "candidat": "+".join(candidats),
          "groupe": True, "accepte": dossier["accepte"], "raison": dossier.get("raison"),
          "mesures": dossier.get("mesures"), "niveaux_par_jeu_graine": dossier.get("niveaux_par_jeu_graine"),
          "empreintes_juge": dossier.get("empreintes")}
    lign["evenements"].append(ev)
    print("GROUPE", candidats, "->", "ACCEPTE" if dossier["accepte"] else "REJETE", dossier.get("raison"),
          {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in (dossier.get("mesures") or {}).items()})
    if dossier["accepte"]:
        ecrire("bassin.json", {"competences": [{"nom": b["nom"], "fichier": b["fichier"]} for b in nouveau]})
    ecrire("lignee.json", lign)


if __name__ == "__main__":
    cmd, k = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    if cmd == "train":
        train(k, args)
    elif cmd == "groupe":
        juger_groupe(k, args)
    elif cmd == "juger":
        juger(k, args)


def posterieurs(k):
    """Postérieur Beta(1,1) par compétence : succès = niveau gagné pendant l'activation ou dans les
    FENETRE_SUCCES actions suivantes. Calculé sur DEV-train (graines 0..2) et DEV-gate (graines 0..4)."""
    bassin = charger_bassin()
    pr = jouer.prieurs_geles() or None
    taches = [(j, g, bassin, pr, False) for j in jouer.DEV_TRAIN for g in GRAINES_TRAIN] + \
             [(j, g, bassin, pr, False) for j in jouer.DEV_GATE for g in range(5)]
    res = jouer.jouer_lot(taches, 2)
    tot = {}
    for r in res:
        for nom, st in r["competences"].items():
            t = tot.setdefault(nom, {"activations": 0, "succes": 0})
            t["activations"] += st["activations"]
            t["succes"] += st["succes"]
    out = {}
    for nom, t in tot.items():
        a, b = 1 + t["succes"], 1 + t["activations"] - t["succes"]
        out[nom] = {**t, "alpha": a, "beta": b, "moyenne": round(a / (a + b), 4),
                    "elagable": t["activations"] >= MIN_ACTIVATIONS and a / (a + b) < SEUIL_ELAGAGE}
    ecrire(f"posterieurs_{k}.json", out)
    print(json.dumps(out, indent=1))
    return out


def elaguer(k):
    """Retire une compétence élagable si le juge (mode retrait) confirme qu'aucun niveau DEV-gate n'est perdu."""
    import juge
    post = lire(f"posterieurs_{k}.json", {})
    bassin = charger_bassin()
    pr = jouer.prieurs_geles() or None
    lign = lire("lignee.json", {"evenements": []})
    for nom, p in post.items():
        if not p["elagable"]:
            continue
        nouveau = [b for b in bassin if b["nom"] != nom]
        d = juge.juger([{"nom": b["nom"], "source": b["source"]} for b in bassin],
                       [{"nom": b["nom"], "source": b["source"]} for b in nouveau], pr, "retrait", {})
        lign["evenements"].append({"date": datetime.now().isoformat(timespec="seconds"), "ronde": k,
                                   "retrait": nom, "accepte": d["accepte"], "raison": d.get("raison"),
                                   "mesures": d.get("mesures"), "posterieur": p})
        print("retrait", nom, "->", d["accepte"], d.get("raison"))
        if d["accepte"]:
            bassin = nouveau
            ecrire("bassin.json", {"competences": [{"nom": b["nom"], "fichier": b["fichier"]} for b in bassin]})
    ecrire("lignee.json", lign)


if __name__ == "__main__" and sys.argv[1] in ("posterieurs", "elaguer"):
    {"posterieurs": posterieurs, "elaguer": elaguer}[sys.argv[1]](sys.argv[2])
