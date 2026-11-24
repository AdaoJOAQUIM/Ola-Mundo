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
    return [{"nom": c["nom"], "fichier": c["fichier"], "source": (SKILLS / c["fichier"]).read_text()}
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


if __name__ == "__main__":
    cmd, k = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    if cmd == "train":
        train(k, args)
    elif cmd == "juger":
        juger(k, args)
