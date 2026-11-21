"""Fusion des gagnants : solveur auto-amélioré (idée SOAR), TRM officiel, CompressARC officiel.

Pour chaque énigme, on classe les réponses candidates :
  1. réponses VÉRIFIÉES : produites par un programme qui reproduit tous les exemples ;
  2. réponses proposées par au moins deux méthodes (accord) ;
  3. réponse de TRM (vote), puis de CompressARC (score) ;
et on garde les 2 premières réponses distinctes (règle d'ARC : 2 tentatives).
Les sorties attendues ne servent qu'à la notation finale, jamais au classement.
"""
import json
import os

ICI = os.path.dirname(os.path.abspath(__file__))
ARC = os.path.dirname(ICI)
METHODES = [("soar_lite", "verifie"), ("trm", "vote"), ("compressarc", "score")]


def cle(tentative):
    return json.dumps(tentative, separators=(",", ":"))


def main():
    taches = json.load(open(os.path.join(ARC, "sous_ensemble_fusion.json")))["taches"]
    sources = {}
    for nom, _ in METHODES:
        chemin = os.path.join(ICI, f"{nom}.json")
        if os.path.exists(chemin):
            sources[nom] = json.load(open(chemin))["taches"]
    lignes, resolus = {}, {n: 0 for n in sources}
    resolu_fusion = plafond = 0
    for t in taches:
        attendu = [x["output"] for x in json.load(open(os.path.join(ARC, "data", "evaluation", f"{t}.json")))["test"]]
        juste = lambda tent: tent == attendu  # noqa: E731
        candidats = {}  # clé -> (tentative, priorité, méthodes)
        for rang_m, (nom, _) in enumerate(METHODES):
            for rang, tent in enumerate(sources.get(nom, {}).get(t, {}).get("tentatives", [])):
                k = cle(tent)
                prio = (0 if nom == "soar_lite" else 2, rang_m, rang)
                if k in candidats:
                    ancien = candidats[k]
                    candidats[k] = (tent, min(ancien[1], prio), ancien[2] | {nom})
                else:
                    candidats[k] = (tent, prio, {nom})
        classes = sorted(candidats.values(),
                         key=lambda c: (c[1][0] if c[1][0] == 0 else (1 if len(c[2]) >= 2 else 2), c[1][1:]))
        choix = [c[0] for c in classes[:2]]
        ok = any(juste(c) for c in choix)
        resolu_fusion += ok
        plafond += any(juste(c[0]) for c in candidats.values())
        for nom in sources:
            resolus[nom] += any(juste(x) for x in sources[nom].get(t, {}).get("tentatives", [])[:2])
        lignes[t] = {"fusion": ok, **{n: any(juste(x) for x in sources[n].get(t, {}).get("tentatives", [])[:2])
                                      for n in sources}}
    res = {"taches": len(taches), "methodes_presentes": list(sources),
           "resolus_pass2": {**resolus, "fusion": resolu_fusion},
           "plafond_au_moins_une_methode": plafond, "detail": lignes}
    json.dump(res, open(os.path.join(ICI, "resultats_fusion.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "detail"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
