#!/usr/bin/env python3
"""Baselines sur 200 mondes : cerveau (sans LLM) et agent aléatoire -> resultats_mondes.json / .md.

  python calibrer.py [--mondes 200] [--graine 1000] [--procs 2]
Les niveaux résolus sont comptés par mécanique (mondes qui contiennent la mécanique), par difficulté
(paramètre `difficulte` du monde) et par rang de niveau dans le monde.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
from evaluer import lancer, resume  # noqa: E402
from generateur import MECANIQUES, Monde  # noqa: E402

# Résultats des 5 jeux de DEV réels (cn04, ls20, sc25, tn36, ka59), lus dans echo/arc3/resultats_arc3.md :
DEV_REEL = {"cerveau": {"niveaux": 2, "total": 33, "jeux": 5},
            "aleatoire": {"niveaux": 0, "total": 33, "jeux": 5}}


def par_groupe(res, cle):
    g = defaultdict(lambda: [0, 0, 0])          # résolus, total, mondes
    for r in res:
        for c in cle(r):
            g[c][0] += r["niveaux"]
            g[c][1] += r["niveaux_total"]
            g[c][2] += 1
    return {str(k): {"resolus": v[0], "total": v[1], "taux": round(v[0] / v[1], 3) if v[1] else None,
                     "mondes": v[2]} for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}


def par_rang(res):
    g = defaultdict(lambda: [0, 0])
    for r in res:
        for i in range(r["niveaux_total"]):
            g[i + 1][1] += 1
            g[i + 1][0] += 1 if i < r["niveaux"] else 0
    return {str(k): {"resolus": v[0], "total": v[1], "taux": round(v[0] / v[1], 3)} for k, v in sorted(g.items())}


def analyser(res):
    return {
        "global": resume(res),
        "par_mecanique": par_groupe(res, lambda r: r["mecaniques"]),
        "par_difficulte": par_groupe(res, lambda r: [r["difficulte"]]),
        "par_but": par_groupe(res, lambda r: [r["but"]]),
        "par_rang_de_niveau": par_rang(res),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mondes", type=int, default=200)
    ap.add_argument("--graine", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--sortie", default=str(ICI / "resultats_mondes"))
    ap.add_argument("--md-seul", action="store_true", help="réécrit le .md depuis le .json existant")
    a = ap.parse_args()
    if a.md_seul:
        ecrire_md(json.loads(Path(a.sortie + ".json").read_text()), Path(a.sortie + ".md"))
        return
    seeds = [a.graine + i for i in range(a.mondes)]
    ds = [1 + i % 5 for i in range(a.mondes)]
    # statistiques du jeu de mondes lui-même (optimum de l'oracle)
    mondes = [Monde(s, d) for s, d in zip(seeds, ds)]
    opts = [o for m in mondes for o in m.optimum()]
    stats = {
        "mondes": len(mondes),
        "niveaux": len(opts),
        "niveaux_par_monde_moy": round(len(opts) / len(mondes), 2),
        "optimum_moy": round(sum(opts) / len(opts), 1),
        "optimum_min": min(opts), "optimum_max": max(opts),
        "mondes_avec_mecanique": {m: sum(1 for w in mondes if m in w.meca) for m in MECANIQUES},
        "mecaniques_par_monde_moy": round(sum(len(w.meca) for w in mondes) / len(mondes), 2),
    }
    out = {"jeu_de_mondes": stats, "dev_reel": DEV_REEL, "agents": {}}
    for nom, spec in (("cerveau", "cerveau"), ("aleatoire", "aleatoire")):
        res = lancer(spec, seeds, ds, procs=a.procs, graine_agent=0)
        out["agents"][nom] = analyser(res)
        out["agents"][nom]["mondes"] = [
            {k: r[k] for k in ("jeu", "but", "difficulte", "etat", "niveaux", "niveaux_total", "actions",
                               "niveaux_actions", "niveaux_reference")} for r in res]
        print(nom, out["agents"][nom]["global"], flush=True)
    Path(a.sortie + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    ecrire_md(out, Path(a.sortie + ".md"))


def pct(x):
    return "—" if x is None else f"{100 * x:.0f} %"


def ecrire_md(out, chemin):
    js, ag = out["jeu_de_mondes"], out["agents"]
    cb, al = ag["cerveau"], ag["aleatoire"]
    L = []
    L.append("# Mondes infinis : résultats des baselines\n")
    L.append("Généré par `calibrer.py`. Mondes procéduraux de type ARC-AGI-3 ; chaque niveau est prouvé soluble par un "
             "oracle A* sur l'état interne, qui donne aussi l'optimum d'actions. Protocole du harnais : courtier + garde, "
             "500 actions par monde, RESET automatique et compté après un game-over.\n")
    L.append("## Le jeu de mondes\n")
    L.append(f"- {js['mondes']} mondes, {js['niveaux']} niveaux ({js['niveaux_par_monde_moy']} par monde, de 3 à 8).")
    L.append(f"- Optimum de l'oracle par niveau : moyenne {js['optimum_moy']} actions (min {js['optimum_min']}, max {js['optimum_max']}).")
    L.append(f"- {js['mecaniques_par_monde_moy']} mécaniques actives par monde en moyenne.\n")
    L.append("## Résultat global\n")
    L.append("| agent | niveaux résolus | taux | mondes gagnés | efficacité vs optimum (niveaux résolus) |")
    L.append("|---|---|---|---|---|")
    for nom, a in (("Cerveau bayésien (sans LLM)", cb), ("Aléatoire", al)):
        g = a["global"]
        eff = "—" if g["efficacite_vs_optimum"] is None else f"{100 * g['efficacite_vs_optimum']:.0f} %"
        L.append(f"| {nom} | {g['niveaux_resolus']}/{g['niveaux_total']} | {pct(g['taux'])} | {g['mondes_gagnes']}/{g['mondes']} | {eff} |")
    d = out["dev_reel"]
    L.append(f"| *Cerveau sur les 5 jeux DEV réels* | {d['cerveau']['niveaux']}/{d['cerveau']['total']} | "
             f"{pct(d['cerveau']['niveaux'] / d['cerveau']['total'])} | 0/5 | — |")
    L.append(f"| *Aléatoire sur les 5 jeux DEV réels* | {d['aleatoire']['niveaux']}/{d['aleatoire']['total']} | "
             f"{pct(d['aleatoire']['niveaux'] / d['aleatoire']['total'])} | 0/5 | — |\n")
    L.append("L'efficacité est le rapport optimum/actions réellement dépensées, sur les niveaux résolus (RESET et "
             "game-over inclus). 100 % = jouer comme l'oracle.\n")

    def tableau(titre, cle):
        L.append(f"## {titre}\n")
        L.append("| groupe | mondes | cerveau | aléatoire |")
        L.append("|---|---|---|---|")
        for k, v in cb[cle].items():
            w = al[cle][k]
            L.append(f"| {k} | {v.get('mondes', '—')} | {v['resolus']}/{v['total']} ({pct(v['taux'])}) | {w['resolus']}/{w['total']} ({pct(w['taux'])}) |")
        L.append("")

    tableau("Par difficulté (paramètre du monde : 1 facile, 5 le plus long)", "par_difficulte")
    tableau("Par rang de niveau dans le monde (1 = premier)", "par_rang_de_niveau")
    tableau("Par type de but", "par_but")
    tableau("Par mécanique (niveaux des mondes qui contiennent la mécanique ; un monde compte pour chacune de ses mécaniques)", "par_mecanique")
    L.append("## Lecture : ces mondes sont-ils aussi difficiles que les jeux DEV réels ?\n")
    L.extend(lecture(out))
    L.append("")
    L.append(USAGE)
    chemin.write_text("\n".join(L), encoding="utf-8")


def lecture(out):
    ag = out["agents"]
    cb, al = ag["cerveau"], ag["aleatoire"]
    d = out["dev_reel"]
    t_dev = d["cerveau"]["niveaux"] / d["cerveau"]["total"]
    gd = cb["par_difficulte"]
    ad = al["par_difficulte"]
    L = []
    L.append(f"- Sur les 5 jeux DEV réels, le cerveau résout {d['cerveau']['niveaux']}/{d['cerveau']['total']} niveaux "
             f"({pct(t_dev)}) et l'aléatoire {d['aleatoire']['niveaux']}/{d['aleatoire']['total']}. Sur l'ensemble des mondes "
             f"générés (difficultés 1 à 5 en proportions égales), le cerveau résout {pct(cb['global']['taux'])} et l'aléatoire "
             f"{pct(al['global']['taux'])}.")
    L.append("- Par palier : " + " ; ".join(
        f"difficulté {k} : cerveau {pct(v['taux'])}, aléatoire {pct(ad[k]['taux'])}" for k, v in gd.items()) + ".")
    proche = [k for k, v in gd.items() if v["taux"] is not None and abs(v["taux"] - t_dev) <= 0.04]
    L.append("- Les paliers dont le taux du cerveau est à moins de 4 points du taux DEV réel : "
             + (", ".join(proche) if proche else "aucun") + ". Les paliers plus faciles servent à mesurer un progrès "
             "que les vrais jeux ne laissent pas voir (le cerveau y est proche de zéro) ; les plus durs rapprochent "
             "l'évaluation de la difficulté réelle.")
    rg = cb["par_rang_de_niveau"]
    L.append("- Le premier niveau d'un monde est nettement plus facile que les suivants (cerveau : "
             + ", ".join(f"niveau {k} : {pct(v['taux'])}" for k, v in list(rg.items())[:4]) + "). "
             "Les niveaux suivants réutilisent les mêmes règles avec plus d'éléments.")
    pm = {k: v for k, v in cb["par_mecanique"].items() if v["total"] >= 40}
    ordre = sorted(pm.items(), key=lambda kv: kv[1]["taux"])
    L.append("- Mécaniques les plus difficiles pour le cerveau : " + ", ".join(f"{k} ({pct(v['taux'])})" for k, v in ordre[:4])
             + ". Les plus faciles : " + ", ".join(f"{k} ({pct(v['taux'])})" for k, v in ordre[-4:]) + ".")
    g = cb["global"]
    L.append(f"- Efficacité : sur les niveaux qu'il résout, le cerveau dépense en moyenne {1 / g['efficacite_vs_optimum']:.0f} fois "
             f"l'optimum de l'oracle (aléatoire : {1 / al['global']['efficacite_vs_optimum']:.0f} fois), loin d'un joueur humain.")
    L.append("- Limite : 5 jeux DEV et 33 niveaux seulement côté réel ; l'écart entre 2/33 et un taux mesuré sur plus de "
             "1000 niveaux générés n'a pas de barre d'erreur serrée côté réel. Les mondes sont plus petits et plus "
             "réguliers que les vrais jeux (grilles de 6 à 10 cases, règles toutes déterministes à l'état complet) : un "
             "agent qui progresse ici peut ne pas progresser sur les vrais jeux. Utiliser ces mondes pour développer, "
             "pas pour prédire le score final.")
    return L


USAGE = """## Contenu des mondes

18 mécaniques composables (3 à 8 niveaux par monde, mêmes règles dans tout le monde) : déplacement, murs, clés et portes,
blocs poussables, collectibles avec compteur (sortie verrouillée), dangers mortels, téléporteurs, interrupteurs qui
basculent des murs, couleurs (cases de peinture, sortie de la bonne couleur), lumières (lights-out), sélection puis
déplacement (clic sur une pièce, puis flèches), motif à recopier, gravité (4 directions possibles, sur les blocs),
énergie limitée (barre en bas), ennemis à patrouille déterministe (+ action d'attente), cibles (blocs sur cibles),
annuler (ACTION7). Règles cachées, tirées de la graine : but, mécaniques actives, correspondance ACTION1..4 vers
directions, couleurs et formes de chaque rôle, sens de la gravité. Aucune consigne n'est affichée ; seuls indices : la
sortie est creuse tant qu'un verrou (gemmes, panneau) n'est pas levé, la barre d'énergie, les pastilles de niveaux.
Les mécaniques sont corrélées à la difficulté et entre elles : le tableau par mécanique n'est pas une expérience contrôlée.

Garanties : chaque niveau est résolu par un A* sur l'état interne (optimum = proxy d'efficacité humaine), la solution
optimale est rejouée par l'interface publique dans les tests, tout est déterministe selon (graine, difficulté).

## Utilisation

- `MondeEnv(seed, difficulte)` : même surface que l'env du SDK vu par le harnais (`reset()`, `step(GameAction, data)`,
  `action1..7`), observations 64x64 en 16 couleurs, états NOT_FINISHED / WIN / GAME_OVER.
- `CourtierMonde(seed, difficulte)` : même protocole que `arc3_local_broker.Broker` (observe/act, lots de 20, 500 actions,
  RESET automatique compté après game-over). `jouer(seed, difficulte, fabrique)` reproduit `harnais.jouer`.
- `python evaluer.py --agent cerveau --mondes 20 --graine 0 [--difficulte 1..5|mix]` : niveaux résolus et actions contre
  l'optimum. L'agent est un module (ou `module:Classe`) avec une classe `Classe(graine).choisir(obs)`.
- `python calibrer.py` : régénère ce rapport (200 mondes, cerveau et aléatoire).
- Tests : `python -m pytest tests -q`.
"""

if __name__ == "__main__":
    main()
