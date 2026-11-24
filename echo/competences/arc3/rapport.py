"""Écrit resultats_competences_arc3.json et .md à partir de res_final.json, lignee.json, posterieurs_*.json."""
import json
from pathlib import Path

ICI = Path(__file__).resolve().parent
ARC3 = ICI.parent.parent / "arc3"
fin = json.loads((ICI / "res_final.json").read_text())
base = {r["prefixe"]: r for r in json.loads((ARC3 / "res_held.json").read_text())}
lign = json.loads((ICI / "lignee.json").read_text())["evenements"]
post = json.loads((ICI / "posterieurs_r4.json").read_text())
bassin = fin["bassin"]


def actions_niveaux(r):
    return [int(a) for a in (r["niveaux_actions"] or [])[:int(r["niveaux"])]]


jeux = []
for r in fin["jeux"]:
    p = r["prefixe"]
    b = base[p]
    jeux.append({"jeu": p, "niveaux_total": r["niveaux_total"], "niveaux_nous": r["niveaux"],
                 "niveaux_baseline": b["niveaux"], "actions_par_niveau_nous": actions_niveaux(r),
                 "actions_par_niveau_baseline": actions_niveaux(b),
                 "competences": {k: {"activations": v["activations"], "succes": v["succes"]} for k, v in r["competences"].items()},
                 "attribution_gains": r["resume"]["attribution_gain"],
                 "actions_competences": r["actions_competences"], "actions_cerveau": r["actions_cerveau"]})
nous = sum(j["niveaux_nous"] for j in jeux)
bl = sum(j["niveaux_baseline"] for j in jeux)
tot = sum(j["niveaux_total"] for j in jeux)
a_n = sum(sum(j["actions_par_niveau_nous"]) for j in jeux)
a_b = sum(sum(j["actions_par_niveau_baseline"]) for j in jeux)
communs_n = communs_b = nc = 0
for j in jeux:
    for i in range(min(j["niveaux_nous"], j["niveaux_baseline"])):
        communs_n += j["actions_par_niveau_nous"][i]
        communs_b += j["actions_par_niveau_baseline"][i]
        nc += 1
gagnes = [(j["jeu"], i + 1) for j in jeux for i in range(j["niveaux_baseline"], j["niveaux_nous"])]
perdus = [(j["jeu"], i + 1) for j in jeux for i in range(j["niveaux_nous"], j["niveaux_baseline"])]
transfert = {}
for nom in bassin:
    act = [j["jeu"] for j in jeux if j["competences"][nom]["activations"] > 0]
    gains = sum(1 for j in jeux for a in j["attribution_gains"] if a == nom)
    transfert[nom] = {"jeux_ou_active": len(act), "jeux": act,
                      "activations_total": sum(j["competences"][nom]["activations"] for j in jeux),
                      "niveaux_gagnes_pendant_ou_juste_apres": gains}
res = {
    "protocole": "20 jeux held-out, 500 actions, graine d'environnement 0, graine du cerveau 0, prieurs gelés (identique à la baseline des 7/150)",
    "bassin_final": bassin,
    "B1": {"seuil": 14, "niveaux_nous": nous, "niveaux_baseline": bl, "niveaux_total": tot, "atteint": nous >= 14},
    "B2": {"actions_par_niveau_nous": round(a_n / nous, 1), "actions_par_niveau_baseline": round(a_b / bl, 1),
           "niveaux_resolus_par_les_deux": nc, "actions_par_niveau_commun_nous": round(communs_n / nc, 1),
           "actions_par_niveau_commun_baseline": round(communs_b / nc, 1), "niveaux_gagnes": gagnes, "niveaux_perdus": perdus},
    "B3": {"analyse_statique_et_test_dynamique": "test_competences.py::test_aucun_reseau_ni_llm_a_l_execution (OK)",
           "actions_competences_total": sum(j["actions_competences"] for j in jeux),
           "actions_cerveau_total": sum(j["actions_cerveau"] for j in jeux)},
    "B4": {"competences_acceptees": len(bassin),
           "activees_au_moins_une_fois_en_held_out": sum(1 for v in transfert.values() if v["jeux_ou_active"] > 0),
           "detail": transfert},
    "jeux": jeux,
    "posterieurs_dev": post,
    "lignee": [{"ronde": e["ronde"], "candidat": e.get("candidat") or ("retrait " + e.get("retrait", "")),
                "accepte": e["accepte"], "raison": e.get("raison"),
                "borne_inf": (e.get("mesures") or {}).get("borne_inf"),
                "niveaux_base": (e.get("mesures") or {}).get("niveaux_base"),
                "niveaux_candidat": (e.get("mesures") or {}).get("niveaux_candidat")} for e in lign],
}
(ICI / "resultats_competences_arc3.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))

L = []
L.append("# Compétences exécutables pour ARC-AGI-3 (Skill-Pro adapté), élève sans modèle de langage\n")
L.append("Un enseignant (Claude) lit les trajectoires de jeux de dev et écrit de petites compétences Python déterministes ; "
         "l'élève (cerveau bayésien de `echo/arc3`, sans LLM) les exécute à pleine vitesse. Un juge épinglé (SHA-256) décide de leur "
         "acceptation sur des jeux de dev réservés. Test final : une seule exécution sur les 20 jeux held-out, protocole identique à celui "
         "de la baseline des 7/150 sur 150.\n")
L.append("## Critères pré-enregistrés (`criteres.md`)\n")
L.append(f"- **B1 (>= 14/150) : NON ATTEINT.** {nous}/{tot} niveaux held-out contre {bl}/{tot} pour la baseline (gain net de {nous - bl}).")
L.append(f"- **B2 (actions par niveau terminé) : PLUS MAUVAIS.** {res['B2']['actions_par_niveau_nous']} actions par niveau contre {res['B2']['actions_par_niveau_baseline']} pour la baseline ; "
         f"sur les {nc} niveaux résolus par les deux : {res['B2']['actions_par_niveau_commun_nous']} contre {res['B2']['actions_par_niveau_commun_baseline']}. "
         "Les compétences allongent le premier niveau des jeux que la baseline résolvait vite (vc33 : 16 -> 142 actions, r11l : 10 -> 169, lp85 : 22 -> 68).")
L.append("- **B3 (aucun réseau ni LLM à l'exécution) : CONFIRMÉ.** Analyse statique des imports (`eleve.py`, `outils.py`, `liste_blanche.py`, `cerveau.py`) "
         "et test dynamique : un jeu entier joué avec `socket.socket` remplacé par une fonction qui lève une erreur (`test_competences.py`).")
L.append(f"- **B4 (transfert) :** {res['B4']['activees_au_moins_une_fois_en_held_out']}/{len(bassin)} compétences acceptées s'activent sur des jeux held-out ; "
         + "; ".join(f"{n} : {v['jeux_ou_active']}/20 jeux, {v['niveaux_gagnes_pendant_ou_juste_apres']} niveaux gagnés pendant ou juste après" for n, v in transfert.items()) + ".\n")
L.append("## Compétences acceptées\n")
L.append("1. `sonde_clics` : au début d'un niveau, clique une fois chaque objet de l'image initiale (couleurs rares d'abord). Acceptée par efficacité (actions dans les niveaux résolus de DEV-gate : 206 -> 50).")
L.append("2. `couloir` : répète une flèche tant qu'elle déplace quelque chose vers un état nouveau (8 actions au plus). Acceptée en groupe à la ronde 3 (borne inférieure 0,016).")
L.append("3. `paires_clics` : après RESET, essaie les paires ordonnées de clics parmi les 6 objets qui ont répondu. Acceptée en groupe à la ronde 3 ; aucun succès attribué sur DEV (0/18 activations), 1 en held-out (lp85).\n")
L.append("Rejetées : tour_objets (acceptée en groupe puis élaguée : 0 succès sur 140 activations, aucun niveau perdu au retrait), repetition_clic, clic_puis_fleches, couverture, visite_objets (jamais activées sur DEV-gate), clic_nouveaux (fait perdre sc25 niveau 1), révisions paires_clics r5 et sonde_clics r5.\n")
L.append("## Niveaux held-out par jeu (nous / baseline)\n")
L.append("| jeu | niveaux nous | niveaux baseline | actions par niveau (nous) | actions par niveau (baseline) | activations (sonde / couloir / paires) |")
L.append("|---|---|---|---|---|---|")
for j in jeux:
    c = j["competences"]
    L.append(f"| {j['jeu']} | {j['niveaux_nous']}/{j['niveaux_total']} | {j['niveaux_baseline']} | {j['actions_par_niveau_nous']} | {j['actions_par_niveau_baseline']} | "
             f"{c['sonde_clics']['activations']} / {c['couloir']['activations']} / {c['paires_clics']['activations']} |")
L.append(f"\nTotal : {nous}/{tot} contre {bl}/{tot}. Niveaux gagnés par rapport à la baseline : {gagnes}. Niveaux perdus : {perdus}.\n")
L.append("## Lignée (jugements sur DEV-gate : tn36, sc25 ; graines 0 à 4 ; 10 paires)\n")
L.append("| ronde | candidat | décision | niveaux base -> candidat | borne inférieure |")
L.append("|---|---|---|---|---|")
for e in res["lignee"]:
    bi = e["borne_inf"]
    L.append(f"| {e['ronde']} | {e['candidat']} | {'accepté' if e['accepte'] else 'rejeté'} | {e['niveaux_base']} -> {e['niveaux_candidat']} | {'' if bi is None else round(bi, 3)} |")
L.append("\nPostérieurs Beta (DEV-train graines 0 à 2 et DEV-gate graines 0 à 4, fin de ronde 4) : "
         + "; ".join(f"{n} {v['succes']}/{v['activations']} (moyenne {v['moyenne']})" for n, v in post.items()) + ".\n")
L.append("## Ce que ces résultats disent, et ne disent pas\n")
L.append("- Le gain brut (+2 niveaux) est dans le bruit : un seul passage, une graine, 20 jeux. Il vient de g50t (couloir, +1) et lp85 (+2, après sonde_clics), contre une perte (su15). Aucun test statistique ne le distingue de zéro.")
L.append("- Le juge n'a que 2 jeux (tn36, sc25), tous deux à clics : il ne peut pas valider des compétences de navigation (couverture, visite_objets n'ont jamais été activées par lui). Les acceptations reposent sur 10 paires et un 3e jugement sur les mêmes jeux (comparaisons multiples) : la borne de 0,016 de la ronde 3 est fragile.")
L.append("- L'efficacité du gate (sonde_clics : 206 -> 50 actions) ne se retrouve pas en held-out : l'efficacité mesurée sur deux jeux ne transfère pas, elle s'inverse sur les jeux que la baseline résolvait vite. Je n'ai pas fait l'ablation sur held-out (ce serait un second passage).")
L.append("- Les trajectoires DEV-train (cn04, ka59) demandent un raisonnement sur les objets (pièce à emboîter, bloc à pousser) que des compétences génériques ne capturent pas ; aucune compétence ne gagne de niveau sur ces deux jeux.")
L.append("- Sur DEV-train, le bassin final gagne le niveau 1 de ls20 en graines 0 et 2 (305 et 427 actions) contre une seule graine pour la baseline (198 actions).\n")
L.append("## Écarts par rapport au plan\n")
L.append("- La liste blanche est un nouveau module (`liste_blanche.py`) qui reprend le schéma de `echo/evolution/garde_fous.py` (Rejet, vérification AST, bornes) sans l'importer ni le modifier : celui-ci ne convient qu'aux expressions numériques `estimer`.")
L.append("- Le substrat de perception `h` (`outils.py` : objets, différences, mouvements, BFS) et le cerveau de repli sont écrits par moi ; seules les politiques sont des compétences. Sans compétence active, l'élève est identique à la baseline (vérifié par test).")
L.append("- Découpe : DEV-train = ls20, cn04, ka59 ; DEV-gate = tn36, sc25 (2 jeux seulement : 5 jeux de dev au total). Les prieurs gelés de la baseline ont été calculés sur les 5 jeux de dev, donc aussi sur le gate.")
L.append("- Jugement de groupe aux rondes 3 et 4 (plusieurs compétences révisées ensemble, 3 au plus), séquentiel aux rondes 1, 2 et 5. 5 rondes sur 6 utilisées : aucune idée nouvelle justifiée par les trajectoires.")
L.append("- `couloir` est gardée malgré une moyenne a posteriori de 0,018 (< 0,05) parce que le juge (mode retrait) constate que son retrait fait perdre sc25 niveau 1.")
L.append("- Succès d'une activation = niveau gagné pendant l'exécution ou dans les 25 actions suivantes ; l'attribution est donc approximative.")
L.append("- Honnêteté : aucun fichier source de jeu n'a été ouvert ni inspecté ; seules les images, actions disponibles, `levels_completed` et `state` ont été utilisés. Les images des jeux held-out et de DEV-gate n'ont pas été examinées avant le test final.\n")
L.append("## Fichiers\n")
L.append("`criteres.md`, `liste_blanche.py`, `outils.py`, `eleve.py`, `jouer.py`, `juge.py` + `empreintes.json` (épinglés), `ronde.py`, `enseignant.py`, `final.py`, `skills/` (toutes les versions), `bassin.json`, `lignee.json`, `gradients/`, `test_competences.py`, `res_final.json`.")
(ICI / "resultats_competences_arc3.md").write_text("\n".join(L) + "\n")
print(json.dumps({k: res[k] for k in ("B1", "B2", "B3")}, indent=1)); print(json.dumps(res["B4"]["detail"], indent=1)); print(gagnes, perdus)
