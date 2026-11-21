"""Agrège les résultats : tableaux, comparaison à Prime Agent, estimation d'économie de calcul."""
import json
import statistics
from pathlib import Path

ICI = Path(__file__).resolve().parent
PRIME = ICI.parent / "arc" / "externes" / "prime-agent" / "results" / "results.json"


def _c(nom, defaut=None):
    p = ICI / nom
    return json.loads(p.read_text()) if p.exists() else defaut


def moyenne(x):
    return sum(x) / len(x) if x else 0.0


def ecrire_rapport():
    from lancer import DEV, HELD, JEUX
    held = {r["prefixe"]: r for r in _c("res_held.json", [])}
    dev = {r["prefixe"]: r for r in _c("res_dev_gele.json", [])}
    nous = {**held, **dev}
    alea = _c("res_aleatoire.json", [])
    prime = json.loads(PRIME.read_text())
    pr = {g["game_id"].split("-")[0]: g for g in prime["runs"]["run-1"]["games"]}
    journal = _c("journal_versions.json", {})

    def agr(jeux, src):
        rs = [src[j] for j in jeux if j in src]
        return {"jeux": len(rs), "niveaux": sum(r["niveaux"] for r in rs),
                "niveaux_total": sum(r["niveaux_total"] or 0 for r in rs),
                "jeux_gagnes": sum(1 for r in rs if r["etat"] == "WIN"),
                "actions": sum(r["actions"] for r in rs),
                "cpu_s": round(sum(r["cpu_s"] for r in rs), 1),
                "score_sdk_moyen": round(moyenne([r["score_sdk"] or 0 for r in rs]), 3)}

    def agr_alea(jeux):
        par_graine = []
        for g in (0, 1, 2):
            src = {r["prefixe"]: r for r in alea if r["graine"] == g}
            par_graine.append(agr(jeux, src))
        res = {"par_graine": par_graine}
        for k in ("niveaux", "jeux_gagnes", "actions", "cpu_s", "score_sdk_moyen"):
            res[k + "_moyenne"] = round(moyenne([p[k] for p in par_graine]), 2)
        res["niveaux_total"] = par_graine[0]["niveaux_total"] if par_graine else 0
        return res

    def agr_prime(jeux):
        gs = [pr[j] for j in jeux]
        return {"niveaux": sum(g["levels_completed"] for g in gs), "niveaux_total": sum(g["total_levels"] for g in gs),
                "jeux_gagnes": sum(1 for g in gs if g["terminal_state"] == "WIN"),
                "actions": sum(g["actions_taken"] for g in gs)}

    synthese = {
        "decoupe": {"graine": 2026, "dev": DEV, "held_out": HELD},
        "held_out": {"nous": agr(HELD, nous), "aleatoire": agr_alea(HELD), "prime_run1": agr_prime(HELD)},
        "dev": {"nous": agr(DEV, nous), "aleatoire": agr_alea(DEV), "prime_run1": agr_prime(DEV)},
        "tous": {"nous": agr(JEUX, nous), "aleatoire": agr_alea(JEUX), "prime_run1": agr_prime(JEUX)},
    }

    # ---- estimation de l'économie de calcul -------------------------------------------------
    cout_total = {r: prime["runs"][r]["estimated_api_cost_usd"] for r in prime["runs"]}
    econ = {}
    for run in prime["runs"]:
        gs = {g["game_id"].split("-")[0]: g for g in prime["runs"][run]["games"]}
        appels = sum(g["model_calls"] for g in gs.values())
        opt = prud = 0.0
        for j in JEUX:
            if j not in nous:
                continue
            g = gs[j]
            part = cout_total[run] * g["model_calls"] / appels      # coût du jeu ~ ses appels modèle
            L = g["total_levels"]
            k = min(nous[j]["niveaux"], g["levels_completed"])      # niveaux que le LLM résolvait aussi
            ref = nous[j]["niveaux_reference"] or []
            opt += part * k / L                                       # tous niveaux de même coût
            poids = ref[:L] if len(ref) >= L else [1] * L
            prud += part * sum(poids[:k]) / sum(poids)                # coût du niveau ~ actions de référence humaines
        econ[run] = {"cout_total_usd": cout_total[run], "economie_usd_niveaux_egaux": round(opt, 1),
                     "economie_pct_niveaux_egaux": round(100 * opt / cout_total[run], 2),
                     "economie_usd_ponderee_reference": round(prud, 1),
                     "economie_pct_ponderee_reference": round(100 * prud / cout_total[run], 2)}
    synthese["economie_calcul"] = econ

    par_jeu = []
    for j in JEUX:
        ar = [r for r in alea if r["prefixe"] == j]
        n = nous.get(j, {})
        par_jeu.append({
            "jeu": j, "split": "dev" if j in DEV else "held-out",
            "niveaux_total": pr[j]["total_levels"],
            "nous_niveaux": n.get("niveaux"), "nous_etat": n.get("etat"), "nous_actions": n.get("actions"),
            "nous_score_sdk": n.get("score_sdk"), "nous_cpu_s": n.get("cpu_s"), "nous_niveaux_actions": n.get("niveaux_actions"),
            "aleatoire_niveaux_par_graine": [r["niveaux"] for r in sorted(ar, key=lambda r: r["graine"])],
            "prime_niveaux": pr[j]["levels_completed"], "prime_etat": pr[j]["terminal_state"],
            "prime_actions": pr[j]["actions_taken"], "prime_appels_modele": pr[j]["model_calls"]})
    synthese["par_jeu"] = par_jeu
    synthese["journal_versions"] = journal
    (ICI / "resultats_arc3.json").write_text(json.dumps(synthese, indent=1, ensure_ascii=False))

    # ---- Markdown -----------------------------------------------------------------------------
    def lig(nom, s):
        return f"| {nom} | {s['niveaux']}/{s['niveaux_total']} | {s['jeux_gagnes']}/{s.get('jeux', len(JEUX))} | {s['actions']} |"
    L = ["# ARC-AGI-3 sans LLM : cerveau bayésien sur CPU", "",
         "Généré par `rapport.py` à partir de `res_*.json`. Protocole Prime Agent (courtier + garde réutilisés), 500 actions/jeu, graine d'environnement 0.", ""]
    for cle, titre in (("held_out", f"Held-out ({len(HELD)} jeux, agent gelé, un seul passage)"),
                       ("dev", f"Dev ({len(DEV)} jeux : utilisés pour le débogage et le réglage, NON indépendants)"),
                       ("tous", "Les 25 jeux (mélange dev + held-out)")):
        s = synthese[cle]
        a = s["aleatoire"]
        L += [f"## {titre}", "", "| agent | niveaux | jeux gagnés | actions totales |", "|---|---|---|---|",
              f"| Cerveau bayésien (nous) | {s['nous']['niveaux']}/{s['nous']['niveaux_total']} | {s['nous']['jeux_gagnes']} | {s['nous']['actions']} |",
              f"| Aléatoire (moy. 3 graines) | {a['niveaux_moyenne']}/{a['niveaux_total']} | {a['jeux_gagnes_moyenne']} | {a['actions_moyenne']} |",
              f"| Prime Agent run-1 (Claude Opus 5) | {s['prime_run1']['niveaux']}/{s['prime_run1']['niveaux_total']} | {s['prime_run1']['jeux_gagnes']} | {s['prime_run1']['actions']} |", "",
              f"Score SDK local moyen par jeu : nous {s['nous']['score_sdk_moyen']}, aléatoire {a['score_sdk_moyen_moyenne']}. CPU total nous : {s['nous']['cpu_s']} s (aléatoire : {a['cpu_s_moyenne']} s).", ""]
    L += ["## Détail par jeu", "", "| jeu | split | niveaux | nous | état | score SDK | aléatoire (3 graines) | Prime | CPU s |", "|---|---|---|---|---|---|---|---|---|"]
    for r in par_jeu:
        L.append(f"| {r['jeu']} | {r['split']} | {r['niveaux_total']} | {r['nous_niveaux']} | {r['nous_etat']} | {r['nous_score_sdk']} | {r['aleatoire_niveaux_par_graine']} | {r['prime_niveaux']} ({r['prime_etat']}) | {r['nous_cpu_s']} |")
    L += ["", "## Harnais continu (journal des versions)", "",
          f"Version finale gelée : **{journal.get('champion')}** (transfert de groupes : {journal.get('groupes_transfert')}).", "",
          "| version | description | test (jeux postérieurs) | décision |", "|---|---|---|---|"]
    for e in journal.get("journal", []):
        t = "-" if "P_meilleur" not in e else f"{e['victoires_candidat']} victoires / {e['defaites_candidat']} défaites par niveau, P(meilleur)={e['P_meilleur']}"
        L.append(f"| {e['version']} | {e['description']} | {t} | {e['decision']} |")
    L += ["", "## Valeur du calcul : économie si l'agent bon marché traite d'abord les niveaux qu'il sait résoudre", "",
          "| run Prime Agent | coût total estimé | économie (niveaux de coût égal) | économie (coût ~ actions de référence) |", "|---|---|---|---|"]
    for run, e in econ.items():
        L.append(f"| {run} | {e['cout_total_usd']} $ | {e['economie_usd_niveaux_egaux']} $ ({e['economie_pct_niveaux_egaux']} %) | {e['economie_usd_ponderee_reference']} $ ({e['economie_pct_ponderee_reference']} %) |")
    L += ["", "Méthode : `results.json` ne donne pas le coût par jeu ; on répartit le coût total du run au prorata des appels modèle de chaque jeu.",
          "Pour un jeu, l'agent bon marché joue d'abord ; les k premiers niveaux qu'il résout (et que Prime résolvait) sont retirés du LLM.",
          "Variante 1 : tous les niveaux coûtent autant. Variante 2 : le coût d'un niveau est proportionnel à son nombre d'actions de référence (scorecard du SDK).", "",
          "Réserves : (i) coût par niveau inconnu, il est modélisé ; (ii) le LLM doit reprendre un jeu déjà avancé : il lui faut rejouer/observer l'état, ce que ce calcul ignore ; "
          "(iii) les actions de l'agent bon marché consomment une partie du plafond de 500 actions ; (iv) les niveaux faciles sont probablement moins chers que la moyenne, d'où la variante 2 ; "
          "(v) 20 jeux held-out seulement + 5 jeux de dev tunés : estimation bruitée ; (vi) coût CPU de notre agent négligeable (quelques secondes par jeu).", ""]
    L += ["## Notes et limites", "",
          "- Le courtier de Prime Agent appelle `env.action1()...` qui n'existe plus dans le SDK installé : `harnais.CourtierCompat` ne remplace que `_apply` (via `env.step(GameAction, data)`). Le reste (dispatch, validation, garde de phase) est réutilisé tel quel, sans copie.",
          "- Découpe dev/held-out : `random.Random(2026).sample(sorted(25 préfixes), 5)`. Réglages faits uniquement sur les jeux de dev (masque HUD, marche aléatoire de repli).",
          "- Le harnais continu est testé sur les jeux de dev (jeux postérieurs, 3 graines du cerveau, appariées) ; le résultat v2 repose sur très peu de comparaisons (3 victoires/0 défaite) et est fragile. Les 20 jeux held-out utilisent les prieurs gelés, sans mise à jour entre jeux.",
          "- Le « score SDK » est le score local du scorecard hors ligne (échelle en %, par jeu) ; il n'est pas comparable au RHAE de Prime Agent.",
          "- Limites du cerveau : l'espace d'états des jeux est vaste, la nouveauté ne suffit pas comme signal de progrès ; pas de modèle d'objets/règles. Le masque HUD ne gère que des barres monotones en ligne/colonne.", ""]
    (ICI / "resultats_arc3.md").write_text("\n".join(L))
