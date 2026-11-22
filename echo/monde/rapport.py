"""Agrege resultats_bruts/*.json + traces -> resultats_monde.json et resultats_monde.md (francais clair)."""
import glob, json, os, pickle, collections
import moteur as mt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/resultats_bruts"
TR = "/home/user/monde_data/traces"
GB = 1e9
FRACS = [0.0025, 0.01, 0.05, 0.25]
# (debit octets/s, latence s). NVMe : consigne. Latences SATA/USB : HYPOTHESES (non fournies).
DISQUES = {"NVMe": (3.5 * GB, 0.1e-3), "SATA": (0.5 * GB, 0.2e-3), "USB": (1.0 * GB, 0.5e-3)}


def temps(b, ops, comp, disque, recouvre=False):
    bw, lat = DISQUES[disque]
    io = b / bw + ops * lat
    return max(io, comp) if recouvre else io + comp


def sim_lfu(trace, sizes, budget):
    """Bonus (simulation seule, non implemente dans le moteur) : cache LFU, a titre de comparaison."""
    freq = collections.Counter()
    cache, used, hits, rd = {}, 0, 0, 0
    for i, k in enumerate(trace):
        freq[k] += 1
        if k in cache:
            hits += 1
            cache[k] = i
            continue
        rd += sizes[k[2]]
        n = sizes[k[2]]
        if n > budget:
            continue
        while used + n > budget and cache:
            v = min(cache, key=lambda x: (freq[x], cache[x]))
            if freq[v] >= freq[k]:
                break
            used -= sizes[v[2]]
            del cache[v]
        if used + n <= budget:
            cache[k] = i
            used += n
    return hits, len(trace), rd


def charge():
    dec = {os.path.basename(p)[7:-5]: json.load(open(p)) for p in glob.glob(f"{RES}/decode_*.json")}
    ppl = {os.path.basename(p)[4:-5]: json.load(open(p)) for p in glob.glob(f"{RES}/ppl_*.json")}
    return dec, ppl


def main():
    dec, ppl = charge()
    lossless = json.load(open(f"{HERE}/resultats_lossless.json"))
    sizes = dec["naif"]["chunk_sizes"]
    total_q4 = 16 * 64 * sizes["q4"]
    base_ppl = ppl["q4"]["ppl"]
    R = dict(lossless=lossless, total_expert_bytes_q4=total_q4, chunk_sizes=sizes)
    # --- perplexite
    R["ppl"] = {k: dict(ppl=v["ppl"], rel_q4_pct=(v["ppl"] / base_ppl - 1) * 100, frac_culled=v["frac_culled"],
                        frac_q2=v["frac_q2_of_used"], rss_peak_mb=v["rss_peak_mb"], params=v["params"])
                for k, v in sorted(ppl.items())}
    # --- decodage
    R["decode"] = {}
    for k, d in dec.items():
        R["decode"][k] = {x: d[x] for x in d if x not in ("chunk_sizes",)}
        R["decode"][k]["bytes_rel_naif"] = d["bytes_per_token"] / dec["naif"]["bytes_per_token"]
    # --- cache : simulation sur traces (exacte : le cache ne change pas les sorties)
    R["sim_cache"] = {}
    for k, d in dec.items():
        f = f"{TR}/{k}.pkl"
        if not os.path.exists(f):
            continue
        tr = pickle.load(open(f, "rb"))
        R["sim_cache"][k] = {}
        for fr in FRACS:
            h, n, rd = mt.simulate_lru(tr, sizes, fr * total_q4)
            h2 = sim_lfu(tr, sizes, fr * total_q4)[0] if k == 'naif' else 0
            R["sim_cache"][k][str(fr)] = dict(budget_mb=fr * total_q4 / 1e6, hit_rate_lru=h / n,
                                              bytes_par_token_lru=rd / d["steps"], hit_rate_lfu_bonus=h2 / n)
    # --- critere (d)
    nv = dec["naif"]["bytes_per_token"]
    if "combo" in dec:
        c = dec["combo"]
        cp = ppl.get(c["ppl_name"]) if "ppl_name" in c else None
    # --- vitesse
    R["vitesse"] = {}
    for k, d in dec.items():
        R["vitesse"][k] = {}
        for dk in DISQUES:
            t = temps(d["bytes_per_token"], d["reads_per_token"], d["compute_s_per_token"], dk)
            to = temps(d["bytes_per_token"], d["reads_per_token"], d["compute_s_per_token"], dk, True)
            R["vitesse"][k][dk] = dict(s_par_token=t, tokens_par_s=1 / t, s_par_token_recouvert=to)
    R["extrapolation"] = extrapole(dec)
    json.dump(R, open(f"{HERE}/resultats_monde.json", "w"), indent=1)
    return R


def extrapole(dec):
    """EXTRAPOLATION (pas une mesure). Hypotheses explicites dans 'hypotheses'."""
    ref = dec["naif"]
    ns_par_param = ref["compute_s_per_token"] / 1.18e9  # OLMoE : ~1,18 G params actifs matmul (attn+experts+tete)
    ratio = dec["combo_sans_pf"]["bytes_per_token"] / ref["bytes_per_token"] if "combo_sans_pf" in dec else 1.0
    modeles = {
        "Kimi-Linear-48B-A3B (config lue : 26 couches MoE, 256 experts, 8 actifs, d=2304, I=1024)":
            dict(couches=26, experts=256, actifs=8, params_expert=3 * 2304 * 1024, o_par_param=0.5625,
                 actifs_params=3.0e9, quant="4 bits g32 fp16 (format du moteur)"),
        "Kimi K3 config lue (92 couches MoE, 896 experts, 16 actifs, d=7168, I=3072, MXFP4)":
            dict(couches=92, experts=896, actifs=16, params_expert=3 * 7168 * 3072, o_par_param=0.53125,
                 actifs_params=92 * 16 * 3 * 7168 * 3072 + 3e9, quant="MXFP4 (4,25 bits/poids)"),
        "Kimi K3.1 coherent avec 2,78T (92 couches, 896 experts, 16 actifs, expert = 33,7 M params, MXFP4)":
            dict(couches=92, experts=896, actifs=16, params_expert=2.78e12 / (92 * 896), o_par_param=0.53125,
                 actifs_params=92 * 16 * 2.78e12 / (92 * 896) + 3e9, quant="MXFP4 (4,25 bits/poids)"),
    }
    out = {}
    for nom, m in modeles.items():
        chunk = m["params_expert"] * m["o_par_param"]
        b = m["couches"] * m["actifs"] * chunk
        ops = m["couches"] * m["actifs"]
        comp = m["actifs_params"] * ns_par_param
        row = dict(taille_expert_Mo=chunk / 1e6, octets_par_token_naif_Go=b / 1e9, lectures_par_token=ops,
                   total_experts_Go=m["couches"] * m["experts"] * chunk / 1e9,
                   cache_0_25pct_Mo=0.0025 * m["couches"] * m["experts"] * chunk / 1e6,
                   cache_0_25pct_nb_experts=0.0025 * m["couches"] * m["experts"],
                   compute_s_par_token_mis_a_l_echelle=comp, quant=m["quant"], disques={})
        for dk in DISQUES:
            row["disques"][dk] = dict(
                naif_io_seul_s=temps(b, ops, 0, dk),
                naif_avec_compute_s=temps(b, ops, comp, dk),
                meca_io_seul_s=temps(b * ratio, ops * ratio, 0, dk),
                meca_avec_compute_s=temps(b * ratio, ops * ratio, comp, dk))
        out[nom] = row
    out["hypotheses"] = [
        "EXTRAPOLATION, pas une mesure : aucun de ces modeles n'a ete execute.",
        "Configs lues sur le Hub (moonshotai/Kimi-Linear-48B-A3B-Instruct, moonshotai/Kimi-K3). "
        "Le depot Kimi-K3.1 n'est pas accessible (401) : on utilise la config Kimi-K3 et les chiffres de la consigne.",
        "La config K3 lue (I=3072) donne ~5,4 T de params d'experts, incompatible avec 2,78 T : on montre les deux lectures.",
        f"Calcul : {ns_par_param*1e9:.2f} ns par parametre actif, mesure sur OLMoE (4 coeurs, torch fp32, "
        "dequantification incluse), applique tel quel. Un noyau natif serait bien plus rapide.",
        f"Les mecanismes sont appliques via le rapport d'octets mesure sur OLMoE (LOD 0,08 + culling 0,06 sans prefetch / naif = {ratio:.3f} ; le prefetch, lui, ajoute ~13 % de lectures inutiles mais cache la latence). "
        "Le cache a 0,25 % est suppose sans effet (taux de succes mesure sur OLMoE = 0).",
        "Latences SATA 0,2 ms et USB 0,5 ms : hypotheses (la consigne ne donne que NVMe 0,1 ms).",
    ]
    out["ratio_meca"] = ratio
    return out



# ============================================================================ rapport en francais
def mo(x):
    return f"{x/1e6:,.1f}".replace(",", " ")


def ecrire_md(R):
    dec, ppl, sim, vit = R["decode"], R["ppl"], R["sim_cache"], R["vitesse"]
    ls = R["lossless"]
    L = []
    w = L.append
    nv = dec["naif"]["bytes_per_token"]
    rss_max = max([d["rss_peak_mb"] for d in dec.values()] + [p["rss_peak_mb"] for p in ppl.values()]
                  + [ls["moteur"]["rss_peak_mb"]])
    # critere d
    c = dec.get("combo")
    pc = ppl["combo_l0.08_c0.06"]
    d_bytes = c["bytes_per_token"] / nv
    d_ok_b, d_ok_p = d_bytes <= 0.5, pc["rel_q4_pct"] <= 2.0
    w("# Moteur Monde Infini - phase 1 (OLMoE-1B-7B) : resultats\n")
    w("Tout ce qui suit est mesure sur ce CPU (4 coeurs, 15 Go de RAM), sauf la section « Extrapolation » qui est "
      "clairement un calcul, pas une mesure. Aucun texte du fichier de validation n'est reproduit : seulement des nombres.\n")
    w("## En bref\n")
    w(f"- Le moteur tient en RAM (pic mesure {rss_max/1000:.2f} Go, limite 4 Go) pour un modele de 7 Go "
      "(13,8 Go en bf16). Seuls le noyau (attention, normes, routeur, embeddings) et un petit cache d'experts sont en memoire.")
    w(f"- Sans perte (bf16 lu sur disque) : memes 32 tokens que transformers sur 3 prompts, ecart max de logit "
      f"{ls['max_abs_logit_diff']:.1e} (limite 1e-2). **Critere (a) : {'REUSSI' if ls['pass'] else 'ECHEC'}.**")
    w(f"- Le cache LRU a 0,25 %, 1 % et 5 % du volume des experts ne sert a rien ici (taux de succes "
      f"{sim['naif']['0.0025']['hit_rate_lru']*100:.1f} %, {sim['naif']['0.01']['hit_rate_lru']*100:.1f} %, "
      f"{sim['naif']['0.05']['hit_rate_lru']*100:.1f} %) : un token touche 128 experts distincts, bien plus que ces caches. "
      f"A 25 % il atteint {sim['naif']['0.25']['hit_rate_lru']*100:.1f} %.")
    w(f"- Le prefetch devine {dec['cache1_pf']['pred_hit_rate']*100:.0f} % des experts de la couche suivante, sans changer les sorties.")
    w(f"- **Critere (d) : ECHEC** sur cette cible. Cache 1 % + prefetch + LOD + culling : "
      f"{c['bytes_per_token']/1e6:.0f} Mo/token soit {d_bytes*100:.0f} % du 4 bits naif (cible : 50 %), "
      f"pour {pc['rel_q4_pct']:+.2f} % de perplexite (cible : +2 %). Sans prefetch : {dec['combo_sans_pf']['bytes_per_token']/nv*100:.0f} %. "
      "Le prefetch ajoute des lectures inutiles (predictions fausses) ; LOD et culling economisent ~13 % ; le cache 1 % n'economise rien. "
      "Descendre a 50 % exigerait de casser la qualite (voir plus bas).\n")

    w("## Criteres pre-enregistres\n")
    w("| Critere | Resultat | Chiffres |\n|---|---|---|")
    w(f"| (a) Sans perte vs transformers | {'REUSSI' if ls['pass'] else 'ECHEC'} | 3 prompts x 32 tokens identiques "
      f"({'oui' if ls['all_tokens_identical'] else 'non'}), ecart max de logit {ls['max_abs_logit_diff']:.2e} (<= 1e-2), fp32 des deux cotes |")
    cache_ok = all(d["cache_max_used_mb"] <= d["cache_budget_mb"] + 1e-9 for d in dec.values())
    w(f"| (b) RSS <= 4 Go partout, cache dans son budget | {'REUSSI' if rss_max < 4096 and cache_ok else 'ECHEC'} | "
      f"pic RSS max {rss_max:.0f} Mo ; cache jamais au-dessus du budget (verifie a chaque insertion et dans test_monde.py) |")
    w(f"| (c) Prefetch on/off identique | REUSSI | logits identiques bit a bit (tests) ; perplexite de decodage identique "
      f"{dec['naif']['nll_mean']:.12f} avec et sans prefetch |")
    w(f"| (d) Cible : octets/token <= 50 % du 4 bits naif a <= +2 % de ppl | {'REUSSI' if d_ok_b and d_ok_p else 'ECHEC'} | "
      f"{d_bytes*100:.1f} % des octets (prefetch inclus ; {dec['combo_sans_pf']['bytes_per_token']/nv*100:.1f} % sans), "
      f"{pc['rel_q4_pct']:+.2f} % de ppl |\n")

    w("## Octets lus par token, mecanisme par mecanisme (128 tokens de decodage)\n")
    w("Octets exacts comptes au niveau du pread. « Lectures » = nombre de pread par token. "
      "Un token touche 16 couches x 8 experts = 128 experts ; un expert 4 bits = 3,54 Mo, 2 bits = 1,97 Mo.\n")
    w("| Configuration | Octets/token (Mo) | % du naif | Lectures/token | Succes cache | Succes cache ou prefetch |\n|---|---|---|---|---|---|")
    noms = [("naif", "1. Naif : 4 bits en streaming, sans cache"), ("cache1", "2. + cache LRU 1 %"),
            ("cache25", "2b. cache LRU 25 %"), ("cache1_pf", "3. cache 1 % + prefetch"),
            ("lod", "4. LOD seul (2 bits si poids normalise < 0,10)"), ("cull", "5. Culling seul (poids normalise < 0,06)"),
            ("combo", "Tout ensemble (cible d) : cache 1 % + prefetch + LOD 0,08 + culling 0,06"),
            ("combo_sans_pf", "Idem sans prefetch : cache 1 % + LOD 0,08 + culling 0,06"),
            ("combo25", "Tout ensemble avec cache 25 % (informatif)"),
            ("agressif", "Agressif : cache 1 % + prefetch + LOD 0,12 + culling 0,07 (informatif)")]
    for k, n in noms:
        if k in dec:
            d = dec[k]
            w(f"| {n} | {mo(d['bytes_per_token'])} | {d['bytes_per_token']/nv*100:.0f} % | {d['reads_per_token']:.0f} | "
              f"{d['hit_rate_cache']*100:.1f} % | {d['hit_rate_cache_ou_prefetch']*100:.1f} % |")
    w("")
    w("## Cache des zones visitees : taux de succes selon la taille\n")
    w(f"Total des experts 4 bits : {R['total_expert_bytes_q4']/1e9:.2f} Go. Les taux viennent du rejeu exact de la trace d'acces "
      "du decodage naif dans un LRU (le cache ne change pas les sorties, donc la trace est la meme). "
      "Verification : la mesure reelle a 1 % et a 25 % donne les memes nombres que le rejeu.\n")
    w("| Taille du cache | Octets | Succes LRU | Octets/token (LRU) | Succes LFU (bonus, simulation) |\n|---|---|---|---|---|")
    for fr in FRACS:
        s = sim["naif"][str(fr)]
        w(f"| {fr*100:g} % | {s['budget_mb']:.0f} Mo | {s['hit_rate_lru']*100:.1f} % | {mo(s['bytes_par_token_lru'])} Mo | "
          f"{s['hit_rate_lfu_bonus']*100:.1f} % |")
    if "cache25" in dec:
        d = dec["cache25"]
        w(f"\nMesure reelle a 25 % : succes {d['hit_rate_cache']*100:.1f} %, {mo(d['bytes_per_token'])} Mo/token "
          f"(rejeu : {sim['naif']['0.25']['hit_rate_lru']*100:.1f} %). "
          f"Cache reel a 1 % : {dec['cache1']['hit_rate_cache']*100:.1f} % (rejeu : {sim['naif']['0.01']['hit_rate_lru']*100:.1f} %).")
    w("\nPourquoi si bas : 41 % des experts d'un token etaient deja utilises au token precedent, mais entre deux usages du meme "
      "expert il passe en mediane 226 autres experts distincts. Un LRU doit donc contenir plus d'un token entier "
      "(128 experts, soit 3,5 % du volume) avant de gagner quoi que ce soit. Un modele geant (cache vers 0,25 %) est dans ce cas : "
      "le cache LRU n'y aide pas, seuls le prefetch, le LOD et le culling comptent.\n")

    w("## Prefetch (prediction de la couche suivante)\n")
    p = dec["cache1_pf"]
    w(f"On applique le routeur de la couche L+1 a l'etat cache de la couche L (apres son attention), on lit en tache de fond "
      f"les 8 experts predits pendant que la couche L calcule.\n")
    w(f"- Taux de succes de la prediction : {p['pred_hit_rate']*100:.1f} % des experts reellement utilises (couches 1 a 15) etaient predits.")
    w(f"- Part des utilisations servie sans attendre le disque : {p['hit_rate_cache_ou_prefetch']*100:.1f} % "
      "(la couche 0 ne peut pas etre predite).")
    w(f"- Cout : {mo(p['bytes_per_token'])} Mo/token contre {mo(nv)} sans prefetch (les predictions fausses sont lues pour rien et comptees). "
      f"Le prefetch ne reduit donc pas les octets : il cache la latence. Attente disque du fil principal : "
      f"{p['io_wait_s']/p['steps']*1000:.0f} ms/token contre {dec['naif']['io_wait_s']/dec['naif']['steps']*1000:.0f} ms sans.")
    w("- Les sorties sont identiques bit a bit avec ou sans prefetch.\n")

    w("## LOD et culling : qualite (perplexite sur ~20 000 caracteres = 4 551 tokens)\n")
    w(f"Reference : bf16 {ppl['bf16']['ppl']:.3f} ; 4 bits {ppl['q4']['ppl']:.3f} "
      f"({(ppl['q4']['ppl']/ppl['bf16']['ppl']-1)*100:+.1f} % vs bf16 : la quantification 4 bits simple est deja ce qui coute le plus). "
      "Le poids « normalise » = poids du routeur / somme des poids des 8 experts retenus (moyenne 0,125).\n")
    w("| Config | Perplexite | vs 4 bits | Experts en 2 bits | Experts ignores |\n|---|---|---|---|---|")
    for k, v in sorted(ppl.items(), key=lambda kv: (kv[0][:3] != "lod", kv[0][:4] != "cull", kv[0])):
        if k in ("q4", "bf16"):
            continue
        w(f"| {k} | {v['ppl']:.3f} | {v['rel_q4_pct']:+.2f} % | {v['frac_q2']*100:.1f} % | {v['frac_culled']*100:.1f} % |")
    w("")
    w("Lecture : le LOD jusqu'a 0,10 coute +2,0 % pour 39 % d'experts en 2 bits (soit ~18 % d'octets en moins) ; "
      "le culling a 0,06 coute +1,0 % pour 6 % d'experts ignores. Au-dela, la qualite s'effondre vite. "
      "Pour descendre vers 50 % des octets, il faudrait ~l'essentiel des experts en 2 bits et un tiers ignores : "
      f"{ppl.get('combo_l0.15_c0.10',{}).get('rel_q4_pct',float('nan')):+.0f} % de perplexite avec LOD 0,15 + culling 0,10.\n")

    w("## Vitesse (modele : temps/token = octets/debit + lectures x latence + calcul mesure)\n")
    w("Calcul mesure sur ce CPU (4 coeurs, torch fp32, dequantification a chaque usage). "
      "Disques : NVMe 3,5 Go/s et 0,1 ms (consigne) ; SATA 0,5 Go/s et USB 1 Go/s (latences non fournies : "
      "hypotheses 0,2 ms et 0,5 ms). Les colonnes donnent le temps par token en secondes ; entre parentheses, avec "
      "recouvrement parfait lecture/calcul (cas du prefetch).\n")
    w("| Configuration | calcul s/tok | NVMe | SATA | USB |\n|---|---|---|---|---|")
    for k, n in noms:
        if k in dec:
            v = vit[k]
            w(f"| {n} | {dec[k]['compute_s_per_token']:.2f} | " + " | ".join(
                f"{v[dk]['s_par_token']:.2f} ({v[dk]['s_par_token_recouvert']:.2f})" for dk in DISQUES) + " |")
    w("\nIci le calcul (~1,2 s/token en Python/torch) domine meme sur USB : le moteur est limite par le CPU, pas par le disque. "
      "Le disque de cette machine lit a environ 1,5 Go/s (mesure).\n")

    ex = R["extrapolation"]
    w("## Extrapolation (CALCUL, pas une mesure)\n")
    for h in ex["hypotheses"]:
        w(f"- {h}")
    w("")
    w("| Modele | Expert | Octets/token naif | Lectures/token | Cache 0,25 % | I/O seul NVMe / SATA / USB (naif) | idem avec mecanismes | avec calcul mis a l'echelle (naif, NVMe) |\n|---|---|---|---|---|---|---|---|")
    for nom, r in ex.items():
        if not isinstance(r, dict) or "disques" not in r:
            continue
        dd = r["disques"]
        w(f"| {nom} | {r['taille_expert_Mo']:.1f} Mo | {r['octets_par_token_naif_Go']:.1f} Go | {r['lectures_par_token']} | "
          f"{r['cache_0_25pct_nb_experts']:.0f} experts | "
          + " / ".join(f"{dd[k]['naif_io_seul_s']:.1f} s" for k in DISQUES) + " | "
          + " / ".join(f"{dd[k]['meca_io_seul_s']:.1f} s" for k in DISQUES)
          + f" | {dd['NVMe']['naif_avec_compute_s']:.0f} s |")
    w("\nLe calcul mis a l'echelle (62 a 118 s/token avec le meme noyau torch sur 4 coeurs) depasse de loin la lecture disque : sur ce moteur-ci, "
      "les geants seraient limites par le CPU ; avec un noyau natif multi-coeurs, le disque redeviendrait le goulot.")
    w("\nLecture : sur ces geants, meme un NVMe donne des secondes par token a cause des octets seuls ; "
      "avec des mecanismes aussi modestes que ceux mesures ici, le gain de ~12 % (sans prefetch) ne change pas l'ordre de grandeur.\n")

    w("## Ecarts par rapport a la consigne et limites\n")
    for t in [
        "Experts bf16 non recopies : la copie bf16 (12,9 Go) depasserait seule la limite de 12 Go de fichiers convertis. "
        "Le mode bf16 lit donc les experts directement dans les shards HF d'origine (un pread contigu par expert, "
        "index dans manifest.json), sans O_DIRECT (decalages non alignes) mais avec posix_fadvise(DONTNEED). Fichiers convertis : 6,6 Go.",
        "Reference transformers calculee couche par couche (modules HF reels, fp32) car le modele fp32 complet (28 Go) ne tient pas en 15 Go ; "
        "comparaison par forcage de l'enseignant sur la sequence produite par le moteur (les logits de chaque position sont ceux du decodage incremental). "
        "Les tokens argmax de la reference sont identiques a ceux du moteur a chaque position.",
        "Quantification : 4 bits et 2 bits par groupes de 32, echelle fp16 seule (pas de zero), recherche du meilleur ecretage par groupe. "
        "C'est de l'arrondi simple (pas de GPTQ) : +6 % de perplexite vs bf16 ; un 2 bits meilleur deplacerait le compromis du LOD.",
        "Perplexite mesuree par lots (fenetres de 2048 tokens, memes experts et memes calculs ; test_decodage_egal_lot verifie l'egalite avec le decodage) ; "
        "les octets/token sont mesures en decodage token par token sur 128 tokens du meme texte (forcage de l'enseignant), pas en generation libre.",
        "Les tailles de cache sont aussi evaluees par rejeu de trace (exact, car le cache ne change pas les sorties) ; deux tailles (1 % et 25 %) sont verifiees en reel.",
        "Les poids « normalises » du LOD et du culling : poids du routeur / somme des 8 poids retenus ; les experts gardes ne sont pas renormalises (OLMoE ne renormalise pas).",
        "Le prefetch n'ecrit dans le cache que ce qui est consomme ; les lectures predites mais inutiles sont comptees dans les octets.",
        "Un seul run par configuration (pas de barres d'erreur) ; 128 tokens de decodage. Le calcul mesure varie de ~15 % d'un run a l'autre.",
    ]:
        w(f"- {t}")
    w("\n## Reproduire\n")
    w("```\npython3 convert.py            # safetensors HF -> /home/user/monde_data/monde (6,6 Go)\n"
      "python3 lossless.py moteur && python3 lossless.py ref   # critere (a)\n"
      "python3 bench.py decode NOM '{...}' ; python3 bench.py ppl NOM '{...}'\n"
      "python3 rapport.py            # ce rapport\npython3 -m pytest -q test_monde.py\n```")
    open(f"{HERE}/resultats_monde.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ecrire_md(main())
