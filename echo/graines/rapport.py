"""Genere resultats_graines.md depuis resultats_graines.json (tableaux + verdict contre le critere)."""
import json, os
D = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(D, 'resultats_graines.json')))
NOMS = {'rtn4': 'RTN 4 bits', 'rtn3': 'RTN 3 bits', 'rtn2': 'RTN 2 bits',
        'sl4_full': 'SeedLM C=8,P=3 (~4 b) [2^16 graines]', 'sl4_p12': 'SeedLM C=8,P=3 (~4 b) [2^12 graines]', 'sl3_p12': 'SeedLM C=12,P=4 (~3 b) [2^12 graines]',
        'sl2_p12': 'SeedLM C=14,P=2 (~2 b) [2^12 graines]', 'sl2_full': 'SeedLM C=14,P=2 (~2 b) [2^16 graines]',
        'sl1_full': 'SeedLM C=24,P=1 (~1 b) [2^16 graines]', 'sl0375_full': 'SeedLM C=64,P=1 (0,375 b) [2^16 graines]',
        'pr1': 'PRANC-blocs 1 bit', 'pr0375': 'PRANC-blocs 0,375 bit', 'hyper25': 'HyperNetwork 0,375 bit'}
ordre = list(NOMS)
L = []
for mod, d in R.items():
    L.append(f"\n### {mod}\n")
    i = d['info']
    L.append(f"MLP comprimes : {i['params_mlp']/1e6:.1f} M / {i['params_total']/1e6:.1f} M parametres = {100*i['part_comprimee']:.1f} % ; "
             f"perplexite d'origine : {d['orig']['ppl']:.2f} ({d['orig']['n_tokens']} tokens predits).\n")
    L.append("| Methode | bits/poids (tout compte) | err. rel. ‖W−Ŵ‖/‖W‖ | perplexite | ppl / origine | critère +10 % |")
    L.append("|---|---|---|---|---|---|")
    for k in ordre:
        if k in d:
            r = d[k]
            L.append(f"| {NOMS[k]} | {r['bpw']:.3f} | {r['err_rel']:.3f} | {r['ppl']:.4g} | {r['ppl_rel']:.4g} | {'oui' if r['ppl_rel'] <= 1.10 else 'non'} |")
    if 'lzma_rtn4' in d:
        z = d['lzma_rtn4']
        L.append(f"\nRTN 4 bits, codes seuls : {z['bpw_codes_brut']:.3f} b/poids bruts → {z['bpw_codes_lzma']:.3f} b/poids après lzma "
                 f"({z['bpw_lzma_avec_echelles']:.3f} avec les échelles 8 bits non compressées).")
open(os.path.join(D, 'tableaux.md'), 'w').write('\n'.join(L))
print('\n'.join(L))
