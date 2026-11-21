"""Orchestrateur : python lancer.py MODELE job1 job2 ...   (resultats cumules dans resultats_graines.json)

jobs : orig, rtn4, rtn3, rtn2, lzma, sl4, sl3, sl2, sl1, sl0375 (+ suffixe _full ou _p12 pour
la recherche de graines : exhaustive 2^16 ou sous-ensemble aleatoire de 2^12), pr0375, pr1, hyper, sens
"""
import sys, os, json, time, torch, numpy as np
from modeles import *
from seedlm import SeedLM
from rtn import rtn, taille_lzma_codes4
from pranc import pranc
import hyper as hy

torch.set_num_threads(4)
SORTIE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultats_graines.json')
SL = {'sl4': (8, 3), 'sl3': (12, 4), 'sl2': (14, 2), 'sl1': (24, 1), 'sl0375': (64, 1)}


def charger_res():
    return json.load(open(SORTIE)) if os.path.exists(SORTIE) else {}


def sauver(res, modele, cle, val):
    res = charger_res(); res.setdefault(modele, {})[cle] = val
    json.dump(res, open(SORTIE, 'w'), indent=1, ensure_ascii=False)


def appliquer(m, f, originaux):
    """f(nom, W, couche_idx) -> (What, bits). Remplace les poids ; retourne bits, n, erreurs."""
    bits = n = 0; num = den = 0.0; errs = []
    for k, (nom, mod) in enumerate(poids_mlp(m)):
        W = originaux[nom]
        Wh, b = f(nom, W)
        mod.weight.data.copy_(Wh)
        bits += b; n += W.numel()
        e2 = ((W - Wh) ** 2).sum().item(); w2 = (W ** 2).sum().item()
        num += e2; den += w2; errs.append(math.sqrt(e2 / w2))
    return dict(bits=bits, n_comprime=n, bpw=bits / n, err_rel=math.sqrt(num / den), err_rel_moy_matrice=float(np.mean(errs)))


import math


def restaurer(m, originaux):
    for nom, mod in poids_mlp(m): mod.weight.data.copy_(originaux[nom])


def main():
    modele, jobs = sys.argv[1], sys.argv[2:]
    m, tok = charger(modele)
    mats = poids_mlp(m)
    orig = {n: mod.weight.data.clone() for n, mod in mats}
    ntot = sum(p.numel() for p in m.parameters()); nmlp = sum(w.numel() for w in orig.values())
    base = dict(params_total=ntot, params_mlp=nmlp, part_comprimee=nmlp / ntot, n_matrices=len(mats))
    sauver(None, modele, 'info', base)
    print(modele, base, flush=True)
    ppl0 = charger_res().get(modele, {}).get('orig', {}).get('ppl')
    for job in jobs:
        t0 = time.time(); cpu0 = time.process_time()
        extra = {}
        if job == 'orig':
            ppl, ntok = perplexite(m, tok); sauver(None, modele, 'orig', dict(ppl=ppl, n_tokens=ntok, bpw_orig=16.0))
            print('orig', ppl, flush=True); continue
        if job == 'lzma':
            allc = []; nb = 0
            for n, W in orig.items():
                _, _, q = rtn(W, 4); allc.append(q.reshape(-1)); 
            lz, brut = taille_lzma_codes4(allc); nn_ = sum(c.numel() for c in allc)
            r = dict(bits_codes_brut=brut, bits_lzma=lz, bpw_codes_brut=brut / nn_, bpw_codes_lzma=lz / nn_,
                     bpw_lzma_avec_echelles=lz / nn_ + 8 / 32)
            sauver(None, modele, 'lzma_rtn4', r); print(job, r, flush=True); continue
        if job == 'sens':
            nom, W = [(n, w) for n, w in orig.items() if 'layers.10.mlp.up_proj' in n][0]
            flat = W.reshape(-1); res = {}
            rng = np.random.RandomState(0)
            for k, (C, P) in SL.items():
                nb = flat.numel() // C; ids = rng.choice(nb, 3000, replace=False)
                blocs = flat[: nb * C].reshape(nb, C)[ids]
                sc = W.std().item(); res[k] = {}
                for ns, tk in [(4096, 8), (16384, 8), (None, 1), (None, 8)]:
                    S = SeedLM(C, P, topk=tk, nseeds=ns)
                    Wh, _ = S.compress(blocs.reshape(-1).clone())
                    e = ((blocs.reshape(-1) - Wh).norm() / blocs.norm()).item()
                    res[k][f'nseeds={ns or 65535},topk={tk}'] = e
                print(k, res[k], flush=True)
            sauver(None, modele, 'sensibilite_seedlm', res); continue

        if job.startswith('rtn'):
            b = int(job[3:]); f = lambda n, W: rtn(W, b)[:2]
            cfg = dict(methode='RTN g32 sym, echelle 8 bits', bits_nominal=b)
        elif job.startswith('sl'):
            nom, _, mode = job.partition('_')
            C, P = SL[nom]; ns = None if mode == 'full' else 4096
            S = SeedLM(C, P, nseeds=ns)
            f = lambda n, W: S.compress(W)[0::1][0:2] if False else (lambda r: (r[0], r[1]['bits']))(S.compress(W))
            cfg = dict(methode='SeedLM', C=C, P=P, nseeds=ns or 65535, topk=8, bits_bloc=S.bits_par_bloc())
        elif job.startswith('pr'):
            bpw = {'pr0375': 0.375, 'pr1': 1.0}[job]; f = lambda n, W: pranc(W, bpw)
            cfg = dict(methode='PRANC par blocs L=4096 (base partagee, alpha fp16 MC)', bpw_cible=bpw)
        elif job.startswith('hyper'):
            minutes = float(job[5:] or 25)
            nl = max(int(n.split('.')[2]) for n in orig) + 1
            lst = [(int(n.split('.')[2]), hy.TYPES[n.rsplit('.', 1)[-1]], W) for n, W in orig.items()]
            rmax = max(W.shape[0] for _, _, W in lst); cmax = max(W.shape[1] // hy.BS for _, _, W in lst)
            coords, cibles, scales = hy.construire_donnees(lst)
            net, H, npar, steps, dt = hy.entrainer(coords, cibles, nl, rmax, cmax, 0.375, minutes, len(lst),
                                                   log=lambda s: print(s, flush=True))
            gen = hy.generer(net, coords)
            off = {}; p = 0
            for k, (n, W) in enumerate(orig.items()):
                cnt = W.numel() // hy.BS
                off[n] = (gen[p:p + cnt].reshape(W.shape) * scales[k]); p += cnt
            f = lambda n, W: (off[n], None)
            cfg = dict(methode='HyperNetwork MLP (coordonnees -> bloc de 32 poids)', H=H, params_fp16=npar,
                       pas_entrainement=steps, minutes_entrainement=dt / 60, bits_params=npar * 16 + 16 * len(lst))
        else:
            raise SystemExit('job inconnu ' + job)
        if job.startswith('hyper'):
            bits_tot = cfg['bits_params']
            r = appliquer(m, lambda n, W: (f(n, W)[0], 0), orig); r['bits'] = bits_tot; r['bpw'] = bits_tot / r['n_comprime']
        else:
            r = appliquer(m, f, orig)
        tc = time.time() - t0
        ppl, _ = perplexite(m, tok)
        ppl0 = charger_res()[modele]['orig']['ppl']
        r.update(cfg); r.update(ppl=ppl, ppl_rel=ppl / ppl0, temps_compression_s=tc, cpu_s=time.process_time() - cpu0)
        sauver(None, modele, job, r)
        print(job, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        restaurer(m, orig)


if __name__ == '__main__':
    main()
