# -*- coding: utf-8 -*-
"""Rondes d'auto-amélioration (entraînement seul), puis évaluation unique figée."""
import sys, os, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bib, solveur
D = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(D, 'rondes')
K_PAR_RONDE = 30
NRONDES = 3


def attendre(motif):
    while subprocess.run(['pgrep', '-f', motif], capture_output=True).stdout.strip():
        time.sleep(60)


def lancer(split, budget=10.0, procs=1, ids=None):
    import multiprocessing as mp
    dos = os.path.join(solveur.DOSSIER, 'data', split)
    fich = sorted(os.path.join(dos, f) for f in os.listdir(dos) if f.endswith('.json'))
    if ids:
        fich = [f for f in fich if os.path.basename(f)[:-5] in ids]
    args = [(f, budget) for f in fich]
    if procs > 1:
        with mp.get_context('fork').Pool(procs) as p:
            return list(p.imap(solveur._travail, args, chunksize=1))
    return [solveur._travail(a) for a in args]


def donnees_ronde(details, train_dir):
    """Données d'apprentissage : programmes réussis + relabeling a posteriori sur les échecs."""
    out = []
    for r in details:
        if r['resolu2']:
            for p, ok in zip(r['programmes'], r['ok']):
                if ok:
                    s = bib.noms_prog(p)
                    if s:
                        out.append((r['id'], 1.0, s))
        elif not r['resolu1']:
            t = json.load(open(os.path.join(train_dir, r['id'] + '.json')))
            tr = [(solveur.en_grille(x['input']), solveur.en_grille(x['output'])) for x in t['train']]
            for s in bib.hindsight(tr):
                out.append((r['id'], 0.5, list(s)))
    return out


def main():
    attendre('eval_mit.py')
    train_dir = os.path.join(solveur.DOSSIER, 'data', 'training')
    base = json.load(open(os.path.join(solveur.DOSSIER, 'res_training.json')))['details']
    courbe = {'0': sum(r['resolu2'] for r in base)}
    cum, lib = [], None
    det = base
    for k in range(1, NRONDES + 1):
        f = os.path.join(R, 'ronde_%d.json' % k)
        if os.path.exists(f):
            d = json.load(open(f)); lib = d['bib']; det = d['details']
        else:
            cum += donnees_ronde(det, train_dir)
            lib = bib.apprendre(cum, K_PAR_RONDE * k, lib)
            bib.appliquer_bib(lib)
            det = lancer('training')
            for r in det: r.pop('_sorties', None)
            json.dump({'bib': lib, 'details': det, 'n_donnees': len(cum)}, open(f, 'w'), ensure_ascii=False)
        courbe[str(k)] = sum(r['resolu2'] for r in det)
        print('ronde', k, 'resolus2', courbe[k and str(k)], 'resolus1', sum(r['resolu1'] for r in det),
              'macros', len(lib['macros']), flush=True)
    json.dump({'courbe': courbe, 'bib_finale': lib}, open(os.path.join(D, 'bib_finale.json'), 'w'), ensure_ascii=False)
    # évaluation unique, bibliothèque figée
    attendre('eval_mit.py')
    bib.appliquer_bib(lib)
    try:
        procs = int(open(os.path.join(D, 'procs.txt')).read().strip())
    except Exception:
        procs = 1
    procs = max(1, min(procs, 2))
    t0 = time.time()
    det = lancer('evaluation', procs=procs)
    out = []
    for r in det:
        e = {k: v for k, v in r.items() if k != '_sorties'}
        e['sorties'] = [[[list(row) for row in g] for g in sor] for _, sor in r['_sorties']]
        out.append(e)
    json.dump({'procs': procs, 'duree_s': time.time() - t0, 'details': out},
              open(os.path.join(D, 'eval_finale.json'), 'w'), ensure_ascii=False)
    print('EVAL', sum(r['resolu2'] for r in det), sum(r['resolu1'] for r in det), flush=True)


if __name__ == '__main__':
    main()
