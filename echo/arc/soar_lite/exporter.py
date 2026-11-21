# -*- coding: utf-8 -*-
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bib
D = os.path.dirname(os.path.abspath(__file__))
ev = json.load(open(os.path.join(D, 'eval_finale.json')))['details']
taches = {}
for r in ev:
    if not r['programmes']:
        continue
    taches[r['id']] = {'tentatives': r['sorties'][:2],
                       'programmes': [bib.developper(p) for p in r['programmes'][:2]],
                       'verifie_sur_exemples': True, 'cpu_s': r['cpu']}
out = {'methode': 'Solveur + auto-amélioration (idée SOAR)', 'taches': taches}
json.dump(out, open(os.path.join(D, '..', 'fusion', 'soar_lite.json'), 'w'), ensure_ascii=False)
sub = json.load(open(os.path.join(D, '..', 'sous_ensemble_fusion.json')))['taches']
ok = {r['id']: r for r in ev}
print(len(taches), 'avec programme;', sum(ok[t]['resolu2'] for t in sub), '/20 sous-ensemble résolus (pass@2)')
