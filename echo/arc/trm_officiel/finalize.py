import json, sys, pickle
sys.path.insert(0, 'code')
K = int(sys.argv[1])
res = json.load(open(f'res_K{K}.json'))
ev = {}
import glob
out = {'methode': 'TRM officiel', 'K': K, 'taches': {}}
p1 = p2 = 0; rows = []
for t, r in sorted(res.items()):
    nt = len(r['per_test'])
    att = []; sc = []
    for rank in range(2):
        grids = []; s = []
        for pt in r['per_test']:
            c = pt[rank] if rank < len(pt) else pt[0]
            grids.append(c['grid']); s.append(c['votes'])
        att.append(grids); sc.append(s)
    out['taches'][t] = {'tentatives': att, 'scores': sc, 'cpu_s': r['cpu_s']}
    gold = [x['output'] for x in json.load(open(f'/home/user/Ola-Mundo/echo/arc/data/evaluation/{t}.json'))['test']]
    ok1 = all(pt[0]['grid'] == g for pt, g in zip(r['per_test'], gold))
    ok2 = all(any(c['grid'] == g for c in pt[:2]) for pt, g in zip(r['per_test'], gold))
    p1 += ok1; p2 += ok2; rows.append((t, nt, ok1, ok2, r['cpu_s'], r['per_test'][0][0]['votes']))
json.dump(out, open('/home/user/Ola-Mundo/echo/arc/fusion/trm.json', 'w'))
for x in rows: print(x)
print('pass@1', p1, 'pass@2', p2, 'of', len(res), 'mean cpu_s', sum(r['cpu_s'] for r in res.values())/len(res))
