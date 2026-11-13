import json,numpy as np,collections
R=[json.loads(l) for l in open('all.jsonl')]
ref={s:np.load(f'out/exact_s{s}_h100.npy').astype(float) for s in (1,2,3)}
by=collections.OrderedDict()
for r in R: by.setdefault(r['name'],[]).append(r)
rows=[]
for n,rs in by.items():
    rs=sorted(rs,key=lambda r:r['seed']); row=dict(name=n,size=rs[0]['size'],ratio=rs[0]['ratio'],mn9=[r.get('mn9') for r in rs],spikes=[r.get('spikes') for r in rs],actifs=[r.get('actifs') for r in rs])
    row['corr']=[float(np.corrcoef(np.load(f"out/{n}_s{r['seed']}_h100.npy").astype(float),ref[r['seed']])[0,1]) if 'mn9' in r else None for r in rs]
    rows.append(row); print(f"{n:16}{row['ratio']:8.1f} {str(row['mn9']):>14} {np.mean(row['spikes']):7.0f} {np.mean(row['actifs']):6.0f} {np.mean(row['corr']):6.3f}")
json.dump(rows,open('results.json','w'),indent=1)
