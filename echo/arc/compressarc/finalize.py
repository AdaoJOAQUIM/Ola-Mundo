import json,os
A='/home/user/Ola-Mundo/echo/arc'
plan=json.load(open(A+'/compressarc/plan_steps.json'))
sol={t:json.load(open(f'{A}/data/evaluation/{t}.json'))['test'] for t in plan}
res={};rows=[]
for t,n in plan.items():
    f=f'{A}/compressarc/out/{t}.json'
    if not os.path.exists(f): continue
    d=json.load(open(f)); gt=[e['output'] for e in sol[t]]
    k=len(gt)
    att=[[d['a1'][i] for i in range(k)],[d['a2'][i] for i in range(k)]]
    p1=all(att[0][i]==gt[i] for i in range(k))
    p2=all(att[0][i]==gt[i] or att[1][i]==gt[i] for i in range(k))
    res[t]={'tentatives':att,'scores':d['scores'],'cpu_s':d['cpu_s']}
    rows.append((t,d['steps'],round(d['wall_s']),p1,p2))
json.dump({'methode':'CompressARC officiel','pas_par_tache':{t:r[1] for t,r in zip([r[0] for r in rows],rows)},'taches':res},open(A+'/fusion/compressarc.json','w'))
n=len(rows);p1=sum(r[3] for r in rows);p2=sum(r[4] for r in rows)
tot=sum(r[2] for r in rows)
md=f"""# CompressARC officiel (CPU, 1 thread) - resultats

Code officiel (iliao2345/CompressARC, ~76K parametres, entraine from scratch par tache), copie patchee dans `compressarc/code/` : seul changement = `torch.set_default_device('cuda')` -> `'cpu'` (arc_compressor.py). Hyperparametres par defaut conserves (Adam lr 0.01, betas 0.5/0.9, perte KL + 10*reconstruction) ; 2 tentatives = les 2 solutions les plus frequentes/scorees (logique `solution_selection.py`). Adaptateur de donnees : `adapter.py` (fichiers fchollet par tache -> JSON combine style Kaggle `dataset/arc-agi_evaluation_*.json`). Scoring avec les sorties officielles uniquement a la fin.

## Reduction par rapport aux defauts du papier (2000 pas/tache, 1500 dans analyze_example)
Mesure CPU (1 thread, machine partagee) : 0.66 s/pas (grille 6x6, 3 exemples), 2.0 s/pas (19x25), 4.1 s/pas (28x29, 5 exemples) ; donc 1500 pas = 17 a 100 min/tache, impossible pour 20 taches en ~3 h. Budget par tache ~500 s : pas = clamp(500/s_pas, 150, 1500), de 150 a 783 pas (soit 8 a 52 % de 1500), voir `pas_par_tache`. Les grandes taches (150 pas) sont tres probablement sous-entrainees : resultats = borne basse de CompressARC.

## Resultats ({n} taches)
pass@2 : {p2}/{n} ; pass@1 : {p1}/{n} ; temps total {tot/3600:.2f} h

| tache | pas | s | pass@1 | pass@2 |
|---|---|---|---|---|
""" + "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {'oui' if r[3] else 'non'} | {'oui' if r[4] else 'non'} |" for r in rows)
open(A+'/compressarc/resultats_compressarc.md','w').write(md+"\n")
print(n,p1,p2,tot)
