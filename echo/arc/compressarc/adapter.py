# Adapter: fchollet/ARC per-task json -> Kaggle-style combined json expected by preprocessing.py
import json,os
A='/home/user/Ola-Mundo/echo/arc'
ids=json.load(open(A+'/sous_ensemble_fusion.json'))['taches']
ch,so={},{}
for t in ids:
    d=json.load(open(f'{A}/data/evaluation/{t}.json'))
    ch[t]={'train':d['train'],'test':[{'input':e['input']} for e in d['test']]}
    so[t]=[e['output'] for e in d['test']]
os.makedirs(A+'/compressarc/code/dataset',exist_ok=True)
json.dump(ch,open(A+'/compressarc/code/dataset/arc-agi_evaluation_challenges.json','w'))
json.dump(so,open(A+'/compressarc/code/dataset/arc-agi_evaluation_solutions.json','w'))
