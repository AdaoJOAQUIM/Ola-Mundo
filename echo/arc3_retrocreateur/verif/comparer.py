import json,sys
ref={(x['seed'],x['difficulte']):x for x in json.load(open('/home/user/Ola-Mundo/echo/arc3_retrocreateur/runs/j1_retro_500.json'))}
for f in sys.argv[1:]:
    new=json.load(open(f)); diff=[]
    for n in new:
        o=ref[(n['seed'],n['difficulte'])]
        k=(n['niveaux']==o['niveaux'], n['validation']['exact']==o['validation']['exact'], n['actions']==o['actions'], n['niveaux_actions']==o['niveaux_actions'])
        if not all(k): diff.append(((n['seed'],n['difficulte']),k,o['niveaux'],n['niveaux'],o['validation']['exact'],n['validation']['exact'],o['actions'],n['actions'],o['stats'].get('cpu_depasse'),n['stats'].get('cpu_depasse')))
    print(f,len(new),'diff',len(diff)); [print(d) for d in diff]
    print('exact',sum(n['validation']['exact'] for n in new),'niv',sum(n['niveaux'] for n in new),sum(n['niveaux_total'] for n in new))
