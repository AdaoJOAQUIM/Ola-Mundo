# Plus petit circuit sucre -> MN9 : on classe chaque connexion par ce qu'elle transporte reellement vers MN9,
# on garde les k meilleures, et on rejoue le cerveau entier avec ces seules connexions.
import numpy as np, simlib as S, json, sys
from multiprocessing import Pool
pre,post,w,sg,ids=S.load('conn.npz'); N=S.N; ws=w*sg[pre]
rate=sum(np.load(f'out/exact_s{s}_h100.npy').astype(float) for s in (1,2,3))/3
act=rate>0; m=act[pre]&act[post]; idx=np.nonzero(m)[0]
p,q=pre[idx],post[idx]; flux=rate[p]*w[idx]                       # synapses x spikes : trafic reel de la connexion
tin=np.bincount(q,flux,N)+1e-9; share=flux/tin[q]
imp=np.zeros(N); imp[S.MN9]=1.0; tot=np.zeros(N)
for _ in range(12):                                                 # remonte l'importance depuis MN9
    tot+=imp; imp=np.bincount(p,share*imp[q],N)
tot+=imp
score_rel=share*tot[q]; score_flux=flux*(tot[q]>0)
print("neurones actifs",int(act.sum()),"connexions entre eux",len(idx),"dont utiles a MN9",int((score_rel>0).sum()),flush=True)
def run(job):
    kind,k,gain,seed=job; sc=score_rel if kind=='rel' else score_flux
    keep=idx[np.argsort(-sc)[:k]]
    Wk=S.csr(pre[keep],post[keep],ws[keep]*gain); sp=S.sim(S.exact_apply(Wk),seed=seed)
    r=dict(kind=kind,k=k,gain=gain,seed=seed,ratio=len(pre)/k,mn9=None if sp is None else int(sp[S.MN9]),neurones=int(len(np.unique(np.concatenate([pre[keep],post[keep]])))))
    print(json.dumps(r),flush=True); return r
if __name__=='__main__':
    jobs=[(kind,k,1.0,s) for kind in ('rel','flux') for k in (8000,4000,2000,1200,794,400,200) for s in (1,2,3)]
    with Pool(4) as pl: res=pl.map(run,jobs,chunksize=1)
    json.dump(res,open('mini.json','w'))
