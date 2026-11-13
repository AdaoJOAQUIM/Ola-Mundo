# Echelle de compression du connectome : chaque modele est une description plus courte du cablage,
# rejouee dans le meme simulateur (sucre -> MN9). Taille = nombre de valeurs stockees.
import numpy as np, simlib as S, json, sys, time
from scipy import sparse
from scipy.sparse.linalg import svds
from multiprocessing import Pool
pre,post,w,sg,ids=S.load('conn.npz'); N=S.N; ws=w*sg[pre]
_g=np.load('groups.npz'); G={k:_g[k] for k in _g.files}; ORIG=3*len(pre)

def blocks(lab,mask=None):
    K=lab.max()+1; p,q,x=(pre,post,ws) if mask is None else (pre[mask],post[mask],ws[mask])
    key=lab[p]*K+lab[q]; u,inv=np.unique(key,return_inverse=True)
    tot=np.bincount(inv,x); cnt=np.bincount(inv); return K,u//K,u%K,tot,cnt,inv

def build(spec):
    kind=spec[0]
    if kind=='exact': W=S.csr(pre,post,ws); return S.exact_apply(W),ORIG
    if kind=='prune':
        m=w>=spec[1]; W=S.csr(pre[m],post[m],ws[m]); return S.exact_apply(W),3*int(m.sum())
    if kind in('mf','hyb'):   # champ moyen : chaque neurone du groupe b recoit S_ab/(n_a n_b) par spike d'un neurone de a
        lab=G[spec[1]]; mask=None; Wk=None; size=N
        if kind=='hyb':
            keep=w>=spec[2]; mask=~keep; Wk=S.csr(pre[keep],post[keep],ws[keep]); size+=3*int(keep.sum())
        K,a,b,tot,cnt,_=blocks(lab,mask); n=np.bincount(lab,minlength=K)
        B=sparse.csr_matrix((0.275*tot/(n[a]*n[b]),(a,b)),shape=(K,K)); size+=min(K*K,3*len(a))
        def ap(h):
            ug,c=np.unique(lab[h],return_counts=True)
            r=(sparse.csr_matrix((c.astype(float),(np.zeros(len(ug),int),ug)),shape=(1,K))@B).toarray().ravel()[lab]
            return r if Wk is None else r+np.asarray(Wk[h].sum(axis=0)).ravel()
        return ap,size
    if kind=='sbm':  # cablage regenere : meme nombre de connexions par paire de groupes, partenaires tires au hasard (graine), poids moyen du bloc
        lab=G[spec[1]]; rng=np.random.default_rng(spec[2]); K,a,b,tot,cnt,inv=blocks(lab)
        order=np.argsort(lab,kind='stable'); n=np.bincount(lab,minlength=K); start=np.concatenate([[0],np.cumsum(n)[:-1]])
        p2=order[start[lab[pre]]+(rng.random(len(pre))*n[lab[pre]]).astype(np.int64)]
        q2=order[start[lab[post]]+(rng.random(len(pre))*n[lab[post]]).astype(np.int64)]
        W=S.csr(p2,q2,(tot/cnt)[inv]); return S.exact_apply(W),N+min(2*K*K,4*len(a))
    if kind=='lr':   # rang faible : W ~ U S V^T
        r=spec[1]; W=sparse.csr_matrix((ws*0.275,(pre,post)),shape=(N,N)); U,s,Vt=svds(W,k=r,random_state=0)
        US=U*s; return (lambda h: US[h].sum(axis=0)@Vt),2*N*r+r
    raise ValueError(kind)

REF={}
def run(job):
    name,spec,seed,hz=job; t=time.time()
    ap,size=build(spec); sp=S.sim(ap,hz=hz,seed=seed)
    size=int(size); r=dict(name=name,spec=[str(x) for x in spec],seed=seed,hz=hz,size=size,ratio=ORIG/size,t=round(time.time()-t))
    if sp is None: r.update(emballement=True)
    else:
        np.save(f'out/{name}_s{seed}_h{hz}.npy',sp.astype(np.int32))
        r.update(mn9=int(sp[S.MN9]),spikes=int(sp.sum()),actifs=int((sp>0).sum()))
    print(json.dumps(r),flush=True); return r

if __name__=='__main__':
    import os; os.makedirs('out',exist_ok=True)
    models=[('exact',('exact',))]
    models+=[(f'prune{t}',('prune',t)) for t in (2,5,10,20,50)]
    models+=[(f'mf_{k}',('mf',k)) for k in ('K30','K100','K300','K1000','K3000','K10000','R1000')]
    models+=[(f'sbm_{k}',('sbm',k,0)) for k in ('K300','K1000','K3000','K10000','R1000')]
    models+=[(f'lr{r}',('lr',r)) for r in (1,4,16,64)]
    models+=[(f'hyb_{k}_t{t}',('hyb',k,t)) for k in ('K1000','K10000') for t in (10,20,50)]
    done={(json.loads(l)['name'],json.loads(l)['seed']) for l in open('all.jsonl') if l.startswith('{')}
    jobs=[(n,s,seed,100) for n,s in models for seed in (1,2,3) if (n,seed) not in done]
    with Pool(4) as p: res=p.map(run,jobs,chunksize=1)
    pass
