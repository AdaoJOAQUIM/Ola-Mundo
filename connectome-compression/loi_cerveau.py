# Loi du cerveau entier pour la famille d'entrees "sucre" : toutes les sorties a la fois, via un petit nombre de
# variables cachees partagees. y_k = max(0, b_k + c_k . z), z = A r  (A : 20 x d, commun a tous les neurones).
import numpy as np, simlib as S, json
d=np.load('lois_data.npz',allow_pickle=True)
R=d['rates']; nt,ns,nn=int(d['n_train']),int(d['n_test']),int(d['n_noise']); runs=len(R)
allidx=np.unique(np.concatenate(list(d['nz'])))
out=np.setdiff1d(allidx,np.array(S.SUGAR))                 # sorties : tous les neurones qui tirent au moins une fois, hors entrees
col={n:i for i,n in enumerate(out)}; M=len(out)
Y=np.zeros((runs,M))
for k in range(runs):
    for n,c in zip(d['nz'][k],d['cnt'][k]):
        if n in col: Y[k,col[n]]=c
print("neurones qui tirent au moins une fois :",len(allidx),"dont sorties (hors 20 entrees) :",M,"| jamais actifs :",S.N-len(allidx))
X=R/200.0; Xtr,Ytr=X[:nt],Y[:nt]; Xte,Yte=X[nt:nt+ns],Y[nt:nt+ns]
s2=((Y[nt+ns:]-Y[:nn])**2).mean(0)/2                          # bruit par neurone (memes motifs, autre graine)
V=Yte.var(0); mn9=col[S.MN9]
def bilan(name,P,params):
    mse=((P-Yte)**2).mean(0)
    feve=1-(mse-s2).sum()/(V-s2).sum(); r2=1-mse.sum()/V.sum(); plaf=1-s2.sum()/V.sum()
    r2m=1-mse[mn9]/V[mn9]
    print(f"{name:34} R2 global {r2:6.3f} (plafond {plaf:5.3f})  part expliquable captee {feve:6.3f}  R2 MN9 {r2m:6.3f}  {params} nombres",flush=True)
    return dict(name=name,r2=float(r2),plafond=float(plaf),feve=float(feve),r2_mn9=float(r2m),params=int(params))
res=[bilan("moyenne par neurone",np.tile(Ytr.mean(0),(ns,1)),M)]
# lineaire complet par neurone (ridge)
Xa=np.hstack([Xtr,np.ones((nt,1))]); B=np.linalg.solve(Xa.T@Xa+1e-2*np.eye(21),Xa.T@Ytr)
res.append(bilan("lineaire, 21 nombres par neurone",np.hstack([Xte,np.ones((ns,1))])@B,21*M))
def entraine(dd,iters=6000,lr=0.01,seed=0):
    rng=np.random.default_rng(seed)
    U,sv,Vt=np.linalg.svd(Xtr@B[:20],full_matrices=False)       # depart : regression de rang reduit
    A=np.linalg.lstsq(Xtr,U[:,:dd]*sv[:dd],rcond=None)[0]; C=Vt[:dd].copy(); b=B[20].copy()
    P={'A':A,'C':C,'b':b}; m={k:np.zeros_like(v) for k,v in P.items()}; v_={k:np.zeros_like(v) for k,v in P.items()}
    for t in range(1,iters+1):
        Z=Xtr@P['A']; H=P['b']+Z@P['C']; act=H>0; E=(np.maximum(H,0)-Ytr)*act/len(Xtr)
        g={'b':E.sum(0),'C':Z.T@E,'A':Xtr.T@(E@P['C'].T)}
        for k in P:
            m[k]=0.9*m[k]+0.1*g[k]; v_[k]=0.999*v_[k]+0.001*g[k]**2
            P[k]-=lr*(m[k]/(1-0.9**t))/(np.sqrt(v_[k]/(1-0.999**t))+1e-8)
    return P
best={}
for dd in (1,2,3,4,6,8):
    P=entraine(dd); pred=np.maximum(P['b']+(Xte@P['A'])@P['C'],0)
    res.append(bilan(f"loi a {dd} variable(s) cachee(s)",pred,20*dd+dd*M+M)); best[dd]=P
np.savez('loi_cerveau_params.npz',out=out,**{f'{k}{dd}':v for dd,P in best.items() for k,v in P.items()})
json.dump(dict(res=res,M=int(M),actifs=int(len(allidx))),open('loi_cerveau_res.json','w'),indent=1)
