# La loi peut-elle se deduire du cablage seul ? Influence de chaque entree sur chaque sortie = somme signee des chemins
# de longueur 1 a 4 (produits de poids en mV). Ce calcul n'est que jointures + sommes : exactement ce que DAX sait faire.
import numpy as np, simlib as S, json
from scipy import sparse
pre,post,w,sg,ids=S.load('conn.npz'); N=S.N
Wsp=sparse.csr_matrix((0.275*w*sg[pre],(pre,post)),shape=(N,N))
d=np.load('lois_data.npz',allow_pickle=True); P=np.load('loi_cerveau_params.npz'); out=P['out']
R=d['rates']; nt,ns,nn=int(d['n_train']),int(d['n_test']),int(d['n_noise'])
col={n:i for i,n in enumerate(out)}; Y=np.zeros((len(R),len(out)))
for k in range(len(R)):
    for n,c in zip(d['nz'][k],d['cnt'][k]):
        if n in col: Y[k,col[n]]=c
X=R/200.0; Xtr,Ytr,Xte,Yte=X[:nt],Y[:nt],X[nt:nt+ns],Y[nt:nt+ns]
s2=((Y[nt+ns:]-Y[:nn])**2).mean(0)/2; V=Yte.var(0); mn9=col[S.MN9]
act=np.unique(np.concatenate(list(d['nz'])))
def chemins(Wm,L=4):
    E=sparse.csr_matrix((np.ones(20),(np.arange(20),np.array(S.SUGAR))),shape=(20,N))
    T=[]; cur=E
    for l in range(L):
        cur=cur@Wm; T.append(np.asarray(cur[:,out].todense()))
    return T
mask=np.zeros(N); mask[act]=1; Dm=sparse.diags(mask)
variants={"cablage complet":chemins(Wsp),"cablage limite aux 480 neurones actifs":chemins(Dm@Wsp@Dm)}
Xa=np.hstack([Xtr,np.ones((nt,1))]); B=np.linalg.solve(Xa.T@Xa+1e-2*np.eye(21),Xa.T@Ytr)[:20]
def bilan(name,pred,params):
    mse=((pred-Yte)**2).mean(0); feve=1-(mse-s2).sum()/(V-s2).sum(); r2m=1-mse[mn9]/V[mn9]
    print(f"   {name:58} part expliquable captee {feve:6.3f}  R2 MN9 {r2m:6.3f}  ({params})",flush=True); return float(feve),float(r2m)
res={}
for vn,T in variants.items():
    print(vn)
    for l,t in enumerate(T,1):
        c=np.corrcoef(t.ravel(),B.ravel())[0,1]; cm=np.corrcoef(t[:,mn9],B[:,mn9])[0,1]
        print(f"   chemins de longueur {l} : correlation avec la loi ajustee {c:6.3f} (toutes les paires)  {cm:6.3f} (vers MN9)")
    # loi structurelle : influence = combinaison des 4 longueurs (4 nombres globaux), puis par neurone un gain et une base
    F=np.stack([t.ravel() for t in T],1); alpha=np.linalg.lstsq(F,B.ravel(),rcond=None)[0]; Bs=(F@alpha).reshape(B.shape)
    print("   melange des longueurs (4 nombres globaux) :",np.round(alpha,4).tolist(),"-> correlation",round(float(np.corrcoef(Bs.ravel(),B.ravel())[0,1]),3))
    Ztr,Zte=Xtr@Bs,Xte@Bs; g=np.zeros(len(out)); b0=np.zeros(len(out))
    for k in range(len(out)):
        A_=np.vstack([Ztr[:,k],np.ones(nt)]).T; g[k],b0[k]=np.linalg.lstsq(A_,Ytr[:,k],rcond=None)[0]
    res[vn]=bilan("cablage + 4 nombres globaux + gain et base par neurone",np.maximum(Zte*g+b0,0),f"{4+2*len(out)} nombres + le cablage")
    # sans aucun ajustement par neurone : base = moyenne d'entrainement, gain global unique
    gg=np.linalg.lstsq((Ztr-Ztr.mean(0)).ravel()[:,None],(Ytr-Ytr.mean(0)).ravel(),rcond=None)[0][0]
    bilan("cablage + 5 nombres globaux (aucun reglage par neurone)",np.maximum(Ytr.mean(0)+gg*(Zte-Ztr.mean(0)),0),"5 nombres + moyennes")
json.dump(res,open('structure_res.json','w'))
