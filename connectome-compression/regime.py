# Regime permanent du cerveau entier : on resout les frequences au lieu de simuler les spikes.
# Chaque neurone : entrees de Poisson (rates r_j, poids W_ij en mV), noyau exact du modele (tm=20, tau=5) ->
# moyenne mu = tau*sum(W r), variance s2 = 0.5*sum(W^2 r) ; frequence par Siegert (LIF a bruit, refractaire 2,2 ms).
import numpy as np, simlib as S, evenementiel as E, json, time, sys
from scipy import sparse
from scipy.special import erfcx
pre,post,w,sg,ids=S.load('conn.npz'); N=S.N
Wm=0.275*w*sg[pre]
WT=sparse.csr_matrix((Wm,(post,pre)),shape=(N,N)); WT2=sparse.csr_matrix((Wm**2,(post,pre)),shape=(N,N))
tm,tau,tref,vth,vr=20.0,5.0,2.2,7.0,0.0     # tensions relatives au repos
gx,gw=np.polynomial.legendre.leggauss(64)
def siegert(mu,s):
    """frequence (Hz) d'un LIF a bruit blanc equivalent ; mu,s en mV (tension relative au repos)."""
    r=np.zeros_like(mu); s=np.maximum(s,1e-6)
    a=(vr-mu)/s; b=(vth-mu)/s
    det=b<-6                      # tres sous le seuil -> 0
    sat=a>=-0.0+0 if False else (mu>vth+8*s)   # tres au-dessus -> deterministe
    m=~det&~sat
    if m.any():
        aa,bb=a[m],b[m]; h=(bb-aa)/2; c=(bb+aa)/2
        u=c[:,None]+h[:,None]*gx[None,:]
        I=(erfcx(-u)*gw[None,:]).sum(1)*h
        r[m]=1000.0/(tref+tm*np.sqrt(np.pi)*I)
    if sat.any(): r[sat]=1000.0/(tref+tm*np.log(mu[sat]/(mu[sat]-vth)))
    return r
def resoudre(hz,stim=S.SUGAR,iters=300,damp=0.5,perdu=True,verbose=False):
    r=np.zeros(N); r[stim]=hz; rs=np.zeros(N)
    for it in range(iters):
        rk=r/1000.0                                  # spikes / ms
        mu=tau*(WT@rk); s2=0.5*(WT2@rk)
        if perdu:                                    # signaux recus en periode refractaire : perdus
            f=np.clip(1-rk*tref,0,1); mu*=f; s2*=f
        new=siegert(mu,np.sqrt(s2)); new[stim]+=hz
        d=np.abs(new-r).max(); r=damp*new+(1-damp)*r
        if verbose and it%25==0: print(it,round(d,4),round(r[S.MN9],2),flush=True)
        if d<1e-3: break
    return r,it+1
if __name__=='__main__':
    W_=S.csr(pre,post,w*sg[pre]); res={}
    for hz in (25,50,75,100,150,200):
        ex=[]
        for seed in (1,2,3):
            T=10000; sp,_=E.evenementiel(W_,S.SUGAR,E.tirages(S.SUGAR,hz,T,seed),T); ex.append(sp)
        ex=np.mean(ex,0)           # spikes en 1 s = Hz moyen
        t=time.time(); r,n=resoudre(hz); dt=time.time()-t
        act=(ex>0.5)|(r>0.5)
        cor=float(np.corrcoef(ex,r)[0,1])
        res[hz]=dict(exact_mn9=float(ex[S.MN9]),regime_mn9=float(r[S.MN9]),exact_actifs=int((ex>0.5).sum()),regime_actifs=int((r>0.5).sum()),
                     exact_spikes=float(ex.sum()),regime_spikes=float(r.sum()),corr=cor,iters=n,secondes=round(dt,1))
        print(hz,json.dumps(res[hz]),flush=True)
    json.dump(res,open('regime_res.json','w'),indent=1)
