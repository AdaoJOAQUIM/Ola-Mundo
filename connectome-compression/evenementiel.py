# Moteur evenementiel du cerveau entier : aucun neurone n'est calcule tant qu'il ne recoit rien.
# Entre deux evenements, la tension suit une forme fermee ; on predit le pas exact du prochain spike.
import numpy as np, simlib as S, time, sys
from scipy import sparse
N=S.N; dt,tm,tau,v0,vth,DELAI=0.1,20.0,5.0,-52.0,-45.0,18
A=tau/(tau-tm); ev=np.exp(-dt/tm); eg=np.exp(-dt/tau); SEUIL=vth-v0; KICK=0.275*250
LEV,LEG=np.log(ev),np.log(eg)

def tirages(stim,hz,T,graine):
    """Generateur de Park-Miller (portable en M : entiers exacts en double precision)."""
    x=graine; out=[[] for _ in range(T)]; p=hz*dt*1e-3
    for n in range(T):
        for i in stim:
            x=(x*48271)%2147483647
            if x/2147483647<p: out[n].append(i)
    return out

def dense(W,stim,kicks,T):
    rfc=np.full(N,22); rfc[stim]=0; v=np.full(N,v0); g=np.zeros(N); dernier=np.full(N,-10**6)
    file=[None]*(T+DELAI+1); spikes=np.zeros(N,np.int64)
    for n in range(T):
        libre=(n-dernier)>=rfc
        v=np.where(libre,v0+(v-v0-A*g)*ev+A*g*eg,v); g=np.where(libre,g*eg,g)
        s=np.nonzero((v>vth)&libre)[0]; h=file[n]
        if h is not None and len(h): g+=np.where(libre,np.asarray(W[h].sum(axis=0)).ravel(),0)
        if kicks[n]: v[kicks[n]]+=KICK
        if len(s): v[s]=v0; g[s]=0; dernier[s]=n; spikes[s]+=1; file[n+DELAI]=s
    return spikes

def prochain(X,G):
    """Premier k>=1 tel que X*ev^k + A*G*eg^k > SEUIL, ou 0 s'il n'y en a pas (vectorise)."""
    f=lambda k: X*ev**k+A*G*eg**k-SEUIL
    res=np.zeros(len(X),np.int64); res[f(1)>0]=1
    with np.errstate(all='ignore'):
        ratio=-(A*G*LEG)/(X*LEV); ks=np.log(ratio)/(LEV-LEG)
    cand=(res==0)&np.isfinite(ks)&(ks>1)
    if cand.any():
        i=np.nonzero(cand)[0]; hi=np.ceil(ks[i]).astype(np.int64); lo_peak=np.floor(ks[i]).astype(np.int64)
        Xi,Gi=X[i],G[i]; fi=lambda k: Xi*ev**k+A*Gi*eg**k-SEUIL
        top=np.where(fi(lo_peak)>=fi(hi),lo_peak,hi); ok=fi(top)>0
        lo=np.ones(len(i),np.int64); hi=top.copy()          # f(lo)<=0 < f(hi), f croissante sur [1, pic]
        while True:
            act=ok&(hi-lo>1)
            if not act.any(): break
            mid=(lo+hi)//2; up=fi(mid)>0
            hi=np.where(act&up,mid,hi); lo=np.where(act&~up,mid,lo)
        res[i[ok]]=hi[ok]
    return res

def evenementiel(Wcsr,stim,kicks,T):
    rfc=np.full(N,22); rfc[stim]=0
    X=np.zeros(N); G=np.zeros(N); tl=np.zeros(N,np.int64); last=np.full(N,-10**6); ver=np.zeros(N,np.int64)
    cal={}; arr={}; spikes=np.zeros(N,np.int64); stats=dict(evenements=0,predictions=0,eveilles=np.zeros(N,bool))
    for n in range(T):
        s=np.array([],np.int64)
        if n in cal:
            c=np.array(cal.pop(n)); ok=ver[c[:,0]]==c[:,1]; s=np.unique(c[ok,0])
        touche=[]; dg=None
        if n in arr:
            h=arr.pop(n); sub=Wcsr[h]; dgv=np.asarray(sub.sum(axis=0)).ravel(); t=np.unique(sub.indices)
            stats['evenements']+=sub.nnz; t=t[(n-last[t])>=rfc[t]]; touche.append(t); dg=dgv
        k=np.array(kicks[n],np.int64)
        if len(k): touche.append(k)
        if touche:
            t=np.setdiff1d(np.unique(np.concatenate(touche)),s)
            if len(t):
                d=n-tl[t]; X[t]*=ev**d; G[t]*=eg**d; tl[t]=n
                if dg is not None:
                    acc=(n-last[t])>=rfc[t]; x=np.where(acc,dg[t],0.0); G[t]+=x; X[t]-=A*x
                if len(k):
                    kk=np.setdiff1d(k,s); X[kk]+=KICK
                ver[t]+=1; stats['eveilles'][t]=True
                kp=prochain(X[t],G[t]); m=kp>0; stats['predictions']+=int(m.sum())
                for j,kk_,vv in zip(t[m].tolist(),kp[m].tolist(),ver[t[m]].tolist()): cal.setdefault(n+kk_,[]).append((j,vv))
        if len(s):
            X[s]=0; G[s]=0; tl[s]=n; last[s]=n; ver[s]+=1; spikes[s]+=1; arr[n+DELAI]=s
    stats['eveilles']=int(stats['eveilles'].sum()); return spikes,stats

if __name__=='__main__':
    pre,post,w,sg,ids=S.load('conn.npz'); W=S.csr(pre,post,w*sg[pre]); T=10000
    for graine in (1,2,3):
        kicks=tirages(S.SUGAR,100,T,graine)
        t=time.time(); a=dense(W,S.SUGAR,kicks,T); ta=time.time()-t
        t=time.time(); b,st=evenementiel(W,S.SUGAR,kicks,T); tb=time.time()-t
        print(f"graine {graine}: dense {a.sum()} spikes, MN9 {a[S.MN9]}, {ta:.1f}s | evenementiel {b.sum()} spikes, MN9 {b[S.MN9]}, {tb:.1f}s | neurones en ecart {(a!=b).sum()} | {st}",flush=True)
