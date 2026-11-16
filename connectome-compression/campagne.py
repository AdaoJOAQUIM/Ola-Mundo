# Campagne d'experiences sur le cerveau entier exact : chaque neurone sucre recoit sa propre frequence (0 a 200 Hz).
# On enregistre MN9 et l'activite complete, pour chercher ensuite une loi courte.
import numpy as np, simlib as S, evenementiel as E, time, sys
from multiprocessing import Pool
N=S.N; T=10000; dt=0.1
def init():
    global W
    pre,post,w,sg,ids=S.load('conn.npz'); W=S.csr(pre,post,w*sg[pre])
def tirages_pr(rates,T,graine):
    x=graine; p=np.asarray(rates)*dt*1e-3; out=[[] for _ in range(T)]; stim=S.SUGAR
    for n in range(T):
        for i in range(len(stim)):
            x=(x*48271)%2147483647
            if x/2147483647<p[i]: out[n].append(stim[i])
    return out
def run(job):
    k,rates,graine=job; t=time.time()
    sp,_=E.evenementiel(W,S.SUGAR,tirages_pr(rates,T,graine),T)
    nz=np.nonzero(sp)[0]
    return k,int(sp[S.MN9]),int(sp.sum()),int(len(nz)),nz.astype(np.int32),sp[nz].astype(np.int16),round(time.time()-t,1)
if __name__=='__main__':
    rng=np.random.default_rng(7); n_train,n_test,n_noise=300,100,40
    R=rng.uniform(0,200,(n_train+n_test,20))*(rng.random((n_train+n_test,20))>0.3)
    jobs=[(k,R[k],1000+k) for k in range(n_train+n_test)]
    jobs+=[(n_train+n_test+k,R[k],5000+k) for k in range(n_noise)]          # memes motifs, autre graine : plancher de bruit
    out=[]; t0=time.time()
    with Pool(4,initializer=init) as p:
        for r in p.imap_unordered(run,jobs,chunksize=1):
            out.append(r)
            if len(out)%40==0: print(len(out),"/",len(jobs),round(time.time()-t0),"s",flush=True)
    out.sort(key=lambda r:r[0])
    rates=np.vstack([R]+[R[:n_noise]])
    np.savez_compressed('lois_data.npz',rates=rates,mn9=np.array([r[1] for r in out]),spikes=np.array([r[2] for r in out]),actifs=np.array([r[3] for r in out]),
        nz=np.array([r[4] for r in out],dtype=object),cnt=np.array([r[5] for r in out],dtype=object),n_train=n_train,n_test=n_test,n_noise=n_noise,secondes=np.array([r[6] for r in out]))
    print("fini",round(time.time()-t0),"s")
