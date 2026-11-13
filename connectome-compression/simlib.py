import numpy as np
from scipy import sparse
N=138639; MN9=138332
SUGAR=[69093,97602,122795,124291,29281,100605,110469,51107,49584,129730,126873,28825,126600,32863,108426,111357,14842,90589,92298,12494]
def load(path):
    d=np.load(path); return d['pre'].astype(np.int64),d['post'].astype(np.int64),d['w'].astype(np.float64),d['sg'].astype(np.float64),d['ids']
def csr(pre,post,wsigned):
    return sparse.csr_matrix((wsigned*0.275,(pre,post)),shape=(N,N))
def exact_apply(W):
    return lambda h: np.asarray(W[h].sum(axis=0)).ravel()
def sim(apply, stim=SUGAR, hz=100, dur_ms=1000, seed=1, cap=400000):
    """Modele LIF de Shiu et al. (ordre exact du code de la notice). apply(h)->courant recu (mV) par chaque neurone."""
    dt,tm,tau,v0,vth,delai=0.1,20.0,5.0,-52.0,-45.0,18
    A=tau/(tau-tm); ev=np.exp(-dt/tm); eg=np.exp(-dt/tau); T=int(dur_ms/dt)
    rng=np.random.default_rng(seed); poi=np.array(stim); rfc=np.full(N,22); rfc[poi]=0
    v=np.full(N,v0); g=np.zeros(N); dernier=np.full(N,-10**6)
    file=[None]*(T+delai+1); spikes=np.zeros(N,np.int64)
    for n in range(T):
        libre=(n-dernier)>=rfc
        v=np.where(libre,v0+(v-v0-A*g)*ev+A*g*eg,v); g=np.where(libre,g*eg,g)
        s=np.nonzero((v>vth)&libre)[0]
        h=file[n]
        if h is not None and len(h): g+=np.where(libre,apply(h),0)
        p=poi[rng.random(len(poi))<hz*dt*1e-3]
        if len(p): v[p]+=0.275*250
        if len(s):
            v[s]=v0; g[s]=0; dernier[s]=n; spikes[s]+=1; file[n+delai]=s
            if spikes.sum()>cap: return None
    return spikes
