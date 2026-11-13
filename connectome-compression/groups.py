# Groupes de neurones decouverts dans le cablage : SVD du connectome puis k-means, separes par signe.
import numpy as np, simlib as S, time
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans
pre,post,w,sg,ids=S.load('conn.npz'); N=S.N
A=sparse.csr_matrix((np.log1p(w),(pre,post)),shape=(N,N))
t=time.time(); U,s,Vt=svds(A,k=64,random_state=0); print("svd",round(time.time()-t),"s")
F=np.hstack([U*np.sqrt(s),Vt.T*np.sqrt(s)])
outd=np.log1p(np.bincount(pre,w,N)); ind=np.log1p(np.bincount(post,w,N))
F=np.hstack([F/ (np.linalg.norm(F,axis=1,keepdims=True)+1e-9), (outd/outd.max())[:,None],(ind/ind.max())[:,None]])
np.save('feat.npy',F)
out={}
for K in [30,100,300,1000,3000,10000]:
    t=time.time(); lab=np.zeros(N,np.int64); base=0
    for sv in [1,-1,0]:
        idx=np.nonzero(sg==sv)[0]; k=max(1,int(round(K*len(idx)/N)))
        km=MiniBatchKMeans(k,batch_size=8192,n_init=1,random_state=0,max_iter=60).fit(F[idx])
        lab[idx]=base+km.labels_; base+=k
    u,lab=np.unique(lab,return_inverse=True); out[f'K{K}']=lab; print(K,"->",len(u),"groupes",round(time.time()-t),"s",flush=True)
rng=np.random.default_rng(0)
for K in [1000]:
    lab=np.zeros(N,np.int64); base=0
    for sv in [1,-1,0]:
        idx=np.nonzero(sg==sv)[0]; k=max(1,int(round(K*len(idx)/N))); lab[idx]=base+rng.integers(0,k,len(idx)); base+=k
    out[f'R{K}']=np.unique(lab,return_inverse=True)[1]
np.savez_compressed('groups.npz',**out)
