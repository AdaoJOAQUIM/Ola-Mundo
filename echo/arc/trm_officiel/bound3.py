import torch, numpy as np, pickle, sys
CK='/root/.cache/huggingface/hub/models--arcprize--trm_arc_prize_verification/snapshots/55ced5dd59de74c52f53d47aa2898232b5a15b7a/arc_v1_public/step_518071'
sd=torch.load(CK,map_location='cpu'); pe=sd[[k for k in sd if k.endswith('puzzle_emb.weights')][0]]
OFF=900; mu=pe[1:200000].float().mean(0)
n=torch.nn.functional.normalize(pe[OFF:397000].float()-mu,dim=1).numpy().astype(np.float64)
cs=np.vstack([np.zeros((1,n.shape[1])),np.cumsum(n,0)])
W=40
def score(s):
    a=s-OFF; l=(cs[a]-cs[a-W])/W; r=(cs[a+W]-cs[a])/W
    return float(l@r/np.linalg.norm(l)/np.linalg.norm(r))
D=pickle.load(open('subset_data.pkl','rb'))
items=sorted(D['idmap'].items(), key=lambda kv: kv[1]); starts=[v for k,v in items if '|||' not in k]


for lo in range(1000,396000,15000):
    hi=lo+15000
    mins=set(s for s in range(lo+1,hi-1) if score(s)<0.15 and score(s)<=score(s-1) and score(s)<=score(s+1))
    ours=[s for s in starts if lo<=s<hi]
    ok=sum(any(abs(s-m)<=1 for m in mins) for s in ours)
    print(lo,len(ours),ok,flush=True)
