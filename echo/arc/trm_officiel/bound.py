import torch, numpy as np, pickle, sys
CK='/root/.cache/huggingface/hub/models--arcprize--trm_arc_prize_verification/snapshots/55ced5dd59de74c52f53d47aa2898232b5a15b7a/arc_v1_public/step_518071'
sd=torch.load(CK,map_location='cpu'); pe=sd[[k for k in sd if k.endswith('puzzle_emb.weights')][0]]
OFF=int(sys.argv[1])-100; mu=pe[1:200000].float().mean(0)
n=torch.nn.functional.normalize(pe[OFF:int(sys.argv[2])+100].float()-mu,dim=1).numpy().astype(np.float64)
cs=np.vstack([np.zeros((1,n.shape[1])),np.cumsum(n,0)])
W=40
def score(s):
    a=s-OFF; l=(cs[a]-cs[a-W])/W; r=(cs[a+W]-cs[a])/W
    return float(l@r/np.linalg.norm(l)/np.linalg.norm(r))
D=pickle.load(open('subset_data.pkl','rb'))
items=sorted(D['idmap'].items(), key=lambda kv: kv[1]); starts=[v for k,v in items if '|||' not in k]
lo,hi=int(sys.argv[1]),int(sys.argv[2])
sc={s:score(s) for s in range(lo,hi)}
# local minima below 0.1
mins=[s for s in range(lo+1,hi-1) if sc[s]<0.12 and sc[s]<=sc[s-1] and sc[s]<=sc[s+1]]
print('ours',[s for s in starts if lo<=s<hi])
print('theirs',mins)
