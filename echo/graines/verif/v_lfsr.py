import numpy as np, torch, json, glob, os
from safetensors import safe_open
# independent LFSR: x^16+x^14+x^13+x^11+1 Fibonacci, shift left, new bit = xor of bits 15,13,12,10
n=65535; seq=np.empty(n,np.int64); x=1
for i in range(n):
    seq[i]=x
    b=((x>>15)^(x>>13)^(x>>12)^(x>>10))&1
    x=((x<<1)|b)&0xFFFF
assert x==1 and len(set(seq.tolist()))==n
pos=np.zeros(65536,np.int64); pos[seq]=np.arange(n)
C=64
seeds=np.arange(1,65536)
idx=(pos[seeds][:,None]+1+np.arange(C)[None,:])%n
U=((seq[idx]-32768.)/32767.).astype(np.float32)          # (65535,64)
Un=torch.from_numpy(U/np.linalg.norm(U,axis=1,keepdims=True))
path=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/*/model.safetensors'))[0]
rng=np.random.RandomState(7)
f=safe_open(path,'pt')
names=[k for k in f.keys() if '.mlp.' in k and k.endswith('weight')]
blocks=[]
for k in rng.choice(len(names),2000):
    W=f.get_tensor(names[k]).float().reshape(-1); b=rng.randint(len(W)//C); blocks.append(W[b*C:(b+1)*C])
Wb=torch.stack(blocks)
def run(Wb):
    Wb=Wb/Wb.std()
    sc=(Wb@Un.T)**2; best=sc.max(1).values; e2=(Wb**2).sum(1)
    rel=(1-best/e2).clamp_min(0).sqrt()
    return rel.mean().item(), ((e2-best).sum()/e2.sum()).sqrt().item(), (best/e2).mean().item()
r_real=run(Wb); r_gauss=run(torch.randn(2000,C))
print('real',r_real,'gauss',r_gauss)
json.dump(dict(real_mean_rel=r_real[0],real_global_rel=r_real[1],real_energy_captured=r_real[2],gauss_mean_rel=r_gauss[0],gauss_global_rel=r_gauss[1],gauss_energy=r_gauss[2],theory_energy_captured_estimate=0.25),open('lfsr_out.json','w'))
