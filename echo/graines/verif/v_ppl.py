import os, json, sys, time, numpy as np, torch
os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(4)
txt = open('/home/user/Ola-Mundo/echo/validation.txt', encoding='utf-8').read()[:50000]
name='HuggingFaceTB/SmolLM2-135M'
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,dtype=torch.float32).eval()
ids=tok(txt,return_tensors='pt').input_ids[0]
def ppl():
    nf=len(ids)//512; nll=0.;c=0
    with torch.no_grad():
        for i in range(nf):
            x=ids[i*512:(i+1)*512][None]
            lg=m(x).logits[0,:-1].float()
            nll+=torch.nn.functional.cross_entropy(lg,x[0,1:],reduction='sum').item(); c+=511
    return float(np.exp(nll/c)),c
p0,c=ppl(); print('baseline',p0,c,flush=True)
mats=[(n,mod) for n,mod in m.named_modules() if isinstance(mod,torch.nn.Linear) and '.mlp.' in n]
print(len(mats))
# my RTN: symmetric, groups of 32 along flattened weight, 4 bit, fp16 scale per group (honest upper bound check variant) 
def rtn_fp16(W,G=32):
    g=W.reshape(-1,G); s=(g.abs().amax(1,keepdim=True)/7).clamp_min(1e-12).half().float()
    return (torch.clamp(torch.round(g/s),-8,7)*s).reshape(W.shape)
orig={n:mod.weight.data.clone() for n,mod in mats}
num=den=0
for n,mod in mats:
    W=orig[n]; Wh=rtn_fp16(W); mod.weight.data.copy_(Wh); num+=((W-Wh)**2).sum().item(); den+=(W**2).sum().item()
p1,_=ppl(); print('rtn4 fp16-scale',p1,p1/p0,'err',(num/den)**.5, 'bpw',4+16/32)
json.dump(dict(baseline=p0,tokens=c,rtn4_ppl=p1,rtn4_ratio=p1/p0,rtn4_err=(num/den)**.5,rtn4_bpw_fp16scale=4.5,n_mats=len(mats)),open('ppl_out.json','w'))
