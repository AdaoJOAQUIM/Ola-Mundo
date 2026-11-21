import glob, os, json
from safetensors import safe_open
def share(files):
    tot=mlp=0
    for p in files:
        f=safe_open(p,'pt')
        for k in f.keys():
            n=1
            for d in f.get_slice(k).get_shape(): n*=d
            tot+=n
            if '.mlp.' in k and k.rsplit('.',2)[-2] in('gate_proj','up_proj','down_proj') and 'visual' not in k: mlp+=n
    return tot,mlp,mlp/tot
h=os.path.expanduser('~/.cache/huggingface/hub/')
out={}
out['smol']=share(glob.glob(h+'models--HuggingFaceTB--SmolLM2-135M/snapshots/*/*.safetensors'))
out['qwen']=share(glob.glob(h+'models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/*.safetensors'))
print(out); json.dump(out,open('params_out.json','w'))
