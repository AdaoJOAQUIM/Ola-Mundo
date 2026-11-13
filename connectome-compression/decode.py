import zipfile,sys,re,numpy as np
z=zipfile.ZipFile(sys.argv[1]); s=z.read('xl/worksheets/sheet8.xml').decode()
N=138639
pat=re.compile(r'<c r="A(\d+)"><v>(\d+)</v></c><c r="B\d+" t="inlineStr"><is><t>(\d+)</t></is></c><c r="C\d+"><v>(-?\d+)</v></c>(?:<c r="D\d+" t="inlineStr"><is><t>([^<]*)</t></is></c>)?')
ids=np.zeros(N,np.int64); sg=np.zeros(N,np.int8); pre=[];post=[];w=[]; n=0
for m in pat.finditer(s):
    i=int(m.group(2)); ids[i]=int(m.group(3)); sg[i]=int(m.group(4)); n+=1
    t=m.group(5)
    if t:
        u=np.frombuffer(t.encode('utf-16-le'),np.uint16).astype(np.int64)
        assert len(u)%3==0,(i,len(u))
        src=(u[0::3]-19968)*373+(u[1::3]-19968); ww=u[2::3]-24576
        pre.append(src); post.append(np.full(len(src),i)); w.append(ww)
pre=np.concatenate(pre);post=np.concatenate(post);w=np.concatenate(w)
print("rows",n,"connexions",len(pre),"synapses",int(w.sum()),"wmax",w.max(),"wmin",w.min(),"src range",pre.min(),pre.max())
print("exc syn",int(w[sg[pre]==1].sum()),"inh syn",int(w[sg[pre]==-1].sum()),"signs",np.unique(sg,return_counts=True))
np.savez_compressed(sys.argv[2],pre=pre.astype(np.int32),post=post.astype(np.int32),w=w.astype(np.int32),sg=sg,ids=ids)
