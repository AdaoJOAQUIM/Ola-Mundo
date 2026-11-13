# Miroir Python, ligne a ligne, de Simuler.pq (memes tables, memes regles), lu depuis le classeur produit.
# Sert a verifier la logique du code M contre la simulation dense ; il ne teste pas Excel lui-meme.
import numpy as np, sys, math, json, zipfile, re, html
import evenementiel as E, simlib as S
ev,eg,A=math.exp(-0.1/20.0),math.exp(-0.1/5.0),5.0/(5.0-20.0); LEV,LEG=-0.1/20.0,-0.1/5.0; SEUIL=7.0
def lire_synapses(path):
    z=zipfile.ZipFile(path); wbx=z.read('xl/workbook.xml').decode(); names=re.findall(r'<sheet[^>]*? name="([^"]+)"',wbx)
    s=z.read(f"xl/worksheets/sheet{names.index('Synapses')+1}.xml").decode()
    codes=['']*S.N; signes=[0]*S.N
    for m in re.finditer(r'<row r="\d+"[^>]*>(.*?)</row>',s,flags=re.S):
        c=re.findall(r'<c r="([A-C])\d+"[^>]*>(?:<v>([^<]*)</v>|<is><t[^>]*>([^<]*)</t></is>)',m.group(1))
        d={a:(v or t) for a,v,t in c}
        if 'A' in d and d['A'].lstrip('-').isdigit():
            i=int(d['A']); signes[i]=int(d.get('B','0')); codes[i]=html.unescape(d.get('C',''))
    return codes,signes
def sorties(codes,signes,i):
    s=codes[i]
    if not s: return []
    c=[ord(ch)-19968 for ch in s]; m=len(c)//2; g=signes[i]*0.275; out=[]; cb=0
    for k in range(m):
        cb+=c[2*k]; w=c[2*k+1]
        if w>0: out.append((cb,w*g))
    return out
def prochain(X,G):
    f=lambda k: X*ev**k+A*G*eg**k-SEUIL
    if f(1)>0: return 1
    ratio=-1 if X==0 else -(A*G*LEG)/(X*LEV)
    ks=math.log(ratio)/(LEV-LEG) if ratio>0 else 0
    if ks<=1: return 0
    bas,haut=math.floor(ks),math.ceil(ks); top=bas if f(bas)>=f(haut) else haut
    if f(top)<=0: return 0
    lo,hi=1,top
    while hi-lo>1:
        mid=(lo+hi)//2
        if f(mid)>0: hi=mid
        else: lo=mid
    return hi
def simuler(codes,signes,stim,hz,duree,graine):
    T=math.floor(duree/0.1); kicks=E.tirages(stim,hz,T,graine); stimset=set(stim)
    St={}; Q=[[] for _ in range(18)]; out=[]
    for n in range(T):
        sIds=[i for i,r in St.items() if r['ts']==n]; h=Q[0]; kn=kicks[n]
        tous={}
        for src in h:
            for j,x in sorties(codes,signes,src):
                t=tous.setdefault(j,[0.0,0.0]); t[0]+=x
        for j in kn:
            t=tous.setdefault(j,[0.0,0.0]); t[1]+=0.275*250
        sset=set(sIds); nouv={}
        for j,(x,kick) in tous.items():
            if j in sset: continue
            r=St.get(j); X0,G0,tl0,last0=(r['X'],r['G'],r['tl'],r['last']) if r else (0,0,n,-1000000)
            libre=(j in stimset) or (n-last0>=22); xin=x if libre else 0; d=n-tl0
            X1=X0*ev**d-A*xin+kick; G1=G0*eg**d+xin; k=prochain(X1,G1)
            if libre: nouv[j]=dict(X=X1,G=G1,tl=n,last=last0,ts=n+k if k>0 else -1)
        for i in sIds: nouv[i]=dict(X=0,G=0,tl=n,last=n,ts=-1)
        St.update(nouv); Q=Q[1:]+[sIds]; out+= [(n,i) for i in sIds]
    return out
if __name__=='__main__':
    codes,signes=lire_synapses(sys.argv[1])
    pre,post,w,sg,ids=S.load('conn.npz'); W=S.csr(pre,post,w*sg[pre])
    nconn=sum(len(sorties(codes,signes,i)) for i in range(S.N)); print("connexions relues du classeur",nconn,"caracteres",sum(len(c) for c in codes))
    res={}
    for duree,graine in ((100,1),(300,1),(1000,1),(1000,2),(1000,3)):
        sp=simuler(codes,signes,S.SUGAR,100,duree,graine); cnt=np.bincount([i for _,i in sp],minlength=S.N)
        T=int(duree/0.1); ref=E.dense(W,S.SUGAR,E.tirages(S.SUGAR,100,T,graine),T)
        res[f"{duree}_{graine}"]=dict(duree=duree,graine=graine,spikes=int(cnt.sum()),actifs=int((cnt>0).sum()),mn9=int(cnt[S.MN9]),ecart=int((cnt!=ref).sum()))
        print(res[f"{duree}_{graine}"],flush=True)
    json.dump(res,open('attendu.json','w'))
