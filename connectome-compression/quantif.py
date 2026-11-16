import numpy as np, simlib as S, json
d=np.load('lois_data.npz',allow_pickle=True); P=np.load('loi_cerveau_params.npz'); out=P['out']
R=d['rates']; nt,ns,nn=int(d['n_train']),int(d['n_test']),int(d['n_noise'])
col={n:i for i,n in enumerate(out)}; Y=np.zeros((len(R),len(out)))
for k in range(len(R)):
    for n,c in zip(d['nz'][k],d['cnt'][k]):
        if n in col: Y[k,col[n]]=c
Xte,Yte=R[nt:nt+ns]/200.0,Y[nt:nt+ns]; s2=((Y[nt+ns:]-Y[:nn])**2).mean(0)/2; V=Yte.var(0); mn9=col[S.MN9]
for dd in (4,6):
    for q,dt in (("64 bits",np.float64),("16 bits",np.float16)):
        A,C,b=(P[f'{k}{dd}'].astype(dt).astype(np.float64) for k in 'ACb')
        pred=np.maximum(b+(Xte@A)@C,0); mse=((pred-Yte)**2).mean(0)
        n=A.size+C.size+b.size; octets=n*(8 if q=="64 bits" else 2)
        print(f"d={dd} {q}: part expliquable captee {1-(mse-s2).sum()/(V-s2).sum():.3f}  R2 MN9 {1-mse[mn9]/V[mn9]:.3f}  {n} nombres = {octets:,} octets -> 92 Mo / taille = {92613300/octets:,.0f} : 1")
