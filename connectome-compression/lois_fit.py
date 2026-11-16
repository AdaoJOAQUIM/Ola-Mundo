import numpy as np, json
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
d=np.load('lois_data.npz',allow_pickle=True); R=d['rates']; y=d['mn9'].astype(float); nt,ns,nn=int(d['n_train']),int(d['n_test']),int(d['n_noise'])
Rtr,ytr=R[:nt],y[:nt]; Rte,yte=R[nt:nt+ns],y[nt:nt+ns]
print("MN9 : moyenne",y[:nt+ns].mean().round(1),"ecart-type",y[:nt+ns].std().round(1),"max",y.max(), " | motifs ou MN9=0 :",int((y[:nt+ns]==0).sum()),"/",nt+ns)
noise=np.sqrt(np.mean((y[nt+ns:]-y[:nn])**2)/2); print(f"plancher de bruit (meme motif, autre graine) : {noise:.1f} Hz d'ecart-type")
def score(name,pred,size):
    r2=1-((pred-yte)**2).sum()/((yte-yte.mean())**2).sum(); mae=np.abs(pred-yte).mean()
    print(f"{name:46} R2 test {r2:6.3f}   erreur moyenne {mae:5.1f} Hz   {size}"); return dict(name=name,r2=float(r2),mae=float(mae),size=size)
res=[]
res.append(score("constante (moyenne)",np.full(ns,ytr.mean()),"1 nombre"))
m=Rtr.mean(1); iso=IsotonicRegression(out_of_bounds='clip').fit(m,ytr); res.append(score("loi 1 : courbe dose-reponse de la frequence moyenne",iso.predict(Rte.mean(1)),"1 courbe monotone"))
# loi 2 : indice a poids par neurone + courbe monotone (modele a un indice), ajuste par alternance
a=Ridge(alpha=1.0).fit(Rtr,ytr).coef_; 
for _ in range(15):
    z=Rtr@a; iso2=IsotonicRegression(out_of_bounds='clip').fit(z,ytr); g=iso2.predict(z)
    # direction de remontee : regression de y sur z autour de la pente locale
    sl=np.gradient(g,z+1e-9*np.arange(len(z))) if False else None
    a=Ridge(alpha=1.0).fit(Rtr,ytr*0.5+g*0.5).coef_
z=Rtr@a; iso2=IsotonicRegression(out_of_bounds='clip').fit(z,ytr); res.append(score("loi 2 : 20 poids + courbe monotone (un indice)",iso2.predict(Rte@a),"20 poids + 1 courbe"))
# loi 3 : seuil lineaire max(0, b + sum a_i r_i), ajuste par descente de gradient
rng=np.random.default_rng(0); best=None
for essai in range(8):
    w_=rng.normal(0,0.02,20); b_=rng.normal(0,1); lr=2e-7
    X=Rtr; 
    for it in range(40000):
        zz=b_+X@w_; p=np.maximum(zz,0); err=p-ytr; act=(zz>0)
        gw=(X*(err*act)[:,None]).mean(0); gb=(err*act).mean()
        w_-=lr*gw*50; b_-=lr*gb*50
    loss=((np.maximum(b_+X@w_,0)-ytr)**2).mean()
    if best is None or loss<best[0]: best=(loss,w_.copy(),b_)
loss,w3,b3=best; res.append(score("loi 3 : max(0, b + somme a_i r_i)  (21 nombres)",np.maximum(b3+Rte@w3,0),"21 nombres"))
mlp=MLPRegressor(hidden_layer_sizes=(32,32),max_iter=3000,random_state=0,alpha=1e-3).fit(Rtr/200,ytr); res.append(score("reference libre : reseau 20-32-32-1 (non compact)",mlp.predict(Rte/200),"1 697 nombres"))
print("poids loi 3 (Hz^-1) :",np.round(w3,3).tolist(),"b =",round(float(b3),2))
json.dump(dict(res=res,bruit=float(noise),w3=w3.tolist(),b3=float(b3),a2=a.tolist(),xk=iso2.X_thresholds_.tolist(),yk=iso2.y_thresholds_.tolist(),pred2=iso2.predict(Rte@a).tolist()),open('lois_res.json','w'),indent=1)
