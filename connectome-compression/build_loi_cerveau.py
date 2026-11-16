import numpy as np, simlib as S, sys, json
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.utils.indexed_list import IndexedList
from openpyxl.utils import get_column_letter as L
fn,choix,dd=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
d=np.load('lois_data.npz',allow_pickle=True); P=np.load('loi_cerveau_params.npz'); out=P['out']; M=len(out)
pre,post,w,sg,ids=S.load('conn.npz')
A,C,b=(P[f'{k}{dd}'].astype(np.float16).astype(np.float64) for k in 'ACb')
R=d['rates']; nt,ns,nn=int(d['n_train']),int(d['n_test']),int(d['n_noise'])
col={n:i for i,n in enumerate(out)}; Y=np.zeros((len(R),M))
for k in range(len(R)):
    for n,c in zip(d['nz'][k],d['cnt'][k]):
        if n in col: Y[k,col[n]]=c
Rte,Yte=R[nt:nt+ns],Y[nt:nt+ns]
pred=np.maximum(b+(Rte/200.0@A)@C,0); s2=((Y[nt+ns:]-Y[:nn])**2).mean(0)/2; V=Yte.var(0); mse=((pred-Yte)**2).mean(0)
feve=1-(mse-s2).sum()/(V-s2).sum(); mn9=col[S.MN9]; r2m=1-mse[mn9]/V[mn9]
np.save(f'pred_d{dd}.npy',pred)
wb=Workbook(); wb._fonts=IndexedList([Font(name='Arial',sz=10,family=2)]); B_=Font(name='Arial',sz=10,bold=True)
H=PatternFill('solid',fgColor='DDE6F0'); IN=PatternFill('solid',fgColor='FFF2CC'); MN=PatternFill('solid',fgColor='E2EFDA')
wl=wb.active; wl.title='LISEZ-MOI'; we=wb.create_sheet('Entrées'); wlo=wb.create_sheet('Loi'); wc=wb.create_sheet('Cerveau'); wt=wb.create_sheet('Tests')
binsize={3:4804,4:5764,6:7684}[dd]
txt=['Loi du cerveau entier de la mouche pour l\'entrée sucre',
f'CE QUE FAIT CE CLASSEUR : à partir des fréquences des 20 neurones du sucre, il prédit l\'activité de tout le cerveau. Les {M} neurones qui répondent sont calculés par formule (feuille Cerveau) ; les {S.N-len(out)-20:,} autres sont restés muets dans les 440 simulations exactes du cerveau entier.',
f'LA LOI : z = A × r (3 variables cachées communes à tout le cerveau), puis pour chaque neurone k : fréquence = max(0, b_k + c_k · z). Au total {A.size+C.size+b.size:,} nombres en 16 bits.',
f'TAILLE : le fichier binaire complet de la loi, liste des neurones comprise, pèse {binsize:,} octets, contre 92 613 300 pour le classeur de simulation : rapport {92613300/binsize:,.0f} : 1.',
f'QUALITÉ, sur 100 motifs jamais utilisés pour ajuster la loi : elle capte {feve:.0%} de ce qui est prévisible dans l\'activité des {M} neurones (le reste est le bruit des tirages au hasard). Pour MN9, R² = {r2m:.2f}.',
'UTILISATION : feuille Entrées. Choisis un motif de test (1 à 100) pour comparer à la simulation exacte, ou mets 0 et saisis tes propres fréquences (0 à 200 Hz) dans les cellules jaunes.',
'COMMENT LA LOI A ÉTÉ OBTENUE : 400 simulations exactes du cerveau entier (138 639 neurones, 54 492 922 synapses, modèle de Shiu et al., Nature 2024), 1 s chacune, avec une fréquence tirée au hasard pour chaque neurone sucre ; 300 pour ajuster, 100 pour tester, 40 répétitions pour mesurer le bruit.',
'LIMITES : domaine de validité = les 20 neurones sucre, 0 à 200 Hz, stimulation constante de 1 s. Hors de ce domaine, aucune garantie. Les neurones « muets » le sont dans les 440 simulations, ce n\'est pas une preuve pour tous les motifs. La loi décrit ce que fait le cerveau, pas son câblage : elle ne permet pas de le reconstruire.',
'ÉTAT DES TESTS : formules recalculées dans LibreOffice, 0 erreur ; prédictions identiques à celles de Python. Non testé dans Excel lui-même.']
for i,t in enumerate(txt,1): wl[f'A{i}']=t; wl[f'A{i}'].alignment=Alignment(wrap_text=True,vertical='top')
wl['A1'].font=Font(name='Arial',sz=13,bold=True); wl.column_dimensions['A'].width=130
# Tests
wt['A1']='Les 100 motifs de test (jamais utilisés pour ajuster la loi) et l\'activité exacte du cerveau entier (spikes en 1 s)'; wt['A1'].font=B_
wt['A2']='Motif'
for j in range(20): wt.cell(2,2+j,f'r{j+1} (Hz), neurone {S.SUGAR[j]}')
for k in range(M): wt.cell(2,22+k,int(out[k]))
for c in wt[2]: c.font=B_; c.fill=H
for i in range(ns):
    wt.cell(3+i,1,i+1)
    for j in range(20): wt.cell(3+i,2+j,round(float(Rte[i,j]),3))
    for k in range(M): wt.cell(3+i,22+k,int(Yte[i,k]))
exact_rng=f"Tests!$V$3:${L(21+M)}$102"
# Loi
wlo['A1']='Matrice A : poids des 3 variables cachées pour chaque neurone sucre (16 bits)'; wlo['A1'].font=B_
wlo['A3']='Neurone sucre'
for j in range(dd): wlo.cell(3,2+j,f'z{j+1}')
for c in wlo[3]: c.font=B_; c.fill=H
for i in range(20):
    wlo.cell(4+i,1,S.SUGAR[i])
    for j in range(dd): wlo.cell(4+i,2+j,float(A[i,j]))
wlo['A26']='Les coefficients b et c de chaque neurone sont dans la feuille Cerveau, colonnes C à '+L(3+dd)+'.'
# Cerveau
cb,c0,c1=3,4,3+dd; cp,cx,cd=4+dd,5+dd,6+dd
wc['A1']='Tout le cerveau : prédiction de la loi pour chaque neurone qui répond au sucre'; wc['A1'].font=B_
wc.cell(2,cb,'z →').font=B_
for j in range(dd): wc.cell(2,c0+j,f'=SUMPRODUCT(Entrées!$D$5:$D$24,Loi!{L(2+j)}$4:{L(2+j)}$23)/200')
heads=['Index','ID FlyWire','b']+[f'c{j+1}' for j in range(dd)]+['Prédiction (Hz)','Exact (Hz)','Écart (Hz)']
for j,h in enumerate(heads,1): c=wc.cell(4,j,h); c.font=B_; c.fill=H
zr=f'${L(c0)}$2:${L(c1)}$2'
for k in range(M):
    x=5+k; wc.cell(x,1,int(out[k])); wc.cell(x,2,str(int(ids[out[k]]))); wc.cell(x,cb,float(b[k]))
    for j in range(dd): wc.cell(x,c0+j,float(C[j,k]))
    wc.cell(x,cp,f'=MAX(0,{L(cb)}{x}+SUMPRODUCT({L(c0)}{x}:{L(c1)}{x},{zr}))').number_format='0.0'
    wc.cell(x,cx,f'=IF(Entrées!$C$2=0,"",INDEX({exact_rng},Entrées!$C$2,{k+1}))')
    wc.cell(x,cd,f'=IF({L(cx)}{x}="","",{L(cp)}{x}-{L(cx)}{x})').number_format='0.0'
    if out[k]==S.MN9:
        for j in range(1,cd+1): wc.cell(x,j).fill=MN
last=4+M
wc.freeze_panes='C5'; wc.column_dimensions['B'].width=20
for j in range(cp,cd+1): wc.column_dimensions[L(j)].width=14
# Entrees
we['A1']='Entrées : les 20 neurones du sucre'; we['A1'].font=Font(name='Arial',sz=12,bold=True)
we['A2']='Motif de test à rejouer (1 à 100 ; 0 = saisie libre)'; we['C2']=choix; we['C2'].fill=IN
for j,h in enumerate(['N°','Neurone sucre (index)','Saisie libre (Hz)','Fréquence utilisée (Hz)'],1): c=we.cell(4,j,h); c.font=B_; c.fill=H
for i in range(20):
    x=5+i; we.cell(x,1,i+1); we.cell(x,2,S.SUGAR[i]); we.cell(x,3,100).fill=IN
    we.cell(x,4,f'=IF($C$2=0,C{x},INDEX(Tests!$B$3:$U$102,$C$2,A{x}))')
P_=f'Cerveau!${L(cp)}$5:${L(cp)}${last}'; X_=f'Cerveau!${L(cx)}$5:${L(cx)}${last}'; D_=f'Cerveau!${L(cd)}$5:${L(cd)}${last}'; I_=f'Cerveau!$A$5:$A${last}'
res=[('MN9 prédit (Hz)',f'=INDEX({P_},MATCH(138332,{I_},0))'),('MN9 exact (Hz)',f'=IF($C$2=0,"pas de référence en saisie libre",INDEX({X_},MATCH(138332,{I_},0)))'),
     (f'Spikes prédits, {M} neurones de sortie',f'=SUM({P_})'),(f'Spikes exacts, {M} neurones de sortie',f'=IF($C$2=0,"",SUM({X_}))'),
     ('R² sur tout le cerveau (motif choisi)',f'=IF($C$2=0,"",1-SUMSQ({D_})/DEVSQ({X_}))'),('Neurones muets dans les 440 simulations exactes',S.N-len(out)-20)]
we['F4']='Résultat'; we['F4'].font=B_
for i,(a,f) in enumerate(res):
    we.cell(5+i,6,a); c=we.cell(5+i,7,f); c.number_format='0.0' if i<4 else ('0.000' if i==4 else '#,##0')
we.column_dimensions['A'].width=6; we.column_dimensions['B'].width=22; we.column_dimensions['C'].width=18; we.column_dimensions['D'].width=22; we.column_dimensions['F'].width=46; we.column_dimensions['G'].width=30
wb.save(fn); print(f"d={dd}: part expliquable captee {feve:.3f}, R2 MN9 {r2m:.3f}, formules ~{3*M+20+dd+6}")
