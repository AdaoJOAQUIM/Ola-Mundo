import numpy as np, json, simlib as S, sys
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.utils.indexed_list import IndexedList
from openpyxl.utils import get_column_letter as L
d=np.load('lois_data.npz',allow_pickle=True); r=json.load(open('lois_res.json')); nt,ns=int(d['n_train']),int(d['n_test'])
R=d['rates'][nt:nt+ns]; y=d['mn9'][nt:nt+ns]; a=np.array(r['a2']); xk=np.array(r['xk']); yk=np.array(r['yk']); K=len(xk)
wb=Workbook(); wb._fonts=IndexedList([Font(name='Arial',sz=10,family=2)]); B=Font(name='Arial',sz=10,bold=True); H=PatternFill('solid',fgColor='DDE6F0'); IN=PatternFill('solid',fgColor='FFF2CC')
wl=wb.active; wl.title='LISEZ-MOI'; wp=wb.create_sheet('Loi'); wt=wb.create_sheet('Tests')
txt=['Loi de comportement du cerveau entier de la mouche : sucre -> MN9, en une formule',
'La loi : MN9 (Hz) = courbe( somme des poids a_i x frequence r_i de chacun des 20 neurones sucre ). Les poids et la courbe monotone sont dans la feuille Loi (cellules jaunes).',
f'Elle remplace la simulation du cerveau entier (138 639 neurones, 54 492 922 synapses) pour cette question : {len(a)} poids et {K} points de courbe, soit {len(a)+2*K} nombres.',
'Feuille Tests : 100 motifs de stimulation tires au hasard (0 a 200 Hz par neurone, un tiers des neurones muets), JAMAIS utilises pour ajuster la loi. Colonne MN9 exact = simulation exacte du cerveau entier (1 s, une graine). Colonne Prediction = formule de la loi.',
'L\'erreur moyenne et le R2 sont calcules par formules en haut de la feuille Tests.',
f'Plancher de bruit : refaire la meme stimulation avec une autre graine change MN9 de {r["bruit"]:.1f} Hz (ecart-type). Une loi ne peut pas faire mieux que ce bruit.',
'LIMITES : une seule sortie (MN9), une seule famille d\'entrees (les 20 neurones sucre), 1 s de stimulation constante, modele de Shiu et al. La loi ne dit rien du cablage : elle ne peut pas le reconstruire. Hors de ce domaine, elle n\'a aucune garantie.',
'ETAT DES TESTS : formules recalculees dans LibreOffice (0 erreur). Les 100 valeurs exactes viennent du moteur evenementiel Python, identique a la simulation complete.']
for i,t in enumerate(txt,1): wl[f'A{i}']=t; wl[f'A{i}'].alignment=Alignment(wrap_text=True,vertical='top')
wl['A1'].font=Font(name='Arial',sz=13,bold=True); wl.column_dimensions['A'].width=130
wp['A1']='Poids par neurone sucre (cellules jaunes)'; wp['A1'].font=B
wp.append([]); wp.append(['Neurone (index)','Poids a_i (par Hz)'])
for c in wp[3]: c.font=B; c.fill=H
for i,(n,w) in enumerate(zip(S.SUGAR,a)): wp.append([n,float(w)]); wp[f'B{4+i}'].fill=IN
wp['D1']='Courbe monotone (interpolation lineaire entre les points)'; wp['D1'].font=B
wp['D3']='Indice z'; wp['E3']='MN9 (Hz)'
for c in (wp['D3'],wp['E3']): c.font=B; c.fill=H
for k in range(K): wp[f'D{4+k}']=float(xk[k]); wp[f'E{4+k}']=float(yk[k]); wp[f'D{4+k}'].fill=IN; wp[f'E{4+k}'].fill=IN
wp.column_dimensions['A'].width=18; wp.column_dimensions['B'].width=20; wp.column_dimensions['D'].width=16; wp.column_dimensions['E'].width=12
last=3+K
wt['A1']='Test hors entrainement : 100 motifs'; wt['A1'].font=B
wt['A2']='Erreur moyenne (Hz)'; wt['B2']=f'=AVERAGE(Y6:Y{5+ns})'
wt['A3']='R2'; wt['B3']=f'=1-SUMSQ(Z6:Z{5+ns})/DEVSQ(W6:W{5+ns})'
wt['D2']='Plancher de bruit des graines (Hz, ecart-type)'; wt['G2']=r['bruit']; wt['D3']='R2 maximal atteignable'; wt['G3']=f'=1-G2^2/VAR(W6:W{5+ns})'
heads=[f'r{i+1} (Hz) n{S.SUGAR[i]}' for i in range(20)]+['Indice z (borné à la courbe)','Rang','MN9 exact (Hz)','Prediction (Hz)','Erreur absolue (Hz)','Erreur (Hz)']
hr=5
for j,h in enumerate(heads,1):
    c=wt.cell(hr,j,h); c.font=B; c.fill=H; c.alignment=Alignment(wrap_text=True,vertical='top')
# colonnes: A..T rates, U z, V rang, W exact, X prediction, Y erreur abs, Z erreur
for i in range(ns):
    x=hr+1+i
    for j in range(20): wt.cell(x,j+1,round(float(R[i,j]),3))
    wt[f'U{x}']=f'=MAX(Loi!$D$4,MIN(Loi!$D${last},'+'+'.join(f'{L(j+1)}{x}*Loi!$B${4+j}' for j in range(20))+'))'
    wt[f'V{x}']=f'=MAX(1,MIN({K-1},MATCH(U{x},Loi!$D$4:$D${last},1)))'
    wt[f'W{x}']=int(y[i])
    wt[f'X{x}']=f'=MAX(0,MIN(INDEX(Loi!$E$4:$E${last},{K}),INDEX(Loi!$E$4:$E${last},V{x})+(U{x}-INDEX(Loi!$D$4:$D${last},V{x}))*(INDEX(Loi!$E$4:$E${last},V{x}+1)-INDEX(Loi!$E$4:$E${last},V{x}))/(INDEX(Loi!$D$4:$D${last},V{x}+1)-INDEX(Loi!$D$4:$D${last},V{x}))))'
    wt[f'Y{x}']=f'=ABS(X{x}-W{x})'; wt[f'Z{x}']=f'=X{x}-W{x}'
    for col in 'XYZ': wt[f'{col}{x}'].number_format='0.0'
    wt[f'U{x}'].number_format='0.00'
wt['B2'].number_format='0.0'; wt['B3'].number_format='0.000'; wt['G3'].number_format='0.000'; wt['G2'].number_format='0.0'
wt.freeze_panes='A6'
wb.save(sys.argv[1]); print('points de courbe',K)
