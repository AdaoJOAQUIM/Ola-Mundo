import numpy as np, simlib as S, sys
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.utils.indexed_list import IndexedList
from openpyxl.workbook.defined_name import DefinedName
pre,post,w,sg,ids=S.load('conn.npz'); N=S.N
o=np.lexsort((post,pre)); pre,post,w=pre[o],post[o],w[o]; start=np.searchsorted(pre,np.arange(N+1))
wb=Workbook(); wb._fonts=IndexedList([Font(name='Arial',sz=10,family=2)]); B=Font(name='Arial',sz=10,bold=True); H=PatternFill('solid',fgColor='DDE6F0'); IN=PatternFill('solid',fgColor='FFF2CC')
wl=wb.active; wl.title='LISEZ-MOI'
wr=wb.create_sheet('Réglages'); wc=wb.create_sheet('Contrôle'); wm=wb.create_sheet('Code M'); wd=wb.create_sheet('DAX')
ws=wb.create_sheet('Synapses'); ws.append(['Index','Signe','Code']); tot=0; sauts=0
for i in range(N):
    t=post[start[i]:start[i+1]]; ww=w[start[i]:start[i+1]]; vals=[]; prev=0
    for a,b in zip(t.tolist(),ww.tolist()):
        d=a-prev
        while d>16383: vals+= [16383,0]; d-=16383; sauts+=1
        vals+=[d,int(b)]; prev=a
    code=''.join(chr(19968+v) for v in vals); tot+=len(code); ws.append([i,int(sg[i]),code])
for c in ws[1]: c.font=B; c.fill=H
ws.freeze_panes='A2'; ws.column_dimensions['C'].width=60
wb.defined_names['syn_plage']=DefinedName('syn_plage',attr_text=f"Synapses!$A$1:$C${N+1}")
wr['A1']='Réglages de la simulation (cellules jaunes modifiables)'; wr['A1'].font=B
for r,(lab,val,name) in enumerate([('Fréquence de stimulation (Hz)',100,'p_hz'),('Durée simulée (ms)',100,'p_duree'),('Graine du hasard (entier de 1 à 2 147 483 646)',1,'p_graine')],3):
    wr[f'A{r}']=lab; wr[f'B{r}']=val; wr[f'B{r}'].fill=IN; wb.defined_names[name]=DefinedName(name,attr_text=f"'Réglages'!$B${r}")
wr['A7']='Neurones stimulés (index, un par ligne, 40 au plus)'; wr['A7'].font=B; wr['B7']='Par défaut : les 20 neurones du sucre'
for k,i in enumerate(S.SUGAR): wr[f'A{8+k}']=i; wr[f'A{8+k}'].fill=IN
for k in range(len(S.SUGAR),40): wr[f'A{8+k}'].fill=IN
wb.defined_names['stim_plage']=DefinedName('stim_plage',attr_text="'Réglages'!$A$8:$A$47")
wr['D3']='Neurone moteur suivi (MN9)'; wr['E3']=S.MN9
wr.column_dimensions['A'].width=52; wr.column_dimensions['B'].width=40; wr.column_dimensions['D'].width=28
import json; att=json.load(open(sys.argv[2])) if len(sys.argv)>2 else {}
wc['A1']='Valeurs attendues (calculées en Python par un miroir du code M, identiques à la simulation complète)'; wc['A1'].font=B
for j,h in enumerate(['Durée (ms)','Graine','Spikes totaux','Neurones actifs','Spikes de MN9','Neurones en écart avec la simulation complète'],1):
    c=wc.cell(3,j,h); c.font=B; c.fill=H
for r,v in enumerate(att.values(),4):
    for j,k in enumerate(['duree','graine','spikes','actifs','mn9','ecart'],1): wc.cell(r,j,v[k])
wc['A11']='Contrôle des données par formules'; wc['A11'].font=B
wc['A12']='Caractères dans la feuille Synapses'; wc['B12']=f'=SUMPRODUCT(LEN(Synapses!C2:C{N+1}))'; wc['C12']=tot; wc['D12']='=B12-C12'
wc['A13']='Connexions codées (caractères / 2, sauts compris)'; wc['B13']='=B12/2'; wc['C13']=tot//2; wc['D13']='=B13-C13'
wc['A14']='Neurones'; wc['B14']=f'=COUNT(Synapses!A2:A{N+1})'; wc['C14']=N; wc['D14']='=B14-C14'
wc['B11']='Valeur'; wc['C11']='Attendu'; wc['D11']='Écart'
for col,wd_ in zip('ABCDEF',[50,14,14,16,14,44]): wc.column_dimensions[col].width=wd_
for name,r0 in (('Simuler',1),):
    pass
r=1
for name in ('Simuler','Spikes'):
    wm.cell(r,1,f'Requête « {name} » : copier les lignes ci-dessous dans l\'Éditeur avancé').font=B; r+=1
    for line in open(f'{name}.pq',encoding='utf8').read().split('\n'):
        c=wm.cell(r,1,line if not line.startswith('=') else "'"+line); c.font=Font(name='Courier New',sz=9); r+=1
    r+=1
wm.column_dimensions['A'].width=150
dax=[('Spikes totaux','COUNTROWS(Spikes)'),('Neurones actifs','DISTINCTCOUNT(Spikes[Neurone])'),('Spikes MN9','CALCULATE(COUNTROWS(Spikes), Spikes[Neurone] = 138332)'),
     ('Durée simulée (s)','(MAX(Spikes[Pas]) + 1) / 10000'),('Fréquence MN9 (Hz)','DIVIDE([Spikes MN9], [Durée simulée (s)])'),('Spikes par neurone actif','DIVIDE([Spikes totaux], [Neurones actifs])')]
wd['A1']='Mesures DAX à créer sur la table Spikes (Power Pivot > Mesures > Nouvelle mesure). Non exécutées ici.'; wd['A1'].font=B
wd['A3']='Nom'; wd['B3']='Formule'; wd['A3'].font=B; wd['B3'].font=B
for r,(a,b) in enumerate(dax,4): wd[f'A{r}']=a; wd[f'B{r}']=b
wd.column_dimensions['A'].width=30; wd.column_dimensions['B'].width=80
txt=['Cerveau entier de la mouche dans Power Query : 138 639 neurones, 54 492 922 synapses, zéro formule de simulation',
 'PRINCIPE : le moteur ne calcule que les neurones qui reçoivent un signal. Entre deux signaux, la tension suit une formule exacte, et le moteur prédit le pas précis du prochain spike. Le circuit se construit tout seul : pas de slots, pas de tours, pas de preuve à refaire.',
 'ÉTAPE 1 : Données > Obtenir des données > À partir d\'autres sources > Requête vide. Dans l\'Éditeur avancé, coller le code de la requête Simuler (feuille Code M). Renommer la requête : Simuler.',
 'ÉTAPE 2 : nouvelle requête vide, coller le code de la requête Spikes, la nommer Spikes. Fermer et charger dans… > Tableau, et cocher « Ajouter ces données au modèle de données ».',
 'ÉTAPE 3 : comparer avec la feuille Contrôle. Réglages par défaut : 100 ms, graine 1. Commencer par cette durée courte pour mesurer la vitesse, puis passer à 1000 ms.',
 'ÉTAPE 4 (facultative) : créer les mesures de la feuille DAX dans Power Pivot, puis un tableau croisé sur Spikes[Neurone].',
 'AUTRE EXPÉRIENCE : changer les neurones stimulés, la fréquence, la durée ou la graine dans la feuille Réglages, puis Données > Actualiser tout.',
 'ÉTAT DES TESTS : la logique du code M est vérifiée par un miroir Python ligne à ligne, qui relit ce classeur et retrouve la simulation complète sans aucun écart. La syntaxe du code M est validée par l\'analyseur de Microsoft. Le code M lui-même n\'a jamais été exécuté : pas d\'Excel dans l\'environnement de construction. Sa vitesse dans Excel est inconnue.',
 'LIMITES : le hasard de la stimulation suit un générateur propre à ce classeur (Park-Miller), donc les spikes diffèrent, tirage par tirage, de ceux des anciens classeurs ; les statistiques sont les mêmes. La feuille Synapses pèse l\'essentiel du fichier : le câblage complet ne se compresse pas davantage sans perte.',
 'Données : connectome FlyWire v783. Modèle : Shiu et al., Nature 2024.']
for i,t in enumerate(txt,1): wl[f'A{i}']=t; wl[f'A{i}'].alignment=Alignment(wrap_text=True,vertical='top')
wl['A1'].font=Font(name='Arial',sz=13,bold=True); wl.column_dimensions['A'].width=130
wb.save(sys.argv[1]); print('caracteres',tot,'sauts',sauts)
