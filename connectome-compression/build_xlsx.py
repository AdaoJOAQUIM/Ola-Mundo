import json,numpy as np,simlib as S,sys
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.utils.indexed_list import IndexedList
from openpyxl.chart import ScatterChart,Reference,Series
from openpyxl.workbook.defined_name import DefinedName
R=json.load(open('results.json')); pre,post,w,sg,ids=S.load('conn.npz'); lab=np.load('groups.npz')['K1000']
wb=Workbook(); wb._fonts=IndexedList([Font(name='Arial',sz=10,family=2)])
B=Font(name='Arial',sz=10,bold=True); H=PatternFill('solid',fgColor='DDE6F0')
fam={'exact':('Référence','Connectome complet'),'prune':('Élagage','Garder les connexions de N synapses ou plus'),
     'mf':('Règles entre groupes (champ moyen)','Chaque paire de groupes réduite à un poids moyen'),
     'sbm':('Règles + graine (câblage régénéré)','Même nombre de connexions par paire de groupes, partenaires tirés au hasard'),
     'lr':('Rang faible','Matrice des poids approchée par U·S·Vᵀ'),
     'hyb':('Hybride','Connexions fortes exactes + règles entre groupes pour le reste')}
def describe(n):
    if n=='exact': return 'exact','Les 15 091 983 connexions'
    if n.startswith('prune'): return 'prune',f"Seuil : {n[5:]} synapses"
    if n.startswith('lr'): return 'lr',f"Rang {n[2:]}"
    p=n.split('_'); g=p[1]; gt=(f"{g[1:]} groupes tirés au hasard (témoin)" if g[0]=='R' else f"{g[1:]} groupes issus du câblage")
    if p[0]=='hyb': return 'hyb',f"{gt} ; exact à partir de {p[2][1:]} synapses"
    return p[0],gt
order={'exact':0,'prune':1,'hyb':2,'mf':3,'sbm':4,'lr':5}
rows=sorted(R,key=lambda r:(order[describe(r['name'])[0]],-r['size']))
ws=wb.active; ws.title='Résultats'
ws['A1']='Compression du connectome de la mouche : le comportement sucre → MN9 survit-il ?'; ws['A1'].font=Font(name='Arial',sz=13,bold=True)
ws['A2']='Stimulation : 20 neurones sucre à 100 Hz pendant 1 s, cerveau entier (138 639 neurones), 3 graines. MN9 = neurone moteur de la trompe (index 138332).'
ws['A3']='Taille du connectome complet (valeurs stockées : source, cible, poids)'; ws['G3']=3*len(pre)
hd=['Famille','Méthode','Réglage','Modèle','Taille (valeurs stockées)','Rapport de compression','MN9 graine 1','MN9 graine 2','MN9 graine 3','MN9 moyen (Hz)','Part de la référence','Verdict','Spikes totaux (moy.)','Neurones actifs (moy.)','Corrélation avec le complet (moy.)']
for j,h in enumerate(hd,1):
    c=ws.cell(5,j,h); c.font=B; c.fill=H; c.alignment=Alignment(wrap_text=True,vertical='top')
r0=6; refrow=None
for i,r in enumerate(rows):
    k,det=describe(r['name']); x=r0+i
    if k=='exact': refrow=x
    vals=[fam[k][0],fam[k][1],det,r['name'],r['size'],f'=$G$3/E{x}']+r['mn9']+[f'=AVERAGE(G{x}:I{x})',None,None,float(np.mean(r['spikes'])),float(np.mean(r['actifs'])),float(np.mean(r['corr']))]
    for j,v in enumerate(vals,1): ws.cell(x,j,v)
last=r0+len(rows)-1
for x in range(r0,last+1):
    ws[f'K{x}']=f'=J{x}/$J${refrow}'; ws[f'K{x}'].number_format='0%'
    ws[f'L{x}']=f'=IF(K{x}>=0.8,"conservé",IF(K{x}>=0.2,"dégradé","perdu"))'
    ws[f'F{x}'].number_format='0.0" : 1"'; ws[f'J{x}'].number_format='0.0'; ws[f'E{x}'].number_format='#,##0'; ws[f'M{x}'].number_format='#,##0'; ws[f'N{x}'].number_format='0'; ws[f'O{x}'].number_format='0.000'
ws['G3'].number_format='#,##0'
n0=last+2
ws[f'A{n0}']='Lecture'; ws[f'A{n0}'].font=B
notes=['Verdict : « conservé » si MN9 garde au moins 80 % de sa fréquence de référence, « perdu » sous 20 %.',
 "Taille : élagage = 3 valeurs par connexion gardée. Règles = 1 étiquette de groupe par neurone + les blocs non vides. Rang r = 2 × 138 639 × r + r. Hybride = somme des deux parties.",
 "Les groupes sont trouvés dans le câblage (décomposition SVD du connectome, puis k-moyennes séparées par signe). Ce ne sont pas les types cellulaires annotés par les biologistes : non testés ici.",
 "Un seul comportement est mesuré (sucre → MN9). La corrélation compare les fréquences des 138 639 neurones avec celles du connectome complet, à graine égale.",
 "Source des données : feuille Cerveau du classeur cerveau_auto_construit.xlsx (FlyWire v783). Simulateur : code Python de la notice (modèle de Shiu et al., Nature 2024)."]
for i,t in enumerate(notes): ws[f'A{n0+1+i}']=t
for col,wd in zip('ABCDEFGHIJKLMNO',[32,44,46,16,16,14,9,9,9,11,11,11,12,12,14]): ws.column_dimensions[col].width=wd
ws.freeze_panes='A6'
ch=ScatterChart(); ch.title='MN9 selon le rapport de compression'; ch.style=13; ch.height=9; ch.width=18
ch.x_axis.title='Rapport de compression (échelle log)'; ch.y_axis.title='MN9 moyen (Hz)'; ch.x_axis.scaling.logBase=10; ch.x_axis.delete=False; ch.y_axis.delete=False
i=0
while i<len(rows):
    k=describe(rows[i]['name'])[0]; j=i
    while j+1<len(rows) and describe(rows[j+1]['name'])[0]==k: j+=1
    s=Series(Reference(ws,min_col=10,min_row=r0+i,max_row=r0+j),Reference(ws,min_col=6,min_row=r0+i,max_row=r0+j),title=fam[k][0])
    s.marker.symbol='circle'; s.marker.size=7; s.graphicalProperties.line.noFill=(k not in('prune',)); ch.series.append(s); i=j+1
ws.add_chart(ch,f'A{n0+8}')
# regles K1000
K=int(lab.max()+1); key=lab[pre]*K+lab[post]; u,inv=np.unique(key,return_inverse=True)
cnt=np.bincount(inv); syn=np.bincount(inv,w).astype(np.int64); sgn=sg[np.array([np.nonzero(lab==a)[0][0] for a in range(K)])].astype(int)
wr=wb.create_sheet('Règles'); wr.append(['Groupe source','Groupe cible','Signe','Connexions','Synapses'])
for a,b,c,s in zip((u//K).tolist(),(u%K).tolist(),cnt.tolist(),syn.tolist()): wr.append([a,b,int(sgn[a]),c,s])
nr=len(u)+1
wg=wb.create_sheet('Groupes'); wg.append(['Index du neurone','ID FlyWire','Signe','Groupe'])
for i in range(S.N): wg.append([i,str(int(ids[i])),int(sg[i]),int(lab[i])])
for s_ in (wr,wg):
    for c in s_[1]: c.font=B; c.fill=H
    s_.freeze_panes='A2'
    for col in 'ABCDE': s_.column_dimensions[col].width=20
wb.defined_names['Regles']=DefinedName('Regles',attr_text=f"'Règles'!$A$1:$E${nr}")
wb.defined_names['Groupes']=DefinedName('Groupes',attr_text=f"Groupes!$A$1:$D${S.N+1}")
wc=wb.create_sheet('Contrôle'); wc['A1']='Contrôles calculés par formules sur les feuilles Règles et Groupes'; wc['A1'].font=B
ctl=[('Blocs (paires de groupes non vides)',f"=COUNT('Règles'!A2:A{nr})",None),('Connexions (attendu : 15 091 983)',f"=SUM('Règles'!D2:D{nr})",15091983),
     ('Synapses (attendu : 54 492 922)',f"=SUM('Règles'!E2:E{nr})",54492922),('Synapses excitatrices (attendu : 32 494 717)',f"=SUMIFS('Règles'!E2:E{nr},'Règles'!C2:C{nr},1)",32494717),
     ('Neurones (attendu : 138 639)',f"=COUNT(Groupes!A2:A{S.N+1})",138639),('Nombre de groupes',f"=MAX(Groupes!D2:D{S.N+1})+1",None),('Groupe de MN9 (index 138332)',"=INDEX(Groupes!D:D,138332+2)",None)]
wc['A2']='Mesure'; wc['B2']='Valeur'; wc['C2']='Attendu'; wc['D2']='Écart'
for c in wc[2]: c.font=B; c.fill=H
for i,(a,f,e) in enumerate(ctl,3):
    wc[f'A{i}']=a; wc[f'B{i}']=f; wc[f'B{i}'].number_format='#,##0'
    if e is not None: wc[f'C{i}']=e; wc[f'C{i}'].number_format='#,##0'; wc[f'D{i}']=f'=B{i}-C{i}'
wc.column_dimensions['A'].width=48; wc.column_dimensions['B'].width=16; wc.column_dimensions['C'].width=16
wl=wb.create_sheet('LISEZ-MOI',0)
txt=['Compression du connectome de la mouche : résultats mesurés',
'QUESTION : peut-on remplacer les 15 millions de connexions par une description beaucoup plus courte (règles entre groupes, rang faible) sans perdre le comportement ?',
'RÉPONSE MESURÉE : non. Dès qu\'on remplace les connexions individuelles par des règles entre groupes ou par un rang faible, MN9 tombe à 0 Hz, même à un rapport de 5 : 1 seulement. Seul l\'élagage des connexions faibles tient, jusqu\'à environ 6 : 1.',
'Résultats : une ligne par modèle, 3 graines. Les colonnes Rapport, MN9 moyen, Part de la référence et Verdict sont des formules.',
'Règles : le modèle à 1 000 groupes, sous forme lisible : pour chaque paire de groupes, le nombre de connexions et de synapses.',
'Groupes : le groupe de chacun des 138 639 neurones (modèle à 1 000 groupes).',
'Contrôle : totaux recalculés par formules, à comparer aux valeurs du connectome.',
'POWER QUERY (non exécuté ici, pas d\'Excel) : Données > Obtenir des données > Requête vide > Éditeur avancé, puis coller :',
'let Source = Excel.CurrentWorkbook(){[Name = "Regles"]}[Content], Entetes = Table.PromoteHeaders(Source, [PromoteAllScalars = true]) in Entetes',
'Même requête avec "Groupes". Charger les deux dans le modèle de données (Power Pivot).',
'DAX (non exécuté ici) : Synapses = SUM(Regles[Synapses]) ; Connexions = SUM(Regles[Connexions]) ; Poids moyen = DIVIDE([Synapses], [Connexions]). Valeurs attendues : feuille Contrôle.',
'LIMITES : un seul comportement (sucre → MN9) ; groupes tirés du câblage, pas les types cellulaires annotés ; champ moyen et tirage au hasard sont les deux façons les plus simples de régénérer un câblage, d\'autres existent.']
for i,t in enumerate(txt,1):
    wl[f'A{i}']=t; wl[f'A{i}'].alignment=Alignment(wrap_text=True,vertical='top')
wl['A1'].font=Font(name='Arial',sz=13,bold=True); wl.column_dimensions['A'].width=130
wb.save(sys.argv[1]); print('blocs',len(u))
