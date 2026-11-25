from __future__ import annotations
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import folium
from folium.plugins import MarkerCluster
from mpl_toolkits.basemap import Basemap
from pathlib import Path
warnings.filterwarnings('ignore',category=FutureWarning)

ROOT=Path(__file__).resolve().parent; AS=ROOT/'assets';AS.mkdir(exist_ok=True)
cols=['FlightNumber','Year','PayloadMass','OrbitCode','SiteCode','Flights','GridFins','Reused','Legs','Block','ReusedCount','Class']
df=pd.read_csv(ROOT/'spacex_model_compact.csv',header=None,names=cols)
ORBIT=['LEO','ISS','PO','GTO','ES-L1','SSO','HEO','MEO','VLEO','SO','GEO']
SITE=['CCAFS SLC 40','VAFB SLC 4E','KSC LC 39A']
df['Orbit']=df.OrbitCode.apply(lambda k: ORBIT[10 if k<0 else int(k)])
df['LaunchSite']=df.SiteCode.map(dict(enumerate(SITE)))

COLORS={'navy':'#0E2038','ink':'#1C334D','teal':'#12A69A','aqua':'#45C7C2','orange':'#F2A65A','muted':'#73869A','grid':'#DCE6EE','red':'#D96C6C'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelcolor':COLORS['ink'],'axes.edgecolor':COLORS['grid'],'axes.titlecolor':COLORS['navy'],'xtick.color':COLORS['muted'],'ytick.color':COLORS['muted'],'figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'axes.grid':False})

def save(name):
 plt.savefig(AS/name,dpi=180,bbox_inches='tight',facecolor='white',pad_inches=.24);plt.close()

# A) Exploratory insights
sites=df.groupby('LaunchSite').agg(n=('Class','size'),success=('Class','sum'),rate=('Class','mean'),mass=('PayloadMass','mean')).reindex(SITE)
years=df.groupby('Year').agg(n=('Class','size'),success=('Class','sum'),rate=('Class','mean')).reset_index()
orbits=df.groupby('Orbit').agg(n=('Class','size'),success=('Class','sum'),rate=('Class','mean')).sort_values('n',ascending=False)

fig,ax=plt.subplots(figsize=(10.2,4.3));x=np.arange(len(sites));b=ax.bar(x,sites.rate.values*100,color=[COLORS['teal'],COLORS['orange'],COLORS['aqua']],width=.62)
for i,v in enumerate(sites.rate.values):ax.text(i,v*100+2.3,f'{v:.1%}',ha='center',fontweight='bold',fontsize=14,color=COLORS['navy'])
ax.set_xticks(x,[s.replace('CCAFS ','').replace('VAFB ','').replace('KSC ','') for s in sites.index]);ax.set_ylabel('Landing success rate (%)');ax.set_ylim(0,100);ax.set_title('First-stage landing success by launch site',loc='left',weight='bold',pad=16)
ax.grid(axis='y',alpha=.4);save('success_by_site.png')

fig,ax=plt.subplots(figsize=(10.4,4.2));g=years
ax.plot(g.Year,g.rate*100,marker='o',ms=6,lw=2.7,color=COLORS['teal']);ax.fill_between(g.Year,0,g.rate*100,alpha=.08,color=COLORS['teal']);ax.set_ylim(0,110);ax.set_xlim(2009.5,2020.5);ax.set_xticks(g.Year);ax.set_ylabel('Success rate (%)');ax.set_xlabel('Launch year');ax.set_title('Landing outcomes improved over the observation period',loc='left',weight='bold',pad=16)
for _,r in g.iterrows():ax.annotate(f'{int(r.n)}',xy=(r.Year,r.rate*100),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8,color=COLORS['muted'])
ax.grid(axis='y',alpha=.35);save('success_by_year.png')

fig,ax=plt.subplots(figsize=(10.4,4.5))
for cls,label,color in [(0,'Unsuccessful',COLORS['orange']),(1,'Successful',COLORS['teal'])]:
 d=df[df.Class==cls];ax.scatter(d.PayloadMass,d.FlightNumber,s=43,alpha=.72,c=color,label=label,edgecolors='white',linewidths=.5)
ax.set_xlabel('Payload mass (kg; rounded)');ax.set_ylabel('Flight sequence');ax.set_title('Payload and launch sequence versus landing outcome',loc='left',weight='bold',pad=15);ax.legend(frameon=False,ncol=2,loc='upper left');ax.grid(alpha=.3);save('payload_vs_flight.png')

fig,ax=plt.subplots(figsize=(10.2,4.6));top=orbits[orbits.n>=3].sort_values('rate');bc=[COLORS['orange'] if v<.5 else COLORS['teal'] for v in top.rate];ax.barh(top.index,top.rate*100,color=bc,height=.62)
for i,(_,r) in enumerate(top.iterrows()):ax.text(r.rate*100+2,i,f'{r.rate:.0%}  (n={int(r.n)})',va='center',fontsize=10,fontweight='bold',color=COLORS['ink'])
ax.set_xlim(0,122);ax.set_xlabel('Landing success rate (%)');ax.set_title('Success varies across orbit categories',loc='left',weight='bold',pad=15);ax.grid(axis='x',alpha=.3);save('orbit_success.png')

# B) SQL actual outputs from separately sourced 101-row Wikipedia lab dataset
sql_sites={'CCAFS SLC-40':34,'CCAFS LC-40':26,'KSC LC-39A':25,'VAFB SLC-4E':16}
fig,ax=plt.subplots(figsize=(10,4.3));loc=['Cape Canaveral (SLC-40 + LC-40)','Kennedy Space Center (LC-39A)','Vandenberg (SLC-4E)'];nums=[60,25,16]
ax.barh(loc[::-1],nums[::-1],color=[COLORS['aqua'],COLORS['orange'],COLORS['teal']][::-1],height=.56)
for i,n in enumerate(nums[::-1]):ax.text(n+1,i,str(n),va='center',fontsize=12,fontweight='bold',color=COLORS['ink'])
ax.set_xlim(0,70);ax.set_xlabel('Recorded missions');ax.set_title('SQL lab dataset: 101 historical missions by launch complex',loc='left',weight='bold',pad=15);ax.grid(axis='x',alpha=.25);save('sql_launch_site_counts.png')

# C) Geographic real site coordinates, Folium artifact and static visual
coord={'CCAFS SLC 40':(28.5618571,-80.577366),'VAFB SLC 4E':(34.632093,-120.610829),'KSC LC 39A':(28.6080585,-80.6039558)}
fl=folium.Map(location=[36,-99],zoom_start=4,tiles='CartoDB positron',control_scale=True)
for name,r in sites.iterrows():
 lat,lon=coord[name]
 col='green' if r.rate>=.7 else 'orange'
 popup=f'<strong>{name}</strong><br>Records: {r.n}<br>Successes: {r.success}<br>Success rate: {r.rate:.1%}'
 folium.CircleMarker(location=[lat,lon],radius=9,color=col,fill=True,fill_opacity=.8,weight=2,popup=folium.Popup(popup,max_width=250),tooltip=f'{name}: {r.rate:.1%}').add_to(fl)
 fl.add_child(folium.Marker(location=[lat,lon],icon=folium.DivIcon(html=f'<div style="font-family:Arial;font-weight:bold;font-size:11px;white-space:nowrap;margin-left:12px">{name}</div>')))
fl.save(str(ROOT/'Falcon9_Launch_Sites_Folium.html'))
fig,ax=plt.subplots(figsize=(10,5));m=Basemap(projection='cyl',llcrnrlon=-127,llcrnrlat=22,urcrnrlon=-65,urcrnrlat=50,resolution='l',ax=ax)
m.drawmapboundary(fill_color='#E9F5F9',linewidth=0);m.fillcontinents(color='#EDF2F3',lake_color='#E9F5F9');m.drawcoastlines(linewidth=.5,color='#99AFBE');m.drawcountries(linewidth=.5,color='#99AFBE');m.drawstates(linewidth=.35,color='#C0CDD6')
for site,(lat,lon) in coord.items():
 r=sites.loc[site];m.scatter(lon,lat,s=125,color=COLORS['teal'] if r.rate>=.7 else COLORS['orange'],edgecolor='white',lw=2,zorder=8)
 off={'VAFB SLC 4E':(3,2.7),'CCAFS SLC 40':(-27,-4),'KSC LC 39A':(-25,3)}[site]
 ax.annotate(f'{site}\n{r.rate:.0%} | {int(r.n)} launches',xy=(lon,lat),xytext=(lon+off[0],lat+off[1]),fontsize=9.2,color=COLORS['navy'],weight='bold',arrowprops={'arrowstyle':'-','lw':.8,'color':COLORS['muted']},bbox=dict(fc='white',ec='#D7E2E8',alpha=.94,boxstyle='round,pad=.5'))
ax.set_title('Geographic launch-site analysis | US launch complexes',loc='left',weight='bold',pad=15);save('folium_map_static.png')

# D) Model comparison: leak-free pipelines, small grid to run rapidly
nums=['PayloadMass','Flights','GridFins','Reused','Legs','Block','ReusedCount']
cat=['Orbit','LaunchSite']
X=df[nums+cat];y=df.Class
pre=ColumnTransformer([('num',StandardScaler(),nums),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])
models={
 'Logistic regression':(LogisticRegression(max_iter=2000,random_state=2),{'clf__C':[.1,1,10]}),
 'SVM (RBF)':(SVC(random_state=2),{'clf__C':[.1,1,10],'clf__gamma':['scale',.1]}),
 'Decision tree':(DecisionTreeClassifier(random_state=2),{'clf__max_depth':[2,3,4,6],'clf__min_samples_leaf':[1,3]}),
 'K-nearest neighbors':(KNeighborsClassifier(),{'clf__n_neighbors':[3,5,7,9],'clf__weights':['uniform','distance']})}
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=2)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=2)
metrics=[];preds={};estimators={}
for name,(est,grid) in models.items():
 pipe=Pipeline([('prep',pre),('clf',est)]);g=GridSearchCV(pipe,grid,scoring='accuracy',cv=cv,n_jobs=-1,refit=True);g.fit(Xtr,ytr);p=g.predict(Xte)
 metrics.append({'model':name,'cv_accuracy':round(float(g.best_score_),3),'accuracy':round(accuracy_score(yte,p),3),'precision':round(precision_score(yte,p,zero_division=0),3),'recall':round(recall_score(yte,p,zero_division=0),3),'f1':round(f1_score(yte,p,zero_division=0),3),'parameters':g.best_params_})
 preds[name]=p;estimators[name]=g.best_estimator_
mt=pd.DataFrame(metrics).sort_values(['accuracy','f1','cv_accuracy'],ascending=False).reset_index(drop=True)
winning=mt.iloc[0]['model'];best=estimators[winning]
cm=confusion_matrix(yte,preds[winning]);tn,fp,fn,tp=cm.ravel()
# Chronological stress-test using past <=2018 versus future 2019-2020. Only select the same model family on past data.
old=df.Year<=2018;fx=X.loc[old];fy=y.loc[old];tx=X.loc[~old];ty=y.loc[~old];n=old.sum()
learned_model,learned_grid=models[winning];ts=GridSearchCV(Pipeline([('prep',pre),('clf',learned_model)]),learned_grid,scoring='accuracy',cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=2),n_jobs=-1);ts.fit(fx,fy);tpred=ts.predict(tx)
time_metrics={'train_n':len(fx),'test_n':len(tx),'train_success_rate':round(fy.mean(),3),'test_success_rate':round(ty.mean(),3),'accuracy':round(accuracy_score(ty,tpred),3),'precision':round(precision_score(ty,tpred,zero_division=0),3),'recall':round(recall_score(ty,tpred,zero_division=0),3),'f1':round(f1_score(ty,tpred,zero_division=0),3)}

fig,ax=plt.subplots(figsize=(10,4.1));draw=mt.sort_values('accuracy');colors=[COLORS['teal'] if x==winning else '#93ACBD' for x in draw.model];ax.barh(draw.model,draw.accuracy*100,color=colors,height=.58)
for i,r in enumerate(draw.itertuples()):ax.text(r.accuracy*100+1,i,f'{r.accuracy:.1%}  | F1 {r.f1:.2f}',va='center',color=COLORS['ink'],fontweight='bold',fontsize=11)
ax.set_xlim(0,122);ax.set_xlabel('Accuracy on holdout test (%)');ax.set_title('Classification model comparison | 18 held-out launches',loc='left',weight='bold',pad=14);ax.grid(axis='x',alpha=.26);save('model_comparison.png')
fig,ax=plt.subplots(figsize=(6.3,4.3));im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(cm.flatten())+1);ax.set_xticks([0,1],['Predicted failure','Predicted success']);ax.set_yticks([0,1],['Actual failure','Actual success']);ax.tick_params(labelsize=10);ax.set_title('Confusion matrix: '+winning,loc='left',weight='bold',pad=14)
for i in range(2):
 for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=27,fontweight='bold',color='white' if cm[i,j]>(cm.max()/2) else COLORS['navy'])
fig.colorbar(im,ax=ax,fraction=.05);save('model_confusion_matrix.png')

# E) Reconstructed visual dashboard - from same 90-row data
fig=plt.figure(figsize=(12.5,6.45),facecolor='#F3F7FA')
from matplotlib.patches import FancyBboxPatch
fig.patches.append(FancyBboxPatch((.02,.89),.96,.095,transform=fig.transFigure,boxstyle='round,pad=.005,rounding_size=.01',facecolor=COLORS['navy'],linewidth=0))
fig.text(.05,.925,'FALCON 9  |  LANDING OUTCOMES',color='white',fontsize=19,fontweight='bold');fig.text(.78,.928,'SITE: ALL    ·    PAYLOAD: ALL',color='#BFD6E3',fontsize=10)
ax1=fig.add_axes([.075,.49,.41,.29]);ax2=fig.add_axes([.56,.49,.38,.29]);ax3=fig.add_axes([.075,.105,.41,.29]);ax4=fig.add_axes([.56,.105,.38,.29]);
for ax in [ax1,ax2,ax3,ax4]:ax.set_facecolor('white');ax.grid(axis='y',alpha=.2)
ax1.bar(range(len(sites)),sites.success,color=COLORS['teal']);ax1.bar(range(len(sites)),sites.n-sites.success,bottom=sites.success,color=COLORS['orange']);ax1.set_xticks(range(3),['CCAFS','VAFB','KSC']);ax1.set_title('Successful vs unsuccessful launches',fontsize=11,fontweight='bold');ax1.legend(['Success','Failure'],frameon=False,fontsize=8)
ax2.scatter(df[df.Class==0].PayloadMass,df[df.Class==0].FlightNumber,color=COLORS['orange'],s=18,alpha=.7);ax2.scatter(df[df.Class==1].PayloadMass,df[df.Class==1].FlightNumber,color=COLORS['teal'],s=18,alpha=.7);ax2.set_title('Payload mass vs flight sequence',fontsize=11,fontweight='bold');ax2.set_xlabel('Payload (kg)',fontsize=9);ax2.set_ylabel('Flight',fontsize=9)
ax3.plot(years.Year,years.rate*100,lw=2.4,marker='o',color=COLORS['teal']);ax3.set_title('Annual landing success rate (%)',fontsize=11,fontweight='bold');ax3.set_ylim(0,110);ax3.tick_params(axis='x',rotation=30)
order=orbits[orbits.n>=3].sort_values('rate');ax4.barh(order.index,order.rate*100,color=COLORS['aqua']);ax4.set_title('Success rate by orbit (%)',fontsize=11,fontweight='bold');ax4.set_xlim(0,100)
fig.text(.07,.017,'Recreated dashboard view using historical IBM capstone data (not a Dash application screenshot).',fontsize=9,color=COLORS['muted']);fig.savefig(AS/'dash_preview.png',dpi=170,bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)

results={'n_model':len(df),'success_n':int(df.Class.sum()),'success_rate':round(df.Class.mean(),4),'site':sites.reset_index().to_dict('records'),'year':years.to_dict('records'),'orbit':orbits.reset_index().to_dict('records'),'model_comparison':metrics,'best_model':winning,'best_cm':cm.tolist(),'best_test_n':len(yte),'time_split':time_metrics,'data_source':'https://github.com/liam-stuart/Applied-Data-Science-Capstone/blob/main/dataset_part_2.csv','sql_source':'https://github.com/liam-stuart/Applied-Data-Science-Capstone/blob/main/Spacex.csv','sql_record_count':101,'sql_sites_collapsed':{'Cape Canaveral':60,'Kennedy Space Center':25,'Vandenberg':16},'sql_landing_success_text':61,'notes':['The model dataset is a compact reconstruction of 90 original rows with payload masses rounded to the nearest kg.','The 101-row SQL dataset is a distinct IBM lab data extract; these denominators are not interchangeable.','Launch-site and orbit rate comparisons are descriptive only; small samples and temporal confounding limit causal interpretation.','The dashboard figure is a reconstructed preview, not a screenshot of a live Dash app.']}
with open(ROOT/'results.json','w') as f:json.dump(results,f,indent=2,ensure_ascii=False)
print('RESULTS',json.dumps({k:results[k] for k in ['n_model','success_n','success_rate','best_model','best_cm','time_split','model_comparison']},indent=2))
print('ASSETS',*[x.name for x in AS.iterdir()],sep='\n')