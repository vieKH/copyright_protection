from pathlib import Path
import json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / 'Results' / 'Parameter_Research'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
s=pd.read_csv(OUT/'summary.csv')
p=pd.read_csv(OUT/'per_image.csv')
assert len(pd.read_csv(OUT/'raw.csv'))==14400
assert p.groupby(['group','B','L','band','attack']).size().eq(100).all()
rows=[]
comparisons=[('L7_to_L41',(256,7,'high','none'),(256,41,'high','none')),('B64_to_B256_L14',(64,14,'high','none'),(256,14,'high','none')),('low_to_high',(128,14,'low','none'),(128,14,'high','none')),('mid_to_high_JPEG90',(128,14,'mid','jpeg90'),(128,14,'high','jpeg90')),('low_to_high_JPEG70',(128,14,'low','jpeg70'),(128,14,'high','jpeg70'))]
for group in [512,1024]:
 for label,a,b in comparisons:
  def get(t):
   B,L,band,attack=t
   return p.query('group==@group and B==@B and L==@L and band==@band and attack==@attack').set_index('file').ber.sort_index()
  d=(get(b)-get(a)).to_numpy()*100; rng=np.random.default_rng(501);boot=d[rng.integers(len(d),size=(5000,len(d)))].mean(axis=1)
  rows.append(dict(group=group,comparison=label,delta_pp=d.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
pd.DataFrame(rows).to_csv(OUT/'paired_effects.csv',index=False)
choices=[]
for L in sorted(s.L.unique()):
 for B in [64,128,256]:
  z=s.query('B==@B and L==@L and band=="high" and attack=="none"')
  if len(z)==2 and z.ber.max()<=.2:choices.append((L,B))
L=max(v[0] for v in choices);B=min(v[1] for v in choices if v[0]==L)
cons=s.query('band=="high" and attack=="none"').groupby(['L','B']).filter(lambda g:len(g)==2 and g.pass80.min()>=.9)
selection={'primary':{'L':int(L),'B':int(B),'bits':int(L*L),'band':'high'},'conservative':{'L':int(cons.L.max()),'B':int(cons[cons.L==cons.L.max()].B.min()),'band':'high'},'capacity_constant_weight_bits':math.lgamma(L*L+1)/math.log(2)-math.lgamma(L*L//2+1)/math.log(2)-math.lgamma(L*L-L*L//2+1)/math.log(2)}
(OUT/'selection.json').write_text(json.dumps(selection,indent=2))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
def finish(fig,name):fig.tight_layout();fig.savefig(FIG/name,dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for ax,group in zip(axs,[512,1024]):
 for B in [64,128,256]:
  z=s.query('group==@group and B==@B and band=="high" and attack=="none"').sort_values('L')
  ax.errorbar(z.L,z.ber*100,yerr=[(z.ber-z.ci_low)*100,(z.ci_high-z.ber)*100],marker='o',capsize=3,label=f'B={B}')
 ax.axhline(20,color='gray',ls='--',lw=1);ax.set(title=f'{group} × {group}',xlabel='L',ylabel='BER, %',xticks=[7,14,21,28,41]);ax.legend();ax.grid(alpha=.2)
finish(fig,'payload.png')
fig,ax=plt.subplots(figsize=(7,3.8))
for group in [512,1024]:
 z=s.query('group==@group and L==14 and band=="high" and attack=="none"').sort_values('B')
 ax.errorbar(z.B,z.ber*100,yerr=[(z.ber-z.ci_low)*100,(z.ci_high-z.ber)*100],marker='o',capsize=4,label=f'{group} × {group}')
ax.set(xlabel='B',ylabel='BER, %',xticks=[64,128,256]);ax.axhline(20,color='gray',ls='--');ax.grid(alpha=.2);ax.legend();finish(fig,'dft_size.png')
fig,axs=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for ax,group in zip(axs,[512,1024]):
 for attack,label in [('none','Без атак'),('jpeg90','JPEG 90'),('jpeg70','JPEG 70')]:
  z=s.query('group==@group and B==128 and L==14 and attack==@attack').set_index('band').loc[['low','mid','high']]
  ax.errorbar(range(3),z.ber*100,yerr=[(z.ber-z.ci_low)*100,(z.ci_high-z.ber)*100],marker='o',capsize=3,label=label)
 ax.set(title=f'{group} × {group}',xticks=range(3),xticklabels=['Низкие','Средние','Высокие'],ylabel='BER, %');ax.legend();ax.grid(alpha=.2)
finish(fig,'frequency.png')
print(json.dumps(selection,indent=2));print(pd.DataFrame(rows).round(3).to_string(index=False))
