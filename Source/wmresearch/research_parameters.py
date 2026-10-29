"""Controlled spectral watermark experiment. No synchronization, no phase decoding.
Python >=3.10; numpy pandas pillow matplotlib. See README.md for protocol.
"""
from pathlib import Path
import argparse,hashlib,json,io,platform
import numpy as np
import pandas as pd
from PIL import Image, __version__ as pillow_version
PROJECT_ROOT = Path(__file__).resolve().parents[2]


BANDS={'low':(.04,.22),'mid':(.22,.40),'high':(.40,.58)}
def positions(B,L,band):
 if band=='legacy':
  left=(L+1)//2; right=L-left
  p=[(B//8+i,B//8+j if j<left else B-B//16-right+j-left) for i in range(L) for j in range(L)]
 else:
  lo,hi=BANDS[band]; f=np.fft.fftfreq(B);p=[(u,v) for u in range(1,B//2) for v in range(1,B) if lo<=np.hypot(f[u],f[v])<hi]
  # Payload size uses a prefix of a fixed randomly ordered frequency pool.
  p=[p[i] for i in np.random.default_rng(728+B).permutation(len(p))]
  if len(p)<L*L:raise ValueError('Insufficient non-conjugate carriers')
  p=p[:L*L]
 s=set(p);c={((-u)%B,(-v)%B) for u,v in p}
 if len(s)!=L*L or s&c:raise ValueError('Carrier collision')
 return np.array(p).T

def make_watermark(B,L,band,seed,D):
 p=positions(B,L,band);n=L*L;k=n//2;rng=np.random.default_rng(seed)
 b=np.zeros(n,dtype=np.uint8);b[rng.permutation(n)[:k]]=1
 phase=np.random.default_rng(seed+10000).uniform(-np.pi,np.pi,n)
 a=B*B*np.sqrt(D/(2*k));S=np.zeros((B,B),complex);z=a*b*np.exp(1j*phase)
 S[tuple(p)]=z;S[(-p[0])%B,(-p[1])%B]=z.conj();w=np.fft.ifft2(S)
 assert np.max(np.abs(w.imag))<1e-10
 assert abs(w.real.mean())<1e-10 and np.isclose(np.mean(w.real**2),D)
 return w.real,b,p,a

def decode(y,B,p,k,coords):
 errors=[];out=[]
 for r,c in coords:
  s=np.abs(np.fft.fft2(y[r:r+B,c:c+B]))[tuple(p)]
  b=np.zeros(len(s),np.uint8);b[np.argsort(-s,kind='stable')[:k]]=1;out.append(b)
 return out

def jpeg(y,q):
 f=io.BytesIO();Image.fromarray(y).save(f,format='JPEG',quality=q,subsampling=0);f.seek(0)
 return np.array(Image.open(f),dtype=np.uint8)

def run(args):
 out=Path(args.out)
 if out.exists() and any(out.iterdir()): raise ValueError('Output must be empty; refusing mixed results')
 out.mkdir(parents=True,exist_ok=True)
 files = sorted(
  p for p in Path(args.data).rglob('*')
  if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'}
 )
 if not files:raise ValueError('No TIFF files')
 configs=[(B,L,'high') for B in (64,128,256) for L in (7,14,21,28,41) if not (B==64 and L>28)]
 configs += [(128,14,b) for b in ('low','mid')]+[(64,21,'legacy'),(128,41,'legacy')]
 rows=[];manifest=[]
 from concurrent.futures import ProcessPoolExecutor
 from functools import partial
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  for j,(part,meta) in enumerate(pool.map(partial(evaluate,args=args,configs=configs),files)):
   rows.extend(part);manifest.extend(meta)
   print(f'{j+1}/{len(files)} {meta[0]["file"]}',flush=True)
 pd.DataFrame(rows).to_csv(out/'raw.csv',index=False)
 pd.DataFrame(manifest).to_csv(out/'manifest.csv',index=False)
 (out/'protocol.json').write_text(json.dumps(dict(code_sha256={str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((PROJECT_ROOT/'Source').rglob('*.py'))},variance=args.variance,seeds=args.seeds,windows=args.windows,decoder='single-window magnitude top floor(L^2/2); windows averaged as repeated trials, never fused',configs=configs,python=platform.python_version(),numpy=np.__version__,pillow=pillow_version),indent=2))
 summarize(out)

def evaluate(f,args,configs):
 checkpoint=Path(args.out)/'checkpoints'/(hashlib.sha256(str(f.resolve()).encode()).hexdigest()[:16]+'.csv')
 rows=[];manifest=[]
 with Image.open(f) as im: image=np.array(im)
 if image.ndim!=2 or image.dtype!=np.uint8: raise ValueError(f'{f}: expected uint8 grayscale TIFF')
 H,W=image.shape;manifest.append({'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'height':H,'width':W})
 for B,L,band in configs:
  if B>min(H,W):continue
  for seed in args.seeds:
   w,b,p,a=make_watermark(B,L,band,seed,args.variance)
   addition=np.tile(w,((H+B-1)//B,(W+B-1)//B))[:H,:W]
   ideal=image.astype(float)+addition;y=np.rint(np.clip(ideal,0,255)).astype(np.uint8)
   mse=np.mean((y.astype(float)-image)**2);psnr=10*np.log10(255**2/mse)
   rng=np.random.default_rng(seed+int(hashlib.sha256(f.name.encode()).hexdigest()[:8],16))
   coords=[(int(rng.integers(H-B+1)),int(rng.integers(W-B+1))) for _ in range(args.windows)]
   attacks={'none':y}
   if B==128 and L==14 and band!='legacy':attacks.update({f'jpeg{q}':jpeg(y,q) for q in (70,90)})
   for attack,z in attacks.items():
    pred=decode(z,B,p,int(b.sum()),coords);bers=[float(np.mean(v!=b)) for v in pred]
    rows.append(dict(file=f.name,group=H,B=B,L=L,bits=L*L,band=band,seed=seed,attack=attack,ber=np.mean(bers),perfect=np.mean([v==0 for v in bers]),mse=mse,psnr=psnr,clip_fraction=np.mean((ideal<0)|(ideal>255)),alpha=a))
 checkpoint.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(checkpoint,index=False)
 return rows,manifest

def summarize(out):
 raw=pd.read_csv(out/'raw.csv'); keys=['group','B','L','bits','band','attack']
 per=raw.groupby(keys+['file'],as_index=False)[['ber','perfect','psnr','mse','clip_fraction']].mean();per.to_csv(out/'per_image.csv',index=False)
 result=[]
 for k,g in per.groupby(keys):
  x=g.ber.to_numpy();rng=np.random.default_rng(999);boot=x[rng.integers(len(x),size=(2000,len(x)))].mean(axis=1)
  row=dict(zip(keys,k));row.update(n=len(x),ber=x.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),pass80=np.mean(x<=.2),perfect=g.perfect.mean(),psnr=g.psnr.mean(),mse=g.mse.mean(),clip_fraction=g.clip_fraction.mean());result.append(row)
 pd.DataFrame(result).to_csv(out/'summary.csv',index=False)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4)
 ap.add_argument('--data', default=str(PROJECT_ROOT / 'Image'))
 ap.add_argument('--out', default=str(PROJECT_ROOT / 'Results' / 'Parameter_Research'))
 ap.add_argument('--variance',type=float,default=.72)
 ap.add_argument('--seeds',type=int,nargs='+',default=[1729,2718,3141])
 ap.add_argument('--windows',type=int,default=3)
 args=ap.parse_args()
 if not np.isfinite(args.variance) or args.variance<=0 or args.windows<1 or args.workers<1:ap.error('variance, windows and workers must be finite and positive')
 if min(args.seeds)<0 or len(set(args.seeds))!=len(args.seeds):ap.error('Seeds must be distinct nonnegative integers')
 run(args)
