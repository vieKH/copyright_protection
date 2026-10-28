"""Plot ONLY actual 200-image results already stored in repository; no simulated data."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'Results/Voting_Research/comparison_200/summary.csv'
OUT=ROOT/'Results/Report_plan/verified_plots'

def main():
    if not SRC.is_file():raise FileNotFoundError(f'Missing observed 200-image CSV: {SRC}')
    with SRC.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    OUT.mkdir(parents=True,exist_ok=True)
    if any(int(r['images'])!=100 for r in rows):
        raise ValueError('Expected 100 images per resolution/method/K; refusing partial plot')
    for size in (512,1024):
        plt.figure(figsize=(7.8,4.8))
        for method in ('magnitude_vote','phase_vote','complex_average'):
            subset=sorted((r for r in rows if int(r['dataset'])==size and r['method']==method),key=lambda r:int(r['blocks_used']))
            if not subset:raise RuntimeError(f'Missing observations: {size}, {method}')
            plt.plot([int(r['blocks_used']) for r in subset], [100*float(r['mean_BER']) for r in subset],marker='o',label=method)
        plt.xlabel('Number of windows K');plt.ylabel('Mean BER (%)')
        plt.title(f'200-image repository dataset: {size}x{size} (n=100)')
        plt.ylim(bottom=0);plt.grid(alpha=.3);plt.legend();plt.tight_layout()
        plt.savefig(OUT/f'fig_ch3_ber_vs_k_{size}.png',dpi=180);plt.close()
    print('Charts saved:',OUT)

if __name__=='__main__':
    main()
