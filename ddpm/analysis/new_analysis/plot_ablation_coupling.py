"""
Plot ablation coupling ranking and diagnostics.

Generates:
 - `ablation_coupling_bar.png`: sorted bar chart of coupling scores
 - `ablation_coupling_scatter.png`: scatter of sep vs angle colored by score
 - `ablation_coupling_topN.csv`: top-N rows CSV

Run from repo root:
    python3 ddpm/analysis/new_analysis/plot_ablation_coupling.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path('/scratch3/shaiq_home/repos/behaviour_ddpm')
IN_CSV = REPO / 'ddpm' / 'analysis' / 'new_analysis' / 'results' / 'ablation_coupling_ranking.csv'
OUT_DIR = REPO / 'ddpm' / 'analysis' / 'new_analysis' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not IN_CSV.exists():
        print('Ranking CSV not found at', IN_CSV)
        return 1

    df = pd.read_csv(IN_CSV)
    # ensure numeric sorting by coupling_score
    df['coupling_score'] = pd.to_numeric(df['coupling_score'], errors='coerce')
    df_sorted = df.sort_values('coupling_score', ascending=False).reset_index(drop=True)

    # Bar plot of coupling scores
    plt.figure(figsize=(10,4))
    plt.bar(range(len(df_sorted)), df_sorted['coupling_score'], color='C0')
    plt.xlabel('Run (sorted)')
    plt.ylabel('Coupling score (std units)')
    plt.title('Ablation coupling ranking (higher = stronger)')
    # annotate with index where available
    labels = df_sorted['index'].fillna('').astype(str).tolist()
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.tight_layout()
    out_bar = OUT_DIR / 'ablation_coupling_bar.png'
    plt.savefig(out_bar, dpi=200)
    plt.close()

    # Scatter plot: prep separation vs target-distractor angle, color by score
    plt.figure(figsize=(6,5))
    x = pd.to_numeric(df_sorted['sep'], errors='coerce')
    y = pd.to_numeric(df_sorted['angle'], errors='coerce')
    s = pd.to_numeric(df_sorted['coupling_score'], errors='coerce')
    sc = plt.scatter(x, y, c=s, cmap='coolwarm', s=80, edgecolor='k')
    plt.colorbar(sc, label='coupling score')
    plt.xlabel('Prep separation')
    plt.ylabel('Target-distractor plane angle (deg)')
    plt.title('Coupling diagnostics')
    for i, txt in enumerate(df_sorted['index'].fillna('')):
        plt.annotate(str(int(txt)) if txt!='' else '', (x.iat[i], y.iat[i]), textcoords='offset points', xytext=(3,3), fontsize=8)
    plt.tight_layout()
    out_scatter = OUT_DIR / 'ablation_coupling_scatter.png'
    plt.savefig(out_scatter, dpi=200)
    plt.close()

    # Save top-N CSV
    topn = df_sorted.head(6)
    out_top = OUT_DIR / 'ablation_coupling_top6.csv'
    topn.to_csv(out_top, index=False)

    print('Wrote:', out_bar, out_scatter, out_top)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
