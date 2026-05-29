import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

datasets = [
    ("MSE", "S&P 500", "SP500_Realized_Volatility"),
    ("QLIKE", "S&P 500", "SP500_Realized_Volatility"),
    ("MSE", "NARMA10", "NARMA10"),
    ("QLIKE", "NARMA10", "NARMA10"),
    ("MSE", "Mackey-Glass", "Mackey_Glass"),
    ("QLIKE", "Mackey-Glass", "Mackey_Glass")
]

os.makedirs("results/dm_heatmaps", exist_ok=True)

for crit, name, prefix in datasets:
    csv_path = f"results/{prefix}_DM_{crit}.csv"
    if not os.path.exists(csv_path):
        continue
    
    df = pd.read_csv(csv_path, index_col=0)
    # Fill diagonal with 0
    df = df.fillna(0)
    
    plt.figure(figsize=(10, 8))
    # Diverging colormap: red means negative (row is better than col), blue means positive (row is worse than col)
    # Wait, usually a positive DM statistic means loss(row) > loss(col), so row is WORSE.
    # Let's make "better" (negative DM statistic) green, and "worse" (positive DM statistic) red.
    # RdYlGn_r: green for low (negative) values, red for high (positive) values.
    
    # We cap the scale at -5 to +5 for visual clarity since DM stats can be large.
    vmin, vmax = -5, 5
    
    # Custom annot: only show significance stars or raw values
    # Let's show raw values
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn_r", center=0, 
                vmin=vmin, vmax=vmax, linewidths=.5, cbar_kws={'label': 'Diebold-Mariano t-statistic'})
    
    plt.title(f"Diebold-Mariano Test: {name} ({crit})\n(Green: Row is better than Column)", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    out_path = f"results/dm_heatmaps/{prefix}_DM_{crit}_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")
