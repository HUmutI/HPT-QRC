import os
import pandas as pd

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

datasets = [
    ("MSE", "S&P 500", "SP500_Realized_Volatility"),
    ("QLIKE", "S&P 500", "SP500_Realized_Volatility"),
    ("MSE", "NARMA10", "NARMA10"),
    ("QLIKE", "NARMA10", "NARMA10"),
    ("MSE", "Mackey-Glass", "Mackey_Glass"),
    ("QLIKE", "Mackey-Glass", "Mackey_Glass")
]

for crit, name, prefix in datasets:
    try:
        df = pd.read_csv(os.path.join(_ROOT, "results", f"{prefix}_DM_{crit}.csv"), index_col=0)
        print(f"\n### Diebold-Mariano Test ({name} - {crit})\n")
        cols = ["Model"] + list(df.columns)
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join(["---"] * len(cols)) + "|")
        
        for idx, row in df.iterrows():
            vals = []
            for v in row:
                if pd.isna(v):
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            print(f"| **{idx}** | " + " | ".join([f"`{v}`" if v else "" for v in vals]) + " |")
    except Exception as e:
        print(f"Skipping {prefix} - {crit}: {e}")
