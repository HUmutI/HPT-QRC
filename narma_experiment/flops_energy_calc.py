import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/advanced", exist_ok=True)

# Training HPT-QRC involves computing a single Ridge regression: (X^T X + alpha I)^{-1} X^T Y
# X is (N, D). D is number of features (e.g. 90), N is samples (e.g. 6000)
# Matrix multiplication X^T X takes O(N * D^2) FLOPs.
# Inversion of DxD takes O(D^3) FLOPs.
# For N=6000, D=90:
# X^T X FLOPs = 6000 * 90^2 = 4.86e7 FLOPs.
# Total is roughly 5e7 FLOPs.

# Training an LSTM with 100 hidden units, window 10, for 100 epochs on 6000 samples.
# Each LSTM cell step takes ~ 4 * (H^2 + H*I) FLOPs = 4 * (10000 + 100) = 40400 FLOPs.
# Backpropagation Through Time (BPTT) takes ~ 3x forward pass FLOPs = 1.2e5 FLOPs per step.
# For a window of 10, 1 sequence = 1.2e6 FLOPs.
# For N=6000 samples = 7.2e9 FLOPs per epoch.
# For 100 epochs = 7.2e11 FLOPs.

data = {
    "Model": ["Classical LSTM (100 Epochs)", "Classical LSTM (10 Epochs)", "HPT-QRC (Single Pass)"],
    "Trainable Parameters": [40800, 40800, 91],  # QRC only trains 90 Ridge weights + 1 bias
    "Estimated FLOPs (Millions)": [720000, 72000, 48.6]
}

df = pd.DataFrame(data)
df.to_csv("results/advanced/flops_comparison.csv", index=False)

plt.figure(figsize=(8, 6))
bars = plt.bar(df["Model"], df["Estimated FLOPs (Millions)"], color=['darkred', 'indianred', 'forestgreen'])
plt.yscale('log')
plt.ylabel("Estimated Training FLOPs (Millions, Log Scale)")
plt.title("The 'Green AI' Quantum Advantage:\nTraining Energy/Compute Cost")

# Add text labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval * 1.2, f"{yval:,.1f} M", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("results/advanced/flops_comparison.png", dpi=150)
plt.close()

print("Saved FLOPs/Energy comparison to results/advanced/")
