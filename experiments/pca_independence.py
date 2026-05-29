import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / 'src'))
del _pathlib, _sys
from data_loader import load_mackey_glass
from multi_qrc import HPT_QRC_Multi
from esn_baseline import EchoStateNetwork

os.makedirs("results/advanced", exist_ok=True)

print("Running PCA / Linear Independence Analysis...")
y_train, y_test, _, _ = load_mackey_glass(n_train=1000)

# 1. HPT-QRC Features
qrc = HPT_QRC_Multi(in_size=1, window=10, photon_list=[2,3,4], seed=42)
F_qrc = qrc.get_features(y_train)

# 2. Classical ESN Features
esn = EchoStateNetwork(in_size=1, res_size=F_qrc.shape[1], spectral_radius=0.9, seed=42)
F_esn_full = esn._run_reservoir(y_train)[0]
F_esn = F_esn_full[-F_qrc.shape[0]:] # Match the same time window

# 3. PCA Analysis
pca_qrc = PCA().fit(F_qrc)
pca_esn = PCA().fit(F_esn)

var_qrc = np.cumsum(pca_qrc.explained_variance_ratio_)
var_esn = np.cumsum(pca_esn.explained_variance_ratio_)

# Plot Cumulative Variance
plt.figure(figsize=(8, 6))
plt.plot(var_esn, label="Classical ESN", color='red', lw=2)
plt.plot(var_qrc, label="HPT-QRC (Photonic)", color='blue', lw=2)
plt.axhline(y=0.95, color='k', linestyle='--', label="95% Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Feature Orthogonality (Quantum vs Classical)\nSlower Rise = Better Linear Independence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/advanced/pca_independence.png", dpi=150)
plt.close()

# 4. Condition Number (Matrix Rank Proxy)
cond_qrc = np.linalg.cond(F_qrc)
cond_esn = np.linalg.cond(F_esn)

print(f"HPT-QRC Condition Number: {cond_qrc:.2e} (Lower = More Independent Features)")
print(f"Classical ESN Condition Number: {cond_esn:.2e}")

with open("results/advanced/pca_stats.txt", "w") as f:
    f.write(f"HPT-QRC Condition Number: {cond_qrc:.2e}\n")
    f.write(f"Classical ESN Condition Number: {cond_esn:.2e}\n")

print("Saved PCA analysis to results/advanced/")
