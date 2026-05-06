import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from data_loader import load_sp500
from multi_qrc import HPT_QRC_Multi

os.makedirs("results/advanced", exist_ok=True)

class RecursiveLeastSquares:
    def __init__(self, n_features, lambda_=0.99, delta=1.0):
        self.w = np.zeros(n_features)
        self.P = np.eye(n_features) / delta
        self.lambda_ = lambda_

    def predict(self, x):
        return np.dot(x, self.w)

    def update(self, x, y):
        # x shape (n_features,)
        pi = np.dot(self.P, x)
        k = pi / (self.lambda_ + np.dot(x, pi))
        error = y - np.dot(x, self.w)
        self.w = self.w + k * error
        self.P = (self.P - np.outer(k, np.dot(x, self.P))) / self.lambda_
        return error

print("Running Online Streaming (RLS) Analysis on S&P 500...")
y_train, y_test, _, _ = load_sp500()

# Concatenate for a continuous streaming scenario
y_full = np.concatenate([y_train, y_test])

qrc = HPT_QRC_Multi(in_size=1, window=3, photon_list=[2,3,4], seed=42)
# Get features for all streaming data
F_full = qrc.get_features(y_full)

# Add bias
F_full = np.hstack([F_full, np.ones((F_full.shape[0], 1))])

# The target for step t is y_full[t] (after the window shift)
y_targets = y_full[qrc.window:]

min_len = min(len(F_full), len(y_targets))
F_full = F_full[:min_len]
y_targets = y_targets[:min_len]

rls = RecursiveLeastSquares(n_features=F_full.shape[1], lambda_=0.99, delta=0.1)

errors = []
preds = []
for t in range(len(F_full)):
    x_t = F_full[t]
    y_t = y_targets[t][0]
    
    # Predict before updating (true streaming out-of-sample)
    pred_t = rls.predict(x_t)
    preds.append(pred_t)
    
    # Update weights
    err = rls.update(x_t, y_t)
    errors.append(err**2)

# Calculate Cumulative MSE
cum_mse = np.cumsum(errors) / np.arange(1, len(errors) + 1)

plt.figure(figsize=(10, 5))
plt.plot(cum_mse, color='darkorange', lw=2)
plt.title("HPT-QRC Online Streaming Performance (Recursive Least Squares)\n(Model updates weights instantly step-by-step)")
plt.xlabel("Streaming Time Steps (S&P 500 Trading Days)")
plt.ylabel("Cumulative Streaming MSE")
plt.grid(True)
plt.axvline(x=len(y_train), color='k', linestyle='--', label="Start of Test Period")
plt.legend()
plt.tight_layout()
plt.savefig("results/advanced/online_rls_learning.png", dpi=150)
plt.close()

print(f"Final Streaming MSE: {cum_mse[-1]:.6f}")
print("Saved Online Learning plot to results/advanced/")
