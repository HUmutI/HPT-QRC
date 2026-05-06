import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from data_loader import load_sp500
from multi_qrc import HPT_QRC_Multi

os.makedirs("results", exist_ok=True)

def qlike(yt, yp):
    eps = 1e-8
    yt = np.abs(np.exp(np.array(yt).flatten())) + eps
    yp = np.abs(np.exp(np.array(yp).flatten())) + eps
    r = yt / yp
    return np.sum(r - np.log(r) - 1)

print("Loading S&P 500...")
y_train, y_test, X_train, X_test = load_sp500()

windows = [1, 2, 3, 5, 7, 10]
results = []

print("Tuning S&P 500 window sizes for HPT-QRC...")
for w in windows:
    # 1. Base HPT-QRC
    qrc = HPT_QRC_Multi(in_size=1, window=w, photon_list=[2,3,4], seed=42, use_har_context=False)
    qrc.fit(y_train)
    p = qrc.predict(y_test)
    mse = mean_squared_error(y_test, p)
    ql = qlike(y_test, p)
    
    # 2. Exogenous HPT-QRC-X (HAR context)
    qrc_in_dim = 1 + X_train.shape[1]
    qrc_x = HPT_QRC_Multi(in_size=qrc_in_dim, window=w, photon_list=[2,3,4], seed=42, use_har_context=True)
    qrc_x.fit(y_train, X_train)
    px = qrc_x.predict(y_test, X_test)
    mse_x = mean_squared_error(y_test, px)
    ql_x = qlike(y_test, px)
    
    results.append({
        "Window": w,
        "HPT-QRC_MSE": mse,
        "HPT-QRC_QLIKE": ql,
        "HPT-QRC-X_MSE": mse_x,
        "HPT-QRC-X_QLIKE": ql_x
    })
    print(f"Window={w:2d} | QRC MSE: {mse:.6f} | QRC-X MSE: {mse_x:.6f}")

df = pd.DataFrame(results)
df.to_csv("results/sp500_window_tuning.csv", index=False)
print("\n[✓] Saved tuning results to results/sp500_window_tuning.csv")
