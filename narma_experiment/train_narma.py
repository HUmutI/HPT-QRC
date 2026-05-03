import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import mean_squared_error

from data_loader import load_narma10, load_sp500, load_mackey_glass
from classical_baselines import ARModel, ARMAXModel, HARModel, HARXModel, LSTMWrapper, RCModel, ClassicalContextRidge
from multi_qrc import HPT_QRC_Multi

matplotlib.use("Agg")

def qlike_loss(y_true, y_pred):
    """
    QLIKE Loss used in volatility forecasting:
    L(y, y_hat) = y / y_hat - log(y / y_hat) - 1
    Uses absolute values and epsilons for stability.
    Matches QRC Paper code implementation: unnormalized np.sum.
    Both arrays are flattened first to avoid NumPy broadcasting creating
    an N×N matrix when shapes differ (e.g. (N,1) vs (N,)).
    """
    eps = 1e-8
    y_t = np.abs(np.array(y_true).flatten()) + eps
    y_p = np.abs(np.array(y_pred).flatten()) + eps
    ratio = y_t / y_p
    loss = ratio - np.log(ratio) - 1
    return np.sum(loss)

def dm_test(y_true, pred1, pred2, crit="MSE", h=1):
    """
    Computes the Diebold-Mariano test statistic between two arrays of predictions.
    Null Hypothesis: The two models have equal predictive accuracy.
    """
    e1 = y_true.flatten() - pred1.flatten()
    e2 = y_true.flatten() - pred2.flatten()
    
    if crit == "MSE":
        d = e1**2 - e2**2
    elif crit == "QLIKE":
        eps = 1e-8
        y = np.abs(y_true.flatten()) + eps
        p1 = np.abs(pred1.flatten()) + eps
        p2 = np.abs(pred2.flatten()) + eps
        d = (y/p1 - np.log(y/p1) - 1) - (y/p2 - np.log(y/p2) - 1)
    else:
        d = np.abs(e1) - np.abs(e2)
        
    d_mean = np.mean(d)
    T = float(len(d))
    
    gamma = []
    for lag in range(0, h):
        gamma.append(np.sum((d[lag:] - d_mean) * (d[:len(d)-lag] - d_mean)) / T)
        
    V_d = gamma[0] + 2 * sum(gamma[1:])
    
    if V_d == 0:
        return 0.0, 1.0
        
    stat = d_mean / np.sqrt(V_d / T)
    p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(stat)))
    return stat, p_value

def generate_dm_table(y_true, preds_dict, crit="MSE"):
    """
    Generates a combined DM table similar to literature:
    Lower triangle = DM stat values
    Upper triangle = p-values
    """
    models = list(preds_dict.keys())
    n = len(models)
    combined = pd.DataFrame(index=models, columns=models, dtype=object)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                combined.iloc[i, j] = ""
            else:
                stat, p = dm_test(y_true, preds_dict[models[i]], preds_dict[models[j]], crit=crit)
                if i > j: # Lower triangular -> DM Stat
                    combined.iloc[i, j] = f"{stat:.3f}"
                else: # Upper triangular -> p-value
                    combined.iloc[i, j] = f"{p:.3f}"
    return combined

def pmcs_score(loss_values):
    """
    Placeholder for Model Confidence Set PMCS score computation.
    Computing a true PMCS requires block bootstrapping over the entire loss matrix. 
    Here we simply normalize the mean loss across models to mimic the tabular values presented.
    In real benchmarks, the 'target' PMCS is an iterative p-value elimination.
    We will just display the mean loss as standard, to replicate the metrics.
    """
    return np.mean(loss_values)

def benchmark_dataset(name, y_train, y_test, X_train=None, X_test=None, is_log_space=False, use_har_context=False):
    print("=" * 60)
    print(f"      Running Extended Benchmarks on {name}")
    print("=" * 60)
    
    results = []
    preds = {}
    
    # Helper: compute QLIKE, exponentiating if data is in log space (e.g. S&P 500 log-RV)
    def qlike(y_true, y_pred):
        if is_log_space:
            return qlike_loss(np.exp(y_true), np.exp(y_pred))
        return qlike_loss(y_true, y_pred)
    
    # 1. AR(1)
    print("-> Training AR(1)...")
    ar1 = ARModel(lags=1).fit(y_train)
    p_ar1 = ar1.predict(y_test)
    preds["AR1"] = p_ar1
    results.append({"Model": "AR1", "MSE": mean_squared_error(y_test, p_ar1), "QLIKE": qlike(y_test, p_ar1)})
    
    # 2. AR(3)
    print("-> Training AR(3)...")
    ar3 = ARModel(lags=3).fit(y_train)
    p_ar3 = ar3.predict(y_test)
    preds["AR3"] = p_ar3
    results.append({"Model": "AR3", "MSE": mean_squared_error(y_test, p_ar3), "QLIKE": qlike(y_test, p_ar3)})
    
    # 3. HAR
    print("-> Training HAR...")
    har = HARModel().fit(y_train)
    p_har = har.predict(y_test)
    preds["HAR"] = p_har
    results.append({"Model": "HAR", "MSE": mean_squared_error(y_test, p_har), "QLIKE": qlike(y_test, p_har)})
    
    # 4. LSTM
    print("-> Training LSTM...")
    lstm = LSTMWrapper(input_dim=1).fit(y_train, y_train)
    p_lstm = lstm.predict(y_test, y_test)
    preds["LSTM"] = p_lstm
    results.append({"Model": "LSTM", "MSE": mean_squared_error(y_test, p_lstm), "QLIKE": qlike(y_test, p_lstm)})
    
    # 5. RC (Echo State Network)
    print("-> Training RC (Classic ESN)...")
    rc = RCModel(in_size=1).fit(y_train)
    p_rc = rc.predict(y_test)
    preds["RC"] = p_rc
    results.append({"Model": "RC", "MSE": mean_squared_error(y_test, p_rc), "QLIKE": qlike(y_test, p_rc)})

    # 6. Classical Context Ridge (no-quantum ablation)
    print("-> Training ClassicalContextRidge (no-quantum ablation)...")
    ccr = ClassicalContextRidge(window=5, ridge_alpha=10.0).fit(y_train)
    p_ccr = ccr.predict(y_test)
    preds["Classical-Ridge"] = p_ccr
    results.append({"Model": "Classical-Ridge", "MSE": mean_squared_error(y_test, p_ccr), "QLIKE": qlike(y_test, p_ccr)})

    # 7. HPT-QRC (Quantum Reservoir)
    print("-> Training HPT-QRC (Quantum Reservoir)...")
    qrc = HPT_QRC_Multi(in_size=1, window=10, photon_list=[2,3,4],
                        use_har_context=use_har_context).fit(y_train)
    p_qrc = qrc.predict(y_test)
    preds["HPT-QRC"] = p_qrc
    results.append({"Model": "HPT-QRC", "MSE": mean_squared_error(y_test, p_qrc), "QLIKE": qlike(y_test, p_qrc)})

    # Exogenous Models (if X is provided)
    if X_train is not None and X_test is not None:
        print("--- Running Exogenous Variants ---")
        
        # 7. ARMAX
        print("-> Training ARMAX(1,0,0)...")
        armax = ARMAXModel(lags=1).fit(y_train, X_train)
        p_armax = armax.predict(y_test, X_test)
        preds["ARMAX"] = p_armax
        results.append({"Model": "ARMAX", "MSE": mean_squared_error(y_test, p_armax), "QLIKE": qlike(y_test, p_armax)})
        
        # 8. HARX
        print("-> Training HARX...")
        harx = HARXModel().fit(y_train, X_train)
        p_harx = harx.predict(y_test, X_test)
        preds["HARX"] = p_harx
        results.append({"Model": "HARX", "MSE": mean_squared_error(y_test, p_harx), "QLIKE": qlike(y_test, p_harx)})
        
        # 9. LSTMX
        print("-> Training LSTMX...")
        lstm_in_dim = 1 + X_train.shape[1]
        concat_train = np.hstack([y_train, X_train])
        concat_test = np.hstack([y_test, X_test])
        lstmx = LSTMWrapper(input_dim=lstm_in_dim).fit(concat_train, y_train)
        p_lstmx = lstmx.predict(concat_test, y_test)
        preds["LSTMX"] = p_lstmx
        results.append({"Model": "LSTMX", "MSE": mean_squared_error(y_test, p_lstmx), "QLIKE": qlike(y_test, p_lstmx)})
        
        # 10. RCX
        print("-> Training RCX (Exogenous ESN)...")
        rc_in_dim = 1 + X_train.shape[1]
        rcx = RCModel(in_size=rc_in_dim).fit(y_train, X_train)
        p_rcx = rcx.predict(y_test, X_test)
        preds["RCX"] = p_rcx
        results.append({"Model": "RCX", "MSE": mean_squared_error(y_test, p_rcx), "QLIKE": qlike(y_test, p_rcx)})
        
        # 11. HPT-QRC-X (Exogenous Quantum Reservoir)
        print("-> Training HPT-QRC-X (Exogenous Quantum)...")
        qrc_in_dim = 1 + X_train.shape[1]
        qrc_x = HPT_QRC_Multi(in_size=qrc_in_dim, window=10, photon_list=[2,3,4],
                              use_har_context=use_har_context).fit(y_train, X_train)
        p_qrc_x = qrc_x.predict(y_test, X_test)
        preds["HPT-QRC-X"] = p_qrc_x
        results.append({"Model": "HPT-QRC-X", "MSE": mean_squared_error(y_test, p_qrc_x), "QLIKE": qlike(y_test, p_qrc_x)})

    df_res = pd.DataFrame(results)
    
    # Generate DM Table for MSE
    dm_table_mse = generate_dm_table(y_test, preds, crit="MSE")
    # Generate DM Table for QLIKE
    dm_table_qlike = generate_dm_table(y_test, preds, crit="QLIKE")
    
    # Save Results
    os.makedirs("results", exist_ok=True)
    df_res.to_csv(f"results/{name}_benchmark.csv", index=False)
    dm_table_mse.to_csv(f"results/{name}_DM_MSE.csv")
    dm_table_qlike.to_csv(f"results/{name}_DM_QLIKE.csv")
    
    # Generate Plot
    plt.figure(figsize=(14, 7))
    plt.plot(y_test[:100], label='Actual Target', color='black', linewidth=2)
    plt.plot(p_rc[:100], label='RC (Classic)', linestyle='--', alpha=0.7)
    plt.plot(p_qrc[:100], label='HPT-QRC (Quantum)', linestyle='-.', alpha=0.9, linewidth=2)
    if X_train is not None:
        plt.plot(p_qrc_x[:100], label='HPT-QRC-X (Exogenous)', linestyle=':', alpha=0.9, linewidth=2)
        
    plt.title(f"{name} Benchmark (First 100 Test Steps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{name}_overlay.png", dpi=150)
    plt.close()
    
    print(f"\n[DONE] {name} Benchmark Results:")
    print(df_res.to_string(index=False))
    print("-" * 60 + "\n")
    
    return df_res

def main():
    # 1. NARMA10 (Univariate)
    print("Loading NARMA10...")
    n_X_train, n_y_train, n_X_test, n_y_test = load_narma10()
    benchmark_dataset("NARMA10", n_y_train, n_y_test, X_train=n_X_train, X_test=n_X_test)
    
    # 2. Mackey-Glass (Univariate)
    print("Loading Mackey-Glass...")
    from data_loader import load_mackey_glass
    m_X_train, m_y_train, m_X_test, m_y_test = load_mackey_glass(n_train=800, n_test=200)
    benchmark_dataset("Mackey_Glass", m_y_train, m_y_test, X_train=m_X_train, X_test=m_X_test)
    
    # 3. S&P 500 (Multivariate with Exogenous Fama-French markers)
    print("Loading S&P 500...")
    # 'load_sp500' returns y (RV targets), and X (Exogenous markers)
    s_y_train, s_y_test, s_X_train, s_X_test = load_sp500()
    # is_log_space=True because the RV column is log-realized volatility (negative values)
    # QLIKE must be computed after exp() to convert back to actual variance space
    benchmark_dataset("SP500_Realized_Volatility", s_y_train, s_y_test, X_train=s_X_train, X_test=s_X_test, is_log_space=True, use_har_context=True)

if __name__ == "__main__":
    main()
