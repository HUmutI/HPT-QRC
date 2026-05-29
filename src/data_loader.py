import os
import numpy as np
import pandas as pd

_DEFAULT_SP500_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'literature', 'qrc_repo', 'Data.CSV'
)

def load_vix(train_ratio=0.8):
    """
    Download VIX daily close prices from Yahoo Finance and compute log-VIX.
    Treats log-VIX as the target (analogous to log-RV for S&P 500).
    Returns (y_train, y_test, X_train, X_test) in the same format as load_sp500().
    X_exog is None since VIX is univariate.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    df = yf.download("^VIX", start="2000-01-01", end="2024-12-31",
                     progress=False, auto_adjust=True)
    close = df["Close"].dropna().values.flatten()
    log_vix = np.log(close).reshape(-1, 1)

    n = len(log_vix)
    n_train = int(n * train_ratio)

    y_train = log_vix[:n_train]
    y_test  = log_vix[n_train:]
    return y_train, y_test, None, None   # no exogenous variables

def load_narma10(n_train=800, n_test=200, warmup=1000, seed=42):
    """
    Generate the classic NARMA10 dataset.
    """
    np.random.seed(seed)
    total_length = warmup + n_train + n_test
    
    u = 0.5 * np.random.rand(total_length)
    y = np.zeros(total_length)
    
    for t in range(9, total_length - 1):
        y[t+1] = (0.3 * y[t] 
                  + 0.05 * y[t] * np.sum(y[t-9:t+1]) 
                  + 1.5 * u[t] * u[t-9] 
                  + 0.1)
        
    u_final = u[warmup:]
    y_final = y[warmup:]
    
    X_train = u_final[:n_train].reshape(-1, 1)
    y_train = y_final[:n_train].reshape(-1, 1)
    X_test = u_final[n_train:].reshape(-1, 1)
    y_test = y_final[n_train:].reshape(-1, 1)
    
    return X_train, y_train, X_test, y_test

def load_sp500(filepath=_DEFAULT_SP500_CSV,
               lag_exog: int = 1):
    """
    Loads the Realized Volatility data from the QRC repository.
    The file 'Data.CSV' contains 'RV' as the target, and 'MKT', 'SMB', 'HML' etc as exogenous regressors.

    Leakage-control: exogenous regressors are *lagged* by `lag_exog` steps (default 1) so that
    the feature row aligned with target t uses information available no later than t-1. The
    Fama-French / macro variables in this dataset are monthly aggregates published with a delay
    in practice; using contemporaneous values would over-state predictive power.

    Returns (y_train, y_test, X_train, X_test). Default 80/20 split (kept for legacy benchmarks).
    For walk-forward CV use `load_sp500_df` + `walk_forward.WalkForwardSplit`.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"S&P 500 Dataset not found at {filepath}")

    df = pd.read_csv(filepath)

    y_target = df['RV'].values.reshape(-1, 1)

    exog_cols = ['MKT', 'SMB', 'HML', 'TB', 'DEF', 'IP', 'INF']
    X_exog = df[exog_cols].values

    # Lag exogenous by `lag_exog` steps to avoid contemporaneous leakage.
    if lag_exog > 0:
        X_exog = np.vstack([np.zeros((lag_exog, X_exog.shape[1])), X_exog[:-lag_exog]])

    n_samples = len(df)
    n_train = int(n_samples * 0.8)

    y_train = y_target[:n_train]
    y_test  = y_target[n_train:]
    X_train = X_exog[:n_train]
    X_test  = X_exog[n_train:]

    return y_train, y_test, X_train, X_test


def load_sp500_df(filepath=_DEFAULT_SP500_CSV,
                  lag_exog: int = 1) -> pd.DataFrame:
    """
    Date-indexed view of the S&P 500 RV dataset for walk-forward CV.

    Columns: 'y' (RV target), plus lagged Fama-French/macro factors X_*.
    Index: pandas DatetimeIndex from the 'Date' column.
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    exog_cols = ['MKT', 'SMB', 'HML', 'TB', 'DEF', 'IP', 'INF']
    out = pd.DataFrame(index=df.index)
    out['y'] = df['RV'].values
    if lag_exog > 0:
        for c in exog_cols:
            out[f'X_{c}'] = df[c].shift(lag_exog).values
    else:
        for c in exog_cols:
            out[f'X_{c}'] = df[c].values
    return out.dropna()


def load_vix_df(start: str = "2000-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Date-indexed view of log-VIX for walk-forward CV. Columns: 'y'. No exog.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance not installed. Run: pip install yfinance") from exc
    df = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    close = df["Close"].dropna()
    out = pd.DataFrame({"y": np.log(close.values.flatten())}, index=close.index)
    return out

def load_mackey_glass(n_train=800, n_test=200, warmup=1000, tau=17, seed=42):
    """
    Generate the classic Mackey-Glass time series dataset.
    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

    Uses TAU-step-ahead prediction (default tau=17), which is the standard
    QRC/ESN benchmark task. 1-step-ahead is trivially solved by linear models
    due to the series' smoothness (lag-1 autocorr ~0.99), so tau-step-ahead
    is required to fairly test temporal memory capacity.

    Returns (X_train, y_train, X_test, y_test) where
      y[i] = x[i + tau]  -- the target is tau steps into the future.
    """
    np.random.seed(seed)

    beta  = 0.2
    gamma = 0.1
    n     = 10
    dt    = 1.0

    # Need extra `tau` points at the end so that y[i+tau] exists for all i
    total_len = warmup + tau + n_train + n_test + tau
    time_series = np.zeros(total_len)

    # Initial conditions
    time_series[:tau] = 1.2 + 0.1 * (np.random.rand(tau) - 0.5)

    for t in range(tau, total_len - 1):
        x_tau = time_series[t - tau]
        delta = (beta * x_tau / (1.0 + x_tau**n)) - (gamma * time_series[t])
        time_series[t + 1] = time_series[t] + delta * dt

    # Discard warmup; y_final has length n_train + n_test + tau
    y_final = time_series[warmup + tau:]

    # X[i] = y_final[i],  target[i] = y_final[i + tau]
    X_train = y_final[:n_train].reshape(-1, 1)
    y_train = y_final[tau : n_train + tau].reshape(-1, 1)
    X_test  = y_final[n_train : n_train + n_test].reshape(-1, 1)
    y_test  = y_final[n_train + tau : n_train + n_test + tau].reshape(-1, 1)

    return X_train, y_train, X_test, y_test
