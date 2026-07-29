"""Benchmark registry with literature-standard protocols.

Each task returns ``(u, y, split)`` where ``u`` is the driving input the model is allowed
to see and ``y`` the aligned target. Two families:

* **Driven tasks** (NARMA-N, Santa Fe): ``u`` is an exogenous signal and the model must
  build the target's own dynamics internally. This is the protocol published NARMA numbers
  use, and it is the one the predecessor code did *not* implement -- ``multi_qrc`` was fed
  the target's own history, which makes the task substantially easier and its numbers
  incomparable with the literature.
* **Autoregressive tasks** (Mackey-Glass, Lorenz, S&P 500 RV, VIX): ``u_t = y_{t-1}``, i.e.
  one-step-ahead forecasting from the observable's own past.

``y_autoregressive_narma10`` is kept so the predecessor's numbers remain reproducible, but
it is labelled clearly and is not the headline task.
"""

from __future__ import annotations

import numpy as np

from .rc_protocol import Split

__all__ = ["TASKS", "load_task", "narma", "mackey_glass", "lorenz63"]


def narma(order: int = 10, n: int = 1000, seed: int = 42, warmup: int = 1000):
    """NARMA-N driven by i.i.d. uniform noise on [0, 0.5].

    Uses the standard coefficients (Atiya & Parlos 2000). For orders above 10 the classic
    recurrence is unstable, so the conventional tanh-stabilised form is used at order 20.
    """
    rng = np.random.default_rng(seed)
    total = n + warmup
    u = rng.uniform(0.0, 0.5, total)
    y = np.zeros(total)
    for t in range(order - 1, total - 1):
        window_sum = np.sum(y[t - order + 1 : t + 1])
        inner = 0.3 * y[t] + 0.05 * y[t] * window_sum + 1.5 * u[t - order + 1] * u[t] + 0.1
        y[t + 1] = np.tanh(inner) if order >= 20 else inner
    if not np.all(np.isfinite(y)):
        raise RuntimeError(f"NARMA-{order} diverged; try a different seed")
    return u[warmup:].reshape(-1, 1), y[warmup:]


def mackey_glass(n: int = 1000, tau: int = 17, horizon: int = 1, seed: int = 42,
                 warmup: int = 1000, dt: float = 1.0):
    """Mackey-Glass series, predicted ``horizon`` steps ahead from its own past."""
    beta, gamma, exponent = 0.2, 0.1, 10
    total = n + warmup + tau + horizon
    rng = np.random.default_rng(seed)
    x = np.zeros(total + tau)
    x[:tau] = 1.2 + 0.01 * rng.standard_normal(tau)
    for t in range(tau, total + tau - 1):
        x[t + 1] = x[t] + dt * (beta * x[t - tau] / (1 + x[t - tau] ** exponent) - gamma * x[t])
    series = x[tau + warmup :]
    u = series[:-horizon].reshape(-1, 1)
    y = series[horizon:]
    return u[:n], y[:n]


def lorenz63(n: int = 1000, horizon: int = 1, dt: float = 0.02, seed: int = 42,
             warmup: int = 2000):
    """Lorenz-63 x-coordinate, predicted ``horizon`` steps ahead from its own past."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    rng = np.random.default_rng(seed)
    state = np.array([1.0, 1.0, 1.0]) + 0.01 * rng.standard_normal(3)
    total = n + warmup + horizon
    traj = np.empty(total)
    for t in range(total):
        x, y_, z = state
        # Fourth-order Runge-Kutta; Euler drifts noticeably at this step size.
        def deriv(s):
            a, b, c = s
            return np.array([sigma * (b - a), a * (rho - c) - b, a * b - beta * c])
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[t] = state[0]
    series = traj[warmup:]
    return series[:-horizon].reshape(-1, 1)[:n], series[horizon:][:n]


def _autoregressive(series: np.ndarray, horizon: int = 1):
    series = np.asarray(series, dtype=float).ravel()
    return series[:-horizon].reshape(-1, 1), series[horizon:]


def _sp500():
    from .data_loader import load_sp500

    y_train, y_test, x_train, x_test = load_sp500()
    series = np.concatenate([np.ravel(y_train), np.ravel(y_test)])
    return _autoregressive(series)


def _vix():
    from .data_loader import load_vix

    y_train, y_test, _, _ = load_vix()
    series = np.concatenate([np.ravel(y_train), np.ravel(y_test)])
    return _autoregressive(series)


def _y_autoregressive_narma10(n: int = 1000, seed: int = 42):
    """The predecessor protocol: predict NARMA-10 output from its *own* history.

    Retained only so ``multi_qrc``-era numbers stay reproducible. It is a much easier task
    than driven NARMA-10 -- a plain ridge on the target's own lags nearly solves it -- and
    it is not comparable to any published NARMA-10 result.
    """
    _, y = narma(10, n=n, seed=seed)
    return _autoregressive(y)


def _split_for(n_total: int, washout: int = 200, train_frac: float = 0.8,
               val_frac: float = 0.15) -> Split:
    n_train = int(round(train_frac * n_total))
    return Split(n_total, washout=washout, n_train=n_train, n_val=int(round(val_frac * n_train)))


TASKS: dict[str, callable] = {
    "narma5": lambda: narma(5, n=1000),
    "narma10": lambda: narma(10, n=1000),
    "narma20": lambda: narma(20, n=1000),
    "narma10_long": lambda: narma(10, n=3000),
    "mackey_glass": lambda: mackey_glass(n=1000, horizon=1),
    "mackey_glass_h17": lambda: mackey_glass(n=1000, horizon=17),
    # Horizon 1 is not a benchmark: at dt=0.02 the next Lorenz sample is almost a linear
    # function of the current one, and both a tuned ESN and a plain ridge score ~1e-4.
    # Horizon 20 is ~0.4 Lyapunov times ahead, where the task actually discriminates.
    "lorenz63": lambda: lorenz63(n=2000, horizon=20),
    "sp500_rv": _sp500,
    "vix": _vix,
    "narma10_autoregressive": _y_autoregressive_narma10,
}


def load_task(name: str, seed: int | None = None):
    """Return ``(u, y, split)`` for a registered task."""
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    if seed is None:
        u, y = TASKS[name]()
    else:
        try:
            u, y = _reseed(name, seed)
        except TypeError:
            u, y = TASKS[name]()
    washout = 200 if len(y) > 600 else max(20, len(y) // 10)
    return u, y, _split_for(len(y), washout=washout)


def _reseed(name: str, seed: int):
    """Regenerate a synthetic task with a different noise realisation."""
    if name.startswith("narma") and name != "narma10_autoregressive":
        order = {"narma5": 5, "narma10": 10, "narma20": 20, "narma10_long": 10}[name]
        n = 3000 if name == "narma10_long" else 1000
        return narma(order, n=n, seed=seed)
    if name == "mackey_glass":
        return mackey_glass(n=1000, horizon=1, seed=seed)
    if name == "mackey_glass_h17":
        return mackey_glass(n=1000, horizon=17, seed=seed)
    if name == "lorenz63":
        return lorenz63(n=2000, horizon=20, seed=seed)
    if name == "narma10_autoregressive":
        return _y_autoregressive_narma10(seed=seed)
    raise TypeError(f"task {name!r} is a fixed empirical dataset and cannot be reseeded")
