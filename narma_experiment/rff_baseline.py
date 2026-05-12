"""
rff_baseline.py
===============
Random Fourier Features (Rahimi & Recht, NeurIPS 2007) + Ridge regression.

This is the must-have classical comparator for HPT-QRC: a *random fixed
nonlinear feature map followed by Ridge*, structurally identical in spirit
to "fixed quantum reservoir + Ridge" but with no quantum component.
If HPT-QRC cannot beat (or honestly tie with) RFF+Ridge at matched feature
dimension, the photonic-feature claim cannot rest on feature quality.

Public API mirrors `HPT_QRC_Multi` so this drops into existing benchmark
loops (`train_narma.benchmark_dataset`, `multi_seed_benchmark.py`):

    rff = RFFRidge(in_size=1, window=10, output_dim=90).fit(y_train)
    p   = rff.predict(y_test)

    rffx = RFFRidge(in_size=1+X_train.shape[1], window=10, output_dim=90,
                    use_har_context=True).fit(y_train, X_train)
    p    = rffx.predict(y_test, X_test)
"""

import numpy as np
from sklearn.linear_model import Ridge


class RFFRidge:
    """Random Fourier Features (Gaussian kernel) + Ridge readout."""

    def __init__(
        self,
        in_size: int = 1,
        window: int = 10,
        output_dim: int = 90,
        gamma: float = 1.0,
        ridge_alpha: float = 1e-4,
        seed: int = 42,
        use_har_context: bool = False,
    ):
        self.in_size = in_size
        self.window = window
        self.output_dim = output_dim
        self.gamma = gamma
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self.use_har_context = use_har_context
        self.input_dim = in_size * window

        rng = np.random.default_rng(seed)
        # For Gaussian kernel k(x,y) = exp(-gamma ||x-y||^2), draw
        #   W ~ N(0, 2*gamma * I_d)  (one row per RFF output)
        #   b ~ U[0, 2*pi)
        self.W = rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 * gamma),
            size=(output_dim, self.input_dim),
        )
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=output_dim)
        self.scale = np.sqrt(2.0 / output_dim)
        self.ridge = Ridge(alpha=ridge_alpha)

    # ------------------------------------------------------------------
    # Robust scaler (mirrors HPT_QRC_Multi for parity)
    # ------------------------------------------------------------------
    def _fit_scaler(self, X_flat):
        p1, p99 = np.percentile(X_flat, [1, 99])
        clipped = np.clip(X_flat, p1, p99)
        med = np.median(clipped)
        iqr = np.percentile(clipped, 75) - np.percentile(clipped, 25)
        iqr = iqr if iqr > 1e-8 else 1.0
        scaled = (clipped - med) / iqr
        xmin, xmax = scaled.min(), scaled.max()
        rng_ = xmax - xmin if xmax > xmin else 1.0
        self._sc = dict(p1=p1, p99=p99, med=med, iqr=iqr, xmin=xmin, rng=rng_)

    def _apply_scaler(self, X):
        sc = self._sc
        Xc = np.clip(X, sc["p1"], sc["p99"])
        Xs = (Xc - sc["med"]) / sc["iqr"]
        return (Xs - sc["xmin"]) / sc["rng"]

    # ------------------------------------------------------------------
    # HAR-style classical context (mirrors HPT_QRC_Multi._build_har_context)
    # ------------------------------------------------------------------
    def _build_har_context(self, y_1d):
        pad = max(22, self.window)
        padded = np.concatenate([np.zeros(pad), y_1d])
        feats = []
        for t in range(pad, len(padded)):
            win = padded[t - self.window : t]
            lag1 = padded[t - 1]
            ma5 = np.mean(padded[max(0, t - 5) : t])
            ma22 = np.mean(padded[max(0, t - 22) : t])
            delt = padded[t - 1] - padded[t - 2]
            feats.append(np.concatenate([win, [lag1, ma5, ma22, delt]]))
        return np.array(feats)

    # ------------------------------------------------------------------
    # Sliding windows + RFF map
    # ------------------------------------------------------------------
    def _windows(self, X):
        X_dim = self.in_size
        padded = np.vstack([np.zeros((self.window - 1, X_dim)), X.reshape(-1, X_dim)])
        Xw = []
        for t in range(self.window - 1, len(padded)):
            chunk = padded[t - self.window + 1 : t + 1].flatten()
            if len(chunk) < self.input_dim:
                chunk = np.pad(chunk, (0, self.input_dim - len(chunk)))
            elif len(chunk) > self.input_dim:
                chunk = chunk[: self.input_dim]
            Xw.append(chunk)
        return np.array(Xw)

    def _rff(self, Xw):
        proj = Xw @ self.W.T + self.b  # (T, output_dim)
        return self.scale * np.cos(proj)

    def _create_features(self, X, y_1d):
        Xw = self._windows(X)
        Q = self._rff(Xw)
        if self.use_har_context:
            classical = self._build_har_context(y_1d)
            classical = classical[-len(Q):]
        else:
            classical = Xw
        return np.hstack([Q, classical])

    # ------------------------------------------------------------------
    # Public API matching HPT_QRC_Multi
    # ------------------------------------------------------------------
    def fit(self, y_train, X_exog=None, discard_steps=100):
        if X_exog is not None:
            features = np.hstack([y_train, X_exog])
        else:
            features = y_train

        X_in = features[:-1]
        y_target = y_train[1:]
        y_1d = y_train.flatten()[:-1]

        F = self._create_features(X_in, y_1d)
        F_fit = F[discard_steps:, :]
        y_fit = y_target[discard_steps:]

        self.ridge.fit(F_fit, y_fit)

        self.last_X = features[-self.window:].copy()
        self.last_y = y_train.flatten()[-max(22, self.window):]
        return self

    def predict(self, y_test, X_exog=None):
        if X_exog is not None:
            features = np.hstack([y_test, X_exog])
        else:
            features = y_test

        concat_features = np.vstack([self.last_X, features[:-1]])
        y_1d_concat = np.concatenate([self.last_y, y_test.flatten()[:-1]])

        F = self._create_features(concat_features, y_1d_concat)
        return self.ridge.predict(F[-len(features):])


# ---------------------------------------------------------------------------
# Convenience: build an RFFRidge whose feature dim matches a given HPT-QRC.
# ---------------------------------------------------------------------------
def match_qrc_feature_dim(hpt_qrc_multi) -> int:
    """
    Return the total Fock-feature output dim of a `HPT_QRC_Multi` instance:
        n_reservoirs * n_virtual_nodes * lex_out
    so the RFF baseline can be sized identically.
    """
    n_res = len(hpt_qrc_multi.photon_list)
    return n_res * hpt_qrc_multi.n_virtual_nodes * hpt_qrc_multi.lex_out


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(0)
    y = rng.standard_normal((500, 1)).cumsum(axis=0)
    y_train, y_test = y[:400], y[400:]
    rff = RFFRidge(in_size=1, window=10, output_dim=90, gamma=0.1).fit(y_train)
    p = rff.predict(y_test)
    print(f"[RFFRidge smoke] y_test shape {y_test.shape}, pred shape {p.shape}")
    print(f"[RFFRidge smoke] MSE = {np.mean((y_test.flatten() - p.flatten())**2):.4f}")
