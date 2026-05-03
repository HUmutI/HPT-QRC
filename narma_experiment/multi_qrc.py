import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
import perceval as pcvl
from merlin import QuantumLayer, ComputationSpace, LexGrouping

class HPT_QRC_Multi:
    def __init__(self, in_size=1, window=5, n_photons=3, n_reservoirs=3, n_virtual_nodes=3,
                 lex_out=10, ridge_alpha=1e-4, seed=42, use_har_context=False,
                 photon_list=None):
        """
        photon_list : list of int, optional
            Photon count for each reservoir (enables heterogeneous ensemble).
            e.g. [2, 3, 4] for three reservoirs with 2, 3, and 4 photons.
            If None, all reservoirs use n_photons (homogeneous).
        """
        self.in_size = in_size
        self.window = window
        self.use_har_context = use_har_context
        self.n_input_modes = 1
        self.n_memory_modes = 7
        self.n_modes = self.n_input_modes + self.n_memory_modes
        self.n_photons = n_photons
        self.n_reservoirs = n_reservoirs
        self.n_virtual_nodes = n_virtual_nodes
        self.lex_out = lex_out

        # photon_list governs each reservoir's photon count
        if photon_list is not None:
            self.photon_list = photon_list
            self.n_reservoirs = len(photon_list)
        else:
            self.photon_list = [n_photons] * n_reservoirs

        self.total_enc = self.in_size * window
        if self.total_enc > 1:
            self.n_input_modes = min(5, self.in_size)
            self.n_memory_modes = max(3, 8 - self.n_input_modes)
            self.n_modes = self.n_input_modes + self.n_memory_modes

        self.reservoirs = self._build_reservoirs(seed)
        self.ridge = Ridge(alpha=ridge_alpha)
        
    def _build_temporal_circuit(self, n_modes, n_input_modes, n_steps, n_virtual_depth):
        circuit = pcvl.Circuit(n_modes)
        c = 0
        for t in range(n_steps):
            circuit.add(0, pcvl.GenericInterferometer(
                n_modes,
                lambda i, _t=t: (pcvl.BS() // pcvl.PS(pcvl.P(f"t_l{_t}_i{i}"))
                                  // pcvl.BS() // pcvl.PS(pcvl.P(f"t_l{_t}_o{i}"))),
                shape=pcvl.InterferometerShape.RECTANGLE,
            ))
            for m in range(n_input_modes):
                circuit.add(m, pcvl.PS(pcvl.P(f"input{c}")))
                c += 1

        for v in range(n_virtual_depth):
            circuit.add(0, pcvl.GenericInterferometer(
                n_modes,
                lambda i, _v=v: (pcvl.BS() // pcvl.PS(pcvl.P(f"t_v{_v}_i{i}"))
                                  // pcvl.BS() // pcvl.PS(pcvl.P(f"t_v{_v}_o{i}"))),
                shape=pcvl.InterferometerShape.RECTANGLE,
            ))
        return circuit, c

    def _build_reservoirs(self, base_seed):
        """Build one family of virtual-depth circuits per photon count in photon_list."""
        reservoirs = []
        for r_idx, n_ph in enumerate(self.photon_list):
            input_state = [1] * n_ph + [0] * (self.n_modes - n_ph)
            for vd in range(1, self.n_virtual_nodes + 1):
                torch.manual_seed(base_seed + r_idx * 1000)
                circ, n_enc = self._build_temporal_circuit(
                    self.n_modes, self.n_input_modes, self.window, vd
                )
                core = QuantumLayer(
                    input_size=n_enc,
                    circuit=circ,
                    input_state=input_state,
                    input_parameters=["input"],
                    trainable_parameters=["t"],
                    computation_space=ComputationSpace.UNBUNCHED,
                    dtype=torch.float32,
                )
                layer = nn.Sequential(core, LexGrouping(core.output_size, self.lex_out))
                layer.eval()
                reservoirs.append(layer)
                self.n_enc_actual = n_enc
        return reservoirs
        
    # ------------------------------------------------------------------
    # Robust preprocessing helpers (fit on training data, apply to all)
    # ------------------------------------------------------------------
    def _fit_scaler(self, X_flat):
        """Fit Winsorize -> IQR scale -> Min-max [0,1] on 1D training data."""
        p1, p99 = np.percentile(X_flat, [1, 99])
        clipped  = np.clip(X_flat, p1, p99)
        med      = np.median(clipped)
        iqr      = np.percentile(clipped, 75) - np.percentile(clipped, 25)
        iqr      = iqr if iqr > 1e-8 else 1.0
        scaled   = (clipped - med) / iqr
        xmin, xmax = scaled.min(), scaled.max()
        rng = xmax - xmin if xmax > xmin else 1.0
        self._sc = dict(p1=p1, p99=p99, med=med, iqr=iqr, xmin=xmin, rng=rng)

    def _apply_scaler(self, X):
        """Apply fitted robust scaler to any array."""
        sc = self._sc
        X_clipped = np.clip(X, sc['p1'], sc['p99'])
        X_scaled  = (X_clipped - sc['med']) / sc['iqr']
        return (X_scaled - sc['xmin']) / sc['rng']

    # ------------------------------------------------------------------
    # HAR classical context
    # ------------------------------------------------------------------
    def _build_har_context(self, y_1d):
        """
        Build enriched classical context for each timestep from a 1D signal.
        Features: [window of y, lag1, MA5, MA22, momentum Δy]
        Returns shape (len(y_1d), window+4).
        """
        pad    = max(22, self.window)
        padded = np.concatenate([np.zeros(pad), y_1d])
        feats  = []
        for t in range(pad, len(padded)):
            win  = padded[t - self.window : t]
            lag1 = padded[t - 1]
            ma5  = np.mean(padded[max(0, t - 5)  : t])
            ma22 = np.mean(padded[max(0, t - 22) : t])
            delt = padded[t - 1] - padded[t - 2]
            feats.append(np.concatenate([win, [lag1, ma5, ma22, delt]]))
        return np.array(feats)

    def _quantum_features_batch(self, X_win):
        xt = torch.tensor(X_win, dtype=torch.float32)
        with torch.no_grad():
            feats = []
            for r in self.reservoirs:
                out = r(xt)
                if out.is_complex():
                    out = out.real
                feats.append(out.numpy())
            return np.concatenate(feats, axis=1)

    def _create_features(self, X, y_1d):
        """Build full feature vector: [Fock quantum features] + [HAR classical context].
        X      : shape (T, in_size) — the raw input (scaled)
        y_1d   : shape (T,)         — univariate y component for HAR features
        """
        X_dim = self.in_size
        padded_X = np.vstack([np.zeros((self.window - 1, X_dim)), X.reshape(-1, X_dim)])
        X_win = []
        for t in range(self.window - 1, len(padded_X)):
            window_slice = padded_X[t - self.window + 1 : t + 1].flatten()
            if len(window_slice) < self.n_enc_actual:
                window_slice = np.pad(window_slice, (0, self.n_enc_actual - len(window_slice)))
            elif len(window_slice) > self.n_enc_actual:
                window_slice = window_slice[:self.n_enc_actual]
            X_win.append(window_slice)

        X_win = np.array(X_win)
        Q     = self._quantum_features_batch(X_win)

        if self.use_har_context:
            # HAR context: designed for financial volatility (MA5/MA22 capture clustering)
            classical = self._build_har_context(y_1d)
            classical = classical[-len(Q):]
        else:
            # Raw window: preserves nonlinear memory signal for synthetic datasets
            classical = X_win

        return np.hstack([Q, classical])

    # ------------------------------------------------------------------
    # Public: extract features only (no Ridge fit) — used for MC analysis
    # ------------------------------------------------------------------
    def get_features(self, y):
        """Return the full feature matrix for input y without fitting the readout.
        y : shape (T, in_size)  — raw input signal
        Returns features of shape (T-1, n_features).
        """
        features = y
        X_in = features[:-1]
        y_1d = y.flatten()[:-1]
        return self._create_features(X_in, y_1d)

    def fit(self, y_train, X_exog=None, discard_steps=100):
        if X_exog is not None:
            features = np.hstack([y_train, X_exog])
        else:
            features = y_train

        # Target is next step y
        X_in     = features[:-1]
        y_target = y_train[1:]
        y_1d     = y_train.flatten()[:-1]   # unscaled y for HAR context

        print(f"[HPT-QRC] Extracting quantum features for {len(X_in)} training steps...")
        Q_features = self._create_features(X_in, y_1d)

        Q_features_fit = Q_features[discard_steps:, :]
        y_fit          = y_target[discard_steps:]

        print("[HPT-QRC] Fitting Ridge readout...")
        self.ridge.fit(Q_features_fit, y_fit)

        self.last_X = features[-self.window:].copy()
        self.last_y = y_train.flatten()[-max(22, self.window):]  # HAR warm-start
        return self

    def predict(self, y_test, X_exog=None):
        if X_exog is not None:
            features = np.hstack([y_test, X_exog])
        else:
            features = y_test

        # Build input: last window from training + test features (shifted by 1)
        concat_features = np.vstack([self.last_X, features[:-1]])
        y_1d_concat     = np.concatenate([self.last_y, y_test.flatten()[:-1]])

        print(f"[HPT-QRC] Extracting quantum features for {len(features)} test steps...")
        Q_features = self._create_features(concat_features, y_1d_concat)

        return self.ridge.predict(Q_features[-len(features):])
