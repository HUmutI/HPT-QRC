"""Recurrent linear-optical reservoir computer.

The predecessor model (``src/multi_qrc.py``) is a *windowed* feature map: it encodes a
sliding window of the series into interferometer phases and reads out Fock probabilities.
It has no state, so its linear memory capacity is pinned to the window length (measured
at exactly 5.0 for window=5 in ``results/mc_scores.csv``) and its features add nothing a
Ridge on the raw window does not already have.

This module adds the missing ingredient -- a state that persists across timesteps::

    e_t = g_in * W_in u_t  +  g_fb * W_fb s_{t-1}          (phases, radians)
    p_t = photonic_layer(e_t)                              (unbunched Fock probabilities)
    s_t = (1 - leak) * s_{t-1} + leak * p_t                (leaky integration)

Only the linear readout on ``[s_t, input window]`` is trained; ``W_in``, ``W_fb`` and the
interferometers are fixed random draws, as in any reservoir.

Two properties matter for the hardware story:

* The feedback is applied to the *encoding*, and is computed classically between shots.
  Each timestep is still one circuit configuration and one batch of samples, so a
  recurrent run costs exactly the same shot budget as the windowed model. Nothing here
  requires an optical memory or fast feed-forward.
* ``g_fb`` controls contractivity. Since ``p_t`` lies on a simplex the map is bounded for
  any gain, but the echo state property needs the feedback to forget perturbations;
  :meth:`TemporalPhotonicQRC.esp_decay` measures that directly.

Set ``feedback=False`` to recover the windowed model exactly, which is the ablation that
isolates what recurrence buys.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .photonic_core import LayeredInterferometer

__all__ = ["PhotonicReservoir", "TemporalPhotonicQRC"]


def _as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, 1) if x.ndim == 1 else x


class PhotonicReservoir:
    """One recurrent photonic reservoir with a fixed interferometer and fixed projections."""

    def __init__(
        self,
        n_modes: int = 8,
        n_photons: int = 2,
        depth: int = 1,
        input_dim: int = 1,
        leak: float = 0.3,
        g_in: float = np.pi,
        g_fb: float = np.pi,
        feedback: bool = True,
        seed: int = 42,
        noise_spec=None,
        n_samples: int | None = None,
        threshold_detectors: bool = True,
    ):
        self.optics = LayeredInterferometer(n_modes, n_photons, depth, seed=seed)
        if noise_spec is not None or n_samples:
            # Any noise at all routes through Perceval, so the reported robustness numbers
            # come from Quandela's own device model rather than an approximation of ours.
            from .noise import IDEAL, PercevalBackend

            self.optics = PercevalBackend(
                self.optics,
                noise_spec if noise_spec is not None else IDEAL,
                threshold_detectors=threshold_detectors,
                n_samples=n_samples,
                seed=seed,
            )
        self.leak = float(leak)
        self.g_in = float(g_in)
        self.g_fb = float(g_fb)
        self.feedback = bool(feedback)
        self.input_dim = int(input_dim)
        self.seed = int(seed)

        rng = np.random.default_rng(seed + 7919)
        n_enc = self.optics.n_encoding
        self.state_size = self.optics.output_size

        # Input projection: uniform in [-1, 1], so g_in is the peak phase swing (radians)
        # that a unit-magnitude input produces.
        self.w_in = rng.uniform(-1.0, 1.0, size=(n_enc, self.input_dim))
        # Feedback projection with variance 1/state_size. The state is fed back as
        # ``state_size * s - 1``, the deviation from the uniform distribution, whose entries
        # are O(1) regardless of Fock-space size. Together these make ``g_fb`` a phase swing
        # in radians that means the same thing at every photon number -- otherwise a larger
        # Fock space would silently receive weaker feedback and the photon-number ablation
        # would confound expressivity with feedback strength.
        self.w_fb = rng.normal(0.0, 1.0 / np.sqrt(self.state_size), size=(n_enc, self.state_size))
        self.bias = rng.uniform(0.0, 2 * np.pi, size=n_enc)

    @property
    def output_size(self) -> int:
        return self.state_size

    def run(self, u: np.ndarray, state0: np.ndarray | None = None) -> np.ndarray:
        """Drive the reservoir with ``u`` of shape ``(T, input_dim)``; return states ``(T, P)``.

        ``u`` is the already-lagged drive: when ``encode_window > 1`` the caller stacks that
        many lags, so the interferometer computes a nonlinear function of a stretch of input
        history rather than of the current sample alone. NARMA-style targets contain explicit
        cross-lag products, which the optics can only form if both lags are present in the
        same encoding.
        """
        u = _as_2d(u)
        if u.shape[1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {u.shape[1]}")

        drive = (self.w_in @ u.T).T * self.g_in + self.bias      # (T, n_enc), input part
        state = np.full(self.state_size, 1.0 / self.state_size) if state0 is None else np.asarray(state0)

        if not self.feedback:
            # Stateless: every timestep is independent, so the whole sequence is one batch.
            return self.optics.probabilities(drive)

        states = np.empty((len(u), self.state_size))
        size = self.state_size
        for t in range(len(u)):
            phases = drive[t] + self.g_fb * (self.w_fb @ (size * state - 1.0))
            p = self.optics.probabilities(phases)
            state = (1.0 - self.leak) * state + self.leak * p
            states[t] = state
        return states


class TemporalPhotonicQRC:
    """Ensemble of recurrent photonic reservoirs with a ridge readout.

    Follows the standard input-driven reservoir protocol: the model sees the driving
    signal ``u`` only and never the true target history, so it is directly comparable to
    published NARMA / Santa Fe numbers.

    Args:
        n_modes: interferometer width.
        photon_list: one reservoir per entry; heterogeneous photon numbers give Fock
            spaces of different sizes and different nonlinearity orders.
        reservoirs_per_photon: independent random draws per photon number.
        depth: number of encoding layers. Deeper is more expressive but lossier on
            hardware, so this is kept small by default.
        leak: leaky-integration rate. Lower means longer memory.
        g_in, g_fb: encoding gains, in radians. ``g_in`` fixes the fraction of the 2*pi
            phase range the input actually uses -- the predecessor model used only 0.58 rad.
        feedback: set False to ablate recurrence and recover a windowed feature map.
        window: how many lags of the raw input to append as an explicit classical block.
        use_quantum / use_classical: toggle each feature block, for the controls.
        standardize: z-score features before the ridge. Fock probabilities and raw inputs
            differ in scale by more than an order of magnitude, which a single ridge
            penalty cannot accommodate.
    """

    def __init__(
        self,
        n_modes: int = 8,
        photon_list: tuple[int, ...] = (2, 3),
        reservoirs_per_photon: int = 2,
        depth: int = 1,
        leak: float = 0.3,
        g_in: float = np.pi,
        g_fb: float = np.pi,
        feedback: bool = True,
        encode_window: int = 1,
        window: int = 10,
        use_quantum: bool = True,
        use_classical: bool = True,
        standardize: bool = True,
        ridge_alpha: float = 1e-6,
        washout: int = 100,
        seed: int = 42,
        noise_spec=None,
        n_samples: int | None = None,
        threshold_detectors: bool = True,
    ):
        if not (use_quantum or use_classical):
            raise ValueError("at least one of use_quantum / use_classical must be True")
        self.n_modes = int(n_modes)
        self.photon_list = tuple(photon_list)
        self.reservoirs_per_photon = int(reservoirs_per_photon)
        self.depth = int(depth)
        self.leak = float(leak)
        self.g_in = float(g_in)
        self.g_fb = float(g_fb)
        self.feedback = bool(feedback)
        self.encode_window = int(encode_window)
        self.window = int(window)
        self.use_quantum = bool(use_quantum)
        self.use_classical = bool(use_classical)
        self.standardize = bool(standardize)
        self.ridge_alpha = float(ridge_alpha)
        self.washout = int(washout)
        self.seed = int(seed)
        self.noise_spec = noise_spec
        self.n_samples = n_samples
        self.threshold_detectors = bool(threshold_detectors)

        self.reservoirs: list[PhotonicReservoir] = []
        self._input_dim: int | None = None
        self.scaler_: StandardScaler | None = None
        self.input_scaler_: StandardScaler | None = None
        self.ridge_: Ridge | None = None

    def _build(self, input_dim: int) -> None:
        self._input_dim = input_dim
        self.reservoirs = []
        if not self.use_quantum:
            return
        k = 0
        for n_ph in self.photon_list:
            for _ in range(self.reservoirs_per_photon):
                self.reservoirs.append(
                    PhotonicReservoir(
                        n_modes=self.n_modes,
                        n_photons=n_ph,
                        depth=self.depth,
                        input_dim=input_dim * self.encode_window,
                        leak=self.leak,
                        g_in=self.g_in,
                        g_fb=self.g_fb,
                        feedback=self.feedback,
                        seed=self.seed + 1000 * k,
                        noise_spec=self.noise_spec,
                        n_samples=self.n_samples,
                        threshold_detectors=self.threshold_detectors,
                    )
                )
                k += 1

    @property
    def feature_dim(self) -> int:
        q = sum(r.output_size for r in self.reservoirs) if self.use_quantum else 0
        c = self.window * (self._input_dim or 0) if self.use_classical else 0
        return q + c

    @staticmethod
    def _lags(u: np.ndarray, k: int) -> np.ndarray:
        """``k`` lags of ``u``, zero-padded at the start; shape ``(T, k * dim)``."""
        t, d = u.shape
        pad = np.vstack([np.zeros((k - 1, d)), u])
        return np.stack([pad[i : i + k].ravel() for i in range(t)])

    def _lag_block(self, u: np.ndarray) -> np.ndarray:
        """``window`` lags of the (scaled) driving input, zero-padded at the start."""
        return self._lags(u, self.window)

    def transform(self, u) -> np.ndarray:
        """Feature matrix for driving input ``u``; shape ``(T, feature_dim)``."""
        u = _as_2d(u)
        if self._input_dim is None:
            raise RuntimeError("call fit() before transform()")
        us = self.input_scaler_.transform(u)
        blocks = []
        if self.use_quantum:
            drive = self._lags(us, self.encode_window) if self.encode_window > 1 else us
            blocks.extend(r.run(drive) for r in self.reservoirs)
        if self.use_classical:
            blocks.append(self._lag_block(us))
        return np.hstack(blocks)

    def build_features(self, u, n_train: int) -> np.ndarray:
        """Feature matrix for the whole series, with the input scaler fitted on train rows.

        The readout is fitted separately (see ``src.rc_protocol``), so this is the natural
        unit of work for hyperparameter search and for the model comparison: every model
        exposes a feature matrix and they all share one readout and one metric.
        """
        u = _as_2d(u)
        self._build(u.shape[1])
        self.input_scaler_ = StandardScaler().fit(u[:n_train])
        return self.transform(u)

    def fit(self, u, y) -> "TemporalPhotonicQRC":
        """Fit the readout. ``u`` is the driving input, ``y`` the target, aligned in time."""
        u = _as_2d(u)
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(u) != len(y):
            raise ValueError(f"u and y must align in time, got {len(u)} and {len(y)}")
        if self.washout >= len(u):
            raise ValueError(f"washout {self.washout} >= sequence length {len(u)}")

        self._build(u.shape[1])
        # Scale the drive to zero mean / unit variance on train only, so g_in is a
        # dataset-independent knob and no test statistics leak backwards.
        self.input_scaler_ = StandardScaler().fit(u)

        features = self.transform(u)
        train = slice(self.washout, None)
        if self.standardize:
            self.scaler_ = StandardScaler().fit(features[train])
            features = self.scaler_.transform(features)
        self.ridge_ = Ridge(alpha=self.ridge_alpha).fit(features[train], y[train])
        return self

    def predict(self, u) -> np.ndarray:
        """Predict on ``u``.

        This restarts the reservoir from its default state. For a train/test split on one
        continuous series use :meth:`fit_predict_split`, which keeps the state warm across
        the boundary as the reservoir-computing protocol requires.
        """
        if self.ridge_ is None:
            raise RuntimeError("call fit() before predict()")
        features = self.transform(u)
        if self.standardize:
            features = self.scaler_.transform(features)
        return self.ridge_.predict(features)

    def fit_predict_split(self, u, y, n_train: int) -> np.ndarray:
        """Fit on the first ``n_train`` steps of one continuous series; predict the rest.

        The reservoir is driven once over the whole series so its state carries across the
        split, which is the standard reservoir-computing protocol -- restarting the state at
        the test boundary would penalise exactly the memory this model is meant to have.

        This is not leakage: the reservoir has no trained parameters, and every quantity
        fitted from data (both scalers and the ridge) is estimated on the training slice
        only. Returns predictions for ``u[n_train:]``.
        """
        u = _as_2d(u)
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(u) != len(y):
            raise ValueError(f"u and y must align in time, got {len(u)} and {len(y)}")
        if not self.washout < n_train < len(u):
            raise ValueError(f"need washout ({self.washout}) < n_train ({n_train}) < T ({len(u)})")

        self._build(u.shape[1])
        self.input_scaler_ = StandardScaler().fit(u[:n_train])

        features = self.transform(u)
        train = slice(self.washout, n_train)
        if self.standardize:
            self.scaler_ = StandardScaler().fit(features[train])
            features = self.scaler_.transform(features)
        self.ridge_ = Ridge(alpha=self.ridge_alpha).fit(features[train], y[train])
        return self.ridge_.predict(features[n_train:])

    def esp_decay(self, u, perturbation: float = 0.1, seed: int = 0) -> np.ndarray:
        """Distance between states from two different initial conditions, per timestep.

        The echo state property holds when this decays to zero: the reservoir forgets its
        initialisation and its state is a function of the input history alone.
        """
        u = _as_2d(u)
        us = self.input_scaler_.transform(u)
        rng = np.random.default_rng(seed)
        decay = np.zeros(len(us))
        for r in self.reservoirs:
            base = np.full(r.state_size, 1.0 / r.state_size)
            other = np.abs(base + rng.normal(0, perturbation, size=r.state_size))
            other /= other.sum()
            a = r.run(us, state0=base)
            b = r.run(us, state0=other)
            decay += np.linalg.norm(a - b, axis=1)
        return decay / max(len(self.reservoirs), 1)
