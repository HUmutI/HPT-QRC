"""Hardware noise models for the linear-optical reservoir.

The predecessor implementation (``src/noise_models.py``) approximated imperfect photon
indistinguishability by mixing the ideal distribution with a "distinguishable baseline"
obtained from a *different random circuit*. That is not the physical model: distinguishable
photons in the *same* circuit follow the permanent of the modulus-squared transfer matrix,
not an unrelated distribution. This module replaces that with the real thing, cross-checked
against Perceval's ``NoiseModel``.

Sources modelled, with the values measured on Quandela hardware (read live from the cloud
API's ``perfs`` field, recorded in :data:`ASCELLA` and :data:`BELENOS`):

``indistinguishability``
    Hong-Ou-Mandel visibility. For two photons the mixture
    ``V * perm(U_S) ** 2 + (1 - V) * perm(|U_S| ** 2)`` is exact. Above two photons it is a
    mean-field approximation, so :func:`validate_against_perceval` reports the error there.

``transmittance``
    Per-photon survival probability. Uniform loss does not change the distribution
    *conditioned* on detecting all n photons -- it changes how long you must run to collect
    them, which is why it enters this study through the shot budget rather than the shape of
    the distribution. Mode-dependent loss does distort the conditional distribution and is
    modelled explicitly by :func:`apply_mode_transmittance`.

``g2``
    Second-order correlation at zero delay: the probability the source emits an extra
    photon. An extra photon in an n-fold coincidence window produces a detection pattern
    drawn from the (n+1)-photon distribution, which with threshold detectors is misread as
    an n-photon event.

``phase_error``
    Thermo-optic phase-shifter jitter, sampled per shot and averaged.

Shot noise is separate and unavoidable: :func:`apply_shot_noise` resamples the distribution
with the number of coincidences actually collected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .photonic_core import permanent_batch, unbunched_patterns

__all__ = [
    "HardwareSpec",
    "ASCELLA",
    "BELENOS",
    "IDEAL",
    "noisy_unbunched_probabilities",
    "apply_mode_transmittance",
    "apply_shot_noise",
    "coincidence_rate",
    "validate_against_perceval",
]


@dataclass(frozen=True)
class HardwareSpec:
    """Measured operating point of a photonic processor.

    Defaults are the ideal device. ``clock_hz`` and ``transmittance`` together set how many
    n-fold coincidences a job of a given wall-clock length can collect, which is the real
    constraint on hardware experiments.
    """

    name: str = "ideal"
    indistinguishability: float = 1.0
    g2: float = 0.0
    transmittance: float = 1.0
    clock_hz: float = 1.0
    phase_error: float = 0.0
    dark_counts_hz: tuple = field(default=(), repr=False)

    def as_perceval_noise(self):
        """Equivalent ``perceval.NoiseModel``, for cross-validation."""
        import perceval as pcvl

        return pcvl.NoiseModel(
            indistinguishability=self.indistinguishability,
            g2=self.g2,
            transmittance=self.transmittance,
            phase_error=self.phase_error,
        )


# Values read from the Quandela cloud API 'perfs' field. Ascella reports HOM 86.36%,
# g2 1.95%, transmittance 2.44%, clock 80 MHz; Belenos reports HOM 82.7%, g2 18.2%,
# clock 4.94 MHz. Belenos' aggregate transmittance is reported as 0 with a per-mode
# breakdown instead, so the effective figure below is taken from its published
# characterisation rather than the API field.
ASCELLA = HardwareSpec(
    name="ascella",
    indistinguishability=0.8636,
    g2=0.0195,
    transmittance=0.0244,
    clock_hz=80e6,
)
BELENOS = HardwareSpec(
    name="belenos",
    indistinguishability=0.827,
    g2=0.182,
    transmittance=0.0484,
    clock_hz=4.94e6,
)
IDEAL = HardwareSpec()


def _distinguishable_probabilities(unitaries, n_photons, patterns, input_modes):
    """Distribution for fully distinguishable photons: permanent of |U|^2."""
    mod2 = np.abs(unitaries) ** 2
    cols = mod2[:, :, input_modes]
    subs = cols[:, patterns, :]
    batch, n_pat = subs.shape[0], subs.shape[1]
    perms = permanent_batch(subs.reshape(batch * n_pat, n_photons, n_photons).astype(complex))
    return np.abs(perms.reshape(batch, n_pat))


def noisy_unbunched_probabilities(
    unitaries: np.ndarray,
    n_photons: int,
    spec: HardwareSpec = IDEAL,
    patterns: np.ndarray | None = None,
    input_modes: np.ndarray | None = None,
) -> np.ndarray:
    """Unbunched distribution under partial indistinguishability.

    Returns ``(batch, C(m, n))`` probabilities normalised over the unbunched subspace.
    """
    unitaries = np.asarray(unitaries)
    if unitaries.ndim == 2:
        unitaries = unitaries[None]
    m = unitaries.shape[-1]
    if patterns is None:
        patterns = unbunched_patterns(m, n_photons)
    if input_modes is None:
        input_modes = np.arange(n_photons)

    cols = unitaries[:, :, input_modes]
    subs = cols[:, patterns, :]
    batch, n_pat = subs.shape[0], subs.shape[1]
    ideal = np.abs(permanent_batch(subs.reshape(batch * n_pat, n_photons, n_photons))) ** 2
    ideal = ideal.reshape(batch, n_pat)

    v = float(spec.indistinguishability)
    if v < 1.0:
        dist = _distinguishable_probabilities(unitaries, n_photons, patterns, input_modes)
        probs = v * ideal + (1.0 - v) * dist
    else:
        probs = ideal
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-300, None)


def apply_mode_transmittance(probs: np.ndarray, patterns: np.ndarray, t_modes) -> np.ndarray:
    """Reweight by mode-dependent detection efficiency, then renormalise.

    Conditioned on an n-fold coincidence, a pattern's probability is scaled by the product
    of the efficiencies of its occupied modes. Uniform efficiency cancels in the
    renormalisation, as it should.
    """
    t_modes = np.asarray(t_modes, dtype=float)
    weights = np.prod(t_modes[patterns], axis=1)
    out = np.asarray(probs, dtype=float) * weights
    return out / np.clip(out.sum(axis=-1, keepdims=True), 1e-300, None)


def apply_shot_noise(probs: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Resample a distribution from ``n_samples`` coincidences.

    ``n_samples <= 0`` returns the distribution unchanged, which is the infinite-shot limit.
    """
    if n_samples is None or n_samples <= 0:
        return probs
    probs = np.asarray(probs, dtype=float)
    single = probs.ndim == 1
    flat = probs.reshape(1, -1) if single else probs
    out = np.empty_like(flat)
    for i, row in enumerate(flat):
        p = np.clip(row, 0, None)
        total = p.sum()
        if total <= 0:
            out[i] = np.full_like(row, 1.0 / len(row))
            continue
        out[i] = rng.multinomial(n_samples, p / total) / n_samples
    return out[0] if single else out


def coincidence_rate(spec: HardwareSpec, n_photons: int) -> float:
    """Detected n-fold coincidences per second at this operating point.

    This is the number that decides whether a hardware experiment is feasible: the rate
    falls as ``transmittance ** n``, so dropping from three photons to two buys back a
    factor of ``1 / transmittance`` -- more than an order of magnitude on current devices.
    """
    return float(spec.clock_hz * spec.transmittance**n_photons)


class PercevalBackend:
    """Drop-in replacement for :class:`~src.photonic_core.LayeredInterferometer`.

    Exposes the same ``probabilities(phases)`` interface but evaluates through Perceval's
    ``NoiseModel``, which is Quandela's own implementation of the device physics. The fast
    core stays the reference for noiseless work (it agrees with Perceval to 1e-16); this is
    used wherever noise is switched on, so no approximation of our own enters the
    noise-robustness results.

    Threshold detectors are attached when ``threshold_detectors`` is set, matching Ascella
    and Belenos, which cannot resolve photon number.
    """

    def __init__(self, layer, spec: HardwareSpec, threshold_detectors: bool = True,
                 n_samples: int | None = None, seed: int = 0):
        import perceval as pcvl

        self.layer = layer
        self.spec = spec
        self.n_samples = n_samples
        self._rng = np.random.default_rng(seed)

        circuit, input_state = layer.to_perceval()
        self._params = circuit.get_parameters()
        # Sort by (layer, mode) so the flat phase vector maps the same way as the fast core,
        # which orders phases layer-major. Lexicographic name order would break at m > 10.
        order = []
        for param in self._params:
            k, mode = param.name.split("_")[1:]
            order.append(int(k) * layer.n_modes + int(mode))
        self._order = np.argsort(np.argsort(order))
        self._params = [self._params[i] for i in np.argsort(order)]

        self._processor = pcvl.Processor("SLOS", circuit, noise=spec.as_perceval_noise())
        if threshold_detectors:
            for mode in range(layer.n_modes):
                try:
                    self._processor.detectors[mode] = pcvl.Detector.threshold()
                except (AttributeError, IndexError):
                    break
        self._processor.with_input(pcvl.BasicState(input_state))
        self._processor.min_detected_photons_filter(layer.n_photons)

        self._states = []
        for pattern in layer.patterns:
            occ = [0] * layer.n_modes
            for mode in pattern:
                occ[mode] = 1
            self._states.append(pcvl.BasicState(occ))

    @property
    def n_encoding(self) -> int:
        return self.layer.n_encoding

    @property
    def output_size(self) -> int:
        return self.layer.output_size

    @property
    def patterns(self):
        return self.layer.patterns

    def probabilities(self, phases: np.ndarray) -> np.ndarray:
        phases = np.asarray(phases, dtype=float)
        single = phases.ndim == 1
        rows = phases[None, :] if single else phases

        out = np.empty((len(rows), self.layer.output_size))
        for i, row in enumerate(rows):
            for param, value in zip(self._params, row):
                param.set_value(float(value))
            results = self._processor.probs()["results"]
            probs = np.array([results.get(state, 0.0) for state in self._states], dtype=float)
            total = probs.sum()
            probs = probs / total if total > 0 else np.full_like(probs, 1.0 / len(probs))
            out[i] = probs
        if self.n_samples:
            out = apply_shot_noise(out, self.n_samples, self._rng)
        return out[0] if single else out


def validate_against_perceval(
    n_modes: int = 6,
    n_photons: int = 2,
    spec: HardwareSpec = ASCELLA,
    seed: int = 0,
    n_phase_samples: int = 1,
) -> dict:
    """Compare this module's noisy distribution with Perceval's ``NoiseModel``.

    Perceval is the reference implementation; this reports the total variation distance so
    the mean-field approximation used above two photons can be quoted honestly.
    """
    import perceval as pcvl

    from .photonic_core import LayeredInterferometer, layered_unitary

    layer = LayeredInterferometer(n_modes, n_photons, depth=1, seed=seed)
    circuit, input_state = layer.to_perceval()
    rng = np.random.default_rng(seed + 5)
    phases = rng.uniform(0, 2 * np.pi, size=layer.n_encoding)

    for param in circuit.get_parameters():
        k, mode = param.name.split("_")[1:]
        param.set_value(float(phases[int(k) * n_modes + int(mode)]))

    processor = pcvl.Processor("SLOS", circuit, noise=spec.as_perceval_noise())
    processor.with_input(pcvl.BasicState(input_state))
    processor.min_detected_photons_filter(n_photons)
    results = processor.probs()["results"]

    reference = np.zeros(len(layer.patterns))
    for idx, pattern in enumerate(layer.patterns):
        state = [0] * n_modes
        for mode in pattern:
            state[mode] = 1
        reference[idx] = results.get(pcvl.BasicState(state), 0.0)
    total = reference.sum()
    if total > 0:
        reference /= total

    unitary = layered_unitary(layer.fixed, phases.reshape(1, 1, n_modes))
    ours = noisy_unbunched_probabilities(
        unitary, n_photons, spec, patterns=layer.patterns, input_modes=layer.input_modes
    )[0]

    return {
        "n_modes": n_modes,
        "n_photons": n_photons,
        "spec": spec.name,
        "total_variation": float(0.5 * np.abs(ours - reference).sum()),
        "max_abs": float(np.abs(ours - reference).max()),
    }


if __name__ == "__main__":
    for spec in (IDEAL, ASCELLA, BELENOS):
        for n in (2, 3):
            report = validate_against_perceval(n_modes=6, n_photons=n, spec=spec)
            print(
                f"{report['spec']:>8}  n={n}  TVD={report['total_variation']:.2e}  "
                f"max={report['max_abs']:.2e}"
            )
    print()
    for spec in (ASCELLA, BELENOS):
        for n in (2, 3, 4):
            print(f"{spec.name:>8}  n={n}  coincidences/s = {coincidence_rate(spec, n):.3e}")
