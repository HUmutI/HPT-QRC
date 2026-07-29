"""Exact linear-optical probability core for layered phase-encoded interferometers.

MerLin's ``QuantumLayer`` is the reference implementation, but a single forward pass
costs ~7 ms regardless of batch size (the cost is fixed overhead, not the permanent).
A recurrent reservoir has to step one timestep at a time, so that overhead sets the
budget for the entire experimental programme: ~7 s per 1000-step sequence per
reservoir, times seeds, times Optuna trials.

The circuits used here have a special structure that makes the overhead avoidable::

    U = A_d . D(phi_{d-1}) . A_{d-1} ... A_1 . D(phi_0) . A_0

where the ``A_k`` are fixed (never trained, never input-dependent) interferometers and
``D(phi)`` is a diagonal layer of phase shifters carrying the encoded input. The ``A_k``
are built once with Perceval; each timestep then costs a handful of ``m x m`` matrix
products plus a batch of small permanents.

``verify_against_merlin`` checks this core reproduces MerLin's probabilities to machine
precision, and ``tests/test_photonic_core.py`` runs that check in CI. Nothing here is a
new physical model -- it is the same boson-sampling distribution, computed faster.

Conventions match ``hardware/README.md`` section 5 so the same spec can be replayed on a QPU:
phases are applied directly in radians, and the unbunched output ordering is
lexicographic over occupied-mode tuples, identical to MerLin's ``output_keys``.
"""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

__all__ = [
    "unbunched_patterns",
    "layered_unitary",
    "permanent_batch",
    "unbunched_probabilities",
    "LayeredInterferometer",
    "haar_unitary",
]


def haar_unitary(m: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an ``m x m`` Haar-random unitary (QR of a complex Ginibre matrix)."""
    z = (rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    # Fix the phase convention so the result is Haar-distributed, not just unitary.
    return q * (np.diagonal(r) / np.abs(np.diagonal(r)))


def unbunched_patterns(m: int, n: int) -> np.ndarray:
    """Occupied-mode tuples for every unbunched ``n``-photon output of an ``m``-mode circuit.

    Returns an ``(C(m, n), n)`` integer array in lexicographic order, matching the
    ordering MerLin exposes as ``output_keys``.
    """
    if not 1 <= n <= m:
        raise ValueError(f"need 1 <= n <= m, got n={n}, m={m}")
    return np.array(list(combinations(range(m), n)), dtype=np.int64)


def layered_unitary(fixed: list[np.ndarray], phases: np.ndarray) -> np.ndarray:
    """Compose ``A_d . D(phi_{d-1}) ... A_1 . D(phi_0) . A_0`` for a batch of phase settings.

    Args:
        fixed: ``d + 1`` fixed ``m x m`` unitaries, in light-propagation order.
        phases: ``(batch, d, m)`` or ``(d, m)`` encoded phases in radians.

    Returns:
        ``(batch, m, m)`` total unitaries.
    """
    phases = np.asarray(phases, dtype=np.float64)
    if phases.ndim == 2:
        phases = phases[None, :, :]
    batch, depth, m = phases.shape
    if len(fixed) != depth + 1:
        raise ValueError(f"expected {depth + 1} fixed unitaries for depth {depth}, got {len(fixed)}")

    u = np.broadcast_to(fixed[0], (batch, m, m)).copy()
    for k in range(depth):
        # Left-multiplying by a diagonal scales rows, so no full matmul is needed here.
        u = np.exp(1j * phases[:, k, :])[:, :, None] * u
        u = fixed[k + 1] @ u
    return u


def permanent_batch(mats: np.ndarray) -> np.ndarray:
    """Permanents of a batch of small square matrices, by direct expansion.

    Args:
        mats: ``(batch, n, n)`` complex array.

    Ryser's algorithm is asymptotically better, but photon numbers reachable on current
    hardware are ``n <= 6``, where expanding ``n!`` products vectorised over the batch is
    faster in NumPy and exact.
    """
    n = mats.shape[-1]
    if n == 0:
        return np.ones(mats.shape[0], dtype=complex)
    if n > 8:
        raise ValueError(f"direct permanent expansion is impractical for n={n}")
    rows = np.arange(n)
    total = np.zeros(mats.shape[0], dtype=complex)
    for perm in permutations(range(n)):
        total += np.prod(mats[:, rows, perm], axis=1)
    return total


def unbunched_probabilities(
    unitaries: np.ndarray,
    n_photons: int,
    patterns: np.ndarray | None = None,
    input_modes: np.ndarray | None = None,
    renormalise: bool = True,
) -> np.ndarray:
    """Probabilities of every unbunched output, for photons injected one per input mode.

    Args:
        unitaries: ``(batch, m, m)`` total circuit unitaries.
        n_photons: photon number ``n``.
        patterns: precomputed output patterns; recomputed from ``m, n`` when omitted.
        input_modes: modes carrying the input photons, default ``0..n-1``.
        renormalise: divide by the total unbunched mass, so the result is a distribution
            over the no-collision subspace. This is what MerLin's
            ``ComputationSpace.UNBUNCHED`` returns, and what post-selecting on
            coincidences gives on hardware.

    Returns:
        ``(batch, C(m, n))`` real probabilities.
    """
    unitaries = np.asarray(unitaries)
    if unitaries.ndim == 2:
        unitaries = unitaries[None]
    m = unitaries.shape[-1]
    if patterns is None:
        patterns = unbunched_patterns(m, n_photons)
    if input_modes is None:
        input_modes = np.arange(n_photons)

    # The amplitude for output pattern S is perm(U[S, T]) with T the input modes: rows are
    # indexed by occupied *output* modes, columns by input modes. U[S, T] and U[T, S] are not
    # transposes of each other unless U is symmetric, so the order here matters.
    cols = unitaries[:, :, input_modes]                        # (batch, m, n)
    subs = cols[:, patterns, :]                                # (batch, P, n, n)
    batch, n_pat = subs.shape[0], subs.shape[1]

    perms = permanent_batch(subs.reshape(batch * n_pat, n_photons, n_photons))
    probs = np.abs(perms.reshape(batch, n_pat)) ** 2
    # Unbunched patterns have all occupations equal to 1, so the 1/prod(n_i!) factor is 1.
    if renormalise:
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-300, None)
    return probs


class LayeredInterferometer:
    """A fixed interferometer stack with ``depth`` layers of input-driven phase shifters.

    The fixed layers are Haar-random and never trained -- this is a reservoir, so only the
    linear readout is fitted. ``encode`` maps a ``(batch, depth * m)`` phase array to
    unbunched output probabilities.
    """

    def __init__(self, n_modes: int, n_photons: int, depth: int = 1, seed: int = 42):
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.n_modes = int(n_modes)
        self.n_photons = int(n_photons)
        self.depth = int(depth)
        self.seed = int(seed)

        rng = np.random.default_rng(seed)
        self.fixed = [haar_unitary(self.n_modes, rng) for _ in range(self.depth + 1)]
        self.patterns = unbunched_patterns(self.n_modes, self.n_photons)
        self.input_modes = np.arange(self.n_photons)

    @property
    def n_encoding(self) -> int:
        """Number of phase shifters available to carry encoded values."""
        return self.depth * self.n_modes

    @property
    def output_size(self) -> int:
        return len(self.patterns)

    def probabilities(self, phases: np.ndarray) -> np.ndarray:
        """Map ``(batch, depth * m)`` phases (radians) to ``(batch, C(m, n))`` probabilities."""
        phases = np.asarray(phases, dtype=np.float64)
        single = phases.ndim == 1
        if single:
            phases = phases[None, :]
        if phases.shape[1] != self.n_encoding:
            raise ValueError(
                f"expected {self.n_encoding} phases (depth {self.depth} x {self.n_modes} modes), "
                f"got {phases.shape[1]}"
            )
        u = layered_unitary(self.fixed, phases.reshape(-1, self.depth, self.n_modes))
        probs = unbunched_probabilities(
            u, self.n_photons, patterns=self.patterns, input_modes=self.input_modes
        )
        return probs[0] if single else probs

    def to_perceval(self):
        """Build the equivalent Perceval circuit, for hardware replay and cross-checking.

        Returns ``(circuit, input_state)``. The fixed layers become ``Unitary`` components
        and the encoded phases become free parameters named ``input_{layer}_{mode}``, so the
        same spec can be pinned and shipped to a ``RemoteProcessor``.
        """
        import perceval as pcvl

        circuit = pcvl.Circuit(self.n_modes)
        circuit.add(0, pcvl.Unitary(pcvl.Matrix(self.fixed[0])))
        for k in range(self.depth):
            for mode in range(self.n_modes):
                circuit.add(mode, pcvl.PS(pcvl.P(f"input_{k}_{mode}")))
            circuit.add(0, pcvl.Unitary(pcvl.Matrix(self.fixed[k + 1])))
        input_state = [1] * self.n_photons + [0] * (self.n_modes - self.n_photons)
        return circuit, input_state


def verify_against_merlin(n_modes=6, n_photons=2, depth=2, seed=0, batch=8, tol=1e-9):
    """Check this core against MerLin's QuantumLayer. Returns the worst absolute deviation."""
    import torch
    from merlin import ComputationSpace, QuantumLayer

    layer = LayeredInterferometer(n_modes, n_photons, depth, seed=seed)
    circuit, input_state = layer.to_perceval()
    rng = np.random.default_rng(seed + 1)
    phases = rng.uniform(0, 2 * np.pi, size=(batch, layer.n_encoding))

    ours = layer.probabilities(phases)

    ql = QuantumLayer(
        input_size=layer.n_encoding,
        circuit=circuit,
        input_parameters=["input"],
        input_state=input_state,
        computation_space=ComputationSpace.UNBUNCHED,
        dtype=torch.float64,
    )
    with torch.no_grad():
        theirs = ql(torch.tensor(phases, dtype=torch.float64)).numpy()

    worst = float(np.abs(ours - theirs).max())
    if worst > tol:
        raise AssertionError(f"photonic_core disagrees with MerLin by {worst:.3e} (tol {tol:.1e})")
    return worst


if __name__ == "__main__":
    import time

    for m, n, d in [(6, 2, 1), (8, 2, 2), (8, 3, 1), (10, 3, 2), (8, 4, 1)]:
        worst = verify_against_merlin(m, n, d, seed=1)
        print(f"m={m} n={n} depth={d}: max |ours - merlin| = {worst:.2e}")

    layer = LayeredInterferometer(8, 2, depth=1, seed=0)
    x = np.random.default_rng(0).uniform(0, 2 * np.pi, size=(1, layer.n_encoding))
    layer.probabilities(x)
    t0 = time.time()
    for _ in range(2000):
        layer.probabilities(x)
    per = (time.time() - t0) / 2000
    print(f"\nper-step batch=1: {per * 1e3:.4f} ms -> 1000 steps = {per * 1000:.2f} s")
