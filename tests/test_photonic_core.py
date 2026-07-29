"""The fast probability core must agree with MerLin and Perceval exactly.

Everything downstream -- the model, the noise study, the hardware specs -- assumes this
core computes the same distribution the reference implementations do. If that stops being
true, every result in the repository is wrong, so this is the gating test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.photonic_core import (
    LayeredInterferometer,
    haar_unitary,
    layered_unitary,
    permanent_batch,
    unbunched_patterns,
    unbunched_probabilities,
    verify_against_merlin,
)

CASES = [(6, 2, 1), (8, 2, 2), (8, 3, 1), (10, 3, 2), (8, 4, 1)]


@pytest.mark.parametrize("n_modes,n_photons,depth", CASES)
def test_matches_merlin(n_modes, n_photons, depth):
    worst = verify_against_merlin(n_modes, n_photons, depth, seed=1, tol=1e-9)
    assert worst < 1e-10, f"deviation from MerLin: {worst:.3e}"


@pytest.mark.parametrize("n_modes,n_photons", [(4, 2), (6, 2), (6, 3)])
def test_matches_perceval_slos(n_modes, n_photons):
    """Cross-check against Perceval's SLOS backend, independent of MerLin."""
    import perceval as pcvl

    layer = LayeredInterferometer(n_modes, n_photons, depth=1, seed=3)
    circuit, input_state = layer.to_perceval()
    phases = np.random.default_rng(4).uniform(0, 2 * np.pi, size=layer.n_encoding)
    for param in circuit.get_parameters():
        k, mode = param.name.split("_")[1:]
        param.set_value(float(phases[int(k) * n_modes + int(mode)]))

    processor = pcvl.Processor("SLOS", circuit)
    processor.with_input(pcvl.BasicState(input_state))
    results = processor.probs()["results"]

    reference = np.array(
        [
            results.get(pcvl.BasicState([1 if m in pattern else 0 for m in range(n_modes)]), 0.0)
            for pattern in layer.patterns
        ]
    )
    reference /= reference.sum()
    assert np.abs(layer.probabilities(phases) - reference).max() < 1e-10


def test_permanent_matches_definition():
    rng = np.random.default_rng(0)
    mats = rng.normal(size=(5, 3, 3)) + 1j * rng.normal(size=(5, 3, 3))
    from itertools import permutations

    expected = np.array(
        [sum(np.prod([m[i, p[i]] for i in range(3)]) for p in permutations(range(3))) for m in mats]
    )
    assert np.abs(permanent_batch(mats) - expected).max() < 1e-12


def test_permanent_is_transpose_invariant():
    rng = np.random.default_rng(1)
    mats = rng.normal(size=(4, 3, 3)) + 1j * rng.normal(size=(4, 3, 3))
    a = permanent_batch(mats)
    b = permanent_batch(np.transpose(mats, (0, 2, 1)))
    assert np.abs(a - b).max() < 1e-12


def test_probabilities_are_normalised_and_nonnegative():
    layer = LayeredInterferometer(8, 3, depth=2, seed=5)
    phases = np.random.default_rng(6).uniform(0, 2 * np.pi, size=(16, layer.n_encoding))
    probs = layer.probabilities(phases)
    assert probs.shape == (16, layer.output_size)
    assert (probs >= 0).all()
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_haar_unitary_is_unitary():
    rng = np.random.default_rng(7)
    for m in (4, 8, 12):
        u = haar_unitary(m, rng)
        assert np.abs(u.conj().T @ u - np.eye(m)).max() < 1e-12


def test_layered_unitary_is_unitary():
    layer = LayeredInterferometer(6, 2, depth=3, seed=8)
    phases = np.random.default_rng(9).uniform(0, 2 * np.pi, size=(3, 3, 6))
    for u in layered_unitary(layer.fixed, phases):
        assert np.abs(u.conj().T @ u - np.eye(6)).max() < 1e-12


def test_patterns_are_lexicographic_and_complete():
    from math import comb

    patterns = unbunched_patterns(6, 3)
    assert len(patterns) == comb(6, 3)
    assert (np.diff(patterns, axis=1) > 0).all(), "each pattern must be strictly increasing"
    as_tuples = [tuple(p) for p in patterns]
    assert as_tuples == sorted(as_tuples)


def test_identity_circuit_keeps_photons_in_input_modes():
    """With no mixing, the only unbunched outcome is the input pattern itself."""
    identity = [np.eye(5, dtype=complex), np.eye(5, dtype=complex)]
    unitary = layered_unitary(identity, np.zeros((1, 1, 5)))
    patterns = unbunched_patterns(5, 2)
    probs = unbunched_probabilities(unitary, 2, patterns=patterns)[0]
    winner = patterns[int(np.argmax(probs))]
    assert tuple(winner) == (0, 1)
    assert probs.max() > 1 - 1e-12


def test_phase_shift_on_single_mode_does_not_change_probabilities():
    """A global phase on one output mode is unobservable in the probabilities."""
    layer = LayeredInterferometer(6, 2, depth=1, seed=11)
    rng = np.random.default_rng(12)
    phases = rng.uniform(0, 2 * np.pi, size=layer.n_encoding)
    before = layer.probabilities(phases)
    layer.fixed[-1] = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, size=6))) @ layer.fixed[-1]
    assert np.abs(layer.probabilities(phases) - before).max() < 1e-12
