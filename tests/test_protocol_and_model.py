"""Protocol correctness: no leakage, deterministic features, honest metrics.

The predecessor code lost credibility on exactly these points -- a scaler that was written
but never called, a fixed ridge penalty shared across models of very different feature
counts, and a QLIKE that summed in some files and averaged in others. These tests pin the
properties down so the numbers in the paper mean what they say.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines_rc import esn_features, lag_features, rff_features, scale_input
from src.rc_protocol import ALPHA_GRID, Split, fit_readout, nrmse
from src.tasks import TASKS, load_task, narma
from src.temporal_qrc import TemporalPhotonicQRC


def _toy():
    u, y = narma(10, n=400, seed=0)
    split = Split(len(y), washout=50, n_train=300, n_val=60)
    return u, y, split


def test_split_slices_do_not_overlap_and_cover_test():
    split = Split(1000, washout=100, n_train=800, n_val=150)
    assert split.fit_slice.stop == split.val_slice.start
    assert split.val_slice.stop == split.n_train == split.test_slice.start
    assert split.test_slice.stop == 1000
    assert split.full_train_slice.stop <= split.test_slice.start


def test_input_scaler_uses_training_rows_only():
    """Statistics must not depend on data after n_train."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(200, 1))
    base = scale_input(u, 100)
    shifted = u.copy()
    shifted[100:] += 50.0            # perturb the test region only
    assert np.allclose(base[:100], scale_input(shifted, 100)[:100])


def test_lag_features_are_causal():
    """Row t must depend only on inputs at or before t.

    The perturbation is placed after ``n_train`` so the input scaler, which is fitted on
    training rows, is identical in both runs and only the lag structure is under test.
    """
    rng = np.random.default_rng(1)
    u = rng.normal(size=(60, 1))
    before = lag_features(u, n_train=40, window=5)
    changed = u.copy()
    changed[40:] += 10.0
    after = lag_features(changed, n_train=40, window=5)
    assert np.allclose(before[:40], after[:40])


def test_photonic_features_are_causal():
    u, _, split = _toy()
    model = TemporalPhotonicQRC(
        n_modes=6, photon_list=(2,), reservoirs_per_photon=1, encode_window=3,
        window=4, washout=split.washout, seed=0,
    )
    before = model.build_features(u, split.n_train)
    changed = u.copy()
    changed[split.n_train :] += 5.0        # perturb test rows only; scaler is unaffected
    after = model.build_features(changed, split.n_train)
    assert np.allclose(before[: split.n_train], after[: split.n_train], atol=1e-10)


def test_photonic_features_are_deterministic():
    u, _, split = _toy()
    kwargs = dict(n_modes=6, photon_list=(2,), reservoirs_per_photon=2,
                  washout=split.washout, seed=7)
    a = TemporalPhotonicQRC(**kwargs).build_features(u, split.n_train)
    b = TemporalPhotonicQRC(**kwargs).build_features(u, split.n_train)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_reservoirs():
    u, _, split = _toy()
    kwargs = dict(n_modes=6, photon_list=(2,), reservoirs_per_photon=1, washout=split.washout)
    a = TemporalPhotonicQRC(seed=1, **kwargs).build_features(u, split.n_train)
    b = TemporalPhotonicQRC(seed=2, **kwargs).build_features(u, split.n_train)
    assert not np.allclose(a, b)


def test_feedback_flag_changes_dynamics():
    """With feedback off the model must reduce to a stateless windowed map."""
    u, _, split = _toy()
    kwargs = dict(n_modes=6, photon_list=(2,), reservoirs_per_photon=1,
                  encode_window=3, window=0, use_classical=False, washout=split.washout, seed=3)
    recurrent = TemporalPhotonicQRC(feedback=True, **kwargs).build_features(u, split.n_train)
    stateless = TemporalPhotonicQRC(feedback=False, **kwargs).build_features(u, split.n_train)
    assert not np.allclose(recurrent, stateless)

    # A stateless map must give identical features for identical inputs, wherever they occur.
    # Rows before the encoding window fills are zero-padded, so comparison starts after it.
    repeated = np.vstack([u[:50], u[:50]])
    feats = TemporalPhotonicQRC(feedback=False, **kwargs).build_features(repeated, 50)
    fill = kwargs["encode_window"]
    assert np.allclose(feats[fill:50], feats[50 + fill :], atol=1e-10)


@pytest.mark.parametrize("encode_window", [1, 10])
def test_esp_decay_vanishes_for_contracting_feedback(encode_window):
    """encode_window > 1 is covered because esp_decay must build the same lagged drive
    transform() does; feeding it the raw input silently mismatches the reservoir width."""
    u, _, split = _toy()
    model = TemporalPhotonicQRC(
        n_modes=6, photon_list=(2,), reservoirs_per_photon=1, g_fb=0.3, leak=0.3,
        encode_window=encode_window, washout=split.washout, seed=4,
    )
    model.build_features(u, split.n_train)
    decay = model.esp_decay(u)
    assert len(decay) == len(u)
    assert decay[-50:].mean() < 0.05 * max(decay.max(), 1e-12)


def test_esp_decay_grows_with_feedback_gain():
    """A larger feedback gain must retain the perturbation longer, or g_fb is not the
    contractivity knob the model claims it is."""
    u, _, split = _toy()

    def tail_fraction(gain):
        model = TemporalPhotonicQRC(
            n_modes=6, photon_list=(2,), reservoirs_per_photon=1, g_fb=gain, leak=0.3,
            encode_window=5, washout=split.washout, seed=4,
        )
        model.build_features(u, split.n_train)
        decay = model.esp_decay(u)
        return decay[-50:].mean() / max(decay.max(), 1e-30)

    assert tail_fraction(3.0) > tail_fraction(0.1)


def test_ridge_penalty_is_selected_not_fixed():
    """Different feature maps must be allowed different penalties."""
    u, y, split = _toy()
    _, alpha_small, _ = fit_readout(lag_features(u, split.n_train, window=3), y, split)
    _, alpha_big, _ = fit_readout(rff_features(u, split.n_train, n_features=400), y, split)
    assert alpha_small in ALPHA_GRID and alpha_big in ALPHA_GRID
    assert alpha_small != alpha_big


def test_readout_never_sees_test_targets():
    """Perturbing test targets must not change the fitted model's predictions."""
    u, y, split = _toy()
    features = lag_features(u, split.n_train, window=8)
    pred_a, _, _ = fit_readout(features, y, split)
    corrupted = y.copy()
    corrupted[split.n_train :] = 0.0
    pred_b, _, _ = fit_readout(features, corrupted, split)
    assert np.allclose(pred_a, pred_b)


def test_nrmse_is_scale_invariant_and_zero_for_perfect():
    rng = np.random.default_rng(5)
    y = rng.normal(size=200)
    assert nrmse(y, y) == pytest.approx(0.0, abs=1e-12)
    noisy = y + 0.1 * rng.normal(size=200)
    assert nrmse(y, noisy) == pytest.approx(nrmse(3 * y, 3 * noisy), rel=1e-9)


def test_predicting_the_mean_gives_nrmse_one():
    rng = np.random.default_rng(6)
    y = rng.normal(size=500)
    assert nrmse(y, np.full_like(y, y.mean())) == pytest.approx(1.0, rel=1e-6)


def test_esn_input_scaling_actually_matters():
    """Guards the baseline fix: the original ESN had no input-scaling knob."""
    u, _, _ = _toy()
    a = esn_features(u, n_train=300, res_size=50, input_scaling=0.1, seed=0)
    b = esn_features(u, n_train=300, res_size=50, input_scaling=2.0, seed=0)
    assert not np.allclose(a, b)


@pytest.mark.parametrize("name", ["narma5", "narma10", "narma20", "mackey_glass_h17", "lorenz63"])
def test_synthetic_tasks_are_finite_and_aligned(name):
    u, y, split = load_task(name)
    assert len(u) == len(y) and np.isfinite(u).all() and np.isfinite(y).all()
    assert np.std(y) > 0
    assert split.washout < split.n_train < split.n_total


def test_task_registry_is_reseedable_for_synthetic_tasks():
    a_u, a_y, _ = load_task("narma10", seed=1)
    b_u, b_y, _ = load_task("narma10", seed=2)
    assert not np.allclose(a_u, b_u) and not np.allclose(a_y, b_y)


def test_driven_narma_is_not_the_autoregressive_variant():
    """The headline task must not hand the model the target's own history."""
    driven_u, driven_y, _ = load_task("narma10")
    auto_u, auto_y, _ = load_task("narma10_autoregressive")
    # In the autoregressive variant the input is the target shifted by one.
    assert np.allclose(auto_u[1:, 0], auto_y[:-1], atol=1e-12)
    # In the driven variant it is not.
    assert not np.allclose(driven_u[1:, 0], driven_y[:-1], atol=1e-6)
    assert set(TASKS) >= {"narma10", "narma10_autoregressive"}


def test_ridge_path_matches_sklearn_ridge():
    """The SVD path must reproduce sklearn's Ridge exactly, or penalty selection is wrong."""
    from sklearn.linear_model import Ridge

    from src.rc_protocol import _RidgePath

    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 25))
    y = x @ rng.normal(size=25) + 0.1 * rng.normal(size=80)
    x_new = rng.normal(size=(12, 25))

    path = _RidgePath(x, y)
    for alpha in (1e-6, 1e-2, 1.0, 100.0, 1e4):
        reference = Ridge(alpha=alpha, fit_intercept=True).fit(x, y).predict(x_new)
        assert np.allclose(path.predict(x_new, alpha), reference, atol=1e-8)


def test_ridge_path_matches_sklearn_when_overparameterised():
    """The regime that matters here: far more features than training rows."""
    from sklearn.linear_model import Ridge

    from src.rc_protocol import _RidgePath

    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 300))
    y = x @ rng.normal(size=300) + 0.1 * rng.normal(size=40)
    x_new = rng.normal(size=(10, 300))

    path = _RidgePath(x, y)
    for alpha in (1e-3, 1.0, 50.0):
        reference = Ridge(alpha=alpha, fit_intercept=True).fit(x, y).predict(x_new)
        assert np.allclose(path.predict(x_new, alpha), reference, atol=1e-7)


def test_r2_oos_is_zero_for_the_benchmark_and_one_for_perfect():
    from src.rc_protocol import r2_oos

    rng = np.random.default_rng(3)
    y = rng.normal(loc=2.0, size=300)
    mean = float(y.mean())
    assert r2_oos(y, np.full_like(y, mean), mean) == pytest.approx(0.0, abs=1e-9)
    assert r2_oos(y, y, mean) == pytest.approx(1.0, abs=1e-12)
    # Worse than the constant benchmark must be negative, not merely small.
    assert r2_oos(y, np.full_like(y, mean + 5.0), mean) < 0
