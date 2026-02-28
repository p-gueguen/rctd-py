import numpy as np

from rctd._likelihood import load_cached_q_matrices


def test_load_cached_q_matrices():
    mats = load_cached_q_matrices()
    assert isinstance(mats, dict)
    assert len(mats) > 0


def test_q_matrices_keys():
    mats = load_cached_q_matrices()
    assert "X_vals" in mats
    # Keys Q_10 through Q_70 (every integer), Q_72 through Q_200 (every even)
    expected_q_keys = [f"Q_{k}" for k in range(10, 71)] + [f"Q_{k}" for k in range(72, 201, 2)]
    for key in expected_q_keys:
        assert key in mats, f"Missing key: {key}"


def test_q_matrices_shapes():
    mats = load_cached_q_matrices()
    n_cols = 439
    n_rows = 1003
    q_keys = [k for k in mats if k.startswith("Q_")]
    assert len(q_keys) > 0
    for key in q_keys:
        assert mats[key].shape == (n_rows, n_cols), (
            f"{key}: expected shape ({n_rows}, {n_cols}), got {mats[key].shape}"
        )


def test_x_vals_properties():
    mats = load_cached_q_matrices()
    x_vals = mats["X_vals"]
    assert x_vals.shape == (439,)
    assert np.all(np.diff(x_vals) > 0), "X_vals should be monotonically increasing"
    assert x_vals[0] < 1e-3
    assert x_vals[-1] > 1000
