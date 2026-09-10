"""Q-matrix provisioning: env-var staging and the offline error (issue #29).

The real file is 404 MB and lives in a GitHub release, so these tests use a tiny
stand-in .npz and never touch the network or the user's ~/.cache/rctd.
"""

import numpy as np
import pytest

from rctd import _likelihood


@pytest.fixture
def fake_npz(tmp_path):
    """A minimal q_matrices.npz stand-in, with the keys the loader returns."""
    path = tmp_path / "staged" / "q_matrices.npz"
    path.parent.mkdir()
    np.savez(path, X_vals=np.linspace(0, 1, 5), Q_10=np.ones((2, 3)))
    return path


@pytest.fixture
def no_network_no_cache(tmp_path, monkeypatch):
    """Point HOME at a scratch dir and make any download attempt fail as if offline."""
    monkeypatch.setattr(_likelihood.Path, "home", staticmethod(lambda: tmp_path / "home"))

    def offline(*_args, **_kwargs):
        raise OSError("[Errno -3] Temporary failure in name resolution")

    monkeypatch.setattr(_likelihood.urllib.request, "urlretrieve", offline)


def test_env_var_file_is_used(fake_npz, no_network_no_cache, monkeypatch):
    """RCTD_Q_MATRICES pointing at the .npz is loaded without any download."""
    monkeypatch.setenv("RCTD_Q_MATRICES", str(fake_npz))
    out = _likelihood.load_cached_q_matrices()
    assert set(out) == {"X_vals", "Q_10"}


def test_env_var_directory_is_used(fake_npz, no_network_no_cache, monkeypatch):
    """A directory holding q_matrices.npz works too, so a cluster can stage one copy."""
    monkeypatch.setenv("RCTD_Q_MATRICES", str(fake_npz.parent))
    assert set(_likelihood.load_cached_q_matrices()) == {"X_vals", "Q_10"}


def test_env_var_wrong_path_names_the_variable(tmp_path, no_network_no_cache, monkeypatch):
    """A typo'd env var fails on the env var, not with a download or a KeyError."""
    monkeypatch.setenv("RCTD_Q_MATRICES", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="RCTD_Q_MATRICES"):
        _likelihood.load_cached_q_matrices()


def test_offline_error_gives_staging_instructions(no_network_no_cache, monkeypatch):
    """On an isolated compute node the failure has to say how to stage the file."""
    monkeypatch.delenv("RCTD_Q_MATRICES", raising=False)
    # data_dir is passed explicitly because a dev checkout has the real 404 MB file
    # sitting in the package data/ directory, which would otherwise satisfy the load.
    with pytest.raises(RuntimeError) as excinfo:
        _likelihood.load_cached_q_matrices(data_dir="/nonexistent-data-dir")
    msg = str(excinfo.value)
    assert _likelihood._Q_MATRICES_URL in msg
    assert "RCTD_Q_MATRICES" in msg
    assert ".cache/rctd" in msg


def test_staged_file_is_not_silently_replaced(tmp_path, no_network_no_cache, monkeypatch):
    """A corrupt file the user staged is reported, never overwritten from the network."""
    bad = tmp_path / "staged_bad" / "q_matrices.npz"
    bad.parent.mkdir()
    bad.write_bytes(b"not an npz")
    monkeypatch.setenv("RCTD_Q_MATRICES", str(bad))
    with pytest.raises(RuntimeError, match="could not be read"):
        _likelihood.load_cached_q_matrices()
    assert bad.read_bytes() == b"not an npz"
