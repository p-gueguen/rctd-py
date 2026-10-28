import numpy as np
import torch

from rctd import _sigma
from rctd._likelihood import load_cached_q_matrices
from rctd._sigma import choose_sigma


def test_choose_sigma_runs(synthetic_data):
    """Test that choose_sigma completes successfully."""
    # We need q_matrices and x_vals
    cache = load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    # remaining cache is exactly q_matrices
    q_matrices = {k.replace("Q_", ""): v for k, v in cache.items()}

    profiles = synthetic_data["profiles"]  # (G, K)
    spatial_adata = synthetic_data["spatial"]
    spatial_counts = spatial_adata.X
    spatial_numi = np.array(spatial_counts.sum(axis=1)).flatten()

    # Run with small n_epoch and n_fit just to check if it runs without crashing
    best_sigma = choose_sigma(
        spatial_counts=spatial_counts,
        spatial_numi=spatial_numi,
        norm_profiles=profiles,  # using original profiles as mock norm_profiles
        q_matrices=q_matrices,
        x_vals=x_vals,
        sigma_init=100,
        min_umi=50,
        n_fit=5,
        n_epoch=2,
    )

    assert isinstance(best_sigma, (int, np.integer))
    assert best_sigma > 0


def test_choose_sigma_stages_only_tables_in_the_candidate_window(monkeypatch):
    """An unused Q/SQ pair must never be copied to the compute device."""
    q_current = np.full((4, 5), 1.0)
    sq_current = np.full((4, 5), 2.0)
    q_unused = np.full((4, 5), 3.0)
    sq_unused = np.full((4, 5), 4.0)
    names = {
        id(q_current): "q_current",
        id(sq_current): "sq_current",
        id(q_unused): "q_unused",
        id(sq_unused): "sq_unused",
    }
    events = []
    real_as_tensor = torch.as_tensor

    def tracking_as_tensor(value, *args, **kwargs):
        if id(value) in names:
            events.append(names[id(value)])
        return real_as_tensor(value, *args, **kwargs)

    def fake_solver(*, P, Y_batch, **_):
        events.append("solver")
        return (
            torch.ones((len(Y_batch), P.shape[1]), dtype=P.dtype, device=P.device),
            torch.ones(len(Y_batch), dtype=torch.bool, device=P.device),
        )

    def fake_likelihood(Y, X, *_):
        events.append("likelihood")
        zeros = torch.zeros_like(X)
        return zeros, zeros, zeros

    monkeypatch.setattr(_sigma.torch, "as_tensor", tracking_as_tensor)
    monkeypatch.setattr(_sigma, "solve_irwls_batch_shared", fake_solver)
    monkeypatch.setattr(_sigma, "_calc_q_all_eager", fake_likelihood)

    result = choose_sigma(
        spatial_counts=np.ones((4, 2)),
        spatial_numi=np.full(4, 100.0),
        norm_profiles=np.full((2, 2), 0.5),
        q_matrices={"10": q_unused, "100": q_current},
        sq_matrices={"10": sq_unused, "100": sq_current},
        x_vals=np.arange(5.0),
        sigma_init=100,
        min_umi=0,
        n_fit=2,
        n_epoch=1,
        device="cpu",
    )

    assert result == 100
    assert "q_unused" not in events
    assert "sq_unused" not in events
    assert events.count("q_current") == 2  # current fit and candidate scoring
    assert events.count("sq_current") == 2
    assert events.count("solver") == 1
    assert events.count("likelihood") == 5  # five scale factors
