import weakref

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
    """Stage only needed Q/SQ pairs, releasing each pair before loading the next."""
    q_matrices = {str(s): np.full((4, 5), float(s)) for s in (10, 100, 102)}
    sq_matrices = {str(s): np.full((4, 5), float(s + 1)) for s in (10, 100, 102)}
    names = {
        id(table): (kind, sigma)
        for kind, tables in (("q", q_matrices), ("sq", sq_matrices))
        for sigma, table in tables.items()
    }
    events = []
    references = []
    real_as_tensor = torch.as_tensor
    real_tensor = torch.tensor

    def track(converter, value, *args, **kwargs):
        name = names.get(id(value))
        if name is not None:
            kind, sigma = name
            assert sigma != "10", "A table outside the candidate window was staged"
            live = sum(ref() is not None for ref in references)
            assert live == (0 if kind == "q" else 1), "Previous Q/SQ pair is still live"
        result = converter(value, *args, **kwargs)
        if name is not None:
            events.append(name)
            references.append(weakref.ref(result))
        return result

    def tracking_as_tensor(value, *args, **kwargs):
        return track(real_as_tensor, value, *args, **kwargs)

    def tracking_tensor(value, *args, **kwargs):
        return track(real_tensor, value, *args, **kwargs)

    def fake_solver(*, P, Y_batch, **_):
        return (
            torch.ones((len(Y_batch), P.shape[1]), dtype=P.dtype, device=P.device),
            torch.ones(len(Y_batch), dtype=torch.bool, device=P.device),
        )

    def fake_likelihood(Y, X, Q, *_):
        # Smaller sigma wins, so the current sigma remains selected.
        return -torch.ones_like(X) * Q[0, 0], torch.zeros_like(X), torch.zeros_like(X)

    monkeypatch.setattr(_sigma.torch, "as_tensor", tracking_as_tensor)
    monkeypatch.setattr(_sigma.torch, "tensor", tracking_tensor)
    monkeypatch.setattr(_sigma, "solve_irwls_batch_shared", fake_solver)
    monkeypatch.setattr(_sigma, "_calc_q_all_eager", fake_likelihood)
    result = choose_sigma(
        spatial_counts=np.ones((4, 2)),
        spatial_numi=np.full(4, 100.0),
        norm_profiles=np.full((2, 2), 0.5),
        q_matrices=q_matrices,
        sq_matrices=sq_matrices,
        x_vals=np.arange(5.0),
        sigma_init=100,
        min_umi=0,
        n_fit=2,
        n_epoch=1,
        device="cpu",
    )
    assert result == 100
    assert events == [
        ("q", "100"),
        ("sq", "100"),
        ("q", "100"),
        ("sq", "100"),
        ("q", "102"),
        ("sq", "102"),
    ]
    assert all(ref() is None for ref in references)
