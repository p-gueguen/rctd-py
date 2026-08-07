"""Scientific test: the protein modality resolves a cell-type split that RNA cannot.

Two cell types share an identical RNA profile (no DE genes between them), so the RNA
likelihood is flat along the (w_0 - w_1) direction and an RNA-only solve cannot recover
their individual weights. A single protein marker separates them. The joint solve must
therefore achieve lower per-type weight error on the degenerate pair.
"""

import numpy as np
import pytest
import torch

from rctd._irwls import solve_irwls_batch, solve_irwls_batch_shared
from rctd._likelihood import build_x_vals, compute_q_matrix, compute_spline_coefficients


@pytest.fixture(scope="module")
def likelihood_tables():
    x_vals = build_x_vals()
    Q_mat = compute_q_matrix(sigma=1.0, x_vals=x_vals, K_val=100)
    SQ_mat = compute_spline_coefficients(Q_mat, x_vals)
    return torch.tensor(Q_mat), torch.tensor(SQ_mat), torch.tensor(x_vals)


@pytest.mark.protein
def test_joint_beats_rna_only_on_degenerate_pair(multimodal_synthetic_data, likelihood_tables):
    d = multimodal_synthetic_data
    Q_mat, SQ_mat, x_vals = likelihood_tables

    P = torch.tensor(d["profiles"])
    Y = torch.tensor(d["counts"], dtype=P.dtype)
    nUMI = torch.tensor(d["nUMI"], dtype=P.dtype)
    P_prot = torch.tensor(d["P_prot"], dtype=P.dtype)
    Y_prot = torch.tensor(d["protein"], dtype=P.dtype)
    inv_tau2 = torch.tensor(d["inv_tau2"], dtype=P.dtype)
    true_w = d["true_weights"]
    i, j = d["degenerate_pair"]

    common = dict(
        P=P,
        Y_batch=Y,
        nUMI_batch=nUMI,
        Q_mat=Q_mat,
        SQ_mat=SQ_mat,
        x_vals=x_vals,
        constrain=True,
    )
    rna = solve_irwls_batch_shared(**common)[0].numpy()
    # lam ~ O(1): with unit-variance standardized protein the WLS block is already
    # on a comparable scale to the RNA likelihood. Larger lam over-weights protein
    # and starts to distort even the identifiable combined mass.
    joint = solve_irwls_batch_shared(
        **common,
        P_prot=P_prot,
        Y_prot_batch=Y_prot,
        inv_tau2=inv_tau2,
        lam=1.0,
    )[0].numpy()

    # Sanity: the IDENTIFIABLE combined mass (w_i + w_j) is recovered by BOTH
    # models on average (this is what RNA *can* see; the per-type split is not).
    sum_true = true_w[:, [i, j]].sum(1)
    assert np.abs(rna[:, [i, j]].sum(1) - sum_true).mean() < 0.05
    assert np.abs(joint[:, [i, j]].sum(1) - sum_true).mean() < 0.05

    # The per-type split is what only protein can resolve.
    mae_rna = np.abs(rna[:, [i, j]] - true_w[:, [i, j]]).mean()
    mae_joint = np.abs(joint[:, [i, j]] - true_w[:, [i, j]]).mean()
    print(f"\nMAE on degenerate pair  rna-only={mae_rna:.4f}  joint={mae_joint:.4f}")
    assert mae_joint < mae_rna, f"joint MAE {mae_joint:.3f} !< rna-only {mae_rna:.3f}"
    # And it should be a substantial improvement, not a marginal one.
    assert mae_joint < 0.5 * mae_rna


@pytest.mark.protein
def test_lambda_zero_matches_rna_only_in_shared_solver(
    multimodal_synthetic_data, likelihood_tables
):
    """Solver-level firewall: passing protein tensors with lam=0 is bit-identical."""
    d = multimodal_synthetic_data
    Q_mat, SQ_mat, x_vals = likelihood_tables
    P = torch.tensor(d["profiles"])
    Y = torch.tensor(d["counts"], dtype=P.dtype)
    nUMI = torch.tensor(d["nUMI"], dtype=P.dtype)

    common = dict(
        P=P,
        Y_batch=Y,
        nUMI_batch=nUMI,
        Q_mat=Q_mat,
        SQ_mat=SQ_mat,
        x_vals=x_vals,
        constrain=True,
    )
    rna = solve_irwls_batch_shared(**common)[0].numpy()
    with_prot_off = solve_irwls_batch_shared(
        **common,
        P_prot=torch.tensor(d["P_prot"], dtype=P.dtype),
        Y_prot_batch=torch.tensor(d["protein"], dtype=P.dtype),
        inv_tau2=torch.tensor(d["inv_tau2"], dtype=P.dtype),
        lam=0.0,
    )[0].numpy()
    np.testing.assert_array_equal(rna, with_prot_off)


@pytest.mark.protein
def test_subset_solver_protein_resolves_split(multimodal_synthetic_data, likelihood_tables):
    """The (N, G, K) subset-variant solver (used by doublet/multi) also fuses protein."""
    d = multimodal_synthetic_data
    Q_mat, SQ_mat, x_vals = likelihood_tables
    P = torch.tensor(d["profiles"])
    G, K = P.shape
    nUMI = torch.tensor(d["nUMI"], dtype=P.dtype)
    N = nUMI.shape[0]
    Y = torch.tensor(d["counts"], dtype=P.dtype)
    S_batch = P.unsqueeze(0) * nUMI[:, None, None]  # (N, G, K)
    P_prot = torch.tensor(d["P_prot"], dtype=P.dtype)  # (M, K)
    P_prot_batch = P_prot.unsqueeze(0).expand(N, -1, -1).contiguous()  # (N, M, K)
    Y_prot = torch.tensor(d["protein"], dtype=P.dtype)
    inv_tau2 = torch.tensor(d["inv_tau2"], dtype=P.dtype)
    tw = d["true_weights"]
    i, j = d["degenerate_pair"]

    common = dict(
        S_batch=S_batch,
        Y_batch=Y,
        nUMI_batch=nUMI,
        Q_mat=Q_mat,
        SQ_mat=SQ_mat,
        x_vals=x_vals,
        constrain=True,
    )
    rna = solve_irwls_batch(**common)[0].numpy()
    # lam=0 firewall for this solver too
    off = solve_irwls_batch(
        **common, P_prot_batch=P_prot_batch, Y_prot_batch=Y_prot, inv_tau2=inv_tau2, lam=0.0
    )[0].numpy()
    np.testing.assert_array_equal(rna, off)

    joint = solve_irwls_batch(
        **common, P_prot_batch=P_prot_batch, Y_prot_batch=Y_prot, inv_tau2=inv_tau2, lam=1.0
    )[0].numpy()
    mae_rna = np.abs(rna[:, [i, j]] - tw[:, [i, j]]).mean()
    mae_joint = np.abs(joint[:, [i, j]] - tw[:, [i, j]]).mean()
    assert mae_joint < 0.5 * mae_rna, f"joint {mae_joint:.3f} vs rna {mae_rna:.3f}"
