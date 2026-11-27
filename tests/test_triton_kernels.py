"""Unit and CUDA parity tests for fused Triton GPU kernels."""

import numpy as np
import pytest
import torch

from rctd import _doublet, _full, _irwls, _likelihood, _multi, _sigma, _triton_kernels
from rctd._types import RCTDConfig

CUDA_TRITON = pytest.mark.skipif(
    not (torch.cuda.is_available() and _triton_kernels.ENABLED),
    reason="CUDA GPU with Triton required",
)


def _synthetic_tables(seed: int = 0):
    cache = _likelihood.load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    q_mat = cache["Q_62"]
    sq_mat = _likelihood.compute_spline_coefficients(q_mat, x_vals)
    return q_mat, sq_mat, x_vals


def test_use_triton_cpu_and_env_toggle(monkeypatch):
    assert _triton_kernels.use_triton(torch.zeros(4)) is False
    monkeypatch.setenv("RCTD_TRITON", "0")
    assert _triton_kernels.use_triton(torch.zeros(4)) is False


def test_prepare_y_int_gpu_validation_cpu_or_cuda():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.tensor([[0.0, 5.0, 200.0]], device=dev)
    y_int = _triton_kernels.prepare_y_int_gpu(y, 100)
    assert y_int.dtype == torch.int32
    assert y_int.tolist() == [[0, 5, 100]]

    with pytest.raises(ValueError, match="finite"):
        _triton_kernels.prepare_y_int_gpu(torch.tensor([float("nan")], device=dev), 100)
    with pytest.raises(ValueError, match="non-negative"):
        _triton_kernels.prepare_y_int_gpu(torch.tensor([-1.0], device=dev), 100)


@CUDA_TRITON
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_triton_shared_derivatives_and_unpack_parity(dtype):
    q_mat, sq_mat, x_vals = _synthetic_tables()
    rng = np.random.default_rng(123)
    N, G, K = 32, 240, 24
    Y = torch.tensor(rng.poisson(1.5, size=(N, G)), dtype=dtype, device="cuda")
    P = torch.tensor(rng.dirichlet(np.ones(K), size=G), dtype=dtype, device="cuda")
    nUMI = Y.sum(dim=1).clamp(min=20.0)
    w = torch.tensor(rng.dirichlet(np.ones(K), size=N), dtype=dtype, device="cuda")

    Q_g = torch.tensor(q_mat, dtype=dtype, device="cuda")
    SQ_g = torch.tensor(sq_mat, dtype=dtype, device="cuda")
    X_g = torch.tensor(x_vals, dtype=dtype, device="cuda")

    thresh = torch.clamp(nUMI * 1e-7, min=1e-4)
    pred_raw = w @ P.T
    pred = torch.clamp(torch.abs(nUMI[:, None] * pred_raw), min=thresh[:, None])

    _, d1_ref, d2_ref = _likelihood._calc_q_all_impl(
        Y.reshape(-1), pred.reshape(-1), Q_g, SQ_g, X_g
    )
    d1_ref = d1_ref.reshape(N, G) * nUMI[:, None]
    d2_ref = -d2_ref.reshape(N, G) * (nUMI * nUMI)[:, None]

    poly_gpu = _triton_kernels.build_poly_table_gpu(Q_g, SQ_g, X_g)
    Y_int = _triton_kernels.prepare_y_int_gpu(Y, poly_gpu[6])
    d1_tri, d2_tri = _triton_kernels.irwls_shared_d1_d2_triton(
        Y_int, pred_raw, nUMI, thresh, poly_gpu, out_dtype=dtype
    )

    atol = 1e-11 if dtype == torch.float64 else 2e-3
    rtol = 1e-11 if dtype == torch.float64 else 2e-5
    assert torch.allclose(d1_tri, d1_ref, rtol=rtol, atol=atol)
    assert torch.allclose(d2_tri, d2_ref, rtol=rtol, atol=atol)

    tri_idx = torch.triu_indices(K, K, device="cuda")
    P_sym = (P[:, tri_idx[0]] * P[:, tri_idx[1]]).contiguous()
    H_tri = d2_ref @ P_sym
    H_ref = torch.empty((N, K, K), dtype=dtype, device="cuda")
    H_ref[:, tri_idx[0], tri_idx[1]] = H_tri
    H_ref[:, tri_idx[1], tri_idx[0]] = H_tri
    H_got = _triton_kernels.unpack_sym_hess_triton(H_tri, K)
    assert torch.equal(H_got, H_ref)


@CUDA_TRITON
@pytest.mark.parametrize("K", [3, 8, 15, 45, 78])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_triton_solve_box_qp_parity(K, dtype):
    rng = np.random.default_rng(77 + K)
    N = 16
    A = rng.normal(size=(N, K, K))
    D = torch.tensor(A @ A.transpose(0, 2, 1) + 0.1 * np.eye(K), dtype=dtype, device="cuda")
    D = D / torch.linalg.matrix_norm(D, ord=2)[:, None, None] + 1e-7 * torch.eye(
        K, dtype=dtype, device="cuda"
    )
    d = torch.tensor(rng.normal(size=(N, K)), dtype=dtype, device="cuda")
    lb = torch.tensor(-rng.uniform(0.0, 0.2, size=(N, K)), dtype=dtype, device="cuda")

    x_ref = _irwls._solve_box_qp_batch_impl(D.clone(), d, lb, n_sweeps=50)
    x_tri = _triton_kernels.solve_box_qp_triton(D.clone(), d, lb, n_sweeps=50)
    atol = 1e-10 if dtype == torch.float64 else 1e-4
    assert torch.allclose(x_tri, x_ref, atol=atol, rtol=atol)


@CUDA_TRITON
def test_triton_box_qp_zero_diagonal_fallback():
    D = torch.eye(4, dtype=torch.float64, device="cuda").unsqueeze(0)
    D[0, 0, 0] = 0.0
    d = torch.ones((1, 4), dtype=torch.float64, device="cuda")
    lb = torch.zeros((1, 4), dtype=torch.float64, device="cuda")
    x = _triton_kernels.solve_box_qp_triton(D, d, lb, n_sweeps=10)
    assert torch.isnan(x[0, 0]) or torch.isinf(x[0, 0])


@CUDA_TRITON
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_triton_pairs_and_singles_parity(dtype, monkeypatch):
    q_mat, sq_mat, x_vals = _synthetic_tables()
    rng = np.random.default_rng(42)
    N, G, K = 12, 180, 6
    P_np = rng.dirichlet(np.ones(K), size=G)
    P_np[:120, :] = 0.0
    P_np = P_np / np.maximum(P_np.sum(axis=0, keepdims=True), 1e-12)
    Y_np = rng.poisson(1.2, size=(N, G))
    nUMI_np = np.maximum(Y_np.sum(axis=1), 50.0)

    P_g = torch.tensor(P_np, dtype=dtype, device="cuda")
    Q_g = torch.tensor(q_mat, dtype=dtype, device="cuda")
    SQ_g = torch.tensor(sq_mat, dtype=dtype, device="cuda")
    X_g = torch.tensor(x_vals, dtype=dtype, device="cuda")
    Y_g = torch.tensor(Y_np, dtype=dtype, device="cuda")
    nUMI_g = torch.tensor(nUMI_np, dtype=dtype, device="cuda")

    triples = np.array(
        [[n, t1, t2] for n in range(N) for t1 in range(K) for t2 in range(t1 + 1, K)]
    )
    singles = np.array([[n, t] for n in range(N) for t in range(K)])

    # PyTorch reference
    monkeypatch.setenv("RCTD_TRITON", "0")
    tr_t = torch.tensor(triples, dtype=torch.long, device="cuda")
    P_pair = torch.stack([P_g[:, tr_t[:, 1]].T, P_g[:, tr_t[:, 2]].T], dim=-1)
    S_pair = nUMI_g[tr_t[:, 0]][:, None, None] * P_pair
    w_ref, conv_ref = _irwls.solve_irwls_batch(
        S_pair, Y_g[tr_t[:, 0]], nUMI_g[tr_t[:, 0]], Q_g, SQ_g, X_g, max_iter=25, constrain=False
    )
    exp_ref = torch.clamp((S_pair * w_ref[:, None, :]).sum(-1), min=1e-4)
    sc_ref = _likelihood.calc_log_likelihood_batch(Y_g[tr_t[:, 0]], exp_ref, Q_g, SQ_g, X_g)

    sg_t = torch.tensor(singles, dtype=torch.long, device="cuda")
    S_sg = nUMI_g[sg_t[:, 0]][:, None, None] * P_g[:, sg_t[:, 1]].T.unsqueeze(-1)
    w_sg_ref, _ = _irwls.solve_irwls_batch(
        S_sg, Y_g[sg_t[:, 0]], nUMI_g[sg_t[:, 0]], Q_g, SQ_g, X_g, max_iter=25, constrain=False
    )
    exp_sg_ref = torch.clamp((S_sg * w_sg_ref[:, None, :]).sum(-1), min=1e-4)
    sc_sg_ref = _likelihood.calc_log_likelihood_batch(Y_g[sg_t[:, 0]], exp_sg_ref, Q_g, SQ_g, X_g)

    # Triton under test
    monkeypatch.setenv("RCTD_TRITON", "1")
    poly_gpu = _triton_kernels.build_poly_table_gpu(Q_g, SQ_g, X_g)
    Y_int = _triton_kernels.prepare_y_int_gpu(Y_g, poly_gpu[6])
    P_KT = P_g.T.contiguous()

    w_tri, sc_tri, conv_tri = _triton_kernels.fit_and_score_pairs_triton(
        tr_t[:, 0],
        tr_t[:, 1],
        tr_t[:, 2],
        Y_int,
        nUMI_g,
        P_KT,
        poly_gpu,
        max_iter=25,
        compute_score=True,
    )
    w_sg_tri, sc_sg_tri = _triton_kernels.fit_and_score_singles_triton(
        sg_t[:, 0],
        sg_t[:, 1],
        Y_int,
        nUMI_g,
        P_KT,
        poly_gpu,
        max_iter=25,
        write_weights=True,
        compute_score=True,
    )

    if dtype == torch.float64:
        assert torch.allclose(w_tri, w_ref, atol=1e-11, rtol=1e-11)
        assert torch.allclose(sc_tri, sc_ref, atol=1e-10, rtol=1e-10)
        assert torch.equal(conv_tri, conv_ref)
        assert torch.allclose(w_sg_tri, w_sg_ref[:, 0], atol=1e-11, rtol=1e-11)
        assert torch.allclose(sc_sg_tri, sc_sg_ref, atol=1e-10, rtol=1e-10)
    else:
        assert torch.allclose(w_tri, w_ref, atol=5e-3, rtol=5e-3)
        assert torch.allclose(sc_tri, sc_ref, atol=5e-2, rtol=1e-3)
        assert torch.allclose(sc_sg_tri, sc_sg_ref, atol=5e-2, rtol=1e-3)


@CUDA_TRITON
@pytest.mark.parametrize("k_sub", [1, 2, 3, 5, 6])
def test_triton_subsets_parity_f64(k_sub, monkeypatch):
    q_mat, sq_mat, x_vals = _synthetic_tables()
    rng = np.random.default_rng(100 + k_sub)
    N, G, K = 8, 128, 8
    P_np = rng.dirichlet(np.ones(K), size=G)
    Y_np = rng.poisson(1.5, size=(N, G))
    nUMI_np = np.maximum(Y_np.sum(axis=1), 40.0)

    P_g = torch.tensor(P_np, dtype=torch.float64, device="cuda")
    Q_g = torch.tensor(q_mat, dtype=torch.float64, device="cuda")
    SQ_g = torch.tensor(sq_mat, dtype=torch.float64, device="cuda")
    X_g = torch.tensor(x_vals, dtype=torch.float64, device="cuda")
    Y_g = torch.tensor(Y_np, dtype=torch.float64, device="cuda")
    nUMI_g = torch.tensor(nUMI_np, dtype=torch.float64, device="cuda")

    tasks = [(n, rng.choice(K, size=k_sub, replace=False).tolist()) for n in range(N)]

    monkeypatch.setenv("RCTD_TRITON", "0")
    sc_ref = _multi._run_batched_scoring(
        tasks, nUMI_g, Y_g, P_g, Q_g, SQ_g, X_g, batch_size=32, n_iter=25
    )
    w_ref = _multi._run_batched_weights(
        tasks, nUMI_g, Y_g, P_g, Q_g, SQ_g, X_g, batch_size=32, n_iter=50
    )

    monkeypatch.setenv("RCTD_TRITON", "1")
    sc_tri = _multi._run_batched_scoring(
        tasks, nUMI_g, Y_g, P_g, Q_g, SQ_g, X_g, batch_size=32, n_iter=25
    )
    w_tri = _multi._run_batched_weights(
        tasks, nUMI_g, Y_g, P_g, Q_g, SQ_g, X_g, batch_size=32, n_iter=50
    )

    assert np.allclose(sc_tri, sc_ref, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.stack(w_tri), np.stack(w_ref), atol=1e-11, rtol=1e-11)


@CUDA_TRITON
def test_triton_score_sigma_5_factors_parity():
    q_mat, sq_mat, x_vals = _synthetic_tables()
    rng = np.random.default_rng(7)
    N = 4096
    X = torch.tensor(np.exp(rng.uniform(-8, 4, size=N)), dtype=torch.float64, device="cuda")
    Y = torch.tensor(rng.poisson(2.0, size=N), dtype=torch.float64, device="cuda")
    Q_g = torch.tensor(q_mat, dtype=torch.float64, device="cuda")
    SQ_g = torch.tensor(sq_mat, dtype=torch.float64, device="cuda")
    X_g = torch.tensor(x_vals, dtype=torch.float64, device="cuda")

    ref_scores = []
    for fac in (0.8, 0.9, 1.0, 1.1, 1.2):
        d0 = _likelihood._calc_q_d0_impl(Y, X * fac, Q_g, SQ_g, X_g, 100)
        ref_scores.append(-float(d0.sum().item()))
    ref_min = min(ref_scores)

    poly_gpu = _triton_kernels.build_poly_table_gpu(Q_g, SQ_g, X_g, K_val=100)
    Y_int = _triton_kernels.prepare_y_int_gpu(Y, 100)
    tri_min = _triton_kernels.score_sigma_5_factors_triton(X, Y_int, poly_gpu)
    assert abs(tri_min - ref_min) <= 1e-8 * max(1.0, abs(ref_min))


@CUDA_TRITON
def test_triton_end_to_end_modes_parity_f64(synthetic_data, monkeypatch):
    q_mat, sq_mat, x_vals = _synthetic_tables()
    sp = synthetic_data["spatial"]
    Y = np.asarray(sp.X, dtype=np.float64)
    nUMI = np.maximum(Y.sum(axis=1), 20.0).astype(np.float64)
    P = synthetic_data["profiles"].astype(np.float64)
    P = P / np.maximum(P.sum(axis=0, keepdims=True), 1e-12)
    names = synthetic_data["cell_type_names"]
    cfg = RCTDConfig(device="cuda", compile=False, dtype="float64")

    monkeypatch.setenv("RCTD_TRITON", "0")
    full_ref = _full.run_full_mode(
        Y, nUMI, P, names, q_mat, sq_mat, x_vals, batch_size=32, device="cuda"
    )
    dbl_ref = _doublet.run_doublet_mode(
        Y, nUMI, P, names, q_mat, sq_mat, x_vals, cfg, batch_size=32, device="cuda"
    )
    mul_ref = _multi.run_multi_mode(
        Y, nUMI, P, names, q_mat, sq_mat, x_vals, cfg, batch_size=32, device="cuda"
    )

    monkeypatch.setenv("RCTD_TRITON", "1")
    full_tri = _full.run_full_mode(
        Y, nUMI, P, names, q_mat, sq_mat, x_vals, batch_size=32, device="cuda"
    )
    dbl_tri = _doublet.run_doublet_mode(
        Y, nUMI, P, names, q_mat, sq_mat, x_vals, cfg, batch_size=32, device="cuda"
    )
    mul_tri = _multi.run_multi_mode(
        Y, nUMI, P, names, q_mat, sq_mat, x_vals, cfg, batch_size=32, device="cuda"
    )

    assert np.allclose(full_tri.weights, full_ref.weights, atol=1e-11, rtol=1e-11)
    assert np.array_equal(dbl_tri.spot_class, dbl_ref.spot_class)
    assert np.array_equal(dbl_tri.first_type, dbl_ref.first_type)
    assert np.array_equal(dbl_tri.second_type, dbl_ref.second_type)
    assert np.allclose(dbl_tri.weights_doublet, dbl_ref.weights_doublet, atol=1e-11, rtol=1e-11)
    assert np.array_equal(mul_tri.n_types, mul_ref.n_types)
    assert np.array_equal(mul_tri.cell_type_indices, mul_ref.cell_type_indices)
    assert np.allclose(mul_tri.sub_weights, mul_ref.sub_weights, atol=1e-6, rtol=1e-6)


@CUDA_TRITON
def test_triton_choose_sigma_parity_f64(synthetic_data, monkeypatch):
    cache = _likelihood.load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    q_matrices = {k.replace("Q_", ""): v for k, v in cache.items()}
    profiles = synthetic_data["profiles"]
    counts = np.asarray(synthetic_data["spatial"].X)
    numi = counts.sum(axis=1).ravel()

    monkeypatch.setenv("RCTD_TRITON", "0")
    s_ref = _sigma.choose_sigma(
        counts,
        numi,
        profiles,
        q_matrices,
        x_vals,
        sigma_init=100,
        min_umi=50,
        n_fit=8,
        n_epoch=2,
        device="cuda",
    )
    monkeypatch.setenv("RCTD_TRITON", "1")
    s_tri = _sigma.choose_sigma(
        counts,
        numi,
        profiles,
        q_matrices,
        x_vals,
        sigma_init=100,
        min_umi=50,
        n_fit=8,
        n_epoch=2,
        device="cuda",
    )
    assert s_tri == s_ref


@CUDA_TRITON
def test_triton_n1_m1_edge_cases(monkeypatch):
    """Verify Triton scalar int(1) specialization does not break N=1 or M=1 launches."""
    q_mat, sq_mat, x_vals = _synthetic_tables()
    rng = np.random.default_rng(999)
    N, G, K = 1, 128, 6

    Q_g = torch.tensor(q_mat, dtype=torch.float64, device="cuda")
    SQ_g = torch.tensor(sq_mat, dtype=torch.float64, device="cuda")
    X_g = torch.tensor(x_vals, dtype=torch.float64, device="cuda")
    Y_g = torch.tensor(rng.poisson(1.5, size=(N, G)), dtype=torch.float64, device="cuda")
    P_g = torch.tensor(rng.dirichlet(np.ones(K), size=G), dtype=torch.float64, device="cuda")
    nUMI_g = Y_g.sum(dim=1).clamp(min=40.0)
    thresh_g = torch.clamp(nUMI_g * 1e-7, min=1e-4)
    w_g = torch.tensor(rng.dirichlet(np.ones(K), size=N), dtype=torch.float64, device="cuda")
    pred_raw = w_g @ P_g.T
    pred = torch.clamp(torch.abs(nUMI_g[:, None] * pred_raw), min=thresh_g[:, None])

    poly_gpu = _triton_kernels.build_poly_table_gpu(Q_g, SQ_g, X_g)
    Y_int = _triton_kernels.prepare_y_int_gpu(Y_g, poly_gpu[6])

    _, d1_ref, d2_ref = _likelihood._calc_q_all_impl(
        Y_g.reshape(-1), pred.reshape(-1), Q_g, SQ_g, X_g
    )
    d1_ref = d1_ref.reshape(N, G) * nUMI_g[:, None]
    d2_ref = -d2_ref.reshape(N, G) * (nUMI_g * nUMI_g)[:, None]
    d1_tri, d2_tri = _triton_kernels.irwls_shared_d1_d2_triton(
        Y_int, pred_raw, nUMI_g, thresh_g, poly_gpu, out_dtype=torch.float64
    )
    assert torch.isfinite(d1_tri).all() and torch.isfinite(d2_tri).all()
    assert torch.allclose(d1_tri, d1_ref, atol=1e-11, rtol=1e-11)
    assert torch.allclose(d2_tri, d2_ref, atol=1e-11, rtol=1e-11)

    tri_idx = torch.triu_indices(K, K, device="cuda")
    H_tri = d2_ref @ (P_g[:, tri_idx[0]] * P_g[:, tri_idx[1]])
    H_ref = torch.empty((N, K, K), dtype=torch.float64, device="cuda")
    H_ref[:, tri_idx[0], tri_idx[1]] = H_tri
    H_ref[:, tri_idx[1], tri_idx[0]] = H_tri
    assert torch.equal(_triton_kernels.unpack_sym_hess_triton(H_tri, K), H_ref)

    A = rng.normal(size=(N, K, K))
    D = torch.tensor(A @ A.transpose(0, 2, 1) + 0.1 * np.eye(K), dtype=torch.float64, device="cuda")
    D = D / torch.linalg.matrix_norm(D, ord=2)[:, None, None] + 1e-7 * torch.eye(
        K, dtype=torch.float64, device="cuda"
    )
    d = torch.tensor(rng.normal(size=(N, K)), dtype=torch.float64, device="cuda")
    lb = torch.tensor(-rng.uniform(0.0, 0.2, size=(N, K)), dtype=torch.float64, device="cuda")
    x_ref = _irwls._solve_box_qp_batch_impl(D.clone(), d, lb, n_sweeps=50)
    x_tri = _triton_kernels.solve_box_qp_triton(D.clone(), d, lb, n_sweeps=50)
    assert torch.allclose(x_tri, x_ref, atol=1e-10, rtol=1e-10)

    P_KT = P_g.T.contiguous()
    pix0 = torch.tensor([0], dtype=torch.long, device="cuda")
    t0 = torch.tensor([0], dtype=torch.long, device="cuda")
    t1 = torch.tensor([1], dtype=torch.long, device="cuda")

    monkeypatch.setenv("RCTD_TRITON", "0")
    S_pair = nUMI_g[pix0][:, None, None] * torch.stack([P_g[:, t0].T, P_g[:, t1].T], dim=-1)
    w_pair_ref, conv_pair_ref = _irwls.solve_irwls_batch(
        S_pair, Y_g[pix0], nUMI_g[pix0], Q_g, SQ_g, X_g, max_iter=25, constrain=False
    )
    sc_pair_ref = _likelihood.calc_log_likelihood_batch(
        Y_g[pix0],
        torch.clamp((S_pair * w_pair_ref[:, None, :]).sum(-1), min=1e-4),
        Q_g,
        SQ_g,
        X_g,
    )
    S_sg = nUMI_g[pix0][:, None, None] * P_g[:, t0].T.unsqueeze(-1)
    w_sg_ref, _ = _irwls.solve_irwls_batch(
        S_sg, Y_g[pix0], nUMI_g[pix0], Q_g, SQ_g, X_g, max_iter=25, constrain=False
    )
    sc_sg_ref = _likelihood.calc_log_likelihood_batch(
        Y_g[pix0],
        torch.clamp((S_sg * w_sg_ref[:, None, :]).sum(-1), min=1e-4),
        Q_g,
        SQ_g,
        X_g,
    )
    sc_sub_ref = _multi._run_batched_scoring(
        [(0, [0, 1, 2])], nUMI_g, Y_g, P_g, Q_g, SQ_g, X_g, batch_size=8, n_iter=25
    )

    monkeypatch.setenv("RCTD_TRITON", "1")
    w_pair_tri, sc_pair_tri, conv_pair_tri = _triton_kernels.fit_and_score_pairs_triton(
        pix0, t0, t1, Y_int, nUMI_g, P_KT, poly_gpu, max_iter=25, compute_score=True
    )
    w_sg_tri, sc_sg_tri = _triton_kernels.fit_and_score_singles_triton(
        pix0, t0, Y_int, nUMI_g, P_KT, poly_gpu, max_iter=25, write_weights=True, compute_score=True
    )
    sc_sub_tri = _multi._run_batched_scoring(
        [(0, [0, 1, 2])], nUMI_g, Y_g, P_g, Q_g, SQ_g, X_g, batch_size=8, n_iter=25
    )

    assert torch.allclose(w_pair_tri, w_pair_ref, atol=1e-11, rtol=1e-11)
    assert torch.allclose(sc_pair_tri, sc_pair_ref, atol=1e-10, rtol=1e-10)
    assert torch.equal(conv_pair_tri, conv_pair_ref)
    assert torch.allclose(w_sg_tri, w_sg_ref[:, 0], atol=1e-11, rtol=1e-11)
    assert torch.allclose(sc_sg_tri, sc_sg_ref, atol=1e-10, rtol=1e-10)
    assert np.allclose(sc_sub_tri, sc_sub_ref, atol=1e-10, rtol=1e-10)


@CUDA_TRITON
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_triton_solve_box_qp_early_exit_kkt(dtype, monkeypatch):
    """Verify early-exit Triton box-QP + _finish_box_qp matches PyTorch + _finish_box_qp at K=78."""
    rng = np.random.default_rng(314)
    N, K = 32, 78
    A = rng.normal(size=(N, K, K))
    D = torch.tensor(A @ A.transpose(0, 2, 1) + 0.1 * np.eye(K), dtype=dtype, device="cuda")
    D = D / torch.linalg.matrix_norm(D, ord=2)[:, None, None] + 1e-7 * torch.eye(
        K, dtype=dtype, device="cuda"
    )
    d = torch.tensor(rng.normal(size=(N, K)), dtype=dtype, device="cuda")
    lb = torch.tensor(-rng.uniform(0.0, 0.2, size=(N, K)), dtype=dtype, device="cuda")

    monkeypatch.setenv("RCTD_TRITON", "0")
    x_ref = _irwls._solve_box_qp_batch(D.clone(), d, lb, n_sweeps=50)
    monkeypatch.setenv("RCTD_TRITON", "1")
    x_tri = _irwls._solve_box_qp_batch(D.clone(), d, lb, n_sweeps=50)
    atol = 1e-10 if dtype == torch.float64 else 1e-4
    assert torch.allclose(x_tri, x_ref, atol=atol, rtol=atol)
