"""Profile individual components of the IRWLS inner loop on GPU.

Usage: uv run python benchmarks/profile_irwls.py --device cuda
"""

import argparse
import time

import numpy as np
import torch

from rctd._irwls import _psd_batch, _solve_box_qp_batch, solve_irwls_batch_shared
from rctd._likelihood import build_x_vals, calc_q_all, compute_q_matrix, compute_spline_coefficients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-pixels", type=int, default=10000)
    parser.add_argument("--n-genes", type=int, default=1000)
    parser.add_argument("--n-types", type=int, default=12)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N, G, K = args.n_pixels, args.n_genes, args.n_types
    print(f"Profiling: {N} pixels, {G} genes, {K} types, device={device}")

    rng = np.random.default_rng(42)
    profiles = rng.exponential(0.01, size=(G, K)).astype(np.float64)
    profiles = profiles / profiles.sum(axis=0, keepdims=True)
    nUMIs = rng.integers(200, 5000, size=N).astype(np.float64)
    counts = np.zeros((N, G), dtype=np.float64)
    for i in range(N):
        true_w = rng.dirichlet(np.ones(K))
        lam = (profiles @ true_w) * nUMIs[i]
        counts[i] = rng.poisson(np.clip(lam, 0, 1e6))

    x_vals = build_x_vals()
    q_mat = compute_q_matrix(1.0, x_vals)
    sq_mat = compute_spline_coefficients(q_mat, x_vals)

    P = torch.tensor(profiles, device=device)
    Y = torch.tensor(counts, device=device)
    nUMI = torch.tensor(nUMIs, device=device)
    Q = torch.tensor(q_mat, device=device)
    SQ = torch.tensor(sq_mat, device=device)
    X = torch.tensor(x_vals, device=device)

    # Warmup
    print("Warmup...")
    solve_irwls_batch_shared(P, Y[:100], nUMI[:100], Q, SQ, X, constrain=False)
    if device == "cuda":
        torch.cuda.synchronize()

    # Full solve timing
    print("\nFull solve:")
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    w, c = solve_irwls_batch_shared(P, Y, nUMI, Q, SQ, X, constrain=False)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  Total: {time.perf_counter() - t0:.4f}s")

    # Component profiling
    print("\nComponent profiling (single iteration, all pixels):")

    w_test = torch.ones(N, K, dtype=P.dtype, device=device) / K
    P_outer = (P[:, :, None] * P[:, None, :]).reshape(G, K * K)
    threshold = torch.clamp(nUMI * 1e-7, min=1e-4)
    eye_K = torch.eye(K, dtype=P.dtype, device=device)

    solution = torch.clamp(w_test, min=0.0)

    # 1. Prediction
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        prediction = torch.abs(nUMI.unsqueeze(1) * (solution @ P.T))
        prediction = torch.clamp(prediction, min=threshold.unsqueeze(1))
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  Prediction:   {(time.perf_counter() - t0)/10*1000:.2f}ms")

    # 2. calc_q_all
    Y_flat = Y.reshape(-1)
    pred_flat = prediction.reshape(-1)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _, d1_flat, d2_flat = calc_q_all(Y_flat, pred_flat, Q, SQ, X)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  calc_q_all:   {(time.perf_counter() - t0)/10*1000:.2f}ms")

    d1_vec = d1_flat.reshape(N, G)
    d2_vec = d2_flat.reshape(N, G)

    # 3. Gradient
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        grad = -((d1_vec * nUMI.unsqueeze(1)) @ P)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  Gradient:     {(time.perf_counter() - t0)/10*1000:.2f}ms")

    # 4. Hessian
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        d2_w = (-d2_vec) * (nUMI**2).unsqueeze(1)
        hess = (d2_w @ P_outer).reshape(N, K, K)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  Hessian:      {(time.perf_counter() - t0)/10*1000:.2f}ms")

    # 5. PSD projection
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        hess_psd, norm_factor = _psd_batch(hess)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  PSD (eigh):   {(time.perf_counter() - t0)/10*1000:.2f}ms")

    norm_factor = torch.clamp(norm_factor, min=1e-10)
    D = hess_psd / norm_factor[:, None, None] + 1e-7 * eye_K.unsqueeze(0)
    d = -grad / norm_factor[:, None]

    # 6. QP solver
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        delta_w = _solve_box_qp_batch(D, d, -solution)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  QP solver:    {(time.perf_counter() - t0)/10*1000:.2f}ms")


if __name__ == "__main__":
    main()
