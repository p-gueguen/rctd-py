#!/usr/bin/env python3
"""Cross-validation: Compare rctd-py against R spacexr reference values.

Run: python scripts/compare_r_python.py
Prerequisites: Run scripts/generate_r_reference.R first.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

# Force CPU for reproducibility
jax.config.update("jax_platform_name", "cpu")

REF_DIR = Path("scripts/r_reference")

def load(name):
    return np.load(REF_DIR / f"{name}.npy", allow_pickle=True)


def check(name, py_val, r_val, atol=1e-6, rtol=1e-5):
    """Compare Python and R values, reporting pass/fail."""
    py_arr = np.asarray(py_val, dtype=np.float64)
    r_arr = np.asarray(r_val, dtype=np.float64)
    if py_arr.shape != r_arr.shape:
        print(f"  FAIL {name}: shape mismatch Python={py_arr.shape} R={r_arr.shape}")
        return False
    diff = np.abs(py_arr - r_arr)
    max_diff = np.max(diff)
    rel_diff = diff / (np.abs(r_arr) + 1e-10)
    max_rel = np.max(rel_diff)
    passed = np.allclose(py_arr, r_arr, atol=atol, rtol=rtol)
    status = "PASS" if passed else "FAIL"
    print(f"  {status} {name}: max_abs={max_diff:.2e}, max_rel={max_rel:.2e}")
    if not passed:
        worst_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    worst at {worst_idx}: Python={py_arr[worst_idx]:.10f} R={r_arr[worst_idx]:.10f}")
    return passed


def main():
    results = []

    # ============================================================
    # TEST 1: X_vals grid
    # ============================================================
    print("\n=== TEST 1: X_vals grid ===")
    from rctd._likelihood import build_x_vals
    x_vals_py = build_x_vals()
    x_vals_r = load("x_vals")
    results.append(check("x_vals", x_vals_py, x_vals_r, atol=1e-10))

    # ============================================================
    # TEST 2: Heavy-tailed PDF
    # ============================================================
    print("\n=== TEST 2: ht_pdf_norm ===")
    from rctd._likelihood import _ht_pdf_norm
    test_x = load("ht_pdf_norm_x")
    ht_r = load("ht_pdf_norm_y")
    ht_py = _ht_pdf_norm(test_x)
    results.append(check("ht_pdf_norm", ht_py, ht_r, atol=1e-12))

    # ============================================================
    # TEST 3: Spline coefficients (solve_sq)
    # ============================================================
    print("\n=== TEST 3: Spline coefficients ===")
    from rctd._likelihood import compute_spline_coefficients
    Q_mat_r = load("Q_mat_final")
    x_vals_final = load("X_vals_final")
    SQ_mat_r = load("SQ_mat_final")
    SQ_mat_py = compute_spline_coefficients(Q_mat_r, x_vals_final)
    results.append(check("spline_coefficients", SQ_mat_py, SQ_mat_r, atol=1e-6, rtol=1e-4))

    # ============================================================
    # TEST 4: calc_Q_all (cubic spline interpolation)
    # ============================================================
    print("\n=== TEST 4: calc_Q_all ===")
    from rctd._likelihood import calc_q_all

    test_Y = load("calc_q_all_Y").astype(int)
    test_lambda = load("calc_q_all_lambda")
    r_d0 = load("calc_q_all_d0")
    r_d1 = load("calc_q_all_d1")
    r_d2 = load("calc_q_all_d2")

    Q_j = jnp.array(Q_mat_r)
    SQ_j = jnp.array(SQ_mat_r)
    xv_j = jnp.array(x_vals_final)

    py_d0, py_d1, py_d2 = calc_q_all(
        jnp.array(test_Y), jnp.array(test_lambda), Q_j, SQ_j, xv_j
    )
    results.append(check("calc_q_all_d0", py_d0, r_d0, atol=1e-6))
    results.append(check("calc_q_all_d1", py_d1, r_d1, atol=1e-4))
    results.append(check("calc_q_all_d2", py_d2, r_d2, atol=1e-4))

    # Batch test
    print("\n  --- Batch test ---")
    batch_Y = load("calc_q_all_batch_Y").astype(int)
    batch_lambda = load("calc_q_all_batch_lambda")
    r_batch_d0 = load("calc_q_all_batch_d0")
    r_batch_d1 = load("calc_q_all_batch_d1")
    r_batch_d2 = load("calc_q_all_batch_d2")

    py_bd0, py_bd1, py_bd2 = calc_q_all(
        jnp.array(batch_Y), jnp.array(batch_lambda), Q_j, SQ_j, xv_j
    )
    results.append(check("batch_d0", py_bd0, r_batch_d0, atol=1e-5))
    results.append(check("batch_d1", py_bd1, r_batch_d1, atol=1e-3))
    results.append(check("batch_d2", py_bd2, r_batch_d2, atol=1e-3))

    # ============================================================
    # TEST 5: Log-likelihood
    # ============================================================
    print("\n=== TEST 5: calc_log_likelihood ===")
    from rctd._likelihood import calc_log_likelihood

    r_nll = float(load("log_likelihood_val"))
    py_nll = float(calc_log_likelihood(
        jnp.array(test_Y), jnp.array(test_lambda), Q_j, SQ_j, xv_j
    ))
    diff_nll = abs(py_nll - r_nll)
    passed = diff_nll < 1e-4
    status = "PASS" if passed else "FAIL"
    print(f"  {status} log_likelihood: Python={py_nll:.6f} R={r_nll:.6f} diff={diff_nll:.2e}")
    results.append(passed)

    r_batch_nll = float(load("batch_log_likelihood_val"))
    py_batch_nll = float(calc_log_likelihood(
        jnp.array(batch_Y), jnp.array(batch_lambda), Q_j, SQ_j, xv_j
    ))
    diff_bnll = abs(py_batch_nll - r_batch_nll)
    passed = diff_bnll < 1e-2
    status = "PASS" if passed else "FAIL"
    print(f"  {status} batch_log_likelihood: Python={py_batch_nll:.4f} R={r_batch_nll:.4f} diff={diff_bnll:.2e}")
    results.append(passed)

    # ============================================================
    # TEST 6: Multi-sigma calc_Q_all
    # ============================================================
    print("\n=== TEST 6: Multi-sigma calc_Q_all ===")
    from rctd._likelihood import load_cached_q_matrices

    cached = load_cached_q_matrices()
    x_vals_cached = cached["X_vals"]

    for sig in [10, 50, 100, 150, 200]:
        q_key = f"Q_{sig}"
        if q_key not in cached:
            print(f"  SKIP sigma={sig}: not in cached Q-matrices")
            continue
        Q_sig = cached[q_key]
        SQ_sig = compute_spline_coefficients(Q_sig, x_vals_cached)
        Q_sig_j = jnp.array(Q_sig)
        SQ_sig_j = jnp.array(SQ_sig)
        xv_j2 = jnp.array(x_vals_cached)

        r_d0 = load(f"sigma_{sig}_d0")
        py_d0, _, _ = calc_q_all(
            jnp.array(test_Y), jnp.array(test_lambda), Q_sig_j, SQ_sig_j, xv_j2
        )
        results.append(check(f"sigma_{sig}_d0", py_d0, r_d0, atol=1e-5))

    # ============================================================
    # TEST 7: Single-pixel IRWLS solver (UNCONSTRAINED - matches R pipeline)
    # ============================================================
    print("\n=== TEST 7: IRWLS single pixel (constrain=False, matching R) ===")
    from rctd._irwls import solve_irwls

    S = jnp.array(load("single_pixel_S"))
    Y = jnp.array(load("single_pixel_Y"))
    nUMI_px = float(load("single_pixel_nUMI"))
    r_weights = load("single_pixel_weights")  # unconstrained

    py_weights, py_converged = solve_irwls(
        S, Y, nUMI_px, Q_j, SQ_j, xv_j,
        max_iter=50, min_change=0.001, constrain=False  # Match R!
    )
    py_w_np = np.array(py_weights)
    print(f"  Python converged: {bool(py_converged)}")
    print(f"  R weights (top 5):      {np.sort(r_weights)[::-1][:5]}")
    print(f"  Python weights (top 5): {np.sort(py_w_np)[::-1][:5]}")
    print(f"  R weight sum:      {r_weights.sum():.4f}")
    print(f"  Python weight sum: {py_w_np.sum():.4f}")

    results.append(check("irwls_weights_unconstrained", py_w_np, r_weights, atol=0.05, rtol=0.2))

    # Correlation check
    nonzero_mask = (r_weights > 0.001) | (py_w_np > 0.001)
    if nonzero_mask.sum() > 1:
        corr = np.corrcoef(py_w_np[nonzero_mask], r_weights[nonzero_mask])[0, 1]
        corr_pass = corr > 0.9
        status = "PASS" if corr_pass else "FAIL"
        print(f"  {status} weight_correlation (nonzero): {corr:.4f}")
        results.append(corr_pass)
    else:
        print("  SKIP correlation: too few nonzero weights")

    # Dominant type agreement
    r_dom = np.argmax(r_weights)
    py_dom = np.argmax(py_w_np)
    dom_pass = r_dom == py_dom
    status = "PASS" if dom_pass else "FAIL"
    print(f"  {status} dominant_type: R={r_dom} Python={py_dom}")
    results.append(dom_pass)

    # ============================================================
    # TEST 8: Full mode weights on vignette data (UNCONSTRAINED)
    # ============================================================
    print("\n=== TEST 8: Full mode on vignette data (constrain=False) ===")
    from rctd._irwls import solve_irwls_batch

    r_full_weights = load("full_weights")
    r_full_pixels = load("full_pixel_names")
    print(f"  R full weights shape: {r_full_weights.shape}")
    r_sums = r_full_weights.sum(axis=1)
    print(f"  R weight sums: min={r_sums.min():.4f} max={r_sums.max():.4f} mean={r_sums.mean():.4f}")

    renorm_profiles = load("renorm_profiles")
    gene_list_reg = load("gene_list_reg")
    gene_list_bulk = load("gene_list_bulk")
    sp_counts = load("sp_counts")
    sp_gene_names = load("sp_gene_names")
    sp_nUMI = load("sp_nUMI")
    sp_pixel_names = load("sp_pixel_names")

    # Map gene_list_reg to indices in renorm_profiles (which is indexed by gene_list_bulk)
    bulk_gene_to_idx = {str(g): i for i, g in enumerate(gene_list_bulk)}
    reg_in_bulk_idx = np.array([bulk_gene_to_idx[str(g)] for g in gene_list_reg if str(g) in bulk_gene_to_idx])
    gene_list_reg_matched = np.array([str(g) for g in gene_list_reg if str(g) in bulk_gene_to_idx])

    profiles_reg = renorm_profiles[reg_in_bulk_idx, :]  # (G_reg_matched, K)

    # Map gene_list_reg to spatial counts
    sp_gene_to_idx = {str(g): i for i, g in enumerate(sp_gene_names)}
    gene_idx_matched = np.array([sp_gene_to_idx[str(g)] for g in gene_list_reg_matched])
    sp_counts_matched = sp_counts[:, gene_idx_matched]

    # Match R pixel list
    r_pixel_set = set(str(p) for p in r_full_pixels)
    py_pixel_indices = [i for i, p in enumerate(sp_pixel_names) if str(p) in r_pixel_set]

    if len(py_pixel_indices) != len(r_full_pixels):
        print(f"  WARNING: matched {len(py_pixel_indices)}/{len(r_full_pixels)} R pixels")

    counts_batch = jnp.array(sp_counts_matched[py_pixel_indices])
    numi_batch = jnp.array(sp_nUMI[py_pixel_indices])
    P = jnp.array(profiles_reg)

    S_batch = numi_batch[:, None, None] * P[None, :, :]

    sigma_int = int(round(float(load("sigma")) * 100))
    sigma_str = str(sigma_int)
    Q_sigma = jnp.array(cached[f"Q_{sigma_str}"])
    SQ_sigma = jnp.array(compute_spline_coefficients(np.array(cached[f"Q_{sigma_str}"]), x_vals_cached))

    py_full_weights, py_full_conv = solve_irwls_batch(
        S_batch, counts_batch, numi_batch,
        Q_sigma, SQ_sigma, jnp.array(x_vals_cached),
        max_iter=50, min_change=0.001, constrain=False  # Match R!
    )
    py_fw = np.array(py_full_weights)
    py_fc = np.array(py_full_conv)

    print(f"  Python full weights shape: {py_fw.shape}")
    print(f"  Python converged: {py_fc.sum()}/{len(py_fc)}")
    py_sums = py_fw.sum(axis=1)
    print(f"  Python weight sums: min={py_sums.min():.4f} max={py_sums.max():.4f} mean={py_sums.mean():.4f}")

    # Per-pixel dominant cell type agreement
    r_dominant = np.argmax(r_full_weights, axis=1)
    py_dominant = np.argmax(py_fw, axis=1)
    agreement = np.mean(r_dominant == py_dominant)
    agree_pass = agreement > 0.7
    status = "PASS" if agree_pass else "FAIL"
    print(f"  {status} dominant_type_agreement: {agreement:.1%} ({int(agreement * len(r_dominant))}/{len(r_dominant)})")
    results.append(agree_pass)

    # Weight correlation across all pixels
    corr_matrix = np.corrcoef(r_full_weights.flatten(), py_fw.flatten())[0, 1]
    corr_pass = corr_matrix > 0.9
    status = "PASS" if corr_pass else "FAIL"
    print(f"  {status} full_weight_correlation: {corr_matrix:.4f}")
    results.append(corr_pass)

    # Mean absolute error on weights
    mae = np.mean(np.abs(r_full_weights - py_fw))
    mae_pass = mae < 0.05
    status = "PASS" if mae_pass else "FAIL"
    print(f"  {status} full_weight_MAE: {mae:.4f}")
    results.append(mae_pass)

    # Normalized weight comparison (what downstream users see)
    r_norm = r_full_weights / np.maximum(r_full_weights.sum(axis=1, keepdims=True), 1e-10)
    py_norm = py_fw / np.maximum(py_fw.sum(axis=1, keepdims=True), 1e-10)
    norm_mae = np.mean(np.abs(r_norm - py_norm))
    norm_pass = norm_mae < 0.05
    status = "PASS" if norm_pass else "FAIL"
    print(f"  {status} normalized_weight_MAE: {norm_mae:.4f}")
    results.append(norm_pass)

    # ============================================================
    # TEST 9: Doublet mode on vignette data
    # ============================================================
    print("\n=== TEST 9: Doublet mode on vignette data ===")
    from rctd._doublet import run_doublet_mode
    from rctd._types import RCTDConfig, SPOT_CLASS_REJECT, SPOT_CLASS_SINGLET, SPOT_CLASS_DOUBLET_CERTAIN, SPOT_CLASS_DOUBLET_UNCERTAIN

    r_doublet_spot_class = load("doublet_spot_class")
    r_doublet_first_type = load("doublet_first_type")
    r_doublet_second_type = load("doublet_second_type")
    r_doublet_pair_weights = load("doublet_pair_weights")
    r_doublet_pixel_names = load("doublet_pixel_names")
    r_doublet_weights = load("doublet_weights")
    cell_type_names_r = [str(x) for x in load("cell_type_names")]

    print(f"  R doublet data: {len(r_doublet_pixel_names)} pixels, {len(cell_type_names_r)} types")
    print(f"  R spot_class distribution: ", end="")
    from collections import Counter
    print(dict(Counter(r_doublet_spot_class)))

    # Map R string spot_class to int
    spot_class_map = {
        "reject": SPOT_CLASS_REJECT,
        "singlet": SPOT_CLASS_SINGLET,
        "doublet_certain": SPOT_CLASS_DOUBLET_CERTAIN,
        "doublet_uncertain": SPOT_CLASS_DOUBLET_UNCERTAIN,
    }
    r_spot_class_int = np.array([spot_class_map[str(s)] for s in r_doublet_spot_class])

    # Match pixels between R and Python
    r_pix_to_idx = {str(p): i for i, p in enumerate(r_doublet_pixel_names)}
    sp_pix_to_idx = {str(p): i for i, p in enumerate(sp_pixel_names)}
    matched_r_idx = []
    matched_py_idx = []
    for pname, r_idx in r_pix_to_idx.items():
        if pname in sp_pix_to_idx:
            matched_r_idx.append(r_idx)
            matched_py_idx.append(sp_pix_to_idx[pname])
    matched_r_idx = np.array(matched_r_idx)
    matched_py_idx = np.array(matched_py_idx)
    print(f"  Matched {len(matched_r_idx)}/{len(r_doublet_pixel_names)} pixels")

    # Run doublet mode with the same data used for full mode
    config = RCTDConfig()
    doublet_res = run_doublet_mode(
        spatial_counts=np.array(sp_counts_matched[matched_py_idx]),
        spatial_numi=np.array(sp_nUMI[matched_py_idx]),
        norm_profiles=np.array(profiles_reg),
        cell_type_names=cell_type_names_r,
        q_mat=np.array(Q_sigma),
        sq_mat=np.array(SQ_sigma),
        x_vals=np.array(x_vals_cached),
        config=config,
    )

    # Compare spot_class
    r_sc = r_spot_class_int[matched_r_idx]
    py_sc = doublet_res.spot_class
    sc_agreement = np.mean(r_sc == py_sc)
    sc_pass = sc_agreement > 0.60  # Allow some deviation due to thresholding
    status = "PASS" if sc_pass else "FAIL"
    print(f"  {status} spot_class_agreement: {sc_agreement:.1%}")
    print(f"    R distribution:  {dict(Counter(r_sc))}")
    print(f"    Py distribution: {dict(Counter(py_sc))}")
    results.append(sc_pass)

    # Map R type names (strings) to indices
    ct_name_to_idx = {name: i for i, name in enumerate(cell_type_names_r)}
    r_first_type_idx = np.array([ct_name_to_idx.get(str(t), -1) for t in r_doublet_first_type[matched_r_idx]])
    r_second_type_idx = np.array([ct_name_to_idx.get(str(t), -1) for t in r_doublet_second_type[matched_r_idx]])

    # Compare first_type for doublets
    doublet_mask = (r_sc == SPOT_CLASS_DOUBLET_CERTAIN) | (r_sc == SPOT_CLASS_DOUBLET_UNCERTAIN)
    if doublet_mask.sum() > 0:
        # For doublets, check if the assigned pair matches (order-insensitive)
        r_pairs = set()
        py_pairs = set()
        pair_match = 0
        for i in np.where(doublet_mask)[0]:
            r_pair = frozenset([r_first_type_idx[i], r_second_type_idx[i]])
            py_pair = frozenset([doublet_res.first_type[i], doublet_res.second_type[i]])
            if r_pair == py_pair:
                pair_match += 1
        pair_agreement = pair_match / doublet_mask.sum()
        pair_pass = pair_agreement > 0.50
        status = "PASS" if pair_pass else "FAIL"
        print(f"  {status} doublet_pair_agreement: {pair_agreement:.1%} ({pair_match}/{doublet_mask.sum()})")
        results.append(pair_pass)

    # Compare pair weights for matched doublet pairs
    r_pw = r_doublet_pair_weights[matched_r_idx]
    py_pw = doublet_res.weights_doublet
    pw_corr = np.corrcoef(r_pw.flatten(), py_pw.flatten())[0, 1]
    pw_pass = pw_corr > 0.50
    status = "PASS" if pw_pass else "FAIL"
    print(f"  {status} pair_weight_correlation: {pw_corr:.4f}")
    results.append(pw_pass)

    # Compare singlet type agreement for singlets
    singlet_mask = r_sc == SPOT_CLASS_SINGLET
    if singlet_mask.sum() > 0:
        r_singlet_types = r_first_type_idx[singlet_mask]
        py_singlet_types = doublet_res.first_type[singlet_mask]
        singlet_type_agree = np.mean(r_singlet_types == py_singlet_types)
        singlet_pass = singlet_type_agree > 0.70
        status = "PASS" if singlet_pass else "FAIL"
        print(f"  {status} singlet_type_agreement: {singlet_type_agree:.1%}")
        results.append(singlet_pass)

    # ============================================================
    # TEST 10: Multi mode on vignette data
    # ============================================================
    print("\n=== TEST 10: Multi mode on vignette data ===")
    r_multi_weights = load("multi_all_weights")
    r_multi_n_types = load("multi_n_types")
    r_multi_pixel_names = load("multi_pixel_names")

    if r_multi_weights.size == 0 or r_multi_pixel_names is None:
        print("  SKIP: R multi reference data is empty (vignette may not produce multi results)")
        print("  Running multi mode to verify it executes without error...")
        from rctd._multi import run_multi_mode
        multi_res = run_multi_mode(
            spatial_counts=np.array(sp_counts_matched[matched_py_idx[:20]]),
            spatial_numi=np.array(sp_nUMI[matched_py_idx[:20]]),
            norm_profiles=np.array(profiles_reg),
            cell_type_names=cell_type_names_r,
            q_mat=np.array(Q_sigma),
            sq_mat=np.array(SQ_sigma),
            x_vals=np.array(x_vals_cached),
            config=config,
        )
        multi_ok = multi_res.n_types.min() >= 1 and multi_res.sub_weights.shape[1] == config.MAX_MULTI_TYPES
        status = "PASS" if multi_ok else "FAIL"
        print(f"  {status} multi_mode_executes: n_types range [{multi_res.n_types.min()}, {multi_res.n_types.max()}]")
        results.append(multi_ok)
    else:
        from rctd._multi import run_multi_mode
        # Match pixels
        r_mpix_to_idx = {str(p): i for i, p in enumerate(r_multi_pixel_names)}
        m_r_idx = []
        m_py_idx = []
        for pname, r_idx in r_mpix_to_idx.items():
            if pname in sp_pix_to_idx:
                m_r_idx.append(r_idx)
                m_py_idx.append(sp_pix_to_idx[pname])
        m_r_idx = np.array(m_r_idx)
        m_py_idx = np.array(m_py_idx)

        multi_res = run_multi_mode(
            spatial_counts=np.array(sp_counts_matched[m_py_idx]),
            spatial_numi=np.array(sp_nUMI[m_py_idx]),
            norm_profiles=np.array(profiles_reg),
            cell_type_names=cell_type_names_r,
            q_mat=np.array(Q_sigma),
            sq_mat=np.array(SQ_sigma),
            x_vals=np.array(x_vals_cached),
            config=config,
        )

        # Compare n_types
        r_nt = r_multi_n_types[m_r_idx].astype(int)
        py_nt = multi_res.n_types
        nt_agreement = np.mean(r_nt == py_nt)
        nt_pass = nt_agreement > 0.50
        status = "PASS" if nt_pass else "FAIL"
        print(f"  {status} n_types_agreement: {nt_agreement:.1%}")
        results.append(nt_pass)

        # Weight correlation
        r_mw = r_multi_weights[m_r_idx]
        py_mw = multi_res.weights
        mw_corr = np.corrcoef(r_mw.flatten(), py_mw.flatten())[0, 1]
        mw_pass = mw_corr > 0.80
        status = "PASS" if mw_pass else "FAIL"
        print(f"  {status} multi_weight_correlation: {mw_corr:.4f}")
        results.append(mw_pass)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    n_pass = sum(results)
    n_total = len(results)
    print(f"SUMMARY: {n_pass}/{n_total} tests passed")
    if n_pass == n_total:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {n_total - n_pass} test(s) failed")
    print("=" * 60)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
