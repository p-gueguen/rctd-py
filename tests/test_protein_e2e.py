"""End-to-end run_rctd wiring for the protein modality (obsm read -> normalize ->
bootstrap profiles -> lambda -> joint solve).

The scientific "protein resolves an RNA-degenerate type" claim is proven at the
solver level in test_protein_recovery (where the true P_prot can be injected).
Here the profiles are BOOTSTRAPPED from RNA-confident singlets, so these tests
exercise the full pipeline on RNA-distinguishable types: the run must complete,
attach profiles, and refine without degrading the RNA-only result.
"""

import anndata
import numpy as np
import pytest

from rctd import RCTDConfig, Reference, run_rctd


def _pure_multimodal(profiles, cell_type_names, n_per_type=100, n_markers=4, seed=11):
    """High-purity spatial pixels (mostly one type) so the RNA-only bootstrap finds
    plenty of confident singlets to build protein profiles from."""
    rng = np.random.default_rng(seed)
    G, K = profiles.shape
    N = n_per_type * K
    true_w = np.zeros((N, K), dtype=np.float32)
    counts = np.zeros((N, G), dtype=np.float32)
    nUMI = rng.integers(2000, 5000, size=N)
    for k in range(K):
        for c in range(n_per_type):
            i = k * n_per_type + c
            w = np.full(K, 0.005, dtype=np.float32)
            w[k] = 1.0 - 0.005 * (K - 1)  # ~0.98 on the dominant type
            true_w[i] = w
            counts[i] = rng.poisson((profiles @ w) * nUMI[i])
    P_prot_true = rng.standard_normal((n_markers, K))
    protein = (true_w @ P_prot_true.T + rng.standard_normal((N, n_markers)) * 0.5).astype("f4")

    spatial = anndata.AnnData(X=counts, obs={"i": np.arange(N)})
    spatial.var_names = [f"Gene_{i}" for i in range(G)]
    spatial.obs_names = [f"Pix_{i}" for i in range(N)]
    spatial.obsm["protein"] = protein
    spatial.uns["protein_feature_names"] = [f"prot_{m}" for m in range(n_markers)]
    return spatial, true_w


def _attach_protein(spatial, true_weights, n_markers=4, seed=3):
    """Protein intensities that agree with the RNA cell-type structure."""
    rng = np.random.default_rng(seed)
    K = true_weights.shape[1]
    P_prot_true = rng.standard_normal((n_markers, K))
    protein = (
        true_weights @ P_prot_true.T + rng.standard_normal((true_weights.shape[0], n_markers)) * 0.5
    )
    sp = spatial.copy()
    sp.obsm["protein"] = protein.astype("f4")
    sp.uns["protein_feature_names"] = [f"prot_{m}" for m in range(n_markers)]
    return sp


@pytest.mark.protein
def test_run_rctd_doublet_with_protein_completes_and_attaches(synthetic_data):
    spatial = _attach_protein(synthetic_data["spatial"], synthetic_data["true_weights"])
    reference = Reference(synthetic_data["reference"], cell_min=10, min_UMI=10)

    res = run_rctd(
        spatial,
        reference,
        mode="doublet",
        config=RCTDConfig(compile=False, protein_weight=1.0),
        batch_size=10,
    )
    # Pipeline completed and produced a result of the right shape.
    assert res.weights.shape[0] == spatial.n_obs
    # Bootstrapped profiles were attached to the reference.
    assert reference.protein_profiles is not None
    assert reference.protein_profiles.shape == (4, len(reference.cell_type_names))
    assert reference.protein_feature_names == [f"prot_{m}" for m in range(4)]


@pytest.mark.protein
def test_protein_consistent_with_rna_does_not_degrade(synthetic_data):
    """Protein that agrees with RNA should keep dominant calls and not raise MAE."""
    spatial = _attach_protein(synthetic_data["spatial"], synthetic_data["true_weights"])
    true_w = synthetic_data["true_weights"]

    rna = run_rctd(
        spatial,
        Reference(synthetic_data["reference"], cell_min=10, min_UMI=10),
        mode="doublet",
        config=RCTDConfig(compile=False),
        batch_size=10,
    )
    joint = run_rctd(
        spatial,
        Reference(synthetic_data["reference"], cell_min=10, min_UMI=10),
        mode="doublet",
        config=RCTDConfig(compile=False, protein_weight=1.0),
        batch_size=10,
    )

    agree = (rna.first_type == joint.first_type).mean()
    assert agree >= 0.8, f"dominant-type agreement {agree:.2f} too low"

    mae_rna = np.abs(rna.weights - true_w).mean()
    mae_joint = np.abs(joint.weights - true_w).mean()
    assert mae_joint <= mae_rna + 0.02, f"joint MAE {mae_joint:.3f} degraded vs {mae_rna:.3f}"


@pytest.mark.protein
def test_auto_lambda_runs_and_is_positive(synthetic_data):
    # High-purity data so the RNA-only bootstrap finds enough confident singlets
    # (>=25 per type) to actually build protein profiles.
    spatial, _ = _pure_multimodal(
        synthetic_data["profiles"], synthetic_data["cell_type_names"], n_per_type=60
    )
    reference = Reference(synthetic_data["reference"], cell_min=10, min_UMI=10)
    from rctd._rctd import RCTD

    rctd = RCTD(
        spatial,
        reference,
        RCTDConfig(compile=False, protein_weight="auto", protein_singlet_purity=0.6),
    )
    rctd.fit_platform_effects()
    pk = rctd.prepare_protein()
    assert pk, "prepare_protein returned empty for an active protein config"
    assert np.isfinite(rctd.protein_lambda) and rctd.protein_lambda > 0
    # Bootstrap built profiles for the (now well-populated) types.
    assert (reference.protein_tau > 0).all()
