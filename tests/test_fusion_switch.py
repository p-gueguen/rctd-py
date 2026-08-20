"""run_rctd_multimodal fusion switch: count vs protein_wls on the same inputs.

Two cell types (A, B) share an IDENTICAL RNA profile, so RNA alone cannot split
them; one protein marker separates them, at a large intensity scale (~1e5) that a
count likelihood mishandles. Through the SAME setup/alignment, only the fusion
solver differs:
  * fusion="count"       treats protein as a 2nd Poisson-count modality -> degenerates
                         (collapses to a near-constant weight vector) on intensity data;
  * fusion="protein_wls" treats protein as a standardized Gaussian/WLS block -> resolves
                         the split and beats RNA-only on the degenerate pair.
"""

import anndata
import numpy as np
import pytest
import scipy.sparse as sp

from rctd import Modality, RCTDConfig, run_rctd_multimodal

TYPES = ["A", "B", "C", "D"]
K, G, M = 4, 24, 4


def _build():
    rng = np.random.default_rng(0)
    base = rng.gamma(2.0, 1.0, size=(K, G))
    base[1] = base[0].copy()  # B == A in RNA (degenerate pair)
    base = base / base.sum(1, keepdims=True)
    prot_mean = np.full((K, M), 2.0e4)
    prot_mean[0, 0] = 1.5e5  # A high on marker0
    prot_mean[1, 0] = 1.0e4  # B low  on marker0
    prot_mean[2, 1] = 1.2e5
    prot_mean[3, 2] = 1.2e5

    def ad(X, names, types=None, obsn=None):
        a = anndata.AnnData(sp.csr_matrix(X.astype(np.float32)))
        a.var_names = list(names)
        if types is not None:
            a.obs["cell_type"] = list(types)
        if obsn is not None:
            a.obs_names = list(obsn)
        return a

    ncell = 60
    rr, pp, labels = [], [], []
    for k, t in enumerate(TYPES):
        depth = rng.integers(400, 900, ncell)
        rr.append(np.vstack([rng.poisson(base[k] * d) for d in depth]))
        pp.append(prot_mean[k][None, :] * rng.lognormal(0.0, 0.25, size=(ncell, M)))
        labels += [t] * ncell
    ref_rna = ad(np.vstack(rr), [f"g{i}" for i in range(G)], labels)
    ref_prot = ad(np.vstack(pp), [f"p{i}" for i in range(M)], labels)

    n_spots, cps = 120, 2
    rs = np.random.default_rng(1)
    Wt = np.zeros((n_spots, K))
    sr = np.zeros((n_spots, G))
    spp = np.zeros((n_spots, M))
    for s in range(n_spots):
        ks = rs.integers(0, K, cps)
        for k in ks:
            Wt[s, k] += 1.0 / cps
        for k, d in zip(ks, rs.integers(400, 900, cps)):
            sr[s] += rng.poisson(base[k] * d)
        for k in ks:
            spp[s] += prot_mean[k] * rng.lognormal(0.0, 0.25, size=M)
    bc = [f"spot{i}" for i in range(n_spots)]
    sp_rna = ad(sr, [f"g{i}" for i in range(G)], obsn=bc)
    sp_prot = ad(spp, [f"p{i}" for i in range(M)], obsn=bc)
    return ref_rna, ref_prot, sp_rna, sp_prot, Wt


def _pair_mae(res, Wt):
    perm = [res.cell_type_names.index(t) for t in TYPES]
    sidx = np.array([int(b[4:]) for b in res.barcodes])
    Wpred = res.weights[:, perm]
    return np.abs(Wpred[:, :2] - Wt[sidx][:, :2]).mean()


@pytest.mark.protein
def test_protein_wls_resolves_split_where_count_degenerates():
    ref_rna, ref_prot, sp_rna, sp_prot, Wt = _build()
    cfg = RCTDConfig(
        device="cpu",
        dtype="float64",
        compile=False,
        UMI_min=5,
        UMI_max=10**12,
        counts_MIN=0,
        UMI_min_sigma=50,
        gene_cutoff=0.0,
        fc_cutoff=0.0,
        gene_cutoff_reg=0.0,
        fc_cutoff_reg=0.0,
    )
    RNA = Modality("RNA", sp_rna, ref_rna)
    PROT = Modality("protein", sp_prot, ref_prot)

    rna = run_rctd_multimodal([RNA], config=cfg, fusion="count")
    wls = run_rctd_multimodal([RNA, PROT], config=cfg, fusion="protein_wls", protein_lam="auto")

    mae_rna = _pair_mae(rna, Wt)
    mae_wls = _pair_mae(wls, Wt)
    n = wls.weights.shape[0]

    # protein_wls resolves the RNA-degenerate A/B split; RNA-only cannot.
    assert mae_wls < mae_rna, f"protein_wls {mae_wls:.3f} !< rna-only {mae_rna:.3f}"

    # ... and its output varies per spot rather than collapsing to one vector.
    distinct_wls = np.unique(np.round(wls.weights, 6), axis=0).shape[0]
    assert distinct_wls > n * 0.5, f"protein_wls degenerate: {distinct_wls}/{n} distinct rows"

    # Count fusion on this same continuous panel used to be reachable and produced
    # exactly that collapse (<= max(2, n/20) distinct rows). It is now refused up
    # front instead of returning a plausible-looking constant answer.
    with pytest.raises(ValueError, match="requires integer counts"):
        run_rctd_multimodal([RNA, PROT], config=cfg, fusion="count")


@pytest.mark.protein
def test_default_fusion_is_protein_wls():
    """The default is protein_wls: correct for continuous intensity AND for protein
    counts. Count fusion stayed the default long enough to ship a silent collapse
    to anyone who omitted the argument, which is why omitting it is now the safe
    path and 'count' is opt-in."""
    ref_rna, ref_prot, sp_rna, sp_prot, _ = _build()
    cfg = RCTDConfig(
        device="cpu",
        dtype="float64",
        compile=False,
        UMI_min=5,
        UMI_max=10**12,
        counts_MIN=0,
        UMI_min_sigma=50,
        gene_cutoff=0.0,
        fc_cutoff=0.0,
        gene_cutoff_reg=0.0,
        fc_cutoff_reg=0.0,
    )
    mods = [Modality("RNA", sp_rna, ref_rna), Modality("protein", sp_prot, ref_prot)]
    default = run_rctd_multimodal(mods, config=cfg)
    explicit = run_rctd_multimodal(mods, config=cfg, fusion="protein_wls")
    np.testing.assert_array_equal(default.weights, explicit.weights)

    with pytest.raises(ValueError, match="unknown fusion"):
        run_rctd_multimodal(mods, config=cfg, fusion="nonsense")
