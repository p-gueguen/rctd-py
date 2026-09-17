"""WNN per-cell modality weights (Hao et al. 2021, Cell) for the protein block.

A modality earns weight at a cell when its own neighbours predict that cell better
than the other modality's neighbours do. The controls: each modality must win
where only it carries structure, and a shuffled protein panel must lose.
"""

import numpy as np
import pytest

from rctd import RCTD, RCTDConfig, Reference, wnn_modality_weights


def _two_regimes(n=600, seed=0):
    """Group A: RNA has two sub-clusters, protein is noise. Group B: the reverse.
    A shared offset dimension keeps both groups apart in both modalities."""
    rng = np.random.default_rng(seed)
    half = n // 2
    grp = np.r_[np.zeros(half, int), np.ones(n - half, int)]
    sub = rng.integers(0, 2, n)
    rna = rng.normal(0, 1, (n, 10))
    prot = rng.normal(0, 1, (n, 8))
    rna[:, 0] += 8 * grp
    prot[:, 0] += 8 * grp
    a, b = grp == 0, grp == 1
    rna[a, 1] += np.where(sub[a] == 1, 6.0, -6.0)
    prot[b, 1] += np.where(sub[b] == 1, 6.0, -6.0)
    return rna, prot, a, b


@pytest.mark.protein
def test_wnn_weights_follow_the_informative_modality():
    rna, prot, a, b = _two_regimes()
    w_rna, w_prot = wnn_modality_weights(rna, prot, k=20)
    np.testing.assert_allclose(w_rna + w_prot, 1.0)
    assert w_prot[b].mean() > 0.6, w_prot[b].mean()
    assert w_prot[a].mean() < 0.4, w_prot[a].mean()


@pytest.mark.protein
def test_wnn_weights_shuffled_protein_loses():
    """NEGATIVE CONTROL: protein rows permuted -> protein neighbourhoods say nothing."""
    rna, prot, _, _ = _two_regimes()
    shuffled = prot[np.random.default_rng(1).permutation(len(prot))]
    _, w_real = wnn_modality_weights(rna, prot, k=20)
    _, w_shuf = wnn_modality_weights(rna, shuffled, k=20)
    assert w_shuf.mean() < 0.5, w_shuf.mean()
    assert w_shuf.mean() < w_real.mean() - 0.05, (w_shuf.mean(), w_real.mean())


@pytest.mark.protein
def test_wnn_weights_respect_sections():
    """Two sections with identical embeddings: with `sample`, neighbours stay inside
    a section, so each section gets exactly the weights it gets on its own."""
    rna, prot, _, _ = _two_regimes(n=300)
    alone = wnn_modality_weights(rna, prot, k=15)[1]
    both = wnn_modality_weights(
        np.vstack([rna, rna]),
        np.vstack([prot, prot]),
        k=15,
        sample=np.r_[np.zeros(300), np.ones(300)],
    )[1]
    np.testing.assert_allclose(both[:300], alone)
    np.testing.assert_allclose(both[300:], alone)


def _synth_with_protein(synthetic_data, n_markers=4, seed=3):
    rng = np.random.default_rng(seed)
    sp = synthetic_data["spatial"].copy()
    tw = synthetic_data["true_weights"]
    P = rng.standard_normal((n_markers, tw.shape[1]))
    sp.obsm["protein"] = (tw @ P.T + rng.standard_normal((tw.shape[0], n_markers)) * 0.5).astype(
        "f4"
    )
    sp.uns["protein_feature_names"] = [f"m{i}" for i in range(n_markers)]
    return sp


@pytest.mark.protein
def test_prepare_protein_applies_wnn_weights(synthetic_data):
    sp = _synth_with_protein(synthetic_data)
    ref = Reference(synthetic_data["reference"], cell_min=10, min_UMI=10)

    def prep(**kw):
        r = RCTD(sp, ref, RCTDConfig(compile=False, protein_weight=1.0, **kw))
        r.fit_platform_effects()
        return r, r.prepare_protein()

    r0, k0 = prep()
    assert getattr(r0, "protein_wnn_weight", None) is None
    assert k0["protein_mask"].dtype == np.bool_

    r1, k1 = prep(protein_modality_weight="wnn", protein_wnn_k=10, protein_wnn_npcs=5)
    w = r1.protein_wnn_weight
    assert w.shape == (k1["protein_intensity"].shape[0],)
    assert np.all((w >= 0) & (w <= 1))
    np.testing.assert_allclose(k1["protein_mask"], 2.0 * w, rtol=1e-6)


@pytest.mark.protein
def test_unknown_modality_weight_raises(synthetic_data):
    sp = _synth_with_protein(synthetic_data)
    ref = Reference(synthetic_data["reference"], cell_min=10, min_UMI=10)
    r = RCTD(
        sp, ref, RCTDConfig(compile=False, protein_weight=1.0, protein_modality_weight="bogus")
    )
    r.fit_platform_effects()
    with pytest.raises(ValueError, match="protein_modality_weight"):
        r.prepare_protein()
