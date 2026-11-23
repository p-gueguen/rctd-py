from itertools import combinations

import numpy as np

from rctd._doublet import _candidate_pairs, _classify_pixels, _select_candidates, run_doublet_mode
from rctd._likelihood import compute_spline_coefficients, load_cached_q_matrices
from rctd._types import RCTDConfig


def test_doublet_mode_runs(synthetic_data):
    """Test that doublet mode runs without errors and produces valid outputs."""
    cache = load_cached_q_matrices()
    x_vals = cache.pop("X_vals")
    q_mat = cache["Q_100"]
    sq_mat = compute_spline_coefficients(q_mat, x_vals)

    profiles = synthetic_data["profiles"]
    spatial_adata = synthetic_data["spatial"]
    spatial_counts = spatial_adata.X
    spatial_numi = np.array(spatial_counts.sum(axis=1)).flatten()
    cell_type_names = synthetic_data["cell_type_names"]

    config = RCTDConfig()

    res = run_doublet_mode(
        spatial_counts=spatial_counts,
        spatial_numi=spatial_numi,
        norm_profiles=profiles,
        cell_type_names=cell_type_names,
        q_mat=q_mat,
        sq_mat=sq_mat,
        x_vals=x_vals,
        config=config,
        batch_size=10,
    )

    N, K = spatial_counts.shape[0], profiles.shape[1]

    assert res.weights.shape == (N, K)
    assert res.weights_doublet.shape == (N, 2)
    assert res.spot_class.shape == (N,)
    assert res.first_type.shape == (N,)
    assert res.second_type.shape == (N,)

    # Values check
    assert np.all((res.spot_class >= 0) & (res.spot_class <= 3))
    assert np.all((res.first_type >= 0) & (res.first_type < K))
    assert np.all((res.second_type >= 0) & (res.second_type < K))
    assert np.all(res.weights_doublet >= 0.0)
    np.testing.assert_allclose(res.weights_doublet.sum(axis=1), 1.0, atol=1e-5)
    assert len(res.cell_type_names) == K


def test_select_candidates_follows_r_order():
    W = np.zeros((5, 5))
    W[1, 0] = W[2, 3] = 0.5  # one candidate: type 1 (or 2) is added after it
    W[3, [1, 4]] = 0.3
    W[4, :] = 0.2
    candidates, n = _select_candidates(W)
    lists = [candidates[i, : n[i]].tolist() for i in range(5)]
    assert lists == [[0, 1, 2], [0, 1], [3, 0], [1, 4], [0, 1, 2, 3, 4]]

    triples, singles = _candidate_pairs(candidates, n)
    expected = [(i, a, b) for i, c in enumerate(lists) for a, b in combinations(c, 2)]
    assert triples.tolist() == [list(t) for t in expected]
    assert singles.tolist() == [[i, t] for i, c in enumerate(lists) for t in c]


def _classify_one_pixel(cands, pair_score, sing, cls, conf, dbl):
    """R's process_bead_doublet for one pixel, as the earlier per-pixel loop computed it."""
    min_p, best = np.inf, (cands[0], cands[1])
    for t1, t2 in combinations(cands, 2):
        if pair_score[(t1, t2)] < min_p:
            min_p, best = pair_score[(t1, t2)], (t1, t2)

    def check(my):
        all_pairs = all_class = True
        others = [my]
        for t1, t2 in combinations(cands, 2):
            if pair_score[(t1, t2)] < min_p + conf:
                all_pairs &= my in (t1, t2)
                all_class &= cls[my] in (cls[t1], cls[t2])
                others += [t for t in (t1, t2) if cls[t] == cls[my] and t not in others]
        s = sing[my]
        if all_class and not all_pairs and len(others) > 1:
            s = min([s] + [sing[t] for t in others[1:]])
        return all_pairs, all_class, s

    (t1, t2), (a1, c1, s1), (a2, c2, s2) = best, check(best[0]), check(best[1])
    if not c1 and not c2:
        out = [0, t1, t2, False, False, min_p + 2 * dbl]
    elif c1 and not c2:
        out = [3, t1, t2, not a1, False, s1]
    elif c2 and not c1:
        out = [3, t2, t1, not a2, False, s2]
    elif s2 < s1:
        out = [2, t2, t1, not a2, not a1, s2]
    else:
        out = [2, t1, t2, not a1, not a2, s1]
    if out[5] - min_p < dbl:
        out[0] = 1
    return out[:5] + [min_p, out[5]]


def test_classify_pixels_matches_per_pixel_loop():
    rng = np.random.default_rng(0)
    N, K = 400, 7
    W = rng.dirichlet(np.full(K, 0.3), size=N)
    candidates, n = _select_candidates(W)
    triples, singles = _candidate_pairs(candidates, n)
    # Integer scores make ties, between pairs and between singlets, common
    pair_scores = rng.integers(0, 12, len(triples)).astype(np.float32)
    singlet_scores = np.full((N, K), 1e18, dtype=np.float32)
    singlet_scores[singles[:, 0], singles[:, 1]] = rng.integers(0, 30, len(singles))
    for type_class in (np.arange(K), np.array([0, 0, 1, 1, 1, 2, 2])):
        for conf, dbl in ((5.0, 20.0), (2.0, 6.0), (8.0, 3.0)):
            config = RCTDConfig(CONFIDENCE_THRESHOLD=conf, DOUBLET_THRESHOLD=dbl)
            got = _classify_pixels(
                triples, pair_scores, singlet_scores, candidates, n, type_class, config
            )
            for i in range(N):
                rows = triples[:, 0] == i
                score = dict(zip(map(tuple, triples[rows, 1:].tolist()), pair_scores[rows]))
                cands = candidates[i, : n[i]].tolist()
                expected = _classify_one_pixel(
                    cands, score, singlet_scores[i], type_class, conf, dbl
                )
                assert [g[i] for g in got] == expected, (i, conf, dbl)
