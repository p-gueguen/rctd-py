"""Write RCTD results into the spatial AnnData (scanpy-style), shared by the API and the CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import anndata

    from rctd._reference import Reference
    from rctd._types import RCTDConfig


# Every key _write_results_to_adata can write, as "<key_added>_<suffix>"; only these are
# replaced on a re-run with the same key_added.
_OBS_KEYS = (
    "dominant_type",
    "converged",
    "spot_class",
    "first_type",
    "second_type",
    "first_class",
    "second_class",
    "first_class_name",
    "second_class_name",
    "n_types",
)
_OBSM_KEYS = ("weights", "weights_doublet", "sub_weights", "cell_type_indices")
_UNS_KEYS = ("mode", "config", "version", "cell_type_names")


def deconvolve(
    adata: anndata.AnnData,
    reference: Reference | anndata.AnnData,
    mode: Literal["full", "doublet", "multi"] = "doublet",
    config: RCTDConfig | None = None,
    batch_size: int | str = "auto",
    sigma_override: int | None = None,
    stream_gpu: bool = False,
    copy: bool = False,
    *,
    cell_type_key: str = "cell_type",
    layer: str | None = None,
    key_added: str = "rctd",
) -> anndata.AnnData | None:
    """Run RCTD and write the results into ``adata``, like ``rctd run`` does for its output file.

    Also available as ``rctd.tl.deconvolve``. Takes the same arguments as :func:`~rctd.run_rctd` and
    computes the same result. Pixels removed by the UMI or count filters get NaN weights and the
    label ``"filtered"``.

    Writes ``obsm["rctd_weights"]`` (pixels x cell types) and ``obs["rctd_dominant_type"]`` in every
    mode, plus ``obs["rctd_converged"]`` (full); ``obs["rctd_spot_class"]``,
    ``obs["rctd_first_type"]``, ``obs["rctd_second_type"]``, ``obs["rctd_first_class"]``,
    ``obs["rctd_second_class"]`` and ``obsm["rctd_weights_doublet"]`` (doublet);
    ``obs["rctd_n_types"]``, ``obsm["rctd_sub_weights"]`` and ``obsm["rctd_cell_type_indices"]``
    (multi), and ``obs["rctd_first_class_name"]`` / ``obs["rctd_second_class_name"]`` when
    ``config.class_df`` is set. ``uns`` gets ``rctd_mode``, ``rctd_config``, ``rctd_version`` and
    ``rctd_cell_type_names`` (the column order of ``obsm["rctd_weights"]``). With another
    ``key_added``, ``rctd`` in every key is replaced by it. These keys from an earlier run with the
    same ``key_added`` are replaced, so a second run in another mode leaves no stale columns; other
    keys, including your own ``rctd_*`` ones, are kept.

    ``batch_size="auto"`` sizes the batch from free GPU memory, while ``rctd run`` uses a fixed
    10000: pass ``batch_size=10000`` to reproduce a CLI output exactly.

    Args:
        adata: Spatial data, pixels x genes, raw counts in ``X`` (or in ``layers[layer]``). Must be
            in memory, not backed.
        reference: Single-cell reference: a :class:`~rctd.Reference`, or an AnnData with raw
            counts in ``X`` and the cell types in ``obs[cell_type_key]``, built with the
            :func:`rctd.pp.reference` defaults. For counts in a layer or other filters, pass
            ``rctd.pp.reference(..., layer=...)`` instead.
        mode: Deconvolution mode: "full", "doublet" or "multi".
        config: Configuration parameters; ``RCTDConfig()`` when None.
        batch_size: GPU batch size for pixel processing, or "auto" to size it from free GPU memory.
        sigma_override: Skip sigma estimation and use this value, as an integer 100 times sigma
            (63 means sigma 0.63).
        stream_gpu: Full mode only: copy one batch of counts at a time to the GPU.
        copy: Return a new AnnData with the results and leave ``adata`` unchanged, instead of
            writing into ``adata`` and returning None.
        cell_type_key: Column of the reference AnnData's ``obs`` with the cell types. Used only
            when ``reference`` is an AnnData.
        layer: Read the spatial counts from ``adata.layers[layer]`` instead of ``adata.X``.
        key_added: Prefix of every key written, so runs with different prefixes can sit side by
            side in one AnnData.

    Returns:
        None, or the annotated copy when ``copy=True``.
    """
    import anndata as ad

    from rctd import __version__
    from rctd._rctd import run_rctd
    from rctd._types import RCTDConfig
    from rctd.pp import reference as build_reference

    if adata.isbacked:
        raise ValueError("adata is backed (read with backed=...): load it with adata.to_memory()")
    if not isinstance(key_added, str) or not key_added:
        raise ValueError("key_added must be a non-empty string")
    if isinstance(reference, ad.AnnData):
        reference = build_reference(reference, cell_type_key)
    config = config or RCTDConfig()
    result = run_rctd(
        _with_counts_from(adata, layer),
        reference,
        mode=mode,
        config=config,
        batch_size=batch_size,
        sigma_override=sigma_override,
        stream_gpu=stream_gpu,
    )
    out = _write_results_to_adata(
        adata,
        result,
        mode,
        result.pixel_mask,
        config._asdict(),
        result.cell_type_names,
        __version__,
        inplace=not copy,
        key_added=key_added,
    )
    return out if copy else None


def _with_counts_from(adata, layer):
    """``adata`` itself, or an AnnData with ``layers[layer]`` as X and the same obs/var."""
    if layer is None:
        return adata
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} not in adata.layers: {list(adata.layers.keys())}")
    import anndata as ad

    return ad.AnnData(X=adata.layers[layer], obs=adata.obs, var=adata.var)


def _write_results_to_adata(
    spatial_adata,
    result,
    mode,
    pixel_mask,
    config_dict,
    cell_type_names,
    version,
    inplace=False,
    key_added="rctd",
):
    """Expand results to full AnnData shape and write slots (into a copy unless ``inplace``)."""
    # .h5ad categoricals cannot mix numeric cell-type labels with "filtered"
    cell_type_names = [str(c) for c in cell_type_names]
    import numpy as np
    import pandas as pd

    from rctd._types import SPOT_CLASS_NAMES

    # Build everything first, so a failure leaves the input as it was
    obs, obsm, uns = {}, {}, {}
    n_total = spatial_adata.n_obs
    n_types = len(cell_type_names)

    # Weights (all modes)
    full_weights = np.full((n_total, n_types), np.nan, dtype=np.float32)
    full_weights[pixel_mask] = result.weights
    obsm["weights"] = full_weights

    # Dominant type (all modes)
    dominant = np.full(n_total, "filtered", dtype=object)
    dominant_idx = result.weights.argmax(axis=1)
    dominant[pixel_mask] = [cell_type_names[i] for i in dominant_idx]
    obs["dominant_type"] = pd.Categorical(dominant)

    if mode == "full":
        converged = np.full(n_total, False)
        converged[pixel_mask] = result.converged
        obs["converged"] = converged

    elif mode == "doublet":
        # Spot class
        spot_class = np.full(n_total, "filtered", dtype=object)
        spot_class[pixel_mask] = [SPOT_CLASS_NAMES[i] for i in result.spot_class]
        obs["spot_class"] = pd.Categorical(spot_class)

        # First/second type
        first_type = np.full(n_total, "filtered", dtype=object)
        first_type[pixel_mask] = [cell_type_names[i] for i in result.first_type]
        obs["first_type"] = pd.Categorical(first_type)

        second_type = np.full(n_total, "filtered", dtype=object)
        second_type[pixel_mask] = [cell_type_names[i] for i in result.second_type]
        obs["second_type"] = pd.Categorical(second_type)

        # Class-level ambiguity flags (True = assignment trustworthy only at class level)
        first_class_flag = np.zeros(n_total, dtype=bool)
        first_class_flag[pixel_mask] = result.first_class
        obs["first_class"] = first_class_flag
        second_class_flag = np.zeros(n_total, dtype=bool)
        second_class_flag[pixel_mask] = result.second_class
        obs["second_class"] = second_class_flag

        # Class names (only when class_df was provided)
        if result.first_class_name is not None:
            first_cn = np.full(n_total, "filtered", dtype=object)
            first_cn[pixel_mask] = result.first_class_name
            obs["first_class_name"] = pd.Categorical(first_cn)
        if result.second_class_name is not None:
            second_cn = np.full(n_total, "filtered", dtype=object)
            second_cn[pixel_mask] = result.second_class_name
            obs["second_class_name"] = pd.Categorical(second_cn)

        # Doublet weights
        full_wt_doublet = np.full((n_total, 2), np.nan, dtype=np.float32)
        full_wt_doublet[pixel_mask] = result.weights_doublet
        obsm["weights_doublet"] = full_wt_doublet

    elif mode == "multi":
        n_types_per_pixel = np.zeros(n_total, dtype=np.int32)
        n_types_per_pixel[pixel_mask] = result.n_types
        obs["n_types"] = n_types_per_pixel

        max_multi = result.sub_weights.shape[1]
        full_sub_wt = np.full((n_total, max_multi), np.nan, dtype=np.float32)
        full_sub_wt[pixel_mask] = result.sub_weights
        obsm["sub_weights"] = full_sub_wt

        full_ct_idx = np.full((n_total, max_multi), -1, dtype=np.int32)
        full_ct_idx[pixel_mask] = result.cell_type_indices
        obsm["cell_type_indices"] = full_ct_idx

    # Metadata
    uns["mode"] = mode
    uns["config"] = config_dict
    uns["version"] = version
    uns["cell_type_names"] = cell_type_names

    adata = spatial_adata if inplace else spatial_adata.copy()
    # Drop this writer's keys from an earlier run, so one mode's keys never sit next to another's
    for slot, keys in ((adata.obs, _OBS_KEYS), (adata.obsm, _OBSM_KEYS), (adata.uns, _UNS_KEYS)):
        for key in keys:
            if f"{key_added}_{key}" in slot:
                del slot[f"{key_added}_{key}"]
    for key, value in obs.items():
        adata.obs[f"{key_added}_{key}"] = value
    for key, value in obsm.items():
        adata.obsm[f"{key_added}_{key}"] = value
    for key, value in uns.items():
        adata.uns[f"{key_added}_{key}"] = value
    return adata
