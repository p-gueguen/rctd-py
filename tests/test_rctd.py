import numpy as np

from rctd._rctd import run_rctd
from rctd._types import FullResult, RCTDConfig


def test_rctd_class_and_run(synthetic_data):
    """Test that RCTD class and run_rctd function work end-to-end."""
    spatial_adata = synthetic_data["spatial"]
    reference_adata = synthetic_data["reference"]

    from rctd._reference import Reference

    reference = Reference(reference_adata, cell_min=10, min_UMI=10)

    config = RCTDConfig()

    res_full = run_rctd(
        spatial=spatial_adata,
        reference=reference,
        mode="full",
        config=config,
        batch_size=10,
    )

    assert isinstance(res_full, FullResult)
    assert res_full.weights.shape[0] == spatial_adata.shape[0]


def test_rctd_float32(synthetic_data):
    """Test that float32 mode runs and produces comparable results to float64."""
    spatial_adata = synthetic_data["spatial"]
    reference_adata = synthetic_data["reference"]

    from rctd._reference import Reference

    reference = Reference(reference_adata, cell_min=10, min_UMI=10)

    # Run with float64 (default)
    config_f64 = RCTDConfig()
    res_f64 = run_rctd(
        spatial=spatial_adata,
        reference=reference,
        mode="full",
        config=config_f64,
        batch_size=10,
    )

    # Run with float32
    config_f32 = RCTDConfig(dtype="float32")
    res_f32 = run_rctd(
        spatial=spatial_adata,
        reference=reference,
        mode="full",
        config=config_f32,
        batch_size=10,
    )

    assert res_f32.weights.shape == res_f64.weights.shape

    # Dominant type agreement should be high
    dom_f64 = np.argmax(res_f64.weights, axis=1)
    dom_f32 = np.argmax(res_f32.weights, axis=1)
    agreement = np.mean(dom_f64 == dom_f32)
    assert agreement > 0.90, f"Dominant type agreement too low: {agreement:.1%}"

    # Weight MAE should be small
    mae = np.mean(np.abs(res_f64.weights - res_f32.weights))
    assert mae < 0.05, f"Float32 vs float64 weight MAE too high: {mae:.4f}"
