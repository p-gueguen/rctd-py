"""RCTDConfig settings that live in module globals must not leak into later runs.

eigh_threshold and compile=False are applied by RCTD.__init__ to module-level flags in
_irwls / _likelihood. A later RCTD built with the defaults must restore the defaults;
otherwise trying `eigh_threshold=64` once in a notebook silently keeps every later run on
GPU eigh (8x slower Step 1 on L40S at K=40), and `compile=False` keeps it eager.
"""

import pytest

from rctd import RCTD, RCTDConfig, Reference, _irwls, _likelihood


@pytest.fixture(autouse=True)
def _restore_globals():
    saved = (_irwls._EIGH_THRESHOLD_OVERRIDE, _irwls._USE_COMPILE, _likelihood._CALC_Q_USE_COMPILE)
    yield
    _irwls._EIGH_THRESHOLD_OVERRIDE, _irwls._USE_COMPILE, _likelihood._CALC_Q_USE_COMPILE = saved


def _rctd(data, **kw):
    ref = Reference(data["reference"], cell_type_col="cell_type")
    return RCTD(data["spatial"], ref, RCTDConfig(device="cpu", **kw))


def test_eigh_threshold_does_not_leak(synthetic_data):
    _rctd(synthetic_data, eigh_threshold=64)
    assert _irwls._EIGH_THRESHOLD_OVERRIDE == 64
    _rctd(synthetic_data)
    assert _irwls._EIGH_THRESHOLD_OVERRIDE is None


def test_compile_false_does_not_leak(synthetic_data):
    _irwls._USE_COMPILE = _likelihood._CALC_Q_USE_COMPILE = None
    _rctd(synthetic_data, compile=False)
    assert _irwls._USE_COMPILE is False and _likelihood._CALC_Q_USE_COMPILE is False
    _rctd(synthetic_data)
    assert _irwls._USE_COMPILE is None and _likelihood._CALC_Q_USE_COMPILE is None


def test_compile_failure_fallback_stays_sticky(synthetic_data):
    """A fallback set because compile FAILED is not undone by a default config."""
    _irwls._USE_COMPILE = _likelihood._CALC_Q_USE_COMPILE = False  # as the auto-fallback leaves it
    _rctd(synthetic_data)
    assert _irwls._USE_COMPILE is False and _likelihood._CALC_Q_USE_COMPILE is False
