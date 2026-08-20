"""Regression test for issue #27: inductor codegen WARNING spam.

torch>=2.10 on some platforms (Apple Silicon CPU) logs a large recoverable
"Error in codegen" dump then falls back to eager silently. The filter must
drop that record and flag it so the solver switches to a quiet fallback.
"""

import logging

from rctd import _likelihood


def _record(msg):
    log = logging.getLogger("torch._inductor.scheduler")
    return log.makeRecord(log.name, logging.WARNING, __file__, 0, msg, None, None)


def test_codegen_error_dropped_and_flagged():
    _likelihood._pop_inductor_codegen_failed()  # clear
    log = logging.getLogger("torch._inductor.scheduler")
    rec = _record("Error in codegen for ComputedBuffer(name='buf0', ...)")

    kept = all(f.filter(rec) for f in log.filters)
    assert kept is False  # noise suppressed
    assert _likelihood._pop_inductor_codegen_failed() is True  # detected
    assert _likelihood._pop_inductor_codegen_failed() is False  # and reset


def test_unrelated_records_pass_through():
    log = logging.getLogger("torch._inductor.scheduler")
    rec = _record("compiling region foo")
    assert all(f.filter(rec) for f in log.filters) is True
    assert _likelihood._pop_inductor_codegen_failed() is False
