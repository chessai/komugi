"""Environment baseline: Cython toolchain, config dataclasses, shapecheck seam.

mypy is deliberately not invoked from here -- CI runs it as its own step
(see ci/check.sh), so a type error fails the type-checking job rather than
hiding inside a test report.
"""

from __future__ import annotations

import importlib.util
import multiprocessing

import numpy as np
import pytest

from komugi import _probe
from komugi.config import SelfPlayConfig, TrainConfig

_CHILD_TIMEOUT_S = 120.0

# The shape checks themselves only exist if the seam's dependencies do; the
# env-var plumbing is asserted either way (see _child).
_SEAM_AVAILABLE: bool = (
    importlib.util.find_spec("jaxtyping") is not None
    and importlib.util.find_spec("beartype") is not None
)


def _child(shape: tuple[int, ...], expect_typecheck: bool) -> None:
    """Spawned-child entry point: check the flag survived, then hit the seam."""
    from komugi import shapecheck

    if shapecheck.TYPECHECK is not expect_typecheck:
        raise SystemExit(
            f"KOMUGI_TYPECHECK plumbing broken: TYPECHECK={shapecheck.TYPECHECK}, "
            f"expected {expect_typecheck}"
        )
    shapecheck.validate_policy_batch(np.zeros(shape, dtype=np.float32))


def _run_child(shape: tuple[int, ...], expect_typecheck: bool) -> int:
    """Run _child in a freshly spawned interpreter, returning its exit code.

    "spawn" (not "fork") on purpose: the child re-imports komugi.shapecheck
    from scratch, which is the only way to prove KOMUGI_TYPECHECK is picked up
    at module bootstrap rather than inherited as already-decorated state.
    """
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_child, args=(shape, expect_typecheck))
    proc.start()
    proc.join(_CHILD_TIMEOUT_S)
    if proc.exitcode is None:
        proc.kill()
        proc.join()
        pytest.fail("spawned child did not exit within the timeout")
    return proc.exitcode


def test_probe_extension_is_compiled_and_callable() -> None:
    """The Cython toolchain produced an importable, callable extension."""
    assert _probe.probe_add(2, 3) == 5


def test_selfplay_config_rejects_unknown_kwargs() -> None:
    """A mistyped key is a constructor error, not a silently ignored setting."""
    cfg = SelfPlayConfig(workers=2, process_batch_size=64)
    assert cfg.gumbel_m_full == 32
    assert cfg.prob_fast == 0.75
    # KOM-036's two schedule knobs default to 1.0 -- OFF -- which is the whole
    # of the claim that landing the ticket leaves self-play bit-identical:
    # `_scale_for` returns exactly 1.0 and GumbelConfig receives the value it
    # already carried (acceptance #18).
    assert cfg.fast_gumbel_scale == 1.0
    assert cfg.draft_gumbel_scale == 1.0
    with pytest.raises(TypeError):
        SelfPlayConfig(workers=2, process_batch_size=64, gumbel_m_fill=16)  # type: ignore[call-arg]


def test_a_non_positive_or_non_finite_scale_is_refused_at_the_config() -> None:
    """Acceptance #19: a degenerate scale is refused where it is configured.

    Zero is refused for the reason ``GumbelConfig`` refuses it -- scale 0 is
    not a noiseless search but a silent collapse of root sampling into top-m
    argmax -- and infinity for the mirror reason. Refusing here rather than in
    the search means the campaign dies at construction rather than on the
    first fast draft ply, hours in.
    """
    for value in (0.0, float("inf"), float("nan"), -1.0):
        with pytest.raises(ValueError, match="fast_gumbel_scale"):
            SelfPlayConfig(
                workers=2, process_batch_size=64, fast_gumbel_scale=value
            )
        with pytest.raises(ValueError, match="draft_gumbel_scale"):
            SelfPlayConfig(
                workers=2, process_batch_size=64, draft_gumbel_scale=value
            )


def test_train_config_rejects_unknown_kwargs() -> None:
    """Same for TrainConfig, whose fields all have defaults."""
    assert TrainConfig().batch_size == 512
    with pytest.raises(TypeError):
        TrainConfig(batch_sise=256)  # type: ignore[call-arg]


def test_typecheck_flag_reaches_spawned_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """KOMUGI_TYPECHECK=1 set in the parent is visible when the child imports."""
    monkeypatch.setenv("KOMUGI_TYPECHECK", "1")
    assert _run_child((4, 7), expect_typecheck=True) == 0


@pytest.mark.skipif(
    not _SEAM_AVAILABLE,
    reason="jaxtyping/beartype unavailable: the seam degrades to identity decorators",
)
def test_enabled_seam_rejects_bad_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """With checking on, a 3-D array fails the (batch, actions) contract."""
    monkeypatch.setenv("KOMUGI_TYPECHECK", "1")
    assert _run_child((2, 3, 4), expect_typecheck=True) != 0


def test_disabled_seam_ignores_bad_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag unset the same call is unchecked, and cheap."""
    monkeypatch.delenv("KOMUGI_TYPECHECK", raising=False)
    assert _run_child((2, 3, 4), expect_typecheck=False) == 0
