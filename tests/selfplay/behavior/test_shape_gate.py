"""Design D7: the spawned-worker shape gate, and the control that gives it meaning.

``KOMUGI_TYPECHECK`` is read once, when :mod:`komugi.shapecheck` is first
imported (``tests/test_env.py`` pins that bootstrap).  A test therefore cannot
arm the seam from inside itself: the flag has to be in a *child's* environment
before anything else happens, which is why every case here is a fresh
interpreter rather than a fixture.

The violation is chosen so that only the jaxtyping seam can see it.  A
``wdl`` of shape ``(N, 4)``:

* is not caught by the search -- the search reads ``policy_logits`` and a
  scalar, and its own ``(A,)`` logits ``ValueError`` never looks at wdl;
* is not caught by numpy -- four components broadcast against three-component
  arithmetic in several plausible ways, and still sum to 1.0;
* *is* caught by the ``@jaxtyped`` annotation on the seam, which says three.

Design F7 asks for the pair, not the single case: a run that merely fails
proves nothing, because a broken stub fails under every setting.  So the three
children are

1. ``KOMUGI_TYPECHECK=1`` + misshaped wdl -> nonzero, traceback names
   beartype/jaxtyping;
2. ``KOMUGI_TYPECHECK=0`` + misshaped wdl -> the beartype signature is
   **absent** (the run may still fail for its own reasons -- an unchecked
   contract violation usually does -- and that is not what is being asserted);
3. ``KOMUGI_TYPECHECK=1`` + a well-shaped stub -> exits 0, so case 1 is
   attributable to the shape and not to the seam rejecting everything.
"""

from __future__ import annotations

import pytest

import worker_rig as rig

sp = pytest.importorskip("komugi.selfplay")

if not hasattr(sp, "SelfPlayWorker"):
    pytest.skip(
        "komugi.selfplay exports no SelfPlayWorker yet (KOM-022)",
        allow_module_level=True,
    )

# Without these the seam degrades to identity decorators by design
# (``shapecheck.checked_seam``), and "no beartype error" would be a property of
# the environment rather than of the worker.
pytest.importorskip("beartype")
pytest.importorskip("jaxtyping")

_MISSHAPE = "import worker_rig; worker_rig.misshape_main()"
_SANE = "import worker_rig; worker_rig.sane_main()"


def test_a_misshaped_wdl_kills_a_typechecked_worker() -> None:
    """The gate: armed, the seam refuses a ``(N, 4)`` wdl and the worker dies."""
    result = rig.run_python(
        _MISSHAPE, env=rig.child_env(KOMUGI_TYPECHECK="1")
    )
    assert result.returncode != 0, (
        "a worker fed wdl of shape (N, 4) exited 0 under KOMUGI_TYPECHECK=1; "
        "the seam either is not @jaxtyped or is not on the path the evaluator "
        f"adapter takes (design D7)\nstdout:\n{result.stdout}"
    )
    assert rig.names_the_seam(result.stderr), (
        "the worker died under KOMUGI_TYPECHECK=1 but its traceback names "
        f"neither beartype nor jaxtyping, so something else failed first:\n"
        f"{result.stderr[-4000:]}"
    )


def test_the_same_misshape_produces_no_beartype_error_with_checking_off() -> None:
    """The control (design F7): the seam is what fired, not an incidental check.

    Only the *absence* of the signature is asserted.  With checking off the
    child is free to die however it likes -- a four-component wdl reaching
    ``wdl_utility`` is a ``ValueError`` -- and that freedom is the point: if
    this run also named beartype, the test above would be measuring an
    always-on check and the ``KOMUGI_TYPECHECK`` bootstrap would be decorative.
    """
    result = rig.run_python(
        _MISSHAPE, env=rig.child_env(KOMUGI_TYPECHECK="0")
    )
    assert not rig.names_the_seam(result.stderr), (
        "beartype/jaxtyping fired with KOMUGI_TYPECHECK=0, so the shape check "
        "is not gated on the env var at all and the gate above proves nothing "
        f"about the seam:\n{result.stderr[-4000:]}"
    )


def test_an_armed_seam_still_passes_a_well_shaped_worker() -> None:
    """The other control: checking on, shapes right, the game plays and exits 0."""
    result = rig.run_python(_SANE, env=rig.child_env(KOMUGI_TYPECHECK="1"))
    assert result.returncode == 0, (
        "a correctly shaped worker failed under KOMUGI_TYPECHECK=1, so the "
        "misshape test above cannot be attributed to the misshape:\n"
        f"{result.stderr[-4000:]}"
    )


def test_the_typecheck_flag_reaches_a_spawned_interpreter() -> None:
    """The bootstrap this whole module rests on, asserted directly.

    ``komugi.shapecheck.TYPECHECK`` is computed at import time from the
    environment, and a spawned worker inherits that environment (design D2/D7).
    If this ever reads ``False`` under ``KOMUGI_TYPECHECK=1`` the gate above is
    silently disarmed.
    """
    probe = "import komugi.shapecheck as s; print(int(s.TYPECHECK))"
    armed = rig.run_python(probe, env=rig.child_env(KOMUGI_TYPECHECK="1"))
    off = rig.run_python(probe, env=rig.child_env(KOMUGI_TYPECHECK="0"))
    assert armed.returncode == 0 and off.returncode == 0, (
        f"{armed.stderr}\n{off.stderr}"
    )
    assert armed.stdout.strip() == "1"
    assert off.stdout.strip() == "0"
