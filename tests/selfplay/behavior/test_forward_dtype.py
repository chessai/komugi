"""The CPU fallback of the fp16 serving forms, pinned to the byte (KOM-039).

``forward_dtype`` is a *compute* knob with a device precondition: the weights
are converted -- and torch's autocast is entered -- only when the model is on
CUDA. Off CUDA both fp16 forms run the f32 forward, and this module pins that
they run it **identically**, not merely similarly:

* ``--serial-debug`` runs on the host. An operator reproducing a divergence
  there must be looking at the numbers production would have produced, and a
  precision knob that quietly changed the host answer would make every serial
  reproduction an argument about tolerances.
* the gate has no GPU. Every other test of this seam -- the equivalence corpus,
  the batch-permutation identity, the pipelined server's bit-identity
  invariant I6 -- runs on the CPU path, so if that path moved under
  ``forward_dtype`` they would all be measuring a different forward than the
  one they were written against.

The one deliberate asymmetry is the KOM-039 tripwire, which runs in both fp16
forms on any device and refuses a non-finite wdl. It cannot fire on a batch
this module evaluates: a real net's answer is finite, and the values compared
below are equal byte for byte. Its own tests live in the unit suite, next to
the helper.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch

import inference_rig as rig

#: The module rather than the package: ``AsyncEvaluator`` is a server-internal
#: seam and is not re-exported by ``komugi.selfplay``, exactly as the pipelined
#: suite next door reaches it.
sp = pytest.importorskip("komugi.selfplay.inference")

if not hasattr(sp, "AsyncEvaluator"):
    pytest.skip(
        "komugi.selfplay.inference exports no AsyncEvaluator yet (KOM-032)",
        allow_module_level=True,
    )

#: The two forms whose CPU behaviour must collapse onto ``"f32"``.
_FP16_FORMS = ["f16", "f16_autocast"]


def _batch(rows: int = 5) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """One deterministic batch, shared by every evaluator in this module."""
    rng = np.random.default_rng(20260829)
    obs = rng.standard_normal((rows, rig.IN_PLANES, rig.BOARD, rig.BOARD)).astype(
        np.float32
    )
    legal = np.ones((rows, rig.NUM_ACTIONS), dtype=np.bool_)
    return obs, legal


def _assert_bit_identical(baseline: object, served: object, form: str) -> None:
    """Every field of two :class:`NetEval` s, compared as bytes.

    ``tobytes()`` rather than ``==`` because the claim is bitwise: array
    equality would accept ``-0.0`` for ``0.0``, and would say nothing at all
    about a NaN. Both arrays come out of the seam contiguous and float32, so
    the buffers are directly comparable.
    """
    for name in ("policy_logits", "wdl", "moves_left"):
        want = np.asarray(getattr(baseline, name))
        got = np.asarray(getattr(served, name))
        assert got.dtype == np.float32, f"{name} at forward_dtype={form!r}"
        assert got.shape == want.shape, f"{name} at forward_dtype={form!r}"
        assert np.all(np.isfinite(got)), f"{name} at forward_dtype={form!r}"
        assert got.tobytes() == want.tobytes(), (
            f"{name} at forward_dtype={form!r} differs from the f32 forward on "
            f"the CPU, where no conversion and no autocast may happen"
        )


@pytest.mark.parametrize("form", _FP16_FORMS)
def test_the_serial_cpu_forward_is_bit_identical_to_f32(form: str) -> None:
    """``InProcessEvaluator`` -- the whole evaluator in ``--serial-debug``."""
    obs, legal = _batch()
    baseline = sp.InProcessEvaluator(rig.random_factory(), rig.make_config()).evaluate(
        obs, legal
    )
    served = sp.InProcessEvaluator(
        rig.random_factory(), rig.make_config(forward_dtype=form)
    ).evaluate(obs, legal)
    _assert_bit_identical(baseline, served, form)


@pytest.mark.parametrize("form", _FP16_FORMS)
def test_the_launched_cpu_forward_is_bit_identical_to_f32(form: str) -> None:
    """``AsyncEvaluator`` too -- on CPU it is the serial forward in a handle.

    The pipelined server's invariant I6 says the launched path and the serial
    one are indistinguishable on the host; ``forward_dtype`` must not be the
    thing that separates them.
    """
    obs, legal = _batch()
    baseline = sp.InProcessEvaluator(rig.random_factory(), rig.make_config()).evaluate(
        obs, legal
    )
    served = (
        sp.AsyncEvaluator(rig.random_factory(), rig.make_config(forward_dtype=form))
        .launch(obs, legal)
        .result()
    )
    _assert_bit_identical(baseline, served, form)


@pytest.mark.parametrize("form", _FP16_FORMS)
def test_a_host_model_is_never_converted(form: str) -> None:
    """The mechanism behind the pin: nothing touched the caller's weights.

    Checked on the model object rather than on the outputs because that is
    where the conversion would be visible even if a forward happened to agree:
    a half parameter here would mean the fallback is bit-identical by luck.
    """
    model = rig.random_factory()
    sp.InProcessEvaluator(model, rig.make_config(forward_dtype=form))
    sp.AsyncEvaluator(model, rig.make_config(forward_dtype=form))
    for name, param in model.named_parameters():
        assert param.dtype == torch.float32, f"{name} converted at {form!r} on CPU"
    for name, buffer in model.named_buffers():
        assert buffer.dtype != torch.float16, f"{name} converted at {form!r} on CPU"


@pytest.mark.parametrize("form", _FP16_FORMS)
def test_the_cpu_forward_still_returns_a_distribution(form: str) -> None:
    """The seam's own contract, restated under the knob: wdl is probabilities."""
    obs, legal = _batch()
    served = sp.InProcessEvaluator(
        rig.random_factory(), rig.make_config(forward_dtype=form)
    ).evaluate(obs, legal)
    wdl = np.asarray(served.wdl, dtype=np.float64)
    assert wdl.shape == (obs.shape[0], 3)
    assert np.all(wdl >= 0.0)
    assert np.allclose(wdl.sum(axis=1), 1.0, atol=1e-5)
