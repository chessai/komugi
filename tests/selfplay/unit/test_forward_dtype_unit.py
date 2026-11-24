"""Unit tests for the fp16 serving forms (KOM-039, ``forward_dtype``).

The knob is the *compute* dtype of the server-side forward, and it is
deliberately not the wire dtype: ``policy_dtype`` names what the policy bytes
are where they cross the shared-memory seam and stays float32-only, while this
one names what the model's own arithmetic runs in. The two are tested together
below precisely because they are easy to conflate.

Four groups, in the order the implementation is built:

* **the knob** -- exactly three accepted values, a loud refusal for anything
  else, and ``policy_dtype``'s reserved NotImplementedError still firing beside
  it (a validation edit is the cheapest way to lose the older refusal).
* **the conversion helper** -- :func:`~komugi.selfplay.inference._half_with_f32_bn`
  is the primary form's whole implementation, so its contract is pinned
  directly on a real (tiny) net: every BatchNorm2d stays float32, everything
  else is half, and a converted BatchNorm still takes half in and gives half
  back.
* **the tripwire** -- an fp16 forward that overflows produces NaN priors, which
  a search does not crash on; it just stops preferring anything. The helper is
  tested directly, in **every** ``forward_dtype`` including ``"f32"``: review
  TS-01 (2026-09-12) reversed KOM-039's f32 exemption, because a NaN prior in
  f32 (from a poisoned checkpoint, not from an overflow) is the same flat run.
  ``test_f32_nonfinite_wdl_is_refused`` records that reversal and replaces the
  pin that asserted the opposite.
* **CUDA parity** -- skipped in the gate, which has no GPU, and the reason the
  CPU-fallback pin next door exists.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn

from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.selfplay import inference
from komugi.selfplay.inference import (
    FORWARD_DTYPES,
    AsyncEvaluator,
    InferenceConfig,
    InProcessEvaluator,
)

IN_PLANES = 4
NUM_ACTIONS = 81
BOARD = 9


def _tiny_net() -> KomugiNet:
    """A one-block, 81-action net -- the SMALL test config used next door."""
    trunk = TrunkConfig(in_planes=IN_PLANES, channels=8, blocks=1, gpool_every=3, seed=7)
    heads = HeadsConfig(channels=8, num_actions=NUM_ACTIONS, policy_planes=1, seed=7)
    net = KomugiNet(NetConfig(trunk=trunk, heads=heads))
    net.eval()
    return net


def _config(forward_dtype: str = "f32") -> InferenceConfig:
    """An :class:`InferenceConfig` shaped for the tiny net."""
    return InferenceConfig(
        in_planes=IN_PLANES, num_actions=NUM_ACTIONS, forward_dtype=forward_dtype
    )


def _batch(rows: int = 4) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """A deterministic batch: the same bytes for every evaluator under test."""
    rng = np.random.default_rng(39)
    obs = rng.standard_normal((rows, IN_PLANES, BOARD, BOARD)).astype(np.float32)
    legal = np.ones((rows, NUM_ACTIONS), dtype=np.bool_)
    return obs, legal


# --------------------------------------------------------------------------
# the knob
# --------------------------------------------------------------------------


def test_the_allowed_forms_are_the_documented_three() -> None:
    """f32, the converted-weight form, and the autocast measurement form."""
    assert FORWARD_DTYPES == ("f32", "f16", "f16_autocast")


@pytest.mark.parametrize("form", list(FORWARD_DTYPES))
def test_every_allowed_form_constructs(form: str) -> None:
    """All three are real configurations, not aspirations: none of them raises."""
    assert _config(form).forward_dtype == form


@pytest.mark.parametrize("bad", ["f64", "fp16", "bf16", "F16", "half", ""])
def test_an_unknown_form_is_refused_by_name(bad: str) -> None:
    """The field and the allowed set are both in the message, not just a repr.

    A silently ignored precision knob is the failure this refusal exists to
    prevent: the caller believes it is serving fp16 and measures f32.
    """
    with pytest.raises(ValueError, match="forward_dtype") as excinfo:
        _config(bad)
    message = str(excinfo.value)
    for form in FORWARD_DTYPES:
        assert form in message
    assert repr(bad) in message


def test_the_default_is_f32() -> None:
    """Merging KOM-039 changes nothing about a config nobody edited."""
    config = InferenceConfig(in_planes=IN_PLANES, num_actions=NUM_ACTIONS)
    assert config.forward_dtype == "f32"


@pytest.mark.parametrize("form", list(FORWARD_DTYPES))
def test_the_wire_knob_is_still_deferred_in_every_compute_form(form: str) -> None:
    """``policy_dtype`` keeps raising, whatever ``forward_dtype`` says.

    The two knobs sit in the same ``__post_init__`` and describe different
    seams -- compute here, wire there -- so the cheapest way to lose D8's
    reserved refusal is to add a validation branch beside it. An fp16 forward
    still writes float32 into the slot.
    """
    with pytest.raises(NotImplementedError, match="policy_dtype"):
        InferenceConfig(
            in_planes=IN_PLANES,
            num_actions=NUM_ACTIONS,
            policy_dtype="f16",
            forward_dtype=form,
        )


# --------------------------------------------------------------------------
# the conversion helper
# --------------------------------------------------------------------------


def test_half_conversion_keeps_every_batchnorm_in_float32() -> None:
    """Half weights everywhere the time goes; float32 where the maths is fragile.

    Eval-mode BN divides by ``sqrt(running_var + eps)``, and a running variance
    that underflows to zero in half turns the normalisation into a division by
    ``eps`` -- finite, plausible, and wrong for the rest of the run.
    """
    net = _tiny_net()
    assert inference._half_with_f32_bn(net) is net

    norms = 0
    weighted = 0
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            norms += 1
            for param in module.parameters(recurse=False):
                assert param.dtype == torch.float32
            for buffer in module.buffers(recurse=False):
                # num_batches_tracked is an integer counter, not a float that
                # half() would have touched.
                assert buffer.dtype in (torch.float32, torch.int64)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            weighted += 1
            for param in module.parameters(recurse=False):
                assert param.dtype == torch.float16
    assert norms > 0, "the tiny net has no BatchNorm2d -- this test proves nothing"
    assert weighted > 0, "the tiny net has no conv/linear -- same"


def test_a_converted_batchnorm_takes_half_in_and_gives_half_back() -> None:
    """The hooks are the seam between the two dtypes, and they run both ways.

    Belt-and-braces across torch versions: current torch's native batch_norm
    already accepts half input with float32 weights and returns half (the impl
    review's M8 mutation -- hooks deleted -- survived on torch 2.12), so on
    this version the hooks are unobservable. They stay because that mixed-mode
    tolerance is a torch implementation detail, not a contract, and the hooks
    make the seam explicit whichever torch is installed.
    """
    net = _tiny_net()
    inference._half_with_f32_bn(net)
    norm = next(m for m in net.modules() if isinstance(m, nn.BatchNorm2d))

    out = norm(torch.ones(2, norm.num_features, BOARD, BOARD, dtype=torch.float16))
    assert out.dtype == torch.float16
    assert bool(torch.isfinite(out).all())


def test_the_conversion_is_never_applied_on_cpu() -> None:
    """The CPU fallback is a property of the *construction*, not of the forward.

    Both fp16 forms leave a host-resident model exactly as they found it, which
    is what makes ``--serial-debug`` and the CPU-only gate bit-identical to
    ``forward_dtype="f32"``.
    """
    for form in ("f16", "f16_autocast"):
        net = _tiny_net()
        InProcessEvaluator(net, _config(form))
        for name, param in net.named_parameters():
            assert param.dtype == torch.float32, f"{name} was converted on CPU"


# --------------------------------------------------------------------------
# the tripwire
# --------------------------------------------------------------------------


class _OverflowOutput(NamedTuple):
    """A KomugiNet-shaped output whose wdl logits have already overflowed."""

    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor


class _OverflowNet(nn.Module):
    """A model that produces exactly what a saturated fp16 forward would.

    ``inf`` logits, which softmax turns into NaN -- the failure mode the
    tripwire exists for, reproduced without needing the GPU that would produce
    it naturally. No parameters, so the evaluator reads its device as CPU and
    the conversion path is not involved.
    """

    def forward(self, x: torch.Tensor) -> _OverflowOutput:
        """Overflowed wdl logits, finite everything else."""
        rows = int(x.shape[0])
        return _OverflowOutput(
            policy_logits=torch.zeros(rows, NUM_ACTIONS),
            wdl_logits=torch.full((rows, 3), float("inf")),
            moves_left=torch.zeros(rows),
        )


def _nan_wdl() -> torch.Tensor:
    """A post-softmax wdl row with one NaN in it."""
    return torch.tensor([[0.5, float("nan"), 0.5], [0.2, 0.3, 0.5]])


@pytest.mark.parametrize("form", ["f16", "f16_autocast"])
def test_the_tripwire_refuses_a_nonfinite_wdl(form: str) -> None:
    """It names the form and the ticket, because that is the whole diagnosis."""
    with pytest.raises(RuntimeError, match="KOM-039") as excinfo:
        inference._refuse_nonfinite_wdl(_nan_wdl(), form)
    assert form in str(excinfo.value)


def test_the_tripwire_passes_a_finite_wdl() -> None:
    """A healthy batch costs one isfinite and nothing else."""
    clean = torch.tensor([[0.25, 0.5, 0.25], [1.0, 0.0, 0.0]])
    inference._refuse_nonfinite_wdl(clean, "f16")


def test_f32_nonfinite_wdl_is_refused() -> None:
    """The tripwire runs on the f32 path too.

    **This REVERSES the KOM-039 decision** that the f32 forward should skip the
    check, which this test replaces: it used to be
    ``test_the_tripwire_does_not_run_on_the_f32_path``, and it asserted that
    the same NaN the fp16 forms refuse went straight through at ``"f32"``
    (review TS-01, 2026-09-12).

    KOM-039's argument was about *overflow*: f32 does not saturate, so the
    precision that cannot produce this failure should not pay for the check.
    But the check is for *non-finiteness*, and f32 reaches it by other roads --
    a net whose weights were already NaN when they were loaded, which is
    exactly what the training-side half of TS-01 shows a poisoned checkpoint
    produces. A NaN prior is the same silent, flat, week-long run in f32 that
    it is in fp16, and the cost of refusing it is one ``isfinite`` over
    ``(N, 3)`` values.
    """
    with pytest.raises(RuntimeError, match="KOM-039") as excinfo:
        inference._refuse_nonfinite_wdl(_nan_wdl(), "f32")
    assert "f32" in str(excinfo.value)
    assert "TS-01" in str(excinfo.value)


def test_every_evaluator_dtype_refuses_a_nan_batch() -> None:
    """The reversal at the call site the server actually runs.

    Both evaluators forward the identical overflowing model on the CPU, and
    **both** now refuse. Before TS-01 the f32 arm of this test asserted the
    opposite -- that the NaN priors reached the caller -- which is the
    behaviour the poisoned-checkpoint witness turns into a flat run.
    """
    obs, legal = _batch(rows=2)
    for form in ("f16", "f32"):
        with pytest.raises(RuntimeError, match="KOM-039"):
            InProcessEvaluator(_OverflowNet(), _config(form)).evaluate(obs, legal)


# --------------------------------------------------------------------------
# CUDA parity -- skipped wherever there is no GPU, which includes the gate
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("form", ["f16", "f16_autocast"])
def test_reduced_precision_agrees_with_f32_on_cuda(form: str) -> None:
    """Knob on vs off, on the same random batch through the same tiny net.

    Loose tolerances on purpose: this is not a bitwise pin (fp16 is not f32),
    it is the assertion that the served evaluations are still *evaluations* --
    finite, a distribution, and near the f32 answer rather than merely
    well-shaped. The tight pin lives on the CPU fallback, where equality is a
    property the implementation actually promises.
    """
    device = torch.device("cuda")
    obs, legal = _batch()

    reference = InProcessEvaluator(_tiny_net().to(device), _config()).evaluate(
        obs, legal
    )
    served = InProcessEvaluator(_tiny_net().to(device), _config(form)).evaluate(
        obs, legal
    )

    wdl = np.asarray(served.wdl, dtype=np.float64)
    policy = np.asarray(served.policy_logits, dtype=np.float64)
    assert np.all(np.isfinite(wdl))
    assert np.all(np.isfinite(policy))
    assert np.allclose(wdl.sum(axis=1), 1.0, atol=1e-3)
    assert np.asarray(served.wdl).dtype == np.float32
    assert np.asarray(served.policy_logits).dtype == np.float32

    reference_wdl = np.asarray(reference.wdl, dtype=np.float64)
    reference_policy = np.asarray(reference.policy_logits, dtype=np.float64)
    assert float(np.mean(np.abs(wdl - reference_wdl))) < 5e-2
    assert float(np.mean(np.abs(policy - reference_policy))) < 5e-2


# --------------------------------------------------------------------------
# the 2026-08-30 impl review's killers (F1, F2, F3, F7)
# --------------------------------------------------------------------------


class _SharpOutput(NamedTuple):
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor


class _HalfLogitNet(nn.Module):
    """Emits float16 wdl logits, exactly as a CUDA f16 forward would.

    Sharp rows: ``softmax([11, 0, -11])`` has a tail probability ~2.7e-10,
    below float16's smallest subnormal (~6e-8) -- an f16 softmax flushes it
    to exactly ``0.0``, an f32 softmax does not. This is the design's D1
    sharp-logit stub, and the executed killer of the review's M2a mutation
    (dropping the ``.float()`` before the wdl softmax survived every other
    test, GPU parity included).
    """

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> _SharpOutput:
        rows = int(x.shape[0])
        wdl = torch.tensor(
            [[11.0, 0.0, -11.0]], dtype=torch.float16, device=x.device
        ).expand(rows, 3)
        return _SharpOutput(
            policy_logits=torch.zeros(
                rows, NUM_ACTIONS, dtype=torch.float16, device=x.device
            ),
            wdl_logits=wdl,
            moves_left=torch.zeros(rows, dtype=torch.float16, device=x.device),
        )


def test_wdl_softmax_runs_in_f32_even_for_half_logits() -> None:
    """The load-bearing cast: wdl logits go to float32 BEFORE the softmax."""
    obs = np.zeros((2, IN_PLANES, BOARD, BOARD), dtype=np.float32)
    legal = np.ones((2, NUM_ACTIONS), dtype=np.bool_)
    out = InProcessEvaluator(_HalfLogitNet(), _config("f16")).evaluate(obs, legal)
    wdl = np.asarray(out.wdl)
    assert wdl.dtype == np.float32
    assert np.all(wdl[:, 2] > 0.0), (
        "softmax ran in float16: the tail probability was flushed to zero"
    )


def test_f32_and_f16_never_construct_an_autocast_region() -> None:
    """Structural conditionality of autocast, checkable without a GPU.

    ``torch.device("cuda")`` is a pure descriptor. The review's M7 mutation
    (autocast whenever the device is cuda, knob ignored) survived the GPU
    parity test because the f32 *reference* autocasts along with the subject;
    this pin cannot be fooled that way. It also pins that ``"f16"`` does not
    double-convert (half weights AND an autocast region).
    """
    cuda = torch.device("cuda")
    assert isinstance(
        inference._forward_autocast(_config("f32"), cuda), nullcontext
    ), "forward_dtype='f32' silently autocasts on cuda"
    assert isinstance(
        inference._forward_autocast(_config("f16"), cuda), nullcontext
    ), "forward_dtype='f16' must not ALSO autocast (double-conversion)"
    assert not isinstance(
        inference._forward_autocast(_config("f16_autocast"), cuda), nullcontext
    ), "forward_dtype='f16_autocast' must construct a real autocast region"


def test_half_conversion_preserves_batchnorm_values_not_just_dtype() -> None:
    """The conversion must not round BatchNorm state through float16.

    Review F3: ``model.half()`` quantizes the norm state, and restoring the
    *dtype* afterwards keeps the quantized values -- a running variance of
    1e-9 comes back as exactly 0.0 (the division-by-eps failure the helper's
    docstring promises to prevent) and everything else moves by ~5e-4. The
    helper snapshots and restores the float32 values; this pins it bitwise.
    """
    net = _tiny_net()
    norms = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    assert norms, "the tiny net is expected to carry BatchNorm2d modules"
    with torch.no_grad():
        for norm in norms:
            assert norm.running_var is not None and norm.running_mean is not None
            assert norm.weight is not None and norm.bias is not None
            norm.running_var.fill_(1e-9)
            norm.running_mean.fill_(0.123456791)
            norm.weight.fill_(5e-5)
    before = {
        norm: {
            name: tensor.detach().clone()
            for name, tensor in [
                *norm.named_parameters(recurse=False),
                *norm.named_buffers(recurse=False),
            ]
        }
        for norm in norms
    }

    inference._half_with_f32_bn(net)

    for norm in norms:
        for name, tensor in [
            *norm.named_parameters(recurse=False),
            *norm.named_buffers(recurse=False),
        ]:
            assert tensor.dtype in (torch.float32, torch.int64)
            assert torch.equal(tensor, before[norm][name]), (
                f"BatchNorm {name} was quantized through float16 by the "
                f"conversion instead of restored bitwise"
            )


@pytest.mark.parametrize("forward_dtype", ["f16", "f16_autocast"])
def test_sparse_and_forward_dtype_compose_on_cpu(forward_dtype: str) -> None:
    """Both knobs set at once: the CPU fallback stays bit-identical to f32.

    The interesting sparse x dtype cells are on the GPU (the gather runs in
    the device-work region); this pins the composition's CPU half so setting
    both knobs cannot perturb the gate's serial path (review F7, CPU cells).
    """
    obs, legal = _batch()
    net = _tiny_net()
    base = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        sparse_policy=True,
        forward_dtype="f32",
    )
    subject = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        sparse_policy=True,
        forward_dtype=forward_dtype,
    )
    want = InProcessEvaluator(net, base).evaluate(obs, legal)
    got = InProcessEvaluator(net, subject).evaluate(obs, legal)
    assert np.array_equal(np.asarray(want.policy_logits), np.asarray(got.policy_logits))
    assert np.array_equal(np.asarray(want.wdl), np.asarray(got.wdl))
    assert np.array_equal(np.asarray(want.moves_left), np.asarray(got.moves_left))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("sparse", [False, True])
def test_launch_softmax_runs_in_f32_on_cuda(sparse: bool) -> None:
    """The launch site's load-bearing cast, on real hardware (review M2b).

    The launch path short-circuits to the serial evaluator off CUDA, so the
    serial sharp-logit pin above cannot reach its softmax; this one does. It
    runs on any GPU-bearing dev box or pod and skips in the CPU gate. The
    sparse cell doubles as the sparse x forward_dtype composition check for
    the gather ordering (the gather must see the packed rows before the
    outbound float casts).
    """
    config = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        sparse_policy=sparse,
        forward_dtype="f16",
    )
    net = _HalfLogitNet().to("cuda")
    obs = np.zeros((2, IN_PLANES, BOARD, BOARD), dtype=np.float32)
    legal = np.ones((2, NUM_ACTIONS), dtype=np.bool_)
    handle = AsyncEvaluator(net, config).launch(obs, legal)
    if sparse:
        assert isinstance(handle, inference._PacksPolicy)
        wdl = np.asarray(handle.packed_result().wdl)
    else:
        wdl = np.asarray(handle.result().wdl)
    assert wdl.dtype == np.float32
    assert np.all(wdl[:, 2] > 0.0), (
        "launch-site softmax ran in float16: the tail was flushed to zero"
    )

