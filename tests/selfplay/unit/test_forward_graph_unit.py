"""Unit tests for the captured serving forward (KOM-048, ``forward_graph``).

The knob turns :class:`~komugi.selfplay.inference.AsyncEvaluator`'s CUDA launch
from "enqueue several dozen short kernels" into "replay one captured graph". It
is therefore a knob whose *whole* value is a CPU-side cost that this repository's
gate cannot measure and whose hardware half this repository's development box
cannot even run -- ROCm's hipGraph core-dumped every gfx1034 probe. So the
tests split along exactly that line, and the split is the point:

* **What runs everywhere** -- the knob's validation, all three refusals, the
  ``serial_debug`` purity property, the row-independence argument that makes
  pad rows safe, and the exclusivity handshake. None of these needs a GPU,
  because none of them is about the graph: they are about the *decisions* taken
  around it, which is where a silent failure would live. A run that believes it
  is graphing and is not would be a bench that lies, and every refusal below
  exists to make that impossible rather than unlikely.
* **What runs on a CUDA pod** -- parity, the descending-N sweep that is the
  only order in which a stale tail can be observed, replay determinism, and the
  two guards firing against real hardware. These skip in the gate *and* on the
  ROCm box, and their skip is honest: there is nothing here that pretends to
  have checked a capture it never made.

The refusal tests are deliberately written to execute on **either** kind of
torch build. :func:`~komugi.selfplay.inference._refuse_ungraphable` takes
``torch.version.hip`` as an argument rather than reading it, which is what lets
the ROCm branch and the CUDA branch both be reachable from one box; the
constructor-level tests monkeypatch the module attribute for the same reason.
A refusal test that skipped on the machine the code actually ships to would be
the one test in this file worth nothing.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn

from komugi.config import SelfPlayConfig
from komugi.net import HeadsConfig, KomugiNet, NetConfig, NetOutput, TrunkConfig
from komugi.selfplay import inference, orchestrate
from komugi.selfplay.inference import (
    FORWARD_GRAPH_MODES,
    AsyncEvaluator,
    InferenceConfig,
    InProcessEvaluator,
    NetEval,
)

IN_PLANES = 4
NUM_ACTIONS = 81
BOARD = 9
MAX_BATCH = 12

#: A CUDA device that is not ROCm. The second half is not pedantry: ROCm's
#: torch reports ``device.type == "cuda"``, so ``is_available()`` alone would
#: run the capture tests on exactly the hardware KOM-048 finding D1 refuses.
_GRAPH_GPU = torch.cuda.is_available() and torch.version.hip is None

_SKIP_NO_GPU = pytest.mark.skipif(
    not _GRAPH_GPU, reason="needs a non-ROCm CUDA device (KOM-048 finding D1)"
)


def _tiny_net() -> KomugiNet:
    """A one-block, 81-action net -- the SMALL test config used next door."""
    trunk = TrunkConfig(in_planes=IN_PLANES, channels=8, blocks=1, gpool_every=3, seed=7)
    heads = HeadsConfig(channels=8, num_actions=NUM_ACTIONS, policy_planes=1, seed=7)
    net = KomugiNet(NetConfig(trunk=trunk, heads=heads))
    net.eval()
    return net


def graph_stub_factory() -> torch.nn.Module:
    """A module-level model factory: what design D2/F18 says one must be."""
    return _tiny_net()


def _config(forward_graph: str = "off", forward_dtype: str = "f32") -> InferenceConfig:
    """An :class:`InferenceConfig` shaped for the tiny net."""
    return InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=MAX_BATCH,
        forward_graph=forward_graph,
        forward_dtype=forward_dtype,
    )


def _batch(
    rows: int, seed: int = 48
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """A deterministic batch whose values are nowhere near zero.

    The offset matters for the pad tests: a batch of small values would leave a
    zeroed pad indistinguishable from a real row, and "the pad was zeroed" is
    precisely what those tests are trying to observe.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((rows, IN_PLANES, BOARD, BOARD)) + 4.0
    obs = raw.astype(np.float32)
    legal = np.ones((rows, NUM_ACTIONS), dtype=np.bool_)
    return obs, legal


# --------------------------------------------------------------------------
# T1 -- the knob
# --------------------------------------------------------------------------


def test_the_allowed_modes_are_off_and_on() -> None:
    """Two values and no third: there is deliberately no ``"auto"``."""
    assert FORWARD_GRAPH_MODES == ("off", "on")


@pytest.mark.parametrize("mode", list(FORWARD_GRAPH_MODES))
def test_every_allowed_mode_constructs(mode: str) -> None:
    """Both are real configurations; validation accepts exactly these."""
    assert _config(mode).forward_graph == mode


@pytest.mark.parametrize("bad", ["auto", "ON", "true", "cudagraph", "1", ""])
def test_an_unknown_mode_is_refused_by_name(bad: str) -> None:
    """The field, the allowed set and the offending value are all in the message.

    A misspelled knob that fell through to "off" would be the quietest possible
    failure: the bench runs, the numbers are right, and the thing being
    measured is the path the operator was trying to leave.
    """
    with pytest.raises(ValueError, match="forward_graph") as excinfo:
        _config(bad)
    message = str(excinfo.value)
    for mode in FORWARD_GRAPH_MODES:
        assert mode in message
    assert repr(bad) in message


def test_the_default_is_off() -> None:
    """Merging KOM-048 changes nothing about a config nobody edited."""
    config = InferenceConfig(in_planes=IN_PLANES, num_actions=NUM_ACTIONS)
    assert config.forward_graph == "off"


def test_the_two_forward_knobs_are_independent_fields() -> None:
    """``forward_dtype`` and ``forward_graph`` are validated separately.

    They are neighbours in one ``__post_init__`` and describe different things
    -- what the arithmetic runs in, and whether it was captured -- so the
    cheapest way to break one is to edit the other.
    """
    config = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        forward_dtype="f16",
        forward_graph="on",
    )
    assert (config.forward_dtype, config.forward_graph) == ("f16", "on")
    with pytest.raises(ValueError, match="forward_dtype"):
        _config("on", forward_dtype="bf16")


# --------------------------------------------------------------------------
# T2 -- the three refusals. These RUN in CI, on either torch build.
# --------------------------------------------------------------------------


def test_autocast_and_graphs_are_refused_together_by_name() -> None:
    """Config-only, so the same refusal on every box (KOM-048, checked first).

    Autocast is a per-call thread-local region with its own cast cache; a
    capture would freeze one instant of that cache into every replay. The two
    knobs are individually supported and the combination is forbidden.
    """
    with pytest.raises(RuntimeError, match="KOM-048") as excinfo:
        inference._refuse_ungraphable(
            _config("on", forward_dtype="f16_autocast"), torch.device("cuda"), None
        )
    message = str(excinfo.value)
    assert "autocast" in message
    assert "f16_autocast" in message


def test_the_config_only_refusal_comes_first() -> None:
    """Order pinned: the box-independent refusal wins over the box-dependent ones.

    Not cosmetic. It is what makes the autocast refusal the *same* refusal on
    a ROCm box, a CUDA pod and a laptop -- and therefore what lets the test
    above assert its message rather than skip.
    """
    with pytest.raises(RuntimeError, match="autocast"):
        inference._refuse_ungraphable(
            _config("on", forward_dtype="f16_autocast"), torch.device("cpu"), "6.2.0"
        )


def test_a_rocm_build_is_refused_by_name() -> None:
    """Finding D1: ROCm reports ``cuda``, so only ``torch.version.hip`` can tell.

    The refusal is this development box's protection, and it is a refusal
    rather than a measurement on purpose: hipGraph capture core-dumped every
    gfx1034 probe, so there is nothing here to measure and fall back from.
    """
    with pytest.raises(RuntimeError, match="KOM-048") as excinfo:
        inference._refuse_ungraphable(_config("on"), torch.device("cuda"), "6.2.0")
    message = str(excinfo.value)
    assert "hip" in message
    assert "'6.2.0'" in message


def test_a_cpu_model_is_refused_by_name() -> None:
    """A bench that thinks graphs are on must not silently time the host."""
    with pytest.raises(RuntimeError, match="KOM-048") as excinfo:
        inference._refuse_ungraphable(_config("on"), torch.device("cpu"), None)
    assert "cpu" in str(excinfo.value).lower()


def test_the_evaluator_refuses_a_cpu_model_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal reaches the constructor, not just the helper.

    ``hip`` is forced to ``None`` so this asserts the *device* refusal on the
    ROCm box too -- otherwise the box's own build would refuse first and this
    test would be checking something else while still passing.
    """
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="KOM-048"):
        AsyncEvaluator(_tiny_net(), _config("on"))


def test_the_evaluator_refuses_a_rocm_build_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same, for the ROCm branch, forced on so a CUDA pod runs it too."""
    monkeypatch.setattr(torch.version, "hip", "6.2.0")
    with pytest.raises(RuntimeError, match="KOM-048") as excinfo:
        AsyncEvaluator(_tiny_net(), _config("on"))
    assert "hip" in str(excinfo.value)


def test_the_evaluator_refuses_autocast_plus_graphs_at_construction() -> None:
    """No monkeypatching needed: the config-only refusal fires on any box."""
    with pytest.raises(RuntimeError, match="autocast"):
        AsyncEvaluator(_tiny_net(), _config("on", forward_dtype="f16_autocast"))


def test_a_refused_evaluator_leaves_the_model_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal precedes every mutation, including KOM-039's conversion.

    An evaluator that cannot graph must not leave a half-converted, silently
    eval-mode model behind it for the caller to trip over: refusing is only
    honest if nothing happened.
    """
    monkeypatch.setattr(torch.version, "hip", None)
    net = _tiny_net()
    net.train()
    with pytest.raises(RuntimeError, match="KOM-048"):
        AsyncEvaluator(net, _config("on", forward_dtype="f16"))
    assert net.training, "the refused constructor still called .eval()"
    for name, param in net.named_parameters():
        assert param.dtype == torch.float32, f"{name} was converted anyway"


def test_the_off_path_never_consults_hip_or_the_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The knob off is today's evaluator: no refusal, on any build, anywhere."""
    monkeypatch.setattr(torch.version, "hip", "6.2.0")
    evaluator = AsyncEvaluator(_tiny_net(), _config("off"))
    assert evaluator.config.forward_graph == "off"
    obs, legal = _batch(3)
    assert evaluator.evaluate(obs, legal).wdl.shape == (3, 3)


def test_the_serial_evaluator_never_graphs_and_never_refuses() -> None:
    """One capture site, and this is not it (KOM-048, single-integration rule).

    :class:`InProcessEvaluator` is the serial reference *and* the instance
    :class:`AsyncEvaluator` holds as its CPU fallback, so a capture here would
    mean two captures of one model. It therefore ignores the knob rather than
    refusing it -- and a host-resident graphs-on config constructs and
    evaluates exactly as it always did.
    """
    obs, legal = _batch(3)
    served = InProcessEvaluator(_tiny_net(), _config("on")).evaluate(obs, legal)
    reference = InProcessEvaluator(_tiny_net(), _config("off")).evaluate(obs, legal)
    assert np.array_equal(
        np.asarray(served.policy_logits), np.asarray(reference.policy_logits)
    )
    assert np.array_equal(np.asarray(served.wdl), np.asarray(reference.wdl))


# --------------------------------------------------------------------------
# T3 -- serial_debug purity
# --------------------------------------------------------------------------


class _CapturedConfig(Exception):
    """Raised by the ``serial_debug`` spy once it has recorded the config.

    The spy stops the call there on purpose: what is under test is the config
    ``serial_debug`` hands down, and playing a game to find out would be a
    slower test of something else.
    """


def _selfplay_config() -> SelfPlayConfig:
    """A two-simulation config: the loop's structure without its cost."""
    return SelfPlayConfig(
        workers=1,
        process_batch_size=1,
        gumbel_m_full=2,
        gumbel_n_full=2,
        gumbel_m_fast=2,
        gumbel_n_fast=2,
        prob_fast=0.5,
        draw_score=0.2,
    )


def test_serial_debug_forces_graphs_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A graphs-on production config still serial-debugs, on a CPU box.

    Two things at once, and both are the property KOM-048 pins rather than
    hopes for: ``--serial-debug`` exists to reproduce a production game on
    whatever machine the operator is at (usually one with no GPU, where a
    graphs-on evaluator refuses by design), and this path's whole claim is
    bit-identical replay -- which a forward captured on particular hardware is
    exactly the sort of thing to weaken.
    """
    seen: list[InferenceConfig] = []

    def spy(model: torch.nn.Module, config: InferenceConfig) -> InProcessEvaluator:
        """Record the config, then abort before a game is played."""
        seen.append(config)
        raise _CapturedConfig

    monkeypatch.setattr(orchestrate, "InProcessEvaluator", spy)
    production = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=MAX_BATCH,
        forward_graph="on",
        forward_dtype="f16",
    )
    with pytest.raises(_CapturedConfig):
        orchestrate.serial_debug(
            graph_stub_factory,
            run_seed=48,
            worker_id=0,
            incarnation=0,
            game_index=0,
            shard_dir=str(tmp_path),
            selfplay=_selfplay_config(),
            inference=production,
        )

    assert len(seen) == 1
    assert seen[0].forward_graph == "off"
    # Only that field moved: serial_debug reproduces the production config, it
    # does not sanitise it. A dtype quietly reset here would make every serial
    # reproduction an argument about which forward it reproduced.
    assert seen[0].forward_dtype == "f16"
    assert seen[0].max_batch == MAX_BATCH
    assert production.forward_graph == "on", "the caller's config was mutated"


# --------------------------------------------------------------------------
# T4 -- the row-independence argument that makes pad rows safe
# --------------------------------------------------------------------------


def _padded(
    obs: npt.NDArray[np.float32], width: int, fill: float
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """`obs` in the first rows of a `width`-row batch, the rest set to `fill`."""
    padded = np.full(
        (width, IN_PLANES, BOARD, BOARD), np.float32(fill), dtype=np.float32
    )
    padded[: obs.shape[0]] = obs
    return padded, np.ones((width, NUM_ACTIONS), dtype=np.bool_)


def _evaluate(
    net: KomugiNet,
    batch: tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]],
) -> NetEval:
    """Evaluate one prepared batch through the plain serial evaluator."""
    return InProcessEvaluator(net, _config()).evaluate(batch[0], batch[1])


def test_the_tail_cannot_reach_the_rows_at_a_fixed_width() -> None:
    """The exact invariant a captured, fixed-shape graph relies on.

    A replay always runs at the captured width, so the hazard is never "a
    different batch size": it is *the same width, a different tail* -- the
    previous, wider batch's rows still sitting in the static input. This pins
    that the answer for rows ``[:N]`` does not depend on a single bit of what
    follows them, and pins it **bitwise**, which is the only strength at which
    the claim is worth anything.

    Zeroing the pad every launch is therefore not what makes the rows correct.
    It is what keeps the *pad's own* outputs finite and deterministic, so that
    KOM-039's tripwire -- which reads the whole ``(N, 3)`` wdl -- stays a
    statement about the batch rather than about its padding.
    """
    net = _tiny_net()
    obs, _ = _batch(5)
    quiet = _evaluate(net, _padded(obs, MAX_BATCH, 0.0))
    loud = _evaluate(net, _padded(obs, MAX_BATCH, -37.5))
    stale, _ = _batch(MAX_BATCH, seed=1234)
    mixed_obs, mixed_legal = _padded(obs, MAX_BATCH, 0.0)
    mixed_obs[obs.shape[0] :] = stale[obs.shape[0] :]
    mixed = _evaluate(net, (mixed_obs, mixed_legal))

    rows = obs.shape[0]
    for other, label in ((loud, "a constant tail"), (mixed, "a stale-batch tail")):
        for field in ("policy_logits", "wdl", "moves_left"):
            want = np.asarray(getattr(quiet, field))[:rows]
            got = np.asarray(getattr(other, field))[:rows]
            assert np.array_equal(want, got), (
                f"{label} changed the served rows' {field}: the network is not "
                f"row-independent in eval mode, and a captured graph's pad "
                f"would be corrupting real evaluations"
            )


def test_the_rows_survive_being_padded_at_all() -> None:
    """The same rows alone, and inside a wider batch, agree.

    Deliberately a tolerance rather than a bitwise pin, and the distinction is
    this repository's own KOM-104 GEMM lesson: changing the batch *dimension*
    can change a linear layer's blocking, so equality across widths is not a
    promise anything here makes -- production games are promised only up to
    evaluator tolerance for exactly this reason. The bitwise claim lives in the
    test above, at a fixed width, which is the shape a replay actually runs at.
    """
    net = _tiny_net()
    obs, legal = _batch(5)
    alone = _evaluate(net, (obs, legal))
    padded = _evaluate(net, _padded(obs, MAX_BATCH, 0.0))
    rows = obs.shape[0]
    for field in ("policy_logits", "wdl", "moves_left"):
        want = np.asarray(getattr(alone, field), dtype=np.float64)
        got = np.asarray(getattr(padded, field), dtype=np.float64)[:rows]
        assert np.allclose(want, got, rtol=1e-5, atol=1e-4), field


# --------------------------------------------------------------------------
# T6 -- the exclusivity handshake, tested where it lives
# --------------------------------------------------------------------------


def test_a_fresh_token_owes_nothing() -> None:
    """Nothing has been replayed, so nothing is waiting to be read."""
    assert not inference._ReplayToken().outstanding


def test_marking_claims_the_buffers_and_consuming_releases_them() -> None:
    """The whole protocol, in the order a launch and a result run it."""
    token = inference._ReplayToken()
    token.mark()
    assert token.outstanding
    token.consume()
    assert not token.outstanding
    token.mark()
    assert token.outstanding


def test_a_second_mark_without_a_consume_is_refused_by_name() -> None:
    """The case a CUDA event structurally cannot catch, which is why this exists.

    ``event.query()`` answers "have N's kernels retired?" -- true long before
    anyone has *read* N's outputs, and therefore true in precisely the
    situation that is dangerous: a caller holding an unread handle and
    launching again. The graph has one set of static output buffers, so the
    second replay would overwrite an answer nobody collected.
    """
    token = inference._ReplayToken()
    token.mark()
    with pytest.raises(RuntimeError, match="KOM-048") as excinfo:
        token.mark()
    message = str(excinfo.value)
    assert "result()" in message
    assert token.outstanding, "a refused mark must not disturb the outstanding one"


def test_consuming_twice_is_not_an_error() -> None:
    """A handle read twice is a caller oddity, not a protocol violation.

    The second read returns the bytes the first one copied to the host, so
    there is nothing to refuse -- and refusing would turn a harmless habit into
    a dead run.
    """
    token = inference._ReplayToken()
    token.mark()
    token.consume()
    token.consume()
    assert not token.outstanding


class _StubEvent:
    """A ``_CudaEventLike`` that needs no CUDA context to exist."""

    def record(self) -> None:
        """Nothing is enqueued; this handle's tensors are already on the host."""

    def query(self) -> bool:
        """Always finished."""
        return True


def test_the_handle_consumes_the_token_after_its_copy() -> None:
    """The handle is the second half of the handshake, and does its half on CPU.

    Constructed directly with host tensors: what is under test is the *order*
    -- the release follows the D2H -- and that is a property of this method,
    not of the device the tensors happen to live on.
    """
    token = inference._ReplayToken()
    token.mark()
    handle = inference._CudaHandle(
        _StubEvent(),
        torch.zeros(2, NUM_ACTIONS),
        torch.full((2, 3), 1.0 / 3.0),
        torch.zeros(2),
        torch.zeros(2, IN_PLANES, BOARD, BOARD),
        None,
        None,
        "f32",
        token,
    )
    evaluation = handle.result()
    assert evaluation.wdl.shape == (2, 3)
    assert not token.outstanding


def test_a_dense_handle_with_no_token_still_works() -> None:
    """The knob off means the handle carries no token and consumes nothing.

    Which is the whole compatibility claim for :class:`_CudaHandle`: KOM-048
    added an optional field, and every existing construction of this class --
    all of them positional -- keeps meaning what it meant.
    """
    handle = inference._CudaHandle(
        _StubEvent(),
        torch.zeros(2, NUM_ACTIONS),
        torch.full((2, 3), 1.0 / 3.0),
        torch.zeros(2),
        torch.zeros(2, IN_PLANES, BOARD, BOARD),
    )
    assert handle.query()
    assert handle.result().policy_logits.shape == (2, NUM_ACTIONS)


# --------------------------------------------------------------------------
# T5 -- the capture itself. CUDA pod only; skipped in the gate and on ROCm.
# --------------------------------------------------------------------------


def _cuda_evaluators() -> tuple[AsyncEvaluator, AsyncEvaluator]:
    """An eager evaluator and a graphed one over identically-seeded nets."""
    device = torch.device("cuda")
    eager = AsyncEvaluator(_tiny_net().to(device), _config("off"))
    graphed = AsyncEvaluator(_tiny_net().to(device), _config("on"))
    return eager, graphed


@_SKIP_NO_GPU
def test_a_replayed_forward_agrees_with_the_eager_one() -> None:
    """Knob on vs off, same weights, same batch, at the captured width."""
    eager, graphed = _cuda_evaluators()
    obs, legal = _batch(MAX_BATCH)
    want = eager.evaluate(obs, legal)
    got = graphed.evaluate(obs, legal)
    for field in ("policy_logits", "wdl", "moves_left"):
        assert np.allclose(
            np.asarray(getattr(want, field), dtype=np.float64),
            np.asarray(getattr(got, field), dtype=np.float64),
            rtol=1e-5,
            atol=1e-5,
        ), field


@_SKIP_NO_GPU
def test_a_descending_n_sweep_never_serves_a_stale_tail() -> None:
    """Widest batch FIRST, then narrower ones -- the only order that can fail.

    An ascending sweep leaves the static input's tail holding nothing but the
    zeros it was allocated with, so it would pass against an implementation
    that never zeroed the pad at all. Descending is what puts a previous, wider
    batch's distinctive rows in the tail before a short batch replays over it.
    """
    eager, graphed = _cuda_evaluators()
    for rows in (MAX_BATCH, 7, 1, 5, MAX_BATCH, 2):
        obs, legal = _batch(rows, seed=rows)
        want = eager.evaluate(obs, legal)
        got = graphed.evaluate(obs, legal)
        assert np.allclose(
            np.asarray(got.wdl, dtype=np.float64),
            np.asarray(want.wdl, dtype=np.float64),
            rtol=1e-5,
            atol=1e-5,
        ), f"N={rows} disagreed with the eager forward"
        assert np.all(np.isfinite(np.asarray(got.policy_logits)))


@_SKIP_NO_GPU
def test_repeated_replays_of_one_batch_are_bitwise_identical() -> None:
    """A replay is a replay: same static input, same kernels, same bytes."""
    _, graphed = _cuda_evaluators()
    obs, legal = _batch(MAX_BATCH)
    first = graphed.evaluate(obs, legal)
    second = graphed.evaluate(obs, legal)
    for field in ("policy_logits", "wdl", "moves_left"):
        assert np.array_equal(
            np.asarray(getattr(first, field)), np.asarray(getattr(second, field))
        ), field


@_SKIP_NO_GPU
def test_launching_twice_without_reading_is_refused() -> None:
    """The handshake, against real hardware and through the public launch path."""
    _, graphed = _cuda_evaluators()
    obs, legal = _batch(4)
    handle = graphed.launch(obs, legal)
    with pytest.raises(RuntimeError, match="KOM-048"):
        graphed.launch(obs, legal)
    handle.result()
    # And the token really was released, not merely reported as released.
    graphed.launch(obs, legal).result()


@_SKIP_NO_GPU
def test_a_batch_wider_than_the_capture_is_refused_by_name() -> None:
    """A fixed-shape capture cannot be re-shaped, and says so.

    Loud here because the alternative is a cryptic slice error from inside a
    copy, at the one moment an operator most needs to be told ``max_batch``.
    """
    _, graphed = _cuda_evaluators()
    obs, legal = _batch(MAX_BATCH + 1)
    with pytest.raises(ValueError, match="max_batch"):
        graphed.launch(obs, legal)


@_SKIP_NO_GPU
def test_moving_the_model_after_capture_is_refused() -> None:
    """The pin, checked before the host fallback could swallow it.

    Without the ordering this is the quietest bug in the ticket: the moved
    model takes ``launch``'s ``device.type != "cuda"`` branch and is served
    correctly -- and eagerly -- forever after.
    """
    device = torch.device("cuda")
    net = _tiny_net().to(device)
    graphed = AsyncEvaluator(net, _config("on"))
    obs, legal = _batch(4)
    graphed.evaluate(obs, legal)
    net.to("cpu")
    with pytest.raises(RuntimeError, match="KOM-048"):
        graphed.launch(obs, legal)


@_SKIP_NO_GPU
@pytest.mark.parametrize("forward_dtype", ["f32", "f16"])
def test_sparse_and_graphs_compose(forward_dtype: str) -> None:
    """The KOM-038 gather stays eager and still sees the replayed rows.

    The sparse x graphs cell: the pack runs after the replay, inside the same
    stream, on views into the static outputs -- so a capture that had swallowed
    the gather, or outputs that were not really the replay's, would show up
    here as a mismatch against the eager packer.

    The fp16 cell gets an fp16 tolerance, and not out of politeness: capture
    warms up before it records, so cuDNN may have chosen a different algorithm
    for the replayed convolution than for the eager one, and at half width that
    is worth thousandths. What is being asserted here is *which values landed
    in which lanes*, which no tolerance can launder.
    """
    device = torch.device("cuda")
    sparse = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=MAX_BATCH,
        sparse_policy=True,
        forward_graph="on",
        forward_dtype=forward_dtype,
    )
    dense = InferenceConfig(
        in_planes=IN_PLANES,
        num_actions=NUM_ACTIONS,
        max_batch=MAX_BATCH,
        forward_graph="off",
        forward_dtype=forward_dtype,
    )
    obs, _ = _batch(6)
    legal = np.zeros((6, NUM_ACTIONS), dtype=np.bool_)
    legal[:, ::3] = True

    handle = AsyncEvaluator(_tiny_net().to(device), sparse).launch(obs, legal)
    assert isinstance(handle, inference._PacksPolicy)
    packed = handle.packed_result()
    reference = AsyncEvaluator(_tiny_net().to(device), dense).evaluate(obs, legal)

    atol = 1e-5 if forward_dtype == "f32" else 2e-2
    for row in range(obs.shape[0]):
        count = int(packed.counts[row])
        want = np.asarray(reference.policy_logits)[row][legal[row]]
        got = np.asarray(packed.values)[row, :count]
        assert count == int(legal[row].sum())
        assert np.allclose(
            want.astype(np.float64), got.astype(np.float64), rtol=1e-3, atol=atol
        ), f"row {row}"


# --------------------------------------------------------------------------
# the 2026-08-30 impl review's killers (G1, G3, G4, G5, G7)
# --------------------------------------------------------------------------


class _FakeStream:
    """A stream that records who waited on whom, and nothing else."""

    def __init__(self, name: str, log: list[tuple[str, str]]) -> None:
        self.name = name
        self._log = log

    def wait_stream(self, other: "_FakeStream") -> None:
        self._log.append((self.name, other.name))


class _FakeGraph:
    """A graph that counts replays and records the inference-mode state.

    The mode recording is the reproducing pin for the 2026-08-30 pod finding:
    the captured outputs are inference tensors, so a replay OUTSIDE
    inference_mode is refused by torch at the first real launch ("Inplace
    update to inference tensor outside InferenceMode") -- a context bug no
    real-tensor CPU test can see, because the fake replay does nothing. The
    fake records what a real replay would have been subject to.
    """

    def __init__(self) -> None:
        self.replays = 0
        self.replayed_in_inference_mode: list[bool] = []

    def replay(self) -> None:
        self.replays += 1
        self.replayed_in_inference_mode.append(torch.is_inference_mode_enabled())


class _CountingNet(nn.Module):
    """The tiny net, with every forward counted (warmup-discipline pin)."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _tiny_net()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> NetOutput:
        self.calls += 1
        result: NetOutput = self.inner(x)
        return result


def _fake_runner(
    monkeypatch: pytest.MonkeyPatch, model: nn.Module | None = None
) -> tuple[inference._GraphRunner, list[tuple[str, str]], _FakeGraph]:
    """A :class:`_GraphRunner` on the CPU, through the module's alias seams.

    The five ``_new_cuda_*``/``_cuda_*`` aliases exist exactly so that the
    capture recipe is drivable without a CUDA device (impl review G3): the
    static buffers, the pad zeroing, the N bound, the warmup discipline and
    the train-mode refusal are all plain tensor logic, and this is what lets
    the GPU-less gate pin them instead of trusting the pod to.
    """
    log: list[tuple[str, str]] = []
    current = _FakeStream("current", log)
    graph = _FakeGraph()
    monkeypatch.setattr(inference, "_new_cuda_stream", lambda: _FakeStream("side", log))
    monkeypatch.setattr(inference, "_current_cuda_stream", lambda: current)
    monkeypatch.setattr(inference, "_cuda_stream_scope", lambda stream: nullcontext())
    monkeypatch.setattr(inference, "_new_cuda_graph", lambda: graph)
    monkeypatch.setattr(inference, "_cuda_graph_capture", lambda g: nullcontext())
    runner = inference._GraphRunner(
        model if model is not None else _tiny_net(),
        _config(forward_graph="on"),
        torch.device("cpu"),
    )
    return runner, log, graph


def test_the_runner_refuses_a_batch_wider_than_the_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N > max_batch is a loud ValueError, not a cryptic slice error.

    ``_validate_batch`` never bounds N by ``max_batch`` -- the runner's check
    is the ONLY guard (impl review G3/M6), so it must exist and it must name
    the captured shape.
    """
    runner, _, _ = _fake_runner(monkeypatch)
    too_wide = torch.zeros(MAX_BATCH + 1, IN_PLANES, BOARD, BOARD)
    with pytest.raises(ValueError, match="captured at a fixed shape"):
        runner.run(too_wide, MAX_BATCH + 1)
    assert not runner.token.outstanding, "a refused launch must not mark the token"


def test_the_runner_zeroes_the_pad_every_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wide batch's rows do not survive in a later, narrower batch's pad.

    White-box on purpose (impl review G4): every public reader slices ``[:N]``
    and the net is row-independent, so a stale pad is observationally
    invisible from outside -- this direct look at the static buffer is the
    only assertion, on any hardware, that fails when the per-launch
    ``zero_()`` is dropped.
    """
    runner, _, _ = _fake_runner(monkeypatch)
    wide = torch.full((MAX_BATCH, IN_PLANES, BOARD, BOARD), 7.0)
    runner.run(wide, MAX_BATCH)
    runner.token.consume()

    narrow = torch.full((3, IN_PLANES, BOARD, BOARD), 2.0)
    runner.run(narrow, 3)
    static = runner._static_in
    assert torch.equal(static[:3], narrow)
    assert bool((static[3:] == 0).all()), (
        "the pad tail still holds the previous, wider batch's rows: "
        "run() stopped zeroing the pad"
    )


def test_a_train_mode_model_refuses_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture in train mode would bake BN batch statistics into every replay.

    Unreachable through ``AsyncEvaluator`` today (its serial evaluator calls
    ``model.eval()`` first), which is exactly why the refusal needs its own
    pin (impl review G5): it exists for the constructor reordering that would
    make it reachable, and that reordering would not add the test.
    """
    net = _CountingNet()
    net.train()
    with pytest.raises(RuntimeError, match="train mode"):
        _fake_runner(monkeypatch, model=net)


def test_warmup_runs_three_times_on_a_side_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capture recipe: 3 warmup forwards, side stream joined both ways.

    The wait order is the recipe's load-bearing part -- side waits on current
    before warming, current waits on side before capturing -- and the capture
    itself is the one extra forward.
    """
    net = _CountingNet()
    net.eval()
    runner, log, graph = _fake_runner(monkeypatch, model=net)
    assert net.calls == inference._GRAPH_WARMUP_ITERS + 1
    assert log == [("side", "current"), ("current", "side")]
    assert graph.replays == 0

    runner.run(torch.zeros(2, IN_PLANES, BOARD, BOARD), 2)
    assert graph.replays == 1
    assert graph.replayed_in_inference_mode == [True], (
        "the replay must run under inference_mode: the captured outputs are "
        "inference tensors, and torch refuses the replay's in-place update "
        "of them outside the mode (2026-08-30 pod finding)"
    )


def test_packed_result_consumes_the_token() -> None:
    """The sparse sync point releases the handshake too (impl review G1/M5).

    Production sparse+graphs serving reads every batch through
    ``packed_result``; if only ``result`` consumed, the server would wedge on
    its second batch with a misleading unread-answer error.
    """
    token = inference._ReplayToken()
    token.mark()
    handle = inference._CudaHandle(
        _StubEvent(),
        torch.zeros(2, 5),
        torch.full((2, 3), 1.0 / 3.0),
        torch.zeros(2),
        torch.zeros(2, IN_PLANES, BOARD, BOARD),
        np.asarray([5, 3], dtype=np.int64),
        torch.zeros(2, 5, dtype=torch.int64),
        "f32",
        token,
    )
    packed = handle.packed_result()
    assert packed.wdl.shape == (2, 3)
    assert not token.outstanding


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip is not None),
    reason="needs a ROCm build with a visible device (the W6400 box)",
)
def test_a_rocm_cuda_resident_model_is_refused_end_to_end() -> None:
    """The D1 refusal, proven on the exact hardware it protects.

    ROCm reports ``device.type == "cuda"``, so this is the one environment
    where a device-first check ordering would sail past the gate and into a
    hipGraph segfault. The refusal must fire at construction, before any
    capture is attempted (impl review G7).
    """
    net = _tiny_net().to("cuda")
    with pytest.raises(RuntimeError, match="hip"):
        AsyncEvaluator(net, _config(forward_graph="on"))
