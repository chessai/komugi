"""The hash gate, at gate scale: it must be able to fail, and it must not lie.

KOM-022 §3a/§3b, acceptances #7, #8, #9, #31, #41 and #43. Every test here bar
one runs the **boundary evaluation the campaign runs at each iteration
boundary** over a planted root -- the same call, the same journal, the same
:func:`~komugi.campaign.boundary.enforce`. What is planted is a *published
checkpoint series*, which is what the gate reads; see ``kom022_rig`` for why
that is the design's own construction rather than a shortcut, and for why no
``lr = 0.0`` trainer appears anywhere (both of this repo's configs refuse to
build one, and acceptance #47 keeps it that way).

The two halves are deliberately two-sided:

* **#41** says the gate *fires* when the digest does not move. A digest that
  passed acceptance #27 and failed this would be a gate nobody had wired up.
* **#43** says the BatchNorm series moves on the *same root, the same
  iterations*, while the parameter series is flat -- which is what makes the
  independence of the two series observable rather than asserted. Rev 2
  conflated them, and a campaign producing this series reported its headline
  liveness invariant as enforced for its whole length.

The exception is **#8**, the aliasing check, which reads *nets* rather than a
published series: ``KL == 0.0`` on every probe position is a statement about
two live modules, so it is measured on them directly and then composed with the
escalation the loop applies to it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import kom022_rig as rig
import pytest
import torch

from komugi.campaign import boundary
from komugi.campaign.factory import (
    IN_PLANES,
    NET_ENV,
    NUM_ACTIONS,
    SMALL_NET,
    new_net,
)
from komugi.campaign.loop import checkpoint_dir, ema_dir, journal_path
from komugi.campaign.pause import CampaignPaused, Mute, read_pause
from komugi.campaign.pinned import (
    CROSS_ROLE,
    EMA_ALIAS,
    FINGERPRINT_SERIES,
    HASH_GATE,
)
from komugi.campaign.probe import ProbeSet, policy_kl
from komugi.telemetry.journal import decode_line, write_record
from komugi.telemetry.liveness import LivenessVerdict, self_play_net_advanced
from komugi.telemetry.records import (
    EMA,
    NET_ROLES,
    SELF_PLAY,
    TRAIN,
    IterationMetrics,
    NetFingerprint,
)
from komugi.train.checkpoint import list_checkpoints, save_checkpoint
from komugi.train.digest import canonical_digest, parameter_keys
from komugi.train.ema import ema_meta, ema_state, publish_ema, reconcile_ema

ITERATIONS = 4
"""Acceptance #41's ``N``. Two would suffice for the run-length gate; four
makes the flatness of the whole series visible and gives #31 a ``3N - 1`` of
eleven rather than of five."""


@pytest.fixture
def param_keys(monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    """The small net's parameter key set, with ``NET_ENV`` set for the process."""
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    return parameter_keys(new_net())


def _small_net(monkeypatch: pytest.MonkeyPatch) -> torch.nn.Module:
    """Return a freshly built small net, with ``NET_ENV`` set for this process."""
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    return new_net()


def _probe_positions(rows: int = 2) -> ProbeSet:
    """A probe set at the **inference seam's** shapes, with a few legal moves.

    ``in_planes`` and ``num_actions`` stay at the production values even for
    the small net: they are the seam's shapes, and a probe set that narrowed
    them would measure a net the real workers cannot talk to.
    """
    legal = torch.zeros(rows, NUM_ACTIONS, dtype=torch.bool)
    legal[:, :8] = True
    return ProbeSet(
        observation=torch.zeros(rows, IN_PLANES, 9, 9), legal=legal
    )


def _shifted(net: torch.nn.Module, delta: float) -> dict[str, torch.Tensor]:
    """Return `net`'s state with every floating-point entry moved by `delta`.

    ``num_batches_tracked`` is left alone: it is an integer counter, and
    :func:`~komugi.train.ema.ema_state` copies it from ``theta`` rather than
    interpolating it, so shifting it here would test nothing and change the
    dtype.
    """
    state: dict[str, torch.Tensor] = {}
    for key, value in net.state_dict().items():
        tensor = value.detach().clone()
        state[key] = tensor + delta if tensor.is_floating_point() else tensor
    return state


# ---------------------------------------------------------------------------
# acceptance #41 -- a flat parameter series must FAIL the hash gate
# ---------------------------------------------------------------------------


def test_a_flat_parameter_series_fails_the_hash_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, param_keys: tuple[str, ...]
) -> None:
    """Acceptance #41: the falsifiable half of #27, planted at gate scale.

    A four-iteration root whose ``<root>/ckpt`` series has the **same parameter
    tensors byte for byte in all four files**, ``running_*`` perturbed and
    ``num_batches_tracked`` advanced at each ``i``, plus the ``<root>/ema``
    series §2's arithmetic produces from it (with the parameters constant, the
    EMA's parameters are constant too, so the ``ema`` and ``self_play`` series
    are flat by the same construction).

    Then: the ``train`` series is flat, ``self_play_net_advanced`` returns a
    **failing** verdict, ``<root>/PAUSED`` exists, and its ``tripwire`` field
    names the hash gate.

    **Red on rev 2's field set**: under it the published digests differed every
    iteration, the series was not flat, and the gate returned ``_passed``
    forever. It uses no ``lr``: ``CampaignConfig(lr=0.0)`` raises, and relaxing
    that would not help because the loop re-validates through ``StepConfig`` in
    a file this ticket puts out of scope. The *series*, not the trainer, is
    what the gate reads.
    """
    net = _small_net(monkeypatch)
    states = rig.flat_parameter_states(net, ITERATIONS)
    rig.plant(tmp_path, states, param_keys)

    fingerprints = rig.recorded_fingerprints(tmp_path)
    train_digests = rig.digests_for(fingerprints, TRAIN)
    assert len(train_digests) == ITERATIONS
    assert len(set(train_digests)) == 1, (
        "the parameters-only digest moved across a series whose parameters are "
        "byte-identical: the hash gate cannot fail"
    )
    # The EMA of a flat-parameter series is flat, so all three series are.
    assert len(set(rig.digests_for(fingerprints, EMA))) == 1
    assert len(set(rig.digests_for(fingerprints, SELF_PLAY))) == 1

    # The run-length walk deduplicates by *iteration number*, not by digest, so
    # a flat series of two or more iterations does not take the `len(unique) <
    # 2` vacuous pass.
    verdict = self_play_net_advanced(fingerprints, role=TRAIN)
    assert not verdict.ok
    assert "train net" in verdict.reason
    assert f"unchanged across {ITERATIONS} consecutive iterations" in verdict.reason

    with pytest.raises(CampaignPaused) as raised:
        rig.gate(tmp_path, ITERATIONS - 1)

    pause = read_pause(tmp_path)
    assert pause is not None, "the durable marker must be on disk before the raise"
    assert pause.tripwire == HASH_GATE
    assert pause.pause_class == "tripwire"
    assert pause.paused_at_iteration == ITERATIONS - 1
    assert HASH_GATE in str(raised.value)


def test_the_flat_series_pauses_at_the_first_boundary_that_can_see_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, param_keys: tuple[str, ...]
) -> None:
    """Two iterations are enough; the gate does not wait for a run to be wasted.

    A freeze is a statement about *change*, so one iteration says nothing and
    the gate passes vacuously on it. Two iterations of one digest is already
    the v1 signature, and iteration 1 is the boundary where the six wasted
    iterations would have been stopped at one.
    """
    net = _small_net(monkeypatch)
    rig.plant(
        tmp_path,
        rig.flat_parameter_states(net, 2),
        param_keys,
    )
    fingerprints = rig.recorded_fingerprints(tmp_path)

    # One iteration alone is not evidence of anything.
    single = [entry for entry in fingerprints if entry.iteration == 0]
    assert self_play_net_advanced(single, role=TRAIN).ok

    verdict = self_play_net_advanced(fingerprints, role=TRAIN)
    assert not verdict.ok
    assert "unchanged across 2 consecutive iterations" in verdict.reason

    with pytest.raises(CampaignPaused):
        rig.gate(tmp_path, 1)
    assert read_pause(tmp_path) is not None


def test_a_healthy_series_passes_every_structural_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, param_keys: tuple[str, ...]
) -> None:
    """The false-alarm side: a gate that fires on a healthy run gets switched off.

    Same rig, parameters drifting. Every gate passes, nothing is written, and
    the cross-role invariant holds at every ``i >= 1``.
    """
    net = _small_net(monkeypatch)
    rig.plant(
        tmp_path,
        rig.advancing_states(net, ITERATIONS),
        param_keys,
    )

    for iteration in range(ITERATIONS):
        outcome = rig.gate(tmp_path, iteration)
        assert outcome.tripwire is None, outcome.detail
        assert all(verdict.ok for verdict in outcome.verdicts), [
            verdict.reason for verdict in outcome.verdicts if not verdict.ok
        ]

    assert read_pause(tmp_path) is None
    assert len(set(rig.digests_for(rig.recorded_fingerprints(tmp_path), TRAIN))) == (
        ITERATIONS
    )


# ---------------------------------------------------------------------------
# acceptance #43 -- the BatchNorm series is separate, recorded, and not a gate
# ---------------------------------------------------------------------------


def test_bn_drift_is_a_separate_series_and_not_a_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, param_keys: tuple[str, ...]
) -> None:
    """Acceptance #43: the two series are independent **on one root**.

    Over #41's flat-parameter series, with no trainer in the loop:

    * ``bn_digest_advanced == 1.0`` on every iteration ``i >= 1`` **while** the
      ``train`` fingerprint series is flat -- the per-iteration ``running_*``
      perturbation is what makes the independence observable at all;
    * ``bn_digest_lo`` and ``bn_digest_advanced`` are present and finite on
      every boundary's extras;
    * **no** ``NetFingerprint`` record carries a BatchNorm role, so #31's
      ``3N - 1`` count is unchanged by the second series existing;
    * ``self_play_net_advanced`` is **never** called with the BatchNorm series,
      asserted by recording the calls rather than by inspection -- being handed
      it is precisely the defect the split exists to fix.
    """
    net = _small_net(monkeypatch)

    calls: list[tuple[str, tuple[str, ...]]] = []
    # The real gate is taken from the module that *defines* it: `boundary`
    # imports the name rather than re-exporting it, so reading it back off
    # `boundary` is an implicit re-export strict mypy refuses. The patch below
    # still targets `boundary`'s namespace, which is the one the call site
    # resolves through.
    real = self_play_net_advanced

    def recorder(
        fingerprints: Sequence[NetFingerprint],
        min_iterations_lag: int = 0,
        role: str = SELF_PLAY,
    ) -> LivenessVerdict:
        """Record which digests each gate call was handed, then delegate."""
        calls.append((role, tuple(f.sha256 for f in fingerprints)))
        return real(fingerprints, min_iterations_lag, role)

    monkeypatch.setattr(boundary, "self_play_net_advanced", recorder)

    planted = rig.plant(
        tmp_path,
        rig.flat_parameter_states(net, ITERATIONS),
        param_keys,
    )

    bn_digests = [entry.fingerprints.bn_sha256 for entry in planted]
    assert len(set(bn_digests)) == ITERATIONS, (
        "the BatchNorm series did not move, so this root cannot demonstrate "
        "the two series being independent"
    )

    for entry in planted:
        extras = entry.fingerprints.extras
        assert boundary.BN_DIGEST_LO in extras
        assert boundary.BN_DIGEST_ADVANCED in extras
        expected = 1.0
        assert extras[boundary.BN_DIGEST_ADVANCED] == expected, (
            f"iteration {entry.iteration} reported the BatchNorm series as "
            f"stalled while its running statistics moved"
        )

    fingerprints = rig.recorded_fingerprints(tmp_path)
    assert {f.role for f in fingerprints} <= NET_ROLES
    assert len(fingerprints) == 3 * ITERATIONS - 1, (
        "a fourth role crept into the fingerprint series, which would break "
        "the completeness count and put the BatchNorm digest one careless "
        "argument away from the run-length gate"
    )

    with pytest.raises(CampaignPaused):
        rig.gate(tmp_path, ITERATIONS - 1)

    assert calls, "the run-length gate was never called; the assertion is vacuous"
    handed = {digest for _role, digests in calls for digest in digests}
    assert handed.isdisjoint(set(bn_digests)), (
        "self_play_net_advanced was handed the BatchNorm series: that series "
        "advances whenever data flows, so a gate reading it can never fail"
    )
    assert {role for role, _ in calls} <= NET_ROLES


def test_a_stalled_bn_series_is_recorded_and_never_pauses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, param_keys: tuple[str, ...]
) -> None:
    """The contrapositive rev 2's field set could not express -- and it is a warning.

    ``bn_digest_advanced == 0.0`` while the ``train`` fingerprint advances
    means the trainer stepped without a train-mode forward: a rig or mocking
    artefact, not a production mode. It is recorded, and it stops nothing. A
    muted observation is worth more than a corrupted gate.
    """
    net = _small_net(monkeypatch)
    states = rig.advancing_states(net, 2)
    # Freeze the buffers by hand: parameters move, nothing else does.
    frozen = dict(states[0])
    for key, value in states[1].items():
        if key not in set(param_keys):
            frozen[key] = states[0][key]
        else:
            frozen[key] = value
    states[1] = frozen

    planted = rig.plant(tmp_path, states, param_keys)

    assert planted[1].fingerprints.bn_sha256 == planted[0].fingerprints.bn_sha256
    assert planted[1].fingerprints.extras[boundary.BN_DIGEST_ADVANCED] == 0.0

    outcome = rig.gate(tmp_path, 1)
    assert outcome.tripwire is None, (
        "a stalled BatchNorm series paused the campaign; it is wired to no gate"
    )


def _frozen_buffers(
    states: list[dict[str, torch.Tensor]], param_keys: tuple[str, ...]
) -> list[dict[str, torch.Tensor]]:
    """Return `states` with every non-parameter entry pinned to state 0's.

    Parameters keep drifting: the series under test is one where the trainer
    is learning and no data flowed through the net in ``train()`` mode, which
    is what ``bn_digest_advanced == 0.0`` is supposed to report.
    """
    params = set(param_keys)
    return [
        {
            key: value if key in params else states[0][key]
            for key, value in state.items()
        }
        for state in states
    ]


def test_the_bn_series_survives_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, param_keys: tuple[str, ...]
) -> None:
    """Review F6: the BatchNorm flag must not lie at the first boundary after a restart.

    ``previous_bn`` is re-initialised to ``None`` on every ``run_campaign``
    call, so a boundary that had nothing in memory to compare against reported
    ``bn_digest_advanced = 1.0`` -- *the buffers moved* -- whatever they did.
    That is one wrong reading per restart, and it lands on exactly the
    iteration a post-crash post-mortem reads first.

    Here the buffers are frozen across three iterations and the journal carries
    iterations 0 and 1's campaign records. Iteration 2 is then fingerprinted
    the way a **fresh process** does it -- ``previous_bn=None`` -- and must
    still report a stalled series, because the previous digest is on disk.

    Red without the fix: ``restarted.extras[BN_DIGEST_ADVANCED] == 0.0`` fires
    (it is ``1.0``), and ``last_bn_digest_lo`` does not exist at all.
    """
    net = _small_net(monkeypatch)
    states = _frozen_buffers(rig.advancing_states(net, 3), param_keys)
    planted = rig.plant(tmp_path, states[:2], param_keys)

    assert planted[1].fingerprints.bn_sha256 == planted[0].fingerprints.bn_sha256, (
        "the planted buffers are not frozen, so this root cannot show the flag "
        "lying about them"
    )
    assert planted[1].fingerprints.extras[boundary.BN_DIGEST_ADVANCED] == 0.0

    # The campaign's own records, as `_close_iteration` writes them: the extras
    # the boundary produced, carrying the marker that tells them from the
    # trainer's.
    journal = journal_path(tmp_path)
    for entry in planted:
        write_record(
            journal,
            IterationMetrics(
                iteration=entry.iteration,
                games=0,
                draw_rate=0.0,
                avg_game_len=0.0,
                policy_entropy=0.0,
                extras={
                    boundary.CAMPAIGN_MARKER: 1.0,
                    **entry.fingerprints.extras,
                },
            ),
        )

    assert boundary.last_bn_digest_lo(journal, 2) == planted[1].fingerprints.bn_lo
    assert boundary.last_bn_digest_lo(journal, 0) is None, (
        "iteration 0 has no predecessor; its flag is 1.0 by definition"
    )

    # The restart: iteration 2 publishes, and the boundary runs with nothing in
    # memory.
    checkpoints = checkpoint_dir(tmp_path)
    ema_directory = ema_dir(tmp_path)
    served = str(list_checkpoints(ema_directory)[0].path)
    save_checkpoint(
        checkpoints,
        3 * rig.STEPS_PER_ITERATION,
        states[2],
        {},
        {"iteration": 2, "role": "train"},
    )
    reconcile_ema(
        checkpoints,
        ema_directory,
        param_keys,
        steps_per_iteration=rig.STEPS_PER_ITERATION,
    )

    restarted = boundary.record_fingerprints(
        journal, 2, checkpoints, ema_directory, served, param_keys, None
    )

    assert restarted.bn_lo == planted[1].fingerprints.bn_lo
    assert restarted.extras[boundary.BN_DIGEST_ADVANCED] == 0.0, (
        "the first boundary after a restart reported the BatchNorm series as "
        "advancing while its buffers were frozen: previous_bn was not read "
        "back from the journal"
    )

    # And the flag still moves when the buffers do -- the fix must not pin it
    # to zero.
    moved = dict(states[2])
    for key, value in states[2].items():
        if key not in set(param_keys) and value.is_floating_point():
            moved[key] = value + rig.BUFFER_DRIFT
    save_checkpoint(
        checkpoints,
        4 * rig.STEPS_PER_ITERATION,
        moved,
        {},
        {"iteration": 3, "role": "train"},
    )
    reconcile_ema(
        checkpoints,
        ema_directory,
        param_keys,
        steps_per_iteration=rig.STEPS_PER_ITERATION,
    )
    advanced = boundary.record_fingerprints(
        journal, 3, checkpoints, ema_directory, served, param_keys, restarted.bn_lo
    )
    assert advanced.extras[boundary.BN_DIGEST_ADVANCED] == 1.0


# ---------------------------------------------------------------------------
# acceptance #8 -- an aliased EMA is an exact zero, and it is a pause
# ---------------------------------------------------------------------------


def test_aliased_ema_pauses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance #8: ``KL == 0.0`` on every probe position pauses, unmutably.

    §3b guard 3. An EMA update that bound ``theta_i`` **by reference** instead
    of copying it is not a smoothed net -- it *is* the trainer's net -- and two
    handles on one set of weights agree exactly on every position. That check
    needs no model of what "close" means, which is why it fires on the first
    occurrence and why no ``ACK`` may mute it.

    Three halves, in the order the loop composes them:

    * the **signal**: an aliased pair reports ``zeros == len(probe)``;
    * the **false alarm it must not raise**: a genuinely smoothed EMA --
      :func:`~komugi.train.ema.ema_state`'s own arithmetic over a previous EMA
      that differs from ``theta`` -- does not, so a gate that could not tell
      them apart would pause every healthy campaign at iteration 2;
    * the **escalation**: the signal arms ``ema_alias``, a mute naming it does
      not suppress the pause, and ``PAUSED`` is on disk before the raise.

    What this composes rather than runs, stated rather than glossed: the
    ``iteration >= 1`` guard in ``loop.py``'s boundary, which exempts iteration
    0 because ``EMA_0 := theta_0`` is bit-identical *by construction* there.
    That condition needs a live campaign and is not asserted here.
    """
    trained = _small_net(monkeypatch)
    probe = _probe_positions()

    # The bug: the EMA is the trainer's own weights under another name.
    aliased = new_net()
    aliased.load_state_dict(_shifted(trained, 0.0))
    mean, zeros = policy_kl(aliased, trained, probe)
    assert mean == 0.0
    assert zeros == len(probe), (
        "an EMA holding the trainer's own weights did not report an exact zero "
        "on every probe position: the aliasing check has nothing to read"
    )

    # The false-alarm side: EMA_i = 0.75*EMA_{i-1} + 0.25*theta_i is a
    # different net from theta_i whenever EMA_{i-1} is, which is every
    # iteration after the first.
    smoothed = new_net()
    smoothed.load_state_dict(
        ema_state(_shifted(trained, 0.05), _shifted(trained, 0.0))
    )
    smoothed_mean, smoothed_zeros = policy_kl(smoothed, trained, probe)
    assert smoothed_mean > 0.0
    assert smoothed_zeros < len(probe), (
        "a smoothed EMA was indistinguishable from an aliased one; a gate that "
        "cannot tell them apart pauses every healthy campaign at iteration 2"
    )

    # And the escalation: unmutable, and durable before the raise.
    outcome = boundary.BoundaryOutcome()
    outcome.fire(
        EMA_ALIAS,
        "KL(pi_ema || pi_train) is exactly 0.0 on every probe position: the "
        "EMA is not a smoothed net but an alias of the trainer's own tensors",
        {EMA_ALIAS: Mute(EMA_ALIAS, 1, 1, 99)},
    )
    assert outcome.tripwire == EMA_ALIAS, "the aliasing check was muted"
    assert boundary.MUTED_FLAG not in outcome.extras

    with pytest.raises(CampaignPaused):
        boundary.enforce(tmp_path, outcome, 1, ())

    pause = read_pause(tmp_path)
    assert pause is not None, "the durable marker must be on disk before the raise"
    assert pause.tripwire == EMA_ALIAS
    assert pause.paused_at_iteration == 1


# ---------------------------------------------------------------------------
# acceptance #31 -- the completeness invariant
# ---------------------------------------------------------------------------


def test_fingerprint_series_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, param_keys: tuple[str, ...]
) -> None:
    """Acceptance #31: ``3N - 1`` records, 2 at iteration 0, 3 after, no gaps."""
    net = _small_net(monkeypatch)
    rig.plant(
        tmp_path,
        rig.advancing_states(net, ITERATIONS),
        param_keys,
    )

    fingerprints = rig.recorded_fingerprints(tmp_path)
    assert len(fingerprints) == 3 * ITERATIONS - 1

    by_iteration: dict[int, set[str]] = {}
    for entry in fingerprints:
        by_iteration.setdefault(entry.iteration, set()).add(entry.role)
    assert by_iteration[0] == {TRAIN, EMA}, (
        "iteration 0 has no served file -- CHECKPOINT_ENV is empty -- so a "
        "self_play fingerprint there would be a fabricated record"
    )
    for iteration in range(1, ITERATIONS):
        assert by_iteration[iteration] == {TRAIN, EMA, SELF_PLAY}

    assert rig.gate(tmp_path, ITERATIONS - 1).tripwire is None


def test_a_missing_fingerprint_write_pauses_rather_than_degrading_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, param_keys: tuple[str, ...]
) -> None:
    """Acceptance #31's failing direction, and it is **not** mutable.

    The run-length gate passes vacuously on a short series and walks the
    recorded entries without checking contiguity, so one missing write would
    silently narrow it. Here the whole of iteration 1 is missing from the
    journal, and the boundary refuses instead of reporting on the iterations
    that happened to be there.
    """
    net = _small_net(monkeypatch)
    rig.plant(
        tmp_path,
        rig.advancing_states(net, 3),
        param_keys,
    )

    journal = journal_path(tmp_path)
    kept: list[str] = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = decode_line(line)
        if isinstance(record, NetFingerprint) and record.iteration == 1:
            continue
        kept.append(line)
    journal.write_text("\n".join(kept) + "\n", encoding="utf-8")

    with pytest.raises(CampaignPaused):
        rig.gate(tmp_path, 2)

    pause = read_pause(tmp_path)
    assert pause is not None
    assert pause.tripwire == FINGERPRINT_SERIES


# ---------------------------------------------------------------------------
# acceptance #9 -- the cross-role invariant, and acceptance #7 -- a frozen EMA
# ---------------------------------------------------------------------------


def test_selfplay_fingerprint_equals_previous_ema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, param_keys: tuple[str, ...]
) -> None:
    """Acceptance #9: ``fingerprint(self_play, i) == fingerprint(ema, i-1)``.

    For every ``i >= 1`` -- not ``i >= 2``. This is the real audit of "did the
    loop point self-play at the EMA at all", and the F4 defect (the
    trainer/serving binding split) shows up here first, at ``i = 1``.
    """
    net = _small_net(monkeypatch)
    planted = rig.plant(
        tmp_path,
        rig.advancing_states(net, ITERATIONS),
        param_keys,
    )

    fingerprints = rig.recorded_fingerprints(tmp_path)
    served = rig.digests_for(fingerprints, SELF_PLAY)
    ema_series = rig.digests_for(fingerprints, EMA)

    assert served == ema_series[:-1], (
        "iteration i did not serve EMA_{i-1}: self-play is not playing the net "
        "this campaign thinks it published"
    )
    assert planted[0].served == "", "iteration 0 must serve fresh weights"
    for entry in planted[1:]:
        assert str(ema_dir(tmp_path)) in entry.served, (
            f"iteration {entry.iteration} served {entry.served!r}, which is "
            f"not under the EMA directory"
        )
        assert str(checkpoint_dir(tmp_path)) not in entry.served


def test_serving_the_train_checkpoint_pauses_on_the_cross_role_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, param_keys: tuple[str, ...]
) -> None:
    """The F4 defect planted: ``latest`` repointed instead of split.

    A loop that never split its ``latest`` binding serves the *trainer's*
    newest checkpoint. Every other gate is happy -- the digests all advance --
    and nothing raises, which is exactly why the cross-role invariant exists
    and why it can never be muted.
    """
    net = _small_net(monkeypatch)
    states = rig.advancing_states(net, 3)
    checkpoints = checkpoint_dir(tmp_path)
    ema_directory = ema_dir(tmp_path)
    journal = journal_path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    previous_train = ""
    for index, state in enumerate(states):
        save_checkpoint(
            checkpoints,
            (index + 1) * rig.STEPS_PER_ITERATION,
            state,
            {},
            {"iteration": index, "role": "train"},
        )
        reconcile_ema(
            checkpoints,
            ema_directory,
            param_keys,
            steps_per_iteration=rig.STEPS_PER_ITERATION,
        )
        boundary.record_fingerprints(
            journal,
            index,
            checkpoints,
            ema_directory,
            previous_train,
            param_keys,
            None,
        )
        previous_train = str(list_checkpoints(checkpoints)[0].path)

    with pytest.raises(CampaignPaused):
        rig.gate(tmp_path, 2)

    pause = read_pause(tmp_path)
    assert pause is not None
    assert pause.tripwire == CROSS_ROLE


def test_frozen_ema_pauses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, param_keys: tuple[str, ...]
) -> None:
    """Acceptance #7, with a **production-realistic** freeze.

    The EMA's parameters are unchanged from the previous iteration *while the
    trainer's steps continue to run* -- so the counters advance, the BatchNorm
    statistics move, and a new file with a new filename digest is published
    every iteration. Rev 1's rig republished the previous EMA *file*, which
    carries the same counters and therefore exercised a freeze mode that cannot
    occur in production while missing the one that can.
    """
    net = _small_net(monkeypatch)
    train_states = rig.advancing_states(net, 3)
    frozen_ema = rig.flat_parameter_states(net, 3)

    checkpoints = checkpoint_dir(tmp_path)
    ema_directory = ema_dir(tmp_path)
    journal = journal_path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    served = ""
    for index in range(3):
        step = (index + 1) * rig.STEPS_PER_ITERATION
        save_checkpoint(
            checkpoints, step, train_states[index], {}, {"iteration": index}
        )
        publish_ema(
            ema_directory,
            step,
            frozen_ema[index],
            ema_meta(
                iteration=index,
                source_step=step,
                train_sha256=canonical_digest(
                    train_states[index], param_keys
                ).sha256,
                ema_sha256=canonical_digest(frozen_ema[index], param_keys).sha256,
            ),
        )
        boundary.record_fingerprints(
            journal, index, checkpoints, ema_directory, served, param_keys, None
        )
        served = str(list_checkpoints(ema_directory)[0].path)

    fingerprints = rig.recorded_fingerprints(tmp_path)
    assert len(set(rig.digests_for(fingerprints, TRAIN))) == 3, (
        "the trainer must be visibly learning, or this tests the wrong freeze"
    )
    assert len(set(rig.digests_for(fingerprints, EMA))) == 1

    with pytest.raises(CampaignPaused):
        rig.gate(tmp_path, 2)

    pause = read_pause(tmp_path)
    assert pause is not None
    assert pause.tripwire == HASH_GATE
    assert "ema net" in pause.detail
