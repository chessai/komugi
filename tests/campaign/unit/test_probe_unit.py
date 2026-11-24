"""The frozen probe set and the masked policy KL it is measured on.

KOM-022 §3b/§7, acceptance #25 and §3b guard 3. Three properties of "the mean
``KL(pi_ema || pi_train)`` over the legal actions of a fixed set of positions"
are decisions rather than details, and each is pinned here:

* **legal, not all.** The unmasked distribution is dominated by ~37,000 logits
  nothing will ever play, so a KL over it measures agreement about moves that
  do not exist;
* **frozen, not this iteration's.** The band gate's reference distribution must
  not move under the gate, or a band exit could not be attributed to the nets
  rather than to the data. The file is written **durably before its digest is
  recorded** -- a torn write would otherwise pin the journal to the digest of a
  truncated file and make every later comparison agree with the wrong thing;
* **exactly zero is a bug, not a small number.** ``KL == 0.0`` on *every* probe
  position means the EMA update bound the trainer's tensor by reference instead
  of copying it. That check needs no model at all, which is why it pauses on
  the first occurrence and can never be muted -- it is the sharpest thing in
  §3b.

Acceptance #24 (``policy_target > 0`` is the engine's legal mask, as a
biconditional) is here, in
:func:`test_legal_mask_from_policy_target`, and it is asserted the only way
that can fail on the property: a real engine position, its own ``valid_moves``
mask, a search target written through the real shard writer at the real
self-play schema, and the derived mask compared to the engine's element for
element. What is still **not** closed is the *schema's guarantee* -- the
missing half of the ``SCHEMA`` docstring in ``komugi/selfplay/records.py``,
which is outside this ticket's file allowlist -- so the biconditional is
pinned for the rows this pipeline produces rather than promised for every row
the schema admits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from komugi.campaign.factory import (
    IN_PLANES,
    NET_ENV,
    NUM_ACTIONS,
    SMALL_NET,
    new_net,
)
from komugi.campaign.probe import (
    OOD_DOMAIN,
    PROBE_DOMAIN,
    ProbeError,
    ProbeSet,
    kl_rows,
    load_probe,
    observation_batch,
    policy_kl,
    probe_digest,
    probe_path,
    probe_seed,
    record_probe_digest,
    sample_probe,
    write_probe,
)
from komugi.engine.action_encoding import ACTION_SIZE
from komugi.engine.game import Game
from komugi.engine.observation import PLANE_MAP_VERSION
from komugi.selfplay.adapter import GameState
from komugi.selfplay.records import SCHEMA
from komugi.selfplay.worker import mover_relative_control
from komugi.train.shards import ShardReader, ShardWriter

ACTIONS = 6
"""A toy action space for the shape-agnostic halves of this module."""

RUN_SEED = 20260830

DRAW_SCORE = 0.20
"""The contempt the adapter is built with. Nothing here plays a game; the
state is used for its legality mask and its observation."""


class _Rows:
    """The smallest thing satisfying ``SampleSource``: a list of dicts."""

    def __init__(self, rows: list[dict[str, torch.Tensor]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        """Return how many samples this source holds."""
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return the sample at `index`."""
        return self._rows[index]


def _row(index: int, legal: tuple[int, ...] = (0, 2, 4)) -> dict[str, torch.Tensor]:
    """One shard row: an observation and a policy target nonzero on `legal`."""
    target = torch.zeros(ACTIONS)
    for action in legal:
        target[action] = 1.0 / len(legal)
    return {
        "observation": torch.full((2, 3, 3), float(index)),
        "policy_target": target,
    }


def _probe(count: int = 4) -> ProbeSet:
    """A toy probe set of `count` rows."""
    return sample_probe(_Rows([_row(index) for index in range(count)]), count, 7)


# ---------------------------------------------------------------------------
# the masked KL
# ---------------------------------------------------------------------------


def test_identical_logits_give_exactly_zero() -> None:
    """§3b guard 3's arithmetic: the aliasing signal is an *exact* zero.

    Not "small". The check is assumption-free precisely because it needs no
    tolerance and therefore no model of what "close" means.
    """
    logits = torch.randn(3, ACTIONS)
    legal = torch.ones(3, ACTIONS, dtype=torch.bool)

    values = kl_rows(logits, logits, legal)

    assert torch.equal(values, torch.zeros(3, dtype=values.dtype))


def test_the_kl_is_measured_over_the_legal_actions_only() -> None:
    """An illegal action's logits must not move the number at all.

    A KL over the unmasked distribution would be dominated by the ~37,000
    actions nothing will ever play, so a disagreement about a move that does
    not exist would read as a disagreement about the position.
    """
    legal = torch.tensor([[True, True, False, False, False, False]])
    left = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    right = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]])

    baseline = kl_rows(left, right, legal)

    # Move the illegal logits arbitrarily: the answer must not change.
    noisy_left = left.clone()
    noisy_right = right.clone()
    noisy_left[0, 2:] = 50.0
    noisy_right[0, 2:] = -50.0

    assert torch.allclose(kl_rows(noisy_left, noisy_right, legal), baseline)
    assert float(baseline.item()) > 0.0


def test_the_kl_is_positive_and_directed() -> None:
    """Direction is fixed: the **generating** net comes first.

    ``KL(pi_selfplay || pi_train)`` weights the regions the generating net
    actually visits, which is the quantity describing the data the trainer will
    receive -- and the asymmetry is why the direction is stated rather than
    inferred.
    """
    legal = torch.ones(1, ACTIONS, dtype=torch.bool)
    left = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    right = torch.tensor([[0.0, 3.0, 0.0, 0.0, 0.0, 0.0]])

    forward = float(kl_rows(left, right, legal).item())
    backward = float(kl_rows(right, left, legal).item())

    assert forward > 0.0
    # Symmetric here only because the fixture is a permutation; what matters is
    # that both are finite and the function does not silently symmetrise.
    assert backward > 0.0
    assert kl_rows(left, right, legal).dtype == torch.float64


def test_the_accumulation_is_float64() -> None:
    """``IterationMetrics`` rejects non-finite extras; float32 underflow is a way
    to get one, so the accumulation is widened before it is summed."""
    legal = torch.ones(2, ACTIONS, dtype=torch.bool)
    values = kl_rows(torch.randn(2, ACTIONS), torch.randn(2, ACTIONS), legal)

    assert values.dtype == torch.float64
    assert torch.isfinite(values).all()


# ---------------------------------------------------------------------------
# the aliasing check, on real nets
# ---------------------------------------------------------------------------


def test_an_aliased_net_reports_zero_on_every_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§3b guard 3: the EMA bound by reference instead of copied.

    Two handles on the same weights agree exactly everywhere, so the zero count
    equals the probe size -- which is what the boundary reads to fire the
    unmutable ``ema_alias`` tripwire.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    net = new_net()
    probe = ProbeSet(
        observation=torch.zeros(2, IN_PLANES, 9, 9),
        legal=_legal_mask(2),
    )

    mean, zeros = policy_kl(net, net, probe)

    assert mean == 0.0
    assert zeros == len(probe) == 2


def _legal_mask(rows: int) -> torch.Tensor:
    """A mask over the real action space with a handful of legal moves."""
    mask = torch.zeros(rows, NUM_ACTIONS, dtype=torch.bool)
    mask[:, :8] = True
    return mask


def test_two_different_nets_disagree_on_at_least_one_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The false-alarm side: a *smoothed* EMA is not an aliased one.

    A gate that could not tell them apart would pause every healthy campaign at
    iteration 2, which is worse than not having the gate.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    generator = new_net()
    reference = new_net()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(0.05)

    probe = ProbeSet(
        observation=torch.zeros(2, IN_PLANES, 9, 9), legal=_legal_mask(2)
    )
    mean, zeros = policy_kl(generator, reference, probe)

    assert mean > 0.0
    assert zeros < len(probe)


def test_the_nets_are_measured_in_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A train-mode forward would update BatchNorm from the probe set.

    That is a diagnostic silently editing the thing it measures -- and worse,
    editing a net the ``bn_digest`` series is watching.
    """
    monkeypatch.setenv(NET_ENV, SMALL_NET)
    net = new_net()
    net.train()
    before = {
        key: value.detach().clone()
        for key, value in net.state_dict().items()
        if key.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    assert before, "the net grew no BatchNorm state; this assertion is vacuous"

    probe = ProbeSet(
        observation=torch.zeros(2, IN_PLANES, 9, 9), legal=_legal_mask(2)
    )
    policy_kl(net, net, probe)

    assert not net.training, "policy_kl left the net in training mode"
    for key, value in before.items():
        assert torch.equal(net.state_dict()[key], value), (
            f"{key} was updated by the measurement"
        )


# ---------------------------------------------------------------------------
# sampling, and the row it refuses
# ---------------------------------------------------------------------------


def test_a_sampled_probe_set_is_a_seeded_permutation() -> None:
    """The same source and seed give the same set on any machine, after any resume."""
    source = _Rows([_row(index) for index in range(20)])

    first = sample_probe(source, 5, 11)
    second = sample_probe(source, 5, 11)

    assert len(first) == 5
    assert torch.equal(first.observation, second.observation), (
        "the same source and seed gave two different probe sets"
    )

    # ... and the seed is doing work: across eight seeds the drawn rows are not
    # always the same five. Asserted over a spread rather than over one pair,
    # because two particular seeds drawing the same 5-of-20 subset is a
    # coincidence, not a bug.
    drawn = {
        tuple(sample_probe(source, 5, seed).observation[:, 0, 0, 0].tolist())
        for seed in range(8)
    }
    assert len(drawn) > 1


def test_a_source_smaller_than_the_probe_size_contributes_every_row() -> None:
    """A short probe set is weaker; refusing over it would refuse over the least
    interesting cause."""
    source = _Rows([_row(index) for index in range(3)])

    assert len(sample_probe(source, 512, 3)) == 3


def test_an_empty_source_is_refused() -> None:
    """The band gate's reference distribution would have no positions."""
    with pytest.raises(ProbeError, match="empty source"):
        sample_probe(_Rows([]), 4, 1)


def test_a_row_with_no_legal_action_is_refused() -> None:
    """A masked softmax over an empty mask is undefined, and would produce a
    ``nan`` that takes the campaign down while it is writing telemetry."""
    rows = [_row(0), _row(1)]
    rows[1]["policy_target"] = torch.zeros(ACTIONS)

    with pytest.raises(ProbeError, match="zero on every"):
        sample_probe(_Rows(rows), 2, 1)


def test_legal_mask_from_policy_target(tmp_path: Path) -> None:
    """Acceptance #24: ``policy_target > 0`` **is** the engine's legal mask.

    A biconditional, and §3b's masked softmax depends on both directions: an
    illegal action with a positive target would enter the KL, and a legal one
    with a zero target would be dropped from it. The sibling test below asserts
    the *derivation* on rows this module constructed, which cannot fail on the
    schema's guarantee -- it is a statement about a fixture (review F5).

    So this one takes the long way round, through the pieces that produce the
    two sides in production and nothing else: a real engine position, its own
    ``valid_moves`` mask, a search target of the shape the worker writes
    (nonzero on every legal action), through the real
    :class:`~komugi.train.shards.ShardWriter` at the real self-play
    :data:`~komugi.selfplay.records.SCHEMA`, read back through the reader the
    campaign's probe sampler is handed. The mask ``sample_probe`` derives must
    be the engine's, element for element.
    """
    state = GameState(Game(), DRAW_SCORE)
    engine_mask = np.asarray(state.legal_mask(), dtype=np.bool_)
    legal_indices = np.flatnonzero(engine_mask)

    # Non-vacuity: a mask that was all-true or all-false would make the
    # comparison below pass for a derivation that ignored its input.
    assert legal_indices.size > 0, "the initial position has no legal move"
    assert legal_indices.size < ACTION_SIZE, (
        "every action is legal in the initial position, so an all-true mask "
        "would satisfy this test without deriving anything"
    )

    target = np.zeros(ACTION_SIZE, dtype=np.float32)
    target[legal_indices] = np.float32(1.0) / np.float32(legal_indices.size)
    observation = np.asarray(state.observation(), dtype=np.float32)
    row: dict[str, torch.Tensor] = {
        "observation": torch.from_numpy(observation),
        "policy_target": torch.from_numpy(target),
        "wdl_target": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
        "mlh_target": torch.tensor(20.0, dtype=torch.float32),
        "policy_kl": torch.tensor(0.0, dtype=torch.float32),
        "root_value": torch.tensor(0.0, dtype=torch.float32),
        "search_mode": torch.tensor(0, dtype=torch.int64),
        "to_play": torch.tensor(state.to_play(), dtype=torch.int64),
        "ply": torch.tensor(0, dtype=torch.int64),
        "game_id": torch.tensor(1, dtype=torch.int64),
        "sims_used": torch.tensor(2, dtype=torch.int64),
        # KOM-002 §4: the writer's schema requires it and the reader refuses a
        # row without it, so a fixture that omits it is not "a row the real
        # pipeline produces" -- it is a row the real pipeline cannot write.
        "plane_map_version": torch.tensor(PLANE_MAP_VERSION, dtype=torch.int64),
        # KOM-010, and produced the way the worker produces it: the engine's
        # absolute map for this position, flipped into the row's own mover's
        # frame. Same reason as the line above -- the schema requires it, so a
        # row without it is one the real pipeline cannot write.
        "final_control": torch.from_numpy(
            mover_relative_control(
                state.final_control(), state.to_play()
            ).astype(np.int64)
        ),
        # KOM-014: the move played from this position -- the first legal one,
        # since nothing here reads it and an illegal id would still be a row the
        # real pipeline cannot produce. Same reason as the two fields above: the
        # schema requires it.
        "action": torch.tensor(int(legal_indices[0]), dtype=torch.int64),
        # KOM-005b's three staged-head columns, at the real shapes and with
        # distinguishable values, for the same reason as the three fields
        # above: the schema requires them, so a row without them is one the
        # real pipeline cannot write. This row is a loss to the mover
        # (``wdl_target`` above), so the game was decisive -- ``termination``
        # is 0 -- and the position is past the draft, so ``draft_outcome`` is
        # the all-zero mask.
        "termination": torch.tensor(0, dtype=torch.int64),
        "short_term_value": torch.tensor(
            [-0.25, -0.5, -1.0], dtype=torch.float32
        ),
        "draft_outcome": torch.zeros(3, dtype=torch.float32),
    }

    directory = tmp_path / "worker_0_r0"
    with ShardWriter(directory, SCHEMA, samples_per_shard=1) as writer:
        writer.append(row)

    reader = ShardReader(directory, schema=SCHEMA)
    assert not reader.skipped, f"the shard did not read back: {reader.skipped}"
    assert len(reader) == 1

    probe = sample_probe(reader, 1, probe_seed(RUN_SEED, PROBE_DOMAIN))
    derived = probe.legal[0].numpy()

    assert derived.dtype == np.bool_
    assert np.array_equal(derived, engine_mask), (
        f"the derived mask differs from the engine's on "
        f"{int(np.count_nonzero(derived != engine_mask))} action(s): "
        f"{int(np.count_nonzero(derived & ~engine_mask))} illegal actions "
        f"carry a positive target and "
        f"{int(np.count_nonzero(engine_mask & ~derived))} legal ones carry "
        f"zero"
    )
    assert np.array_equal(probe.observation[0].numpy(), observation), (
        "the row read back is not the position whose mask was compared"
    )


def test_the_mask_is_derived_from_the_policy_target() -> None:
    """``legal = policy_target > 0``, on rows this module built.

    The cheap half, kept for the shape and dtype it pins on a toy action space.
    It cannot fail on the *schema's* guarantee -- it is a statement about a
    fixture -- which is why
    :func:`test_legal_mask_from_policy_target` above makes the same claim over
    a real shard row and the engine's own mask.
    """
    probe = _probe(3)

    assert probe.legal.dtype == torch.bool
    assert probe.legal.shape == (3, ACTIONS)
    assert probe.legal[0].tolist() == [True, False, True, False, True, False]


def test_observation_batch_shares_the_probe_sets_shape() -> None:
    """§7's fresh sample goes through the same KL routine as the frozen one."""
    batch = observation_batch([_row(0), _row(1)])

    assert len(batch) == 2
    assert batch.legal.shape == (2, ACTIONS)

    with pytest.raises(ProbeError, match="no fresh rows"):
        observation_batch([])


# ---------------------------------------------------------------------------
# acceptance #25 -- the probe file is durable and its digest is stable
# ---------------------------------------------------------------------------


def test_probe_digest_is_stable(tmp_path: Path) -> None:
    """Acceptance #25: the digest recorded at iteration 0 is the one recomputed later.

    A swapped probe set is then detectable rather than a silent change of what
    the band *means*, which is the same treatment the pre-registered constants
    get in #40.
    """
    probe = _probe(4)
    path = probe_path(tmp_path)

    digest = write_probe(path, probe)

    assert path.is_file()
    assert digest == probe_digest(path)
    assert len(digest) == 64
    # Recomputed at "every later iteration": same bytes, same answer.
    assert probe_digest(path) == digest

    loaded = load_probe(path)
    assert torch.equal(loaded.observation, probe.observation)
    assert torch.equal(loaded.legal, probe.legal)
    assert loaded.legal.dtype == torch.bool


def test_the_digest_is_recorded_beside_the_file_it_describes(tmp_path: Path) -> None:
    """The journal carries the low 52 bits; the full digest lives next to the file."""
    probe = _probe(2)
    digest = write_probe(probe_path(tmp_path), probe)

    record_probe_digest(tmp_path, digest)

    stamped = (tmp_path / "probe" / "probe.sha256.json").read_text(encoding="utf-8")
    assert digest in stamped


def test_a_swapped_probe_set_changes_the_digest(tmp_path: Path) -> None:
    """The whole point: the reference distribution cannot be replaced quietly."""
    first = write_probe(probe_path(tmp_path), _probe(4))
    second = write_probe(probe_path(tmp_path), _probe(2))

    assert first != second


def test_an_unreadable_probe_file_is_refused(tmp_path: Path) -> None:
    """A truncated probe set must raise, not load as a shorter reference."""
    path = probe_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not an npz archive")

    with pytest.raises(ProbeError, match="not a readable probe set"):
        load_probe(path)


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------


def test_the_probe_and_ood_samplers_cannot_collide() -> None:
    """Separate spawn-key domains, so neither stream coincides with the window's.

    Reusing ``iteration_seed`` would correlate the probe sampler with the batch
    sampler, which is the class of collision the arena's own config docstring
    warns about.
    """
    frozen = probe_seed(RUN_SEED, PROBE_DOMAIN)
    fresh = probe_seed(RUN_SEED, OOD_DOMAIN, 0)

    assert frozen != fresh
    assert probe_seed(RUN_SEED, PROBE_DOMAIN) == frozen, "the seed is not a function"
    assert len({probe_seed(RUN_SEED, OOD_DOMAIN, i) for i in range(8)}) == 8, (
        "two iterations drew the same fresh sample"
    )
