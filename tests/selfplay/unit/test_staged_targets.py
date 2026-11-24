"""``finalize_rows``' three new columns, and the schema tail they append to.

Every target here is computed on the **write** side, and that is a decision
rather than a convenience: batches are drawn one row at a time from a window,
so row ``i + 4`` of the same game is not in the batch and is not reachable from
row ``i``. A short-term value target cannot be built at training time at all.

Three failure modes drive the nodes below, and each is silent:

* **the buffer/ply-count confusion.** ``rows`` is a buffer indexed
  ``0 … len(rows) - 1``; ``total_plies`` is a *ply counter*. They coincide only
  for an unbooked game -- a book prefix is played and not recorded, so the
  counter runs ahead of the row count by the prefix length ``L``. A horizon
  loop bounded on ``total_plies`` raises ``IndexError`` inside the self-play
  worker, after a full game of compute, on **every** booked game; and the
  in-range cases it does not raise on fire the terminal clamp ``L`` rows too
  late.
* **the parity-keyed flip.** The draft lets one side move twice, so the movers
  of consecutive rows need not alternate. Every horizon here is even, so a
  parity-keyed and a ``to_play``-keyed flip agree on every row whose horizon
  window does not straddle a repeat -- which makes a test without a repeat
  strictly inside an asserted window *vacuous*.
* **the WDL index semantics.** Index 0 means "the row's mover won". Nothing
  asserted that end to end until the uncertainty head started subtracting
  index 2 from index 0, where a swap flips every residual's sign invisibly
  because the residual is squared.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest
import torch

from komugi.config import SelfPlayConfig
from komugi.engine.action_encoding import C_START, SHU_BLOCK
from komugi.engine.game import Game
from komugi.net import HeadsConfig, KomugiNet, NetConfig, TrunkConfig
from komugi.selfplay.adapter import GameState
from komugi.selfplay.inference import NetEval, wdl_utility
from komugi import terminations
from komugi.selfplay import worker as worker_module
from komugi.selfplay.records import SCHEMA, SHORT_TERM_HORIZONS, num_actions
from komugi.selfplay.worker import (
    MODE_FULL,
    TERMINATION_DECISIVE,
    TERMINATION_DRAW,
    TERMINATION_RESIGN,
    PendingRow,
    SelfPlayWorker,
    WorkerConfig,
    finalize_rows,
)
from komugi.train.shards import ShardReader, ShardWriter

_WHITE: Final[int] = 0
_BLACK: Final[int] = 1

_WHITE_WINS: Final[tuple[int, int, int]] = (1, 0, 0)
_BLACK_WINS: Final[tuple[int, int, int]] = (0, 1, 0)
_DRAWN: Final[tuple[int, int, int]] = (0, 0, 1)
"""The engine's absolute ``[white, black, draw]`` outcome, as ``win_state()``."""


def _outcome(values: tuple[int, int, int]) -> npt.NDArray[np.uint8]:
    """The ``(3,)`` ``uint8`` the record seam type-checks for."""
    return np.asarray(values, dtype=np.uint8)


def _empty_control() -> npt.NDArray[np.int8]:
    """An all-empty absolute control map.

    The one map invariant under the mover flip, so nothing here can pass or
    fail because of the ownership column it was handed.
    """
    return np.full(81, 2, dtype=np.int8)


def _pending(
    *,
    ply: int,
    to_play: int,
    root_value: float = 0.0,
    in_draft: bool = False,
) -> PendingRow:
    """A minimal buffered row. Only the four named fields carry meaning."""
    return PendingRow(
        observation=np.zeros(Game.observation_size(), dtype=np.float32),
        policy_target=np.zeros(num_actions(), dtype=np.float32),
        policy_kl=0.0,
        root_value=root_value,
        search_mode=MODE_FULL,
        to_play=to_play,
        ply=ply,
        sims_used=2,
        action=ply,
        in_draft=in_draft,
    )


def _stv(samples: list[dict[str, torch.Tensor]], index: int) -> list[float]:
    """The three short-term-value targets of row `index`, as plain floats."""
    return [float(value) for value in samples[index]["short_term_value"]]


# --------------------------------------------------------------------------
# Short-term value: the frozen horizons, the lookup, the flip, and the bound
# --------------------------------------------------------------------------

_SMALL_IN_PLANES: Final[int] = 4
_SMALL_CHANNELS: Final[int] = 16
_SMALL_BOARD: Final[int] = 9

_SMALL_CONFIG: Final[NetConfig] = NetConfig(
    trunk=TrunkConfig(
        in_planes=_SMALL_IN_PLANES,
        channels=_SMALL_CHANNELS,
        blocks=1,
        gpool_every=3,
        seed=17,
    ),
    heads=HeadsConfig(
        channels=_SMALL_CHANNELS,
        num_actions=_SMALL_BOARD * _SMALL_BOARD,
        policy_planes=1,
        seed=17,
    ),
)
"""A one-block net, built only to read the short-term head's output width."""


def test_the_short_term_horizons_are_the_pre_registered_values() -> None:
    """``(4, 16, 64)``, pinned as VALUES, and the head's width pinned to them.

    ``SHORT_TERM_HORIZONS`` is the one constant KOM-005b §1.3.1 froze *before
    any data exists*: "a later change is a new pre-registration and a
    fingerprint bump, not a re-analysis". Nothing enforced that. The tuple's
    length is what the schema's ``shape=(3,)`` is built from, so a change from
    ``(4, 16, 64)`` to ``(4, 16, 32)`` moves neither name, nor dtype, nor
    shape -- and :attr:`~komugi.train.shards.Schema.fingerprint` is a function
    of exactly those three (impl review F2, mutant M1). Shards written under
    two different horizon sets would then mix inside one replay window, each
    row's third component meaning something different, with no bump and no
    warning: the *identical* hazard ``records.py``'s plane-map note records for
    the observation encoding, arriving through this ticket's own new constant.

    A value assertion is the whole fix, and it is deliberately spelled as
    literals rather than derived from anything: the point is that changing the
    definition must cost a red test and a conscious re-registration.

    The second clause ties the *head* to the tuple. ``ShortTermValueHead``
    carries its own width, and the constant's docstring claims "the head's
    width is 3 because this tuple is 3" -- a claim no test made. Read off the
    real net, so a head widened to four outputs without a horizon (or a fourth
    horizon without a head output) is red here rather than at a gate.
    """
    assert SHORT_TERM_HORIZONS == (4, 16, 64), (
        "the short-term horizons are pre-registered (KOM-005b §1.3.1); "
        "changing them is a new pre-registration and a fingerprint bump, not "
        "an edit"
    )

    net = KomugiNet(_SMALL_CONFIG)
    net.eval()
    with torch.no_grad():
        out = net(
            torch.zeros(1, _SMALL_IN_PLANES, _SMALL_BOARD, _SMALL_BOARD)
        )
    assert out.short_term_value.shape[-1] == len(SHORT_TERM_HORIZONS), (
        "the short-term head emits "
        f"{int(out.short_term_value.shape[-1])} values for "
        f"{len(SHORT_TERM_HORIZONS)} horizons: the writer and the head hold "
        "different copies of the target's definition"
    )

    # ... and the column the two meet on is the same width again.
    field = next(spec for spec in SCHEMA.fields if spec.name == "short_term_value")
    assert field.shape == (len(SHORT_TERM_HORIZONS),)


def test_short_term_value_flips_on_to_play_across_a_repeated_mover() -> None:
    """The sign comes from ``to_play``, never from ply parity.

    Eight rows with a **repeated mover** at buffer index 3 -- the draft case
    where one side moves twice -- so ``to_play`` is ``[0, 1, 0, 0, 1, 0, 1, 0]``
    and the parity of a row's ply stops predicting its mover from index 3 on.

    Three constraints, and without any of them the node is red against a
    correct implementation or blind to the bug it is for:

    * **scoped to the lookup branch.** ``stv[h] == ±root_value`` is false
      wherever the horizon runs off the buffer, where the terminal clamp makes
      the entry ``z``. Only horizon 4 is unclamped here and only for rows 0-3,
      so only those are asserted; the clamp is the next node's subject. (The
      alternative -- building 65 rows so all three horizons are unclamped -- is
      not hand-checkable, which is the whole point of a golden.)
    * **the repeat falls strictly inside an asserted window.** Every horizon is
      even, so a parity-keyed flip and a ``to_play``-keyed flip **agree** on
      every row whose window ``(i, i + h]`` does not straddle the repeat. Rows
      0, 1 and 2 all satisfy ``i < 3 <= i + 4``, so a parity implementation is
      individually red on three of the four asserted rows -- and would be
      vacuously green without them.
    * **blind to the buffer/ply-count bound.** This buffer is passed
      ``total_plies == len(rows)``, so it exercises only the regime where the
      two coincide. That is the right scope here; the bound is the next node's.
    """
    movers = [0, 1, 0, 0, 1, 0, 1, 0]
    values = [0.1 * (index + 1) for index in range(len(movers))]
    rows = [
        _pending(ply=index, to_play=mover, root_value=values[index])
        for index, mover in enumerate(movers)
    ]

    samples = finalize_rows(
        rows, _outcome(_WHITE_WINS), _empty_control(), len(rows), 7
    )

    horizon = SHORT_TERM_HORIZONS[0]
    assert horizon == 4
    for index in range(len(rows) - horizon):
        future = index + horizon
        expected = values[future]
        if rows[future].to_play != rows[index].to_play:
            expected = -expected
        assert _stv(samples, index)[0] == pytest.approx(expected, abs=1e-6), (
            f"row {index} at horizon {horizon}: the flip is keyed off "
            "something other than to_play"
        )

    # Non-vacuity, asserted rather than assumed: at least one asserted row's
    # window straddles the repeat, so a parity-keyed implementation is red.
    straddling = [
        index
        for index in range(len(rows) - horizon)
        if index < 3 <= index + horizon
        and (index % 2 == (index + horizon) % 2)
        and rows[index].to_play != rows[index + horizon].to_play
    ]
    assert straddling, (
        "no asserted row has a window straddling the repeated mover: a "
        "parity-keyed implementation would pass this node"
    )


def test_short_term_value_past_the_last_ply_is_the_games_result() -> None:
    """Past the end of the buffer, the label is the game's result.

    A five-row drawn game. **Row 0** is the only row whose first horizon is a
    lookup -- ``0 + 4 = 4 < 5`` -- so its three targets are
    ``(±root_value[4], z, z)``, with the sign taken from whether ``rows[4]``
    shares row 0's mover. Horizons 16 and 64 both run off the buffer and clamp.
    Every other row clamps at every horizon, because ``i + 4 >= 5`` for
    ``i >= 1``.

    On a drawn game ``z`` is ``0.0`` **exactly**, with no ``draw_score``
    anywhere in the row: contempt is a search-time preference and design D1
    keeps it out of every target. That is asserted as an exact float, not
    approximately -- ``0.2`` would be a very visible failure and ``1e-9`` of it
    would not.
    """
    movers = [0, 1, 0, 1, 0]
    values = [0.125, 0.25, 0.375, 0.5, 0.625]
    rows = [
        _pending(ply=index, to_play=mover, root_value=values[index])
        for index, mover in enumerate(movers)
    ]

    samples = finalize_rows(rows, _outcome(_DRAWN), _empty_control(), len(rows), 3)

    # rows[4].to_play == rows[0].to_play, so the lookup keeps its sign.
    assert rows[4].to_play == rows[0].to_play
    first = _stv(samples, 0)
    assert first[0] == pytest.approx(values[4], abs=1e-6)
    assert first[1] == 0.0
    assert first[2] == 0.0

    for index in range(1, len(rows)):
        assert _stv(samples, index) == [0.0, 0.0, 0.0], (
            f"row {index} did not clamp at every horizon: 4 + {index} >= 5"
        )

    # No draw_score anywhere: the clamp is the pure result.
    for index in range(len(rows)):
        for value in _stv(samples, index):
            assert value != pytest.approx(0.2, abs=1e-9)
            assert value != pytest.approx(-0.2, abs=1e-9)


def test_short_term_value_is_bounded_by_the_buffer_not_the_ply_count() -> None:
    """The booked-game shape: ``total_plies != len(rows)``.

    Twenty rows carrying absolute plies ``12 … 31`` and a ``total_plies`` of
    ``32`` -- the shape a book prefix of length ``L = 12`` produces, where the
    prefix is played but not recorded. ``L = 12`` is retained for exactly one
    job: making ``total_plies`` differ from ``len(rows)``. It has nothing to do
    with how many rows clamp.

    Clause (a) is that **no exception is raised**. A horizon loop bounded on
    ``total_plies`` evaluates ``rows[19 + 4] == rows[23]`` against a buffer of
    20 and raises ``IndexError`` -- inside the worker, after a whole game of
    search, on every booked game.

    Clause (b) is the partition, at each horizon ``h`` independently: the rows
    with ``i + h >= len(rows)`` carry the terminal clamp and the rest carry the
    signed lookup. The root values are all inside ``(0, 0.21]`` and the game is
    decisive, so ``z = ±1`` can never coincide with a lookup value and the two
    branches are distinguishable row by row -- which a fixture whose recorded
    values happened to equal the result would not be.
    """
    length = 20
    prefix = 12
    movers = [index % 2 for index in range(length)]
    values = [0.01 * (index + 1) for index in range(length)]
    rows = [
        _pending(ply=prefix + index, to_play=movers[index], root_value=values[index])
        for index in range(length)
    ]
    total_plies = length + prefix

    # (a) -- the bound is the buffer's, so this does not raise.
    samples = finalize_rows(
        rows, _outcome(_WHITE_WINS), _empty_control(), total_plies, 11
    )
    assert len(samples) == length

    # (b) -- the partition, horizon by horizon.
    clamped = 0
    looked_up = 0
    for index in range(length):
        targets = _stv(samples, index)
        result = 1.0 if rows[index].to_play == _WHITE else -1.0
        for slot, horizon in enumerate(SHORT_TERM_HORIZONS):
            future = index + horizon
            if future >= length:
                assert targets[slot] == pytest.approx(result, abs=1e-6), (
                    f"row {index}, horizon {horizon}: past the end of the "
                    "buffer the label is the game's result"
                )
                clamped += 1
            else:
                expected = values[future]
                if rows[future].to_play != rows[index].to_play:
                    expected = -expected
                assert targets[slot] == pytest.approx(expected, abs=1e-6), (
                    f"row {index}, horizon {horizon}: the lookup is bounded "
                    "by the ply counter rather than by the buffer"
                )
                looked_up += 1

    # Both branches are populated, or the partition proves nothing.
    assert clamped > 0 and looked_up > 0
    # At horizon 4 exactly the last 4 rows clamp -- not the last 12. The
    # prefix length does not appear in this arithmetic at all.
    assert all(
        _stv(samples, index)[0] == pytest.approx(
            1.0 if rows[index].to_play == _WHITE else -1.0, abs=1e-6
        )
        for index in range(length - 4, length)
    )
    assert _stv(samples, length - 5)[0] == pytest.approx(values[length - 1], abs=1e-6)


# --------------------------------------------------------------------------
# Draft outcome and termination
# --------------------------------------------------------------------------


def test_draft_outcome_equals_wdl_target_on_draft_rows_and_is_zero_off_them() -> None:
    """The column *is* ``wdl_target``, gated by a per-row boolean.

    One flip, one perspective, no second derivation -- which is what makes the
    head's label rule on resigned, adjudicated and truncated games identical to
    ``wdl_target``'s **by construction** rather than by a paragraph. Asserted
    with ``torch.equal`` on the draft rows, because "the same vector" is the
    claim, and with an exact zero sum off them, because an all-zero row is the
    mask and a nearly-zero one is not.

    The buffer deliberately carries both movers among the draft rows, so a
    column that derived its own perspective instead of copying ``wdl_target``'s
    would disagree on one of them.
    """
    rows = [
        _pending(ply=0, to_play=_WHITE, in_draft=True),
        _pending(ply=1, to_play=_BLACK, in_draft=True),
        _pending(ply=2, to_play=_WHITE, in_draft=True),
        _pending(ply=3, to_play=_BLACK, in_draft=False),
        _pending(ply=4, to_play=_WHITE, in_draft=False),
    ]

    samples = finalize_rows(
        rows, _outcome(_WHITE_WINS), _empty_control(), len(rows), 5
    )

    for index, row in enumerate(rows):
        draft_outcome = samples[index]["draft_outcome"]
        assert draft_outcome.shape == (3,)
        assert draft_outcome.dtype == torch.float32
        if row.in_draft:
            assert torch.equal(draft_outcome, samples[index]["wdl_target"]), (
                f"row {index} is a draft row, so its draft_outcome must be "
                "its wdl_target exactly"
            )
        else:
            assert float(draft_outcome.abs().sum()) == 0.0, (
                f"row {index} is not a draft row, so its target must have no "
                "mass at all -- the zeros ARE the mask"
            )

    # Non-vacuity: both kinds of row are present, and the draft rows do not
    # all share one mover.
    assert any(row.in_draft for row in rows)
    assert any(not row.in_draft for row in rows)
    assert len({row.to_play for row in rows if row.in_draft}) == 2

    # ... and the mask really is masking something: white's draft rows and
    # black's carry different vectors, so an unflipped copy would be red.
    assert not torch.equal(samples[0]["draft_outcome"], samples[1]["draft_outcome"])


def test_termination_is_zero_on_a_decisive_game_and_one_on_a_draw() -> None:
    """A game-level value, stamped on every row of the game.

    ``0`` decisive, ``1`` draw. ``2`` is reserved for resign v2 and is
    **never** written here: P0b's worker cannot resign, so the resigned
    fraction of every window this campaign produces is identically zero. That
    the value exists at all is what makes "ownership is withheld on resigned
    games" a query over a column rather than a claim about a worker version.
    """
    rows = [_pending(ply=index, to_play=index % 2) for index in range(4)]

    for outcome, expected in (
        (_WHITE_WINS, TERMINATION_DECISIVE),
        (_BLACK_WINS, TERMINATION_DECISIVE),
        (_DRAWN, TERMINATION_DRAW),
    ):
        samples = finalize_rows(
            rows, _outcome(outcome), _empty_control(), len(rows), 9
        )
        values = {int(sample["termination"]) for sample in samples}
        assert values == {expected}, (
            f"outcome {outcome} gave terminations {sorted(values)}; the value "
            "is game-level and must be constant within a game"
        )
        for sample in samples:
            assert sample["termination"].dtype == torch.int64
            assert int(sample["termination"]) != TERMINATION_RESIGN

    assert TERMINATION_DECISIVE == 0
    assert TERMINATION_DRAW == 1
    assert TERMINATION_RESIGN == 2


def test_the_worker_re_exports_the_one_termination_vocabulary() -> None:
    """KOM-160: ``worker``'s three codes ARE the leaf's, not a second copy.

    The trainer keys a per-row WDL weight on this column and cannot import
    :mod:`komugi.selfplay.worker` to learn its vocabulary -- the package cycle
    :mod:`komugi.modes` documents -- so the codes moved to
    :mod:`komugi.terminations`, a leaf at the package root, and ``worker``
    re-exports them for every existing importer.

    Equality alone cannot pin this. ``TERMINATION_DRAW == 1`` is satisfied by a
    restated literal, and so is ``is`` -- CPython interns 0, 1 and 2, so the two
    copies would be the same object anyway. The claim is therefore made against
    ``worker``'s **source**: no module-level assignment to any ``TERMINATION_*``
    name. Two copies would agree on every value until the day one of them gained
    a fourth code, which is exactly when a trainer refusing an unknown
    termination needs them not to have diverged.
    """
    assert TERMINATION_DECISIVE == terminations.TERMINATION_DECISIVE
    assert TERMINATION_DRAW == terminations.TERMINATION_DRAW
    assert TERMINATION_RESIGN == terminations.TERMINATION_RESIGN

    assigned = {
        target.id
        for node in ast.walk(ast.parse(inspect.getsource(worker_module)))
        if isinstance(node, ast.AnnAssign | ast.Assign)
        for target in (
            [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        )
        if isinstance(target, ast.Name) and target.id.startswith("TERMINATION_")
    }
    assert assigned == set(), (
        f"komugi.selfplay.worker assigns {sorted(assigned)}; the termination "
        f"vocabulary has two sources of truth again, and the trainer keys a "
        f"per-row loss weight on the one it can import"
    )


# --------------------------------------------------------------------------
# The wire semantics, end to end
# --------------------------------------------------------------------------


def test_index_zero_of_the_wdl_target_is_the_movers_win_and_reads_as_plus_one() -> None:
    """Wire -> target -> utility, as one chain.

    The uncertainty head's target is ``(p[0] - p[2] - (t[0] - t[2]))²``, which
    makes the (win, draw, loss) index order arithmetically load-bearing for the
    first time -- and squaring hides a *consistent* swap completely. The loss
    suite pins that ``v̂`` and ``z`` agree with each other; this node pins what
    index 0 means on the **wire**, which is the half a single-ended swap would
    break and the other node cannot see.

    Two anchors, asserted as one chain: ``finalize_rows`` writes ``[1, 0, 0]``
    for a row whose own mover won, and ``wdl_utility`` reads that same one-hot
    back as ``+1.0``. Both directions of the loss case too, so a swap is red
    from either end.
    """
    rows = [
        _pending(ply=0, to_play=_WHITE),
        _pending(ply=1, to_play=_BLACK),
    ]

    samples = finalize_rows(
        rows, _outcome(_WHITE_WINS), _empty_control(), len(rows), 13
    )

    white_row = samples[0]["wdl_target"]
    black_row = samples[1]["wdl_target"]
    assert torch.equal(
        white_row, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    ), "white moved in row 0 and white won, so index 0 must carry the mass"
    assert torch.equal(
        black_row, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    ), "black moved in row 1 and black lost, so index 2 must carry the mass"

    # ... and the utility seam reads the same indices the same way.
    utility = wdl_utility(
        np.stack(
            (white_row.numpy(), black_row.numpy()),
        ).astype(np.float32),
        0.2,
    )
    assert float(utility[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(utility[1]) == pytest.approx(-1.0, abs=1e-6)

    # The draw slot is index 1, and it is the only one draw_score can touch.
    drawn = finalize_rows(rows, _outcome(_DRAWN), _empty_control(), len(rows), 14)
    assert torch.equal(
        drawn[0]["wdl_target"], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    )
    assert float(
        wdl_utility(drawn[0]["wdl_target"].numpy().reshape(1, 3), 0.2)[0]
    ) == pytest.approx(0.2, abs=1e-6)


# --------------------------------------------------------------------------
# The schema tail
# --------------------------------------------------------------------------


def test_the_schema_ends_with_the_three_new_columns_in_order() -> None:
    """The append order, pinned in one place and stable under the next append.

    A suffix plus relative indices, never absolute positions. That is the
    lesson of the node this one supersedes: KOM-014's tail pin asserted
    ``names[-1] == "action"``, which any correct implementation of the design's
    own "append LAST" rule necessarily falsifies the moment another ticket
    appends -- and it had already been amended once for the same reason before
    this design amended it again.

    The relative chain is the claim every appender needs to keep making
    (*every new field lands after every field that predates it*); the suffix is
    the claim this design adds. Pinning both here is what stops the *next*
    appender from being in the position KOM-014 was.
    """
    names = SCHEMA.names
    assert names[-3:] == ("termination", "short_term_value", "draft_outcome")
    assert (
        names.index("plane_map_version")
        < names.index("final_control")
        < names.index("action")
        < names.index("termination")
    )

    by_name = {field.name: field for field in SCHEMA.fields}
    assert by_name["termination"].dtype == "int64"
    assert by_name["termination"].shape == ()
    assert by_name["short_term_value"].dtype == "float32"
    assert by_name["short_term_value"].shape == (len(SHORT_TERM_HORIZONS),)
    assert by_name["draft_outcome"].dtype == "float32"
    assert by_name["draft_outcome"].shape == (3,)


def test_finalize_rows_emits_every_schema_column() -> None:
    """A finished row carries the whole schema and nothing else.

    ``Schema.validate`` is exact key equality -- a missing field is a hole in
    the batch and an extra one is a name that would be silently dropped at seal
    time -- so this is the shape ``ShardWriter.append`` will hold the worker to.
    """
    rows = [_pending(ply=index, to_play=index % 2) for index in range(3)]
    samples = finalize_rows(
        rows, _outcome(_WHITE_WINS), _empty_control(), len(rows), 21
    )
    for sample in samples:
        assert set(sample) == set(SCHEMA.names)


# --------------------------------------------------------------------------
# The recorded draft flag is the engine's own predicate
# --------------------------------------------------------------------------


class _DraftBoundaryGame(Game):
    """The real engine, started in **advanced** mode, with a short turn cap.

    Advanced is the mode that actually has a draft: its start FEN is an empty
    board with both hands full and both draft bits set. Intro -- the process
    default and what ``GUNGI_MODE`` leaves this class at otherwise -- starts
    from a laid-out board with the draft field ``-``, so *no* ply of an intro
    game is a draft ply and there is no boundary here to cross at all.

    Set as a **class attribute**, which is the whole reason this subclass
    exists: ``Game.__init__`` reads ``type(self)._START_MODE``, so overriding it
    here changes this game and nothing else. ``Game.set_start_mode`` would
    mutate the shared base class for the rest of the process -- a global whose
    restoration this module would then own, and whose leak no assertion here
    would notice.

    The cap is long enough for both sides to finish drafting and for the
    movement phase to produce rows, short enough that a two-simulation search
    finishes it in seconds.
    """

    _START_MODE: str = "advanced"

    @staticmethod
    def max_turns() -> int:
        """Return the shortened turn cap."""
        return 40


class _FinishTheDraftEvaluator:
    """A stub evaluator that walks the two sides out of the draft *apart*.

    Uniform logits everywhere except the action encoding's block C -- "arata +
    draft-finish", ``SHU_BLOCK``, which opens at ``C_START`` and runs to the end
    of the action space. A flat boost there ends both drafts, but it ends them
    on adjacent plies with the same rights on every recorded row, which leaves
    the non-vacuity clause below with nothing to see. So the bias is chosen per
    row, from two planes of the observation the worker hands the evaluator:

    * **plane 114** -- side to move, ``1.0`` for black;
    * **plane 85** -- black's marshal-in-hand count (hand plane
      ``84 + piece_idx * 2 + color_idx``, marshal being piece 0).

    Both are read off the *absolute* observation -- the encoder does not
    canonicalise to the mover -- and both are constant over the board, so one
    cell reads each. Reading them per row rather than shaping the root by hand
    is what makes the bias mean the same thing at every depth of the search.

    Boost block C for white, and for black once its marshal is down; **penalise**
    it for black while the marshal is still in hand. The engine's marshal-first
    rule (``cymovegen``: "if marshal is still in hand, only marshal may be
    placed") then leaves black exactly one other kind of legal id -- the block-B
    placement of that same marshal, which keeps its draft open. The ply table
    that falls out:

    ===  ======  =====================================  =================
    ply  mover   action                                 rights (w, b) at
                                                        capture time
    ===  ======  =====================================  =================
    0    white   block C: marshal placed, draft ends    (True, True)
    1    black   block B: marshal placed, still drafts  (False, True)
    2    black   block C: places and ends its draft     (False, True)
    3+   either  movement, out to the 40-turn cap       (False, False)
    ===  ======  =====================================  =================

    Plies 1 and 2 are the asymmetric ones, and they are recorded rows. Row 0 is
    a draft row too, so the recorded flag has both values and the boundary is
    crossed inside the buffer. Nothing about the recorded flag depends on the
    network being any good; the bias exists only to reach that shape inside a
    test's time budget.
    """

    _SIDE_TO_MOVE_PLANE: Final[int] = 114
    _BLACK_MARSHAL_IN_HAND_PLANE: Final[int] = 85

    _BIAS: Final[float] = 16.0
    """Logit gap between block C and the rest, in both directions.

    Root actions are drawn by Gumbel top-k over these logits, and the draft
    plies offer at most 81 placement squares per block, whose maximum Gumbel is
    about ``log 81 ~ 4.4``. A gap of ``2 * _BIAS`` between the boosted and the
    penalised branch is far outside what any draw of that noise closes, so the
    ply table above is the game the seed plays rather than the game it usually
    plays.
    """

    def __init__(self, actions: int) -> None:
        self.actions = actions
        # Block C is written by name below and runs to the end of the action
        # space, so the encoder's own bound and the width this stub emits must
        # be the same number -- otherwise the slice shapes the wrong ids and
        # the ply table above is quietly fiction.
        assert SHU_BLOCK == (C_START, actions), (
            f"block C is {SHU_BLOCK}, but this evaluator was built for "
            f"{actions} actions"
        )

    def evaluate(
        self, obs: npt.NDArray[np.float32], legal: npt.NDArray[np.bool_]
    ) -> NetEval:
        """Return one evaluation per row of `obs`."""
        rows = int(obs.shape[0])
        black_to_move = obs[:, self._SIDE_TO_MOVE_PLANE, 0, 0] > 0.5
        marshal_in_hand = obs[:, self._BLACK_MARSHAL_IN_HAND_PLANE, 0, 0] > 0.0
        bias: npt.NDArray[np.float32] = np.where(
            black_to_move & marshal_in_hand, -self._BIAS, self._BIAS
        ).astype(np.float32)

        logits = np.zeros((rows, self.actions), dtype=np.float32)
        logits[:, C_START:] = bias[:, None]
        return NetEval(
            policy_logits=logits,
            wdl=np.tile(
                np.asarray([0.4, 0.2, 0.4], dtype=np.float32), (rows, 1)
            ),
            moves_left=np.zeros(rows, dtype=np.float32),
        )


def _drafting_rights(state: GameState) -> tuple[bool, bool]:
    """Return ``(white_is_drafting, black_is_drafting)`` from the board itself.

    Reached through the engine's private board on purpose: it is the
    *independent* oracle this node needs. The worker derives its flag from the
    encoder's drafting-rights planes (``komugi.engine.game.Game`` exposes no
    drafting accessor), and asking the encoder again would make the comparison
    circular. These are the bits the encoder is written from.
    """
    board: Any = state.game._board  # noqa: SLF001
    return bool(board.draft_w), bool(board.draft_b)


def test_the_recorded_draft_flag_is_the_engines_in_draft_at_capture_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``PendingRow.in_draft`` is ``Game.in_draft()``, not a hand-set flag.

    The column node above pins ``draft_outcome`` against the *flag*, so it
    cannot see a wrong choice of predicate at all -- both the broad reading
    (**either** side retains drafting rights, which is this design's) and the
    narrow one (*this row's mover* is still drafting) satisfy it. This node
    pins the flag against the **engine**, which is what makes the definition
    testable rather than editorial.

    **The non-vacuity clause is required.** ``Game.in_draft()`` is
    ``self._drafting["b"] or self._drafting["w"]``, so the broad and narrow
    predicates agree on every ply where both sides have the same drafting
    status. On a game where both sides finish together this node passes against
    the narrow implementation and the definition is back to being prose. So the
    asserted plies must include at least one where the two sides' drafting
    rights **differ**, and that is asserted before the flag is.

    The recorded flag is recovered from the shard rather than from the worker's
    private buffer: ``draft_outcome`` is ``wdl_target`` on a draft row and all
    zeros off it, and ``wdl_target`` is never all-zero, so "has mass" *is* the
    flag. The engine's own answer is recomputed by replaying the recorded
    ``action`` column from a fresh game -- which is exactly the replay the
    ``action`` column exists to make possible.
    """
    monkeypatch.setattr("komugi.selfplay.worker.Game", _DraftBoundaryGame)
    directory = tmp_path / "worker_0_r0"
    writer = ShardWriter(directory, SCHEMA)
    evaluator = _FinishTheDraftEvaluator(num_actions())
    worker = SelfPlayWorker(
        evaluator,
        SelfPlayConfig(
            workers=1,
            process_batch_size=1,
            gumbel_m_full=2,
            gumbel_n_full=2,
            gumbel_m_fast=2,
            gumbel_n_fast=2,
            prob_fast=0.0,
            draw_score=0.2,
            games_in_flight=1,
        ),
        WorkerConfig(
            worker_id=0,
            incarnation=0,
            run_seed=20260906,
            games=1,
            shard_dir=str(directory),
        ),
        writer,
    )
    written = worker.play_game(0)
    writer.close()
    assert written > 0

    reader = ShardReader(directory, schema=SCHEMA)
    assert reader.skipped == ()
    rows = sorted(reader, key=lambda row: int(row["ply"]))
    assert len(rows) == written

    # Replay the recorded actions and ask the engine at every ply.
    state = GameState(_DraftBoundaryGame(), 0.2)
    engine_says: list[bool] = []
    asymmetric = 0
    for row in rows:
        white, black = _drafting_rights(state)
        if white != black:
            asymmetric += 1
        engine_says.append(white or black)
        state.play(int(row["action"]))

    recorded = [float(row["draft_outcome"].abs().sum()) > 0.0 for row in rows]

    # Non-vacuity first: without an asymmetric ply the narrow predicate would
    # pass this node and the definition would be prose again.
    assert asymmetric > 0, (
        "no recorded ply has the two sides at different drafting statuses, so "
        "this node cannot distinguish 'either side is drafting' from 'the "
        "row's mover is drafting' -- it is void as written"
    )
    assert any(engine_says) and not all(engine_says), (
        "the game did not cross the draft boundary, so there is nothing here "
        "to pin"
    )

    assert recorded == engine_says, (
        "the recorded draft flag disagrees with the engine's own in_draft() "
        f"at plies {[index for index, (a, b) in enumerate(zip(recorded, engine_says, strict=True)) if a != b]}"
    )
