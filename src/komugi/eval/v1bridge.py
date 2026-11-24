"""The cross-engine bridge: one v1 subprocess, one wire, and no silent scores.

``docs/design/kom-023-cross-engine-harness.md`` is the pre-registration this
module implements. Two of its sections, and only those two:

* **§D1 -- the adapter boundary.** The seam is at the *move*, not at the
  evaluator: a long-lived subprocess per match, speaking line-delimited JSON
  over pipes, carrying **engine move records** (§D2.3's ten structural fields
  plus ``san``) rather than action ids or observations. §D1's protocol table is
  reproduced here request by request, ``hello`` through ``bye``, including the
  three ``pick`` response types -- ``move``, ``no_move`` and, per §D3.5 as
  revised in rev 3, ``budget_breach``.
* **§D6 -- the failure modes.** Every one of M1--M10 is an *explicit outcome
  type* (:class:`BridgeVoid`, keyed by :data:`M1` .. :data:`M10`), never a
  result. §D6's opening line is the rule this module exists to keep: "A harness
  that scores a crash as a loss measures the harness."

What is deliberately **not** here, and where it goes instead:

* **v1 itself.** Nothing in this module imports, spawns or names a v1 checkout.
  :func:`spawn_v1` takes the argv to run from its caller, so the engine on the
  far end is the caller's business; the suite proves this module against a fake
  v1 stub that speaks the same wire (``tests/eval/test_v1bridge_behavior.py``).
* **§D3's instrumentation.** The *checks* §D3.6 states over ``evals_used`` and
  ``sims_used`` are enforced here, at the wire boundary, because §D6 M8/M9/M10
  are the modes that carry them. Building the counting evaluator that produces
  those numbers on **Komugi's** side (§D3.5) belongs with the ``Mover``
  refactor of ``arena.py``; this module accepts that side's breach as
  :class:`BudgetBreach` and resolves it as M8, which is the boundary half of
  the same rule.

The altitude, restated because it is the design's central finding
--------------------------------------------------------------------
§D1: "``ArenaPlayer`` + ``BatchEvaluator`` is the wrong altitude. It must not
be used for v1." Mounting v1 at the evaluator seam would drive **v1's weights
with Komugi's Gumbel search** -- a hybrid nobody trained and nobody ships. So
:meth:`BridgeEntrant.arena_player` builds a real
:class:`~komugi.eval.arena.ArenaPlayer` for everything an ``ArenaPlayer`` is
consulted for -- the rateable identity, the family, the per-player budget, and
therefore the whole of §D4's pairing -- and gives it an ``evaluator_factory``
that **raises** :class:`WrongAltitude`. A caller who seats a v1 entrant in
:func:`~komugi.eval.arena.play_match` today gets that exception naming §D1,
which is the correct answer until ``arena.py`` grows the ``Mover`` seam §D1
specifies. Bridged games are played by :func:`play_bridged_match`, whose loop
is ``arena.py:489-527`` with the search call replaced by :meth:`Mover.pick`.
"""

from __future__ import annotations

import base64
import json
import queue
import subprocess
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

from komugi.engine import action_encoding
from komugi.engine.cyboard import board_to_fen
from komugi.engine.cymovegen import generate_all_moves
from komugi.engine.game import Game
from komugi.eval.arena import (
    DEFAULT_GUMBEL_SCALE,
    DEFAULT_M,
    DEFAULT_N,
    ArenaConfig,
    ArenaPlayer,
    ArenaSummary,
    GameOutcome,
    MatchOutcome,
    game_seeds,
    new_arena_game,
    pair_id_for,
    replay_prefix,
)
from komugi.eval.records import (
    BLACK_WIN,
    DRAW,
    WHITE_WIN,
    GameRecord,
    Player,
    roster_families,
)
from komugi.search import modes
from komugi.selfplay.adapter import GameState
from komugi.selfplay.records import move_seed

# ---------------------------------------------------------------------------
# the pre-registered numbers (§D3.3, §D6)
# ---------------------------------------------------------------------------

PROTOCOL_VERSION: Final[int] = 1
"""Harness protocol version, sent in ``hello`` and echoed by the adapter.

A mismatch is a **startup** failure (§D1's handshake), not a per-move surprise:
§D6 M4 exists to catch version skew that reaches a move record, and the cheaper
place to catch it is before the first game.
"""

EVAL_CEILING: Final[int] = 257
"""Evaluations per move, per side -- §D3.3's ceiling, as a literal.

§D3.4 pins the anchor "to the numbers, not to the symbols": derived from
``DEFAULT_N`` it would silently redefine itself if the production budget moved,
and every previously measured anchor would become incomparable with no diff
anywhere. :func:`assert_pinned_budget` is the other half of that rule (A15).
"""

PICK_DEADLINE_S: Final[float] = 60.0
"""§D6 M2's per-``pick`` deadline: the same order as
``InferenceConfig.request_timeout_s`` (``inference.py:505``). Fatal by policy
and never retried -- retrying a search after a timeout is the stale-answer race
``inference.py:321-328`` exists to forbid."""

HANDSHAKE_DEADLINE_S: Final[float] = 120.0
"""Deadline for ``hello``. Wider than :data:`PICK_DEADLINE_S` because the far
side loads a checkpoint under it."""

CRASH_ABORT_THRESHOLD: Final[int] = 3
"""§D6: "three crashes in one match abort the match". An **absolute** count, and
per the precedence rules it fires first and is terminal."""

VOID_RATE_GATE: Final[float] = 0.02
"""§D6's admissibility gate: a match whose void rate exceeds 2% of pairs
attempted is not admissible as anchor evidence."""

V1_FAMILY: Final[str] = "v1"
"""The entrant family for a v1 anchor, one of
:data:`~komugi.eval.records.KNOWN_FAMILIES` (``records.py:39-41``).

§D4: v1 entrants are constructed directly and **never** through
``ladder_players``, which hardcodes ``LADDER_FAMILY`` and the checkpoint
evaluator (``arena.py:735-736``).
"""

_WHITE: Final[int] = 0
"""The engine's index for the white seat, as ``arena.py:155-156`` reads it."""

_DRAW_INDEX: Final[int] = 2
"""Draw slot of the engine's absolute ``[white, black, draw]`` outcome."""

# ---------------------------------------------------------------------------
# §D1's protocol table: request and response type tags
# ---------------------------------------------------------------------------

HELLO: Final[str] = "hello"
RESET: Final[str] = "reset"
OBSERVE: Final[str] = "observe"
PICK: Final[str] = "pick"
PROBE: Final[str] = "probe"
BYE: Final[str] = "bye"

REQUESTS: Final[frozenset[str]] = frozenset({HELLO, RESET, OBSERVE, PICK, PROBE, BYE})
"""Every request §D1's protocol table names, and nothing else."""

OK: Final[str] = "ok"
MOVE: Final[str] = "move"
NO_MOVE: Final[str] = "no_move"
BUDGET_BREACH: Final[str] = "budget_breach"
OBSERVATION: Final[str] = "observation"

PICK_RESPONSES: Final[frozenset[str]] = frozenset({MOVE, NO_MOVE, BUDGET_BREACH})
"""§D1: ``pick`` answers with **one of three response types**.

``no_move`` and ``budget_breach`` are first-class response types rather than
fields on a ``move`` for the same structural reason, one layer apart (review
F5, re-review F1): in both states there **is** no move record to send, so a
detector hung on a field of a ``move`` response could never fire, and the event
would be misattributed to ``crash``/``protocol_error`` -- taking with it the
match-level remedy that exists solely for the side that can trip it.
"""

# ---------------------------------------------------------------------------
# §D6's modes, as identifiers rather than prose
# ---------------------------------------------------------------------------

M1: Final[str] = "M1"
"""v1 process exits or segfaults mid-game."""
M2: Final[str] = "M2"
"""v1 hangs past :data:`PICK_DEADLINE_S`."""
M3: Final[str] = "M3"
"""v1 proposes an illegal move."""
M4: Final[str] = "M4"
"""v1 proposes a malformed record, or speaks non-protocol."""
M5: Final[str] = "M5"
"""Rules divergence: v1 thinks the game is over, Komugi does not."""
M6: Final[str] = "M6"
"""Rules divergence the other way. **Not a failure** -- a diagnostic."""
M7: Final[str] = "M7"
"""Komugi's engine raises on the decoded action."""
M8: Final[str] = "M8"
"""A 258th evaluation is requested in one move, on either side."""
M9: Final[str] = "M9"
"""v1 carried a tree across a move boundary."""
M10: Final[str] = "M10"
"""Per-move identity violated on either side."""


class VoidScope(Enum):
    """What a void takes with it (§D6, "Precedence between the two thresholds").

    ``MATCH`` is not a bigger ``PAIR``: a match-void **is an abort** (rev 3,
    re-review F8). Its already-completed pairs are discarded, no ``MatchSet`` is
    emitted, and the run is reported as ``voided(P2|P3, pairs_completed)``.
    M8's own argument is why -- "voiding only the affected pair would keep
    exactly the games where the discrepancy happened not to show".
    """

    PAIR = "pair"
    MATCH = "match"


class VoidBucket(Enum):
    """§D6's void buckets, plus the two the match-level modes report under.

    Voids are **never** folded into ``games``, ``draws`` or ``decisive``
    (§D6 "Void accounting"); they are counted here and nowhere else.
    """

    CRASH = "crash"
    TIMEOUT = "timeout"
    ILLEGAL_MOVE = "illegal_move"
    PROTOCOL_ERROR = "protocol_error"
    RULES_DIVERGENCE = "rules_divergence"
    BUDGET_INVARIANT = "budget_invariant"
    BUDGET_BREACH = "budget_breach"
    TREE_REUSE = "tree_reuse"


@dataclass(frozen=True, slots=True)
class BridgeVoid:
    """One §D6 failure, with everything a rated harness needs to void on.

    Constructed at the point of detection and carried out of the game loop by
    :class:`BridgeVoided`. Nothing here is optional decoration: ``stderr`` and
    ``exit_status`` are what §D6 M1 requires "in the record", ``fen`` and
    ``record`` are what M3 requires logged, and the three budget fields are the
    quantities §D3.6's P1--P3 are stated over -- the ones that make a breach
    *quantified* rather than merely flagged.

    Args:
        mode: The §D6 mode, :data:`M1` .. :data:`M10`.
        bucket: The void bucket the mode books into.
        scope: Pair or match (§D6's precedence section).
        detail: Human-readable cause, always naming the observed values.
        ply: The ply at which it was detected, when a game was in progress.
        invariant: ``"P1"``, ``"P2"`` or ``"P3"`` for the modes that carry one.
        exit_status: The subprocess's exit status for M1, when it had one.
        stderr: v1's captured stderr, verbatim (§D1).
        evals_used: As reported on the offending ``pick``, when reported.
        sims_used: Likewise.
        tree_nodes_at_entry: Likewise -- the wire-level proof of §D3.2 clause 2.
        record: The offending move record, for M3/M4.
        fen: The position the offence was committed in, for M3.
        player_id: Which entrant produced it.
        pair: Pair index of the game.
        seat: Seat index within the pair.
    """

    mode: str
    bucket: VoidBucket
    scope: VoidScope
    detail: str
    ply: int | None = None
    invariant: str | None = None
    exit_status: int | None = None
    stderr: str = ""
    evals_used: int | None = None
    sims_used: int | None = None
    tree_nodes_at_entry: int | None = None
    record: MoveRecord | None = None
    fen: str | None = None
    player_id: str = ""
    pair: int | None = None
    seat: int | None = None


class BridgeVoided(Exception):
    """Raised at the wire boundary to void a pair or a match.

    Carries the :class:`BridgeVoid` rather than a message, because every caller
    that catches it needs the structured detail and none of them should be
    parsing a string to get it.
    """

    def __init__(self, void: BridgeVoid) -> None:
        super().__init__(f"{void.mode} ({void.bucket.value}): {void.detail}")
        self.void = void


class BudgetBreach(Exception):
    """Komugi's side of §D3.5's report-and-refuse: a 258th ask in one move.

    §D3.5: the Komugi-side counting decorator raises this in-process and the
    game loop catches it at the arena boundary, resolving M8 directly -- "there
    is no process boundary to lose it across, and it must not be allowed to
    propagate as a crash of the arena". The decorator that raises it is §D3.5's
    Komugi-side instrumentation and lands with the ``arena.py`` ``Mover``
    refactor; :func:`play_bridged_game` already resolves it, so the two halves
    of M8 -- a live-connection ``budget_breach`` response and this exception --
    reach the same match-level void from either side.

    Args:
        evals_used: The count at the point of refusal. The refusal is a
            *construction check*, not a truncation mechanism: nothing truncated
            is ever scored, because the refusal voids the match, not the move.
        player_id: Which entrant asked.
    """

    def __init__(self, evals_used: int, player_id: str = "") -> None:
        super().__init__(
            f"{player_id or 'a player'} asked for evaluation {evals_used} within "
            f"one move; the pre-registered ceiling is {EVAL_CEILING} (§D3.3)"
        )
        self.evals_used = evals_used
        self.player_id = player_id


class HandshakeRefused(Exception):
    """The adapter is refused at match construction, before any game is played.

    §D1's capability gate (A13). "An adapter that does not declare
    ``supports_budget_breach`` is refused at match construction" -- without that
    gate a non-conforming adapter that simply dies on a breach re-creates the
    re-review F1 collision exactly: the breach becomes an M1 ``crash``, a pair
    void that counts toward "three crashes abort", and the match-level remedy
    never fires for the only side that can trip it. With it, that outcome is a
    **startup** failure rather than a silently mis-bucketed rated match.
    """


class WrongAltitude(Exception):
    """Raised by a bridged entrant's ``evaluator_factory`` -- see §D1.

    Not a stub and not a TODO: mounting v1 at the ``BatchEvaluator`` seam is a
    design error this module refuses to make representable. See the module
    docstring.
    """


class ProtocolViolation(Exception):
    """Internal: the wire carried something that is not this protocol.

    Always converted to an M4 :class:`BridgeVoid` at the boundary that caught
    it, so no caller outside this module ever sees one.
    """


class _TransportEOF(Exception):
    """Internal: the far end closed its pipe or died. Becomes M1."""

    def __init__(self, exit_status: int | None) -> None:
        super().__init__(f"v1 process ended, exit status {exit_status!r}")
        self.exit_status = exit_status


class _TransportTimeout(Exception):
    """Internal: no answer within the deadline. Becomes M2."""


# ---------------------------------------------------------------------------
# §D2.3's wire form: the move record, minus nothing, plus a checked san
# ---------------------------------------------------------------------------

STRUCTURAL_FIELDS: Final[tuple[str, ...]] = (
    "from_r",
    "from_f",
    "to_r",
    "to_f",
    "to_tier",
    "move_type",
    "piece_type_index",
    "color",
    "captured",
    "draft_finished",
)
"""§D2.3's ten structural fields, in ``_make_move_record``'s own order.

The two engines' move-record dicts are structurally identical field for field
(``src/komugi/engine/cymovegen.pyx:481-492`` against
``docs/reference-v1/alphazero-env/cymovegen.pyx:429-440``), and the record --
not the action id -- is what each engine actually applies (``game.py:358-367``;
v1 identically at ``gungi.py:248-254``). That is what makes the record the wire
form: the two action spaces are **not** interchangeable (15390 against 37341),
Komugi -> v1 is lossy *and partial*, and no id ever crosses this wire.
"""

_MAX_SQUARE: Final[int] = 8
_MAX_TIER: Final[int] = 3
_MAX_MOVE_TYPE: Final[int] = 4
_MAX_PIECE_TYPE_INDEX: Final[int] = 13
_MAX_CONVERTIBLE_TIER: Final[int] = 2
"""A conversion above tier 2 is unreachable (``action_encoding.py:145-149``)."""


def _require_int(payload: Mapping[str, Any], field: str) -> int:
    """Return `payload[field]` as an ``int``, or raise :class:`ProtocolViolation`.

    ``bool`` is rejected rather than coerced: ``True`` is a perfectly good
    ``int`` in Python and would otherwise pass silently as square 1, or as an
    ``evals_used`` of 1.
    """
    if field not in payload:
        raise ProtocolViolation(f"message is missing required field {field!r}")
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolViolation(f"field {field!r} must be an int, got {value!r}")
    return int(value)


def _require_range(value: int, field: str, low: int, high: int) -> int:
    """Return `value`, or raise :class:`ProtocolViolation` if outside ``[low, high]``."""
    if not low <= value <= high:
        raise ProtocolViolation(
            f"field {field!r} is {value}, outside [{low}, {high}]"
        )
    return value


def _parse_captured(value: object) -> tuple[tuple[int, int], ...]:
    """Parse the ``captured`` field: a list of ``(type_index, tier)`` pairs.

    The shape is ``sel_from_captured``'s cymovegen shape
    (``action_encoding.py:133-151``), which is what the conversion selector --
    and therefore the whole betray family of §D2.3 -- is keyed on. An empty
    list is selector 0, i.e. a plain tsuke or a plain drop.
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ProtocolViolation(f"move record field 'captured' must be a list, got {value!r}")
    parsed: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ProtocolViolation(
                f"captured[{index}] must be a (type_index, tier) pair, got {item!r}"
            )
        type_index, tier = item
        if isinstance(type_index, bool) or not isinstance(type_index, int):
            raise ProtocolViolation(f"captured[{index}] type index is {type_index!r}")
        if isinstance(tier, bool) or not isinstance(tier, int):
            raise ProtocolViolation(f"captured[{index}] tier is {tier!r}")
        _require_range(int(type_index), f"captured[{index}].type", 0, _MAX_PIECE_TYPE_INDEX)
        _require_range(int(tier), f"captured[{index}].tier", 1, _MAX_CONVERTIBLE_TIER)
        parsed.append((int(type_index), int(tier)))
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class MoveRecord:
    """§D2.3's wire form: the ten structural fields, with ``san`` alongside.

    ``san`` is carried as a **redundant cross-check** (acceptance A3), never as
    the identity of the move: :meth:`key` -- what an incoming record is matched
    against ``generate_all_moves`` by -- is the ten structural fields and does
    not include it. A disagreeing ``san`` is a reportable diagnostic, not a
    void, because the two engines' notation writers are not part of the
    pre-registered contract while the ten fields are.
    """

    from_r: int
    from_f: int
    to_r: int
    to_f: int
    to_tier: int
    move_type: int
    piece_type_index: int
    color: int
    captured: tuple[tuple[int, int], ...]
    draft_finished: bool
    san: str = ""

    def key(self) -> tuple[Any, ...]:
        """Return the identity of this move: the ten structural fields."""
        return (
            self.from_r,
            self.from_f,
            self.to_r,
            self.to_f,
            self.to_tier,
            self.move_type,
            self.piece_type_index,
            self.color,
            self.captured,
            self.draft_finished,
        )

    def to_wire(self) -> dict[str, Any]:
        """Return the JSON-safe payload: ten fields plus ``san``."""
        return {
            "from_r": self.from_r,
            "from_f": self.from_f,
            "to_r": self.to_r,
            "to_f": self.to_f,
            "to_tier": self.to_tier,
            "move_type": self.move_type,
            "piece_type_index": self.piece_type_index,
            "color": self.color,
            "captured": [list(pair) for pair in self.captured],
            "draft_finished": self.draft_finished,
            "san": self.san,
        }

    def to_engine(self) -> dict[str, Any]:
        """Return the engine's own record dict, for ``encode_cython_move``."""
        engine = self.to_wire()
        engine["captured"] = [tuple(pair) for pair in self.captured]
        return engine

    @classmethod
    def from_engine(cls, record: Mapping[str, Any]) -> MoveRecord:
        """Build from a ``cymovegen`` record dict (the engine's own shape).

        The engine is trusted about its own records -- this is the *encode*
        direction, where the record was produced by ``generate_all_moves`` or
        ``decode_cython_action`` a moment ago. :meth:`from_wire` is the
        untrusted direction and validates every field.
        """
        captured = record.get("captured") or ()
        return cls(
            from_r=int(record["from_r"]),
            from_f=int(record["from_f"]),
            to_r=int(record["to_r"]),
            to_f=int(record["to_f"]),
            to_tier=int(record["to_tier"]),
            move_type=int(record["move_type"]),
            piece_type_index=int(record["piece_type_index"]),
            color=int(record["color"]),
            captured=tuple(
                (int(item[0]), int(item[1])) for item in captured
            ),
            draft_finished=bool(record["draft_finished"]),
            san=str(record.get("san", "")),
        )

    @classmethod
    def from_wire(cls, payload: object) -> MoveRecord:
        """Parse and **validate** a record off the wire -- §D6 M4's schema check.

        Every failure here is M4 (``protocol_error``) and not M3
        (``illegal_move``): §D6 distinguishes them because "M3 is a rules
        question and M4 is a version-skew question", and folding the two would
        make a field the two engines disagree about the *shape* of read as a
        disagreement about the *rules*.

        Raises:
            ProtocolViolation: On a missing field, a wrong type, or a square,
                tier, move type or piece index outside the engine's range.
        """
        if not isinstance(payload, dict):
            raise ProtocolViolation(f"move record must be an object, got {payload!r}")
        for field in STRUCTURAL_FIELDS:
            if field not in payload:
                raise ProtocolViolation(f"move record is missing {field!r}")
        draft_finished = payload["draft_finished"]
        if not isinstance(draft_finished, bool):
            raise ProtocolViolation(
                f"move record field 'draft_finished' must be a bool, got "
                f"{draft_finished!r}"
            )
        san = payload.get("san", "")
        if not isinstance(san, str):
            raise ProtocolViolation(f"move record field 'san' must be a string, got {san!r}")
        # A drop has no origin square and says so with -1 (cymovegen.pyx:481-492).
        from_r = _require_range(_require_int(payload, "from_r"), "from_r", -1, _MAX_SQUARE)
        from_f = _require_range(_require_int(payload, "from_f"), "from_f", -1, _MAX_SQUARE)
        return cls(
            from_r=from_r,
            from_f=from_f,
            to_r=_require_range(_require_int(payload, "to_r"), "to_r", 0, _MAX_SQUARE),
            to_f=_require_range(_require_int(payload, "to_f"), "to_f", 0, _MAX_SQUARE),
            to_tier=_require_range(
                _require_int(payload, "to_tier"), "to_tier", 0, _MAX_TIER
            ),
            move_type=_require_range(
                _require_int(payload, "move_type"), "move_type", 0, _MAX_MOVE_TYPE
            ),
            piece_type_index=_require_range(
                _require_int(payload, "piece_type_index"),
                "piece_type_index",
                0,
                _MAX_PIECE_TYPE_INDEX,
            ),
            color=_require_range(_require_int(payload, "color"), "color", 0, 1),
            captured=_parse_captured(payload["captured"]),
            draft_finished=draft_finished,
            san=san,
        )


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------


def encode_message(payload: Mapping[str, Any]) -> str:
    """Return one wire line: JSON, one object, newline-terminated (§D1).

    ``sort_keys`` so that two harnesses at the same protocol version produce
    byte-identical lines for the same message, which is what makes a recorded
    transcript comparable. JSON escapes any embedded newline, so the framing
    cannot be broken by a payload.

    Raises:
        ProtocolViolation: If `payload` carries no ``type``.
    """
    if "type" not in payload:
        raise ProtocolViolation(f"a message must carry a 'type', got {dict(payload)!r}")
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def decode_message(line: str) -> dict[str, Any]:
    """Parse one wire line into a message dict -- the M4 boundary.

    Raises:
        ProtocolViolation: On invalid JSON, a non-object, or a missing or
            non-string ``type``. Every one of those is §D6 M4.
    """
    text = line.strip()
    if not text:
        raise ProtocolViolation("empty line: the protocol is one message per line")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProtocolViolation(f"not JSON: {error} in {text[:200]!r}") from error
    if not isinstance(parsed, dict):
        raise ProtocolViolation(f"a message must be a JSON object, got {parsed!r}")
    kind = parsed.get("type")
    if not isinstance(kind, str):
        raise ProtocolViolation(f"a message must carry a string 'type', got {kind!r}")
    return {str(key): value for key, value in parsed.items()}


# ---------------------------------------------------------------------------
# §D1's handshake blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityBlock:
    """``hello``'s capability block (§D1, new in rev 3).

    It exists so the two report-and-refuse detectors cannot be silently
    downgraded by an adapter that does not implement them. ``memo_separable``
    answers §D3.2 clause 1's memo-shape question and carries the ``file:line``
    the U-A report cites; it is recorded, not gated, because the ceiling
    survives either shape -- what it changes is whether acceptance A7e is a
    witness or is recorded **vacuous**.
    """

    supports_no_move: bool
    supports_budget_breach: bool
    memo_separable: bool
    memo_citation: str = ""

    @classmethod
    def from_wire(cls, payload: object) -> CapabilityBlock:
        """Parse the block, raising :class:`ProtocolViolation` on a bad shape."""
        if not isinstance(payload, dict):
            raise ProtocolViolation(f"capability block must be an object, got {payload!r}")
        for field in ("supports_no_move", "supports_budget_breach", "memo_separable"):
            if not isinstance(payload.get(field), bool):
                raise ProtocolViolation(
                    f"capability block field {field!r} must be a bool, got "
                    f"{payload.get(field)!r}"
                )
        citation = payload.get("memo_citation", "")
        return cls(
            supports_no_move=bool(payload["supports_no_move"]),
            supports_budget_breach=bool(payload["supports_budget_breach"]),
            memo_separable=bool(payload["memo_separable"]),
            memo_citation=str(citation) if citation is not None else "",
        )


@dataclass(frozen=True, slots=True)
class DeviceBlock:
    """``hello``'s device + determinism block (§D1, §D5; review F19).

    ``pin_determinism()`` is process-global torch state (``arena.py:165-192``)
    and the v1 subprocess is a different process, so none of it reaches v1.
    Without this block an A9 failure caused by cuDNN autotuning in the v1
    process is indistinguishable from the RNG-discipline failure U-D predicts,
    and the degraded determinism claim gets shipped where a two-line fix in the
    adapter would have earned the strong one.
    """

    torch_version: str = ""
    numpy_version: str = ""
    device: str = ""
    cudnn_benchmark: bool = False
    deterministic_algorithms: bool = False

    @classmethod
    def from_wire(cls, payload: object) -> DeviceBlock:
        """Parse the block; absent fields read as unstated rather than raising."""
        if not isinstance(payload, dict):
            raise ProtocolViolation(f"device block must be an object, got {payload!r}")
        return cls(
            torch_version=str(payload.get("torch_version", "")),
            numpy_version=str(payload.get("numpy_version", "")),
            device=str(payload.get("device", "")),
            cudnn_benchmark=bool(payload.get("cudnn_benchmark", False)),
            deterministic_algorithms=bool(
                payload.get("deterministic_algorithms", False)
            ),
        )


@dataclass(frozen=True, slots=True)
class HelloResponse:
    """What the adapter answers ``hello`` with -- §D1's protocol table, row 1."""

    protocol_version: int
    checkpoint_digest: str
    action_size: int
    gungi_mode: str
    capabilities: CapabilityBlock
    device: DeviceBlock

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> HelloResponse:
        """Parse the response, raising :class:`ProtocolViolation` on a bad shape."""
        for field in ("protocol_version", "action_size"):
            value = payload.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ProtocolViolation(
                    f"hello field {field!r} must be an int, got {value!r}"
                )
        return cls(
            protocol_version=int(payload["protocol_version"]),
            checkpoint_digest=str(payload.get("checkpoint_digest", "")),
            action_size=int(payload["action_size"]),
            gungi_mode=str(payload.get("gungi_mode", "")),
            capabilities=CapabilityBlock.from_wire(payload.get("capabilities")),
            device=DeviceBlock.from_wire(payload.get("device", {})),
        )


# ---------------------------------------------------------------------------
# §D1's three pick responses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PickMove:
    """``pick`` -> ``move``: a record, the budget triple, and v1's own view.

    ``win_state`` is carried **only** as §D6's divergence detector, never as an
    adjudication: Komugi's ``Game`` is the sole referee (§D6, "The referee
    rule").
    """

    record: MoveRecord
    evals_used: int
    sims_used: int
    tree_nodes_at_entry: int
    win_state: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True, slots=True)
class PickNoMove:
    """``pick`` -> ``no_move``: §D6 M5, and it is **expected to occur**.

    v1's ``valid_moves`` returns an all-zero mask where its cells-only key has
    repeated four times and Komugi's repaired 274-byte key has not
    (``rules-audit.md`` F1, Scenarios A and B), so v1's search has no move to
    return at all. A distinct response type is the only way this reaches the
    harness as itself: with an all-zero mask v1's MCTS raises, divides by zero,
    or returns an arbitrary index, and the harness would observe M1, M4 or M3.
    """

    win_state: tuple[int, int, int]
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PickBudgetBreach:
    """``pick`` -> ``budget_breach``: §D3.5's report-and-refuse, §D6 M8.

    The counted evaluator refused the 258th ask, unwound the search, discarded
    the move and reported. **The process does not exit** -- that is the whole
    point of the response type (re-review F1): a raise inside the v1 process is
    a process death, which arrives as M1, a *pair* void counting toward "three
    crashes abort", and M8 -- the match-level remedy that exists solely for the
    only side that can trip P2 -- would be unreachable.
    """

    evals_used: int
    sims_used: int
    tree_nodes_at_entry: int


PickResponse = PickMove | PickNoMove | PickBudgetBreach
"""§D1: ``pick`` answers with exactly one of three types."""


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------


class _Channel:
    """Line-oriented, deadline-bounded IO over one subprocess's pipes.

    Both pipes are drained by reader threads rather than read inline, for two
    independent reasons that would each be a bug on their own:

    * **M2 needs a deadline.** A blocking ``readline`` on a hung child hangs the
      harness with it, and there is no portable timeout on a pipe read. A queue
      with a timeout gives §D6 M2 a real deadline on every platform.
    * **stderr must not deadlock the wire.** §D1 requires v1's stderr captured
      verbatim into the void record. An undrained stderr pipe fills its buffer
      and blocks the child *inside its write*, which would then be observed as
      M2 -- a hang the harness caused itself.
    """

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._errors: list[str] = []
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout, name="v1bridge-stdout", daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr, name="v1bridge-stderr", daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _pump_stdout(self) -> None:
        """Feed every stdout line into the queue; ``None`` marks EOF."""
        stream = self._process.stdout
        if stream is not None:
            for line in stream:
                self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self) -> None:
        """Accumulate stderr verbatim for the void record (§D1)."""
        stream = self._process.stderr
        if stream is None:
            return
        for line in stream:
            self._errors.append(line)

    @property
    def stderr_text(self) -> str:
        """Return everything the child has written to stderr so far."""
        return "".join(self._errors)

    def exit_status(self) -> int | None:
        """Return the child's exit status, or ``None`` while it is alive."""
        return self._process.poll()

    def is_alive(self) -> bool:
        """Return whether the child is still running."""
        return self._process.poll() is None

    def send(self, payload: Mapping[str, Any]) -> None:
        """Write one message. A closed pipe is a death, i.e. §D6 M1.

        Raises:
            _TransportEOF: If the child is gone or its stdin is closed.
        """
        stream = self._process.stdin
        if stream is None:
            raise _TransportEOF(self.exit_status())
        try:
            stream.write(encode_message(payload))
            stream.flush()
        except (BrokenPipeError, ValueError, OSError) as error:
            raise _TransportEOF(self._reap()) from error

    def receive(self, deadline_s: float) -> dict[str, Any]:
        """Read one message, waiting at most `deadline_s`.

        Raises:
            _TransportTimeout: No line within the deadline -- §D6 M2.
            _TransportEOF: The child closed stdout or died -- §D6 M1.
            ProtocolViolation: The line was not one protocol message -- M4.
        """
        try:
            line = self._lines.get(timeout=deadline_s)
        except queue.Empty:
            raise _TransportTimeout(
                f"no response within {deadline_s:.1f}s"
            ) from None
        if line is None:
            raise _TransportEOF(self._reap())
        return decode_message(line)

    def _reap(self) -> int | None:
        """Return the child's exit status, waiting briefly for it to settle."""
        try:
            return int(self._process.wait(timeout=5.0))
        except subprocess.TimeoutExpired:
            return self._process.poll()

    def kill(self) -> int | None:
        """Kill the child and return its exit status (M2's remedy)."""
        if self._process.poll() is None:
            self._process.kill()
        return self._reap()

    def close(self) -> None:
        """Close the pipes and make sure the child is gone."""
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        if self._process.poll() is None:
            self._process.kill()
            self._reap()
        self._stdout_thread.join(timeout=1.0)
        self._stderr_thread.join(timeout=1.0)


@dataclass(frozen=True, slots=True)
class V1Config:
    """Everything ``hello`` tells the adapter, plus how to start it.

    Args:
        argv: The command that starts the adapter. **Supplied by the caller**:
            nothing in this module knows where a v1 checkout lives, which is
            what keeps §0.2's U-C (v1's torch/numpy pins) a non-question and
            what lets the test suite prove the protocol against a stub.
        gungi_mode: The harness's ``GUNGI_MODE``. The adapter echoes its
            **resolved** mode and a mismatch is a fatal startup error, not a
            warning (§D4 amendment 2, given §0.2 U-E).
        checkpoint: The v1 checkpoint path, opaque to this module.
        ceiling: Per-move evaluation ceiling, §D3.3's 257.
        sims_allowance: ``S₁`` -- the simulation allowance whose arithmetic
            maximum is `ceiling`. 256 if the adapter evaluates its root outside
            the simulation loop as Komugi does (``gumbel.py:499``), 257 if the
            root is expanded inside the first simulation. Which one holds is a
            U-A fact the adapter's report must state with a ``file:line``.
        seed_policy: How the harness seeds the adapter, for the record. §D5
            seeds **per ply** from ``move_seed_for``, not per game.
        env: Extra environment for the child, merged over the parent's.
        cwd: Working directory for the child.
        pick_deadline_s: §D6 M2's deadline.
    """

    argv: tuple[str, ...]
    gungi_mode: str
    checkpoint: str = ""
    ceiling: int = EVAL_CEILING
    sims_allowance: int = 256
    seed_policy: str = "per-ply"
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    pick_deadline_s: float = PICK_DEADLINE_S


def resolved_gungi_mode() -> str:
    """Return the start mode this process's engine resolved ``GUNGI_MODE`` to.

    Read from the engine rather than from the environment: ``game.py:198``
    reads the variable **at import time**, so a test that set it afterwards
    would otherwise be told a mode the engine is not playing.
    """
    return str(Game._START_MODE)


def bridged_book_id(mode: str) -> str:
    """Return §D4 amendment 2's ``book_id``: ``"empty-<mode>"``.

    ``EMPTY_BOOK`` means "the engine's empty board" (``arena.py:136``), but
    *which* position that is depends on ``GUNGI_MODE``. With two processes two
    values are possible, and a bare ``"empty"`` would then be a lie shared by
    both games of every pair -- undetectable by the audit, which only checks
    that the two agree (``seats.py:143-149``).
    """
    if not mode.strip():
        raise ValueError("a start mode must be a non-empty identifier")
    return f"empty-{mode}"


def assert_pinned_budget() -> None:
    """Assert §D3.3's literals on the cross-engine path -- acceptance A15.

    The arena's derivation idiom (``arena.py:111-124``) is right for a *ladder*
    and exactly wrong for a **frozen** anchor: if ``gumbel_n_full`` ever moves,
    the anchor's compute silently redefines itself while the design still says
    257, every previously measured anchor becomes incomparable, and no diff
    anywhere shows it. One line, and it converts a silent invalidation of four
    frozen anchors into a build break.

    Raises:
        ValueError: If the production defaults have drifted from the values
            §D3.3 was pre-registered against.
    """
    if (DEFAULT_M, DEFAULT_N, DEFAULT_GUMBEL_SCALE) != (32, 256, 1.0):
        raise ValueError(
            f"the cross-engine path is pre-registered at m=32, n=256, "
            f"gumbel_scale=1.0 with a ceiling of {EVAL_CEILING} evaluations per "
            f"move (kom-023 §D3.3/§D3.4, acceptance A15), but this build's "
            f"arena defaults are m={DEFAULT_M}, n={DEFAULT_N}, "
            f"gumbel_scale={DEFAULT_GUMBEL_SCALE}. A frozen anchor measured "
            f"under the new numbers is not comparable to one measured under the "
            f"old; re-pre-register §D3 rather than relaxing this check"
        )


def spawn_v1(config: V1Config) -> V1Connection:
    """Start the adapter, shake hands, and gate on its capability block.

    Every refusal here is a **startup** failure, before any game is played,
    which is the entire design intent of §D1's capability block: a
    non-conforming adapter that would have mis-bucketed a real breach as a
    crash never reaches a rated game.

    Raises:
        HandshakeRefused: On a protocol version mismatch, a ``GUNGI_MODE``
            mismatch (§D4 amendment 2), an action-size mismatch, an undeclared
            ``supports_budget_breach`` or ``supports_no_move`` (A13), or a
            handshake that does not arrive.
    """
    process = subprocess.Popen(  # noqa: S603 - argv is the caller's own
        list(config.argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=dict(config.env) if config.env is not None else None,
        cwd=config.cwd,
    )
    channel = _Channel(process)
    try:
        channel.send(
            {
                "type": HELLO,
                "protocol_version": PROTOCOL_VERSION,
                "gungi_mode": config.gungi_mode,
                "checkpoint": config.checkpoint,
                "ceiling": config.ceiling,
                "sims_allowance": config.sims_allowance,
                "seed_policy": config.seed_policy,
            }
        )
        message = channel.receive(HANDSHAKE_DEADLINE_S)
        hello = HelloResponse.from_wire(message)
    except (_TransportEOF, _TransportTimeout, ProtocolViolation) as error:
        stderr = channel.stderr_text
        channel.close()
        raise HandshakeRefused(
            f"the v1 adapter {list(config.argv)!r} did not complete the §D1 "
            f"handshake: {error}"
            + (f"\n-- v1 stderr --\n{stderr}" if stderr else "")
        ) from error

    def refuse(reason: str) -> HandshakeRefused:
        """Close the connection and build the refusal."""
        stderr = channel.stderr_text
        channel.close()
        return HandshakeRefused(
            reason + (f"\n-- v1 stderr --\n{stderr}" if stderr else "")
        )

    if hello.protocol_version != PROTOCOL_VERSION:
        raise refuse(
            f"protocol version mismatch: the harness speaks {PROTOCOL_VERSION}, "
            f"the adapter speaks {hello.protocol_version}"
        )
    if hello.gungi_mode != config.gungi_mode:
        raise refuse(
            f"GUNGI_MODE mismatch: the harness starts games from "
            f"{config.gungi_mode!r}, the adapter resolved {hello.gungi_mode!r}. "
            f"§D4 amendment 2 makes this fatal at startup rather than a warning: "
            f"both games of every pair would otherwise share a book_id that names "
            f"a position only one of them was played from"
        )
    if hello.action_size <= 0:
        raise refuse(f"the adapter reports action_size={hello.action_size}")
    if not hello.capabilities.supports_budget_breach:
        raise refuse(
            "the adapter does not declare supports_budget_breach, so a 258th "
            "evaluation in one move would reach this harness as a process death "
            "-- §D6 M1, bucket 'crash', a pair void counting toward 'three "
            "crashes abort the match' -- and M8, the match-level remedy that "
            "exists solely for the side that can trip P2, could never fire "
            "(kom-023 §D1 capability block, §D3.5, acceptance A13)"
        )
    if not hello.capabilities.supports_no_move:
        raise refuse(
            "the adapter does not declare supports_no_move, so a rules "
            "divergence (§D6 M5, expected to occur) would be misattributed to "
            "crash, protocol_error or illegal_move"
        )
    return V1Connection(channel, hello, config)


class V1Connection:
    """A live v1 adapter: one process, one match (§D1).

    "Process construction is amortised the same way the arena already amortises
    network construction" (``arena.py:598-601``): **one v1 process per match**,
    not per game, with :meth:`reset` starting the next game inside the live
    process.

    Built by :func:`spawn_v1`, which is the only place the handshake and its
    refusals live.
    """

    def __init__(
        self, channel: _Channel, hello: HelloResponse, config: V1Config
    ) -> None:
        self._channel = channel
        self._hello = hello
        self._config = config

    @property
    def hello(self) -> HelloResponse:
        """Return the handshake, for the anchor's configuration block (P5)."""
        return self._hello

    @property
    def stderr_text(self) -> str:
        """Return v1's captured stderr (§D1; §D6 M1's void record)."""
        return self._channel.stderr_text

    def is_alive(self) -> bool:
        """Return whether the adapter process is still running."""
        return self._channel.is_alive()

    def exit_status(self) -> int | None:
        """Return the adapter's exit status, or ``None`` while it lives."""
        return self._channel.exit_status()

    def request(self, payload: Mapping[str, Any], deadline_s: float) -> dict[str, Any]:
        """Send one request and return the response message."""
        self._channel.send(payload)
        return self._channel.receive(deadline_s)

    def expect_ok(self, payload: Mapping[str, Any], deadline_s: float) -> None:
        """Send a request whose only valid answer is ``ok``.

        Raises:
            ProtocolViolation: On any other response type -- §D6 M4.
        """
        message = self.request(payload, deadline_s)
        kind = message["type"]
        if kind != OK:
            raise ProtocolViolation(
                f"expected {OK!r} in reply to {payload['type']!r}, got {kind!r}"
            )

    def reset(self, *, pair: int, seat: int) -> None:
        """Start a new game inside the live process (§D1's ``reset``)."""
        self.expect_ok(
            {"type": RESET, "pair": pair, "seat": seat}, self._config.pick_deadline_s
        )

    def observe(self, record: MoveRecord) -> None:
        """Tell the adapter one played move -- either side's (§D1).

        Not optional and not a convenience: the v1 process holds its own engine
        state (§D2.2) and has **no other way** to learn what the opponent
        played.
        """
        self.expect_ok(
            {"type": OBSERVE, "record": record.to_wire()},
            self._config.pick_deadline_s,
        )

    def pick(self, *, ply: int, seed: int) -> PickResponse:
        """Ask for a move, and parse whichever of the three responses arrives.

        Raises:
            ProtocolViolation: If the response is not one of §D1's three types,
                or is one of them malformed -- §D6 M4 in both cases.
        """
        message = self.request(
            {"type": PICK, "ply": ply, "seed": seed}, self._config.pick_deadline_s
        )
        kind = message["type"]
        if kind not in PICK_RESPONSES:
            raise ProtocolViolation(
                f"a 'pick' is answered by one of {sorted(PICK_RESPONSES)} "
                f"(§D1), got {kind!r}"
            )
        if kind == BUDGET_BREACH:
            return PickBudgetBreach(
                evals_used=_require_int(message, "evals_used"),
                sims_used=_require_int(message, "sims_used"),
                tree_nodes_at_entry=_require_int(message, "tree_nodes_at_entry"),
            )
        if kind == NO_MOVE:
            return PickNoMove(
                win_state=_parse_win_state(message.get("win_state")),
                reason=str(message.get("reason", "")),
            )
        return PickMove(
            record=MoveRecord.from_wire(message.get("record")),
            evals_used=_require_int(message, "evals_used"),
            sims_used=_require_int(message, "sims_used"),
            tree_nodes_at_entry=_require_int(message, "tree_nodes_at_entry"),
            win_state=_parse_win_state(message.get("win_state")),
        )

    def probe(self) -> npt.NDArray[np.float32]:
        """Return v1's current observation -- §D2.4's differential probe (A4).

        §D1 added this request because rev 1's protocol had none that returns an
        observation, which made acceptance A4 -- the sole refutation of RISK C3
        -- unimplementable against the pre-registered protocol.

        Raises:
            ProtocolViolation: On a malformed or wrongly-shaped array.
        """
        message = self.request({"type": PROBE}, self._config.pick_deadline_s)
        if message["type"] != OBSERVATION:
            raise ProtocolViolation(
                f"expected {OBSERVATION!r} in reply to 'probe', got {message['type']!r}"
            )
        shape = message.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or not all(isinstance(dim, int) and dim > 0 for dim in shape)
        ):
            raise ProtocolViolation(f"probe shape must be three positive ints, got {shape!r}")
        payload = message.get("data")
        if not isinstance(payload, str):
            raise ProtocolViolation("probe data must be a base64 float32 string")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (ValueError, TypeError) as error:
            raise ProtocolViolation(f"probe data is not base64: {error}") from error
        flat = np.frombuffer(raw, dtype=np.float32)
        expected = int(shape[0]) * int(shape[1]) * int(shape[2])
        if flat.size != expected:
            raise ProtocolViolation(
                f"probe carries {flat.size} floats, shape {shape!r} needs {expected}"
            )
        return np.asarray(flat.reshape(int(shape[0]), int(shape[1]), int(shape[2])))

    def close(self) -> None:
        """Say ``bye`` if the adapter is alive, then release the process."""
        if self._channel.is_alive():
            try:
                self._channel.send({"type": BYE})
                self._channel.receive(5.0)
            except (_TransportEOF, _TransportTimeout, ProtocolViolation):
                pass
        self._channel.close()

    def kill(self) -> int | None:
        """Kill the adapter -- §D6 M2's remedy."""
        return self._channel.kill()


def _parse_win_state(payload: object) -> tuple[int, int, int]:
    """Parse v1's ``[white, black, draw]`` view, the M5/M6 divergence detector."""
    if payload is None:
        return (0, 0, 0)
    if not isinstance(payload, (list, tuple)) or len(payload) != 3:
        raise ProtocolViolation(f"win_state must be three ints, got {payload!r}")
    values: list[int] = []
    for slot in payload:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise ProtocolViolation(f"win_state must be three ints, got {payload!r}")
        values.append(int(slot))
    return (values[0], values[1], values[2])


# ---------------------------------------------------------------------------
# §D1's Mover seam
# ---------------------------------------------------------------------------


class Mover(Protocol):
    """Chooses one action in a position. The player seam a match dispatches on.

    §D1's seam, verbatim in intent. One refinement, and its reason: ``observe``
    takes the **record** as well as the action id, because decoding an id to a
    record needs the board the move was played *from* (``game.py:358-367``) and
    the loop has already advanced past it by the time both movers are notified.
    §D2.3 puts the decode on the playing side anyway -- "Each side decodes its
    own action id to a record before sending" -- so passing it costs nothing and
    saves every implementation from keeping a mirror board just to re-derive it.

    A ``Mover`` implementation may raise :class:`BridgeVoided` from any method:
    that is how §D6's modes leave the wire, and :func:`play_bridged_game` is
    where they are caught.
    """

    def pick(self, state: GameState, *, ply: int, seed: int) -> int:
        """Return a legal action id for `state`."""
        ...

    def observe(self, action: int, record: MoveRecord) -> None:
        """Told every action actually played, including the opponent's."""
        ...

    def reset(self, *, pair: int, seat: int) -> None:
        """Start a new game from the arena's start position."""
        ...

    def close(self) -> None:
        """Release whatever this holds (a subprocess, a socket, nothing)."""
        ...


def position_fen(state: GameState) -> str:
    """Return the FEN of `state`'s position, for §D6 M3's void record."""
    return str(board_to_fen(state.game._board))


def legal_records(state: GameState) -> dict[tuple[Any, ...], int]:
    """Return ``{move-record key: action id}`` for every legal move here.

    This is §D6 M3's referee test in its exact pre-registered form -- "the
    record does not match any entry of Komugi's ``generate_all_moves``" -- with
    the action id the arena needs to play the move falling out of the same
    lookup, so an accepted record and the id played are the same move by
    construction rather than by a second, separately-fallible decode.

    Raises:
        BridgeVoided: If the engine's own encoder raises on its own generated
            record -- §D6 M7, "M3's second line of defence", which "must not be
            allowed to propagate as a crash of the arena".
    """
    return _legal_index(state)[0]


def _legal_index(
    state: GameState,
) -> tuple[dict[tuple[Any, ...], int], dict[tuple[Any, ...], str]]:
    """Return ``(key -> action id, key -> san)`` in **one** pass of the engine.

    The two maps come out of the same ``generate_all_moves`` call because the
    per-ply cost of this bridge is dominated by move generation, and A3's
    ``san`` cross-check is not worth a second full pass over a 229-wide legal
    set at every remote ply.
    """
    index: dict[tuple[Any, ...], int] = {}
    sans: dict[tuple[Any, ...], str] = {}
    for record in generate_all_moves(state.game._board):
        try:
            action = int(action_encoding.encode_cython_move(record))
        except (ValueError, KeyError, AssertionError, IndexError) as error:
            raise BridgeVoided(
                BridgeVoid(
                    mode=M7,
                    bucket=VoidBucket.ILLEGAL_MOVE,
                    scope=VoidScope.PAIR,
                    detail=f"Komugi's encoder raised on its own record: {error}",
                    fen=position_fen(state),
                )
            ) from error
        parsed = MoveRecord.from_engine(record)
        index[parsed.key()] = action
        sans[parsed.key()] = parsed.san
    return index, sans


def record_of_action(state: GameState, action: int) -> MoveRecord:
    """Return the move record `action` names in `state` -- §D2.3's encode side.

    Must be called **before** the action is played: the decode is against the
    board the move is played from (``game.py:358-367``).

    Raises:
        BridgeVoided: §D6 M7 if the engine's decoder raises.
    """
    try:
        decoded = action_encoding.decode_cython_action(int(action), state.game._board)
    except (ValueError, KeyError, IndexError) as error:
        raise BridgeVoided(
            BridgeVoid(
                mode=M7,
                bucket=VoidBucket.ILLEGAL_MOVE,
                scope=VoidScope.PAIR,
                detail=f"decode_cython_action({action}) raised: {error}",
                fen=position_fen(state),
            )
        ) from error
    return MoveRecord.from_engine(decoded)


class V1RemoteMover:
    """A :class:`Mover` backed by one v1 subprocess: §D1 and §D6, together.

    Every §D6 mode that can be observed at the wire is detected here and leaves
    as a :class:`BridgeVoided` carrying an explicit :class:`BridgeVoid`. The
    order the checks run in is itself pre-registered, and two orderings are
    load-bearing rather than incidental:

    * **``budget_breach`` outranks ``crash``** (§D6 precedence rule 5, A8/A11).
      A ``pick`` that returned ``budget_breach`` is M8 "even if the v1 process
      dies immediately afterwards"; M1 applies only to a ``pick`` for which no
      ``budget_breach`` was received. Without this the breach detector is
      defeated by the breach handler.
    * **Match-level before pair-level.** On a ``move`` response the checks run
      P3 (M9, match) before P1 (M10, pair) before legality (M3, pair), because a
      match-level defect means the per-move budget was not a per-move budget
      *retroactively, for every move already played* -- so it cannot be reported
      as the narrower fault of the ply that happened to expose it.

    Args:
        connection: A live, handshaken adapter (:func:`spawn_v1`).
        player_id: The rateable identity, for the void records.
        ceiling: §D3.3's per-move ceiling; ``evals_used`` above it is P1.
    """

    def __init__(
        self,
        connection: V1Connection,
        player_id: str,
        ceiling: int = EVAL_CEILING,
    ) -> None:
        self._connection = connection
        self._player_id = player_id
        self._ceiling = int(ceiling)
        self._pair: int | None = None
        self._seat: int | None = None
        self.evals: list[int] = []
        """Per-move ``evals_used``, in ply order -- §D3.6's P4/P5 material."""
        self.sims: list[int] = []
        """Per-move ``sims_used``, in ply order."""
        self.san_mismatches: int = 0
        """Records whose ``san`` disagreed with Komugi's (A3's cross-check)."""
        self.last_win_state: tuple[int, int, int] = (0, 0, 0)
        """v1's most recent view, for the M6 ``referee_ahead`` diagnostic."""

    @property
    def connection(self) -> V1Connection:
        """Return the underlying connection (diagnostics and tests)."""
        return self._connection

    @property
    def player_id(self) -> str:
        """Return the rateable identity this mover plays as."""
        return self._player_id

    def _void(self, **kwargs: Any) -> BridgeVoided:
        """Build a :class:`BridgeVoided` stamped with this mover's coordinates."""
        void = BridgeVoid(
            player_id=self._player_id,
            pair=self._pair,
            seat=self._seat,
            stderr=self._connection.stderr_text,
            **kwargs,
        )
        return BridgeVoided(void)

    def _died(self, error: _TransportEOF, ply: int | None) -> BridgeVoided:
        """Return the M1 void for a process that ended (§D6 M1)."""
        return self._void(
            mode=M1,
            bucket=VoidBucket.CRASH,
            scope=VoidScope.PAIR,
            detail=f"the v1 process ended mid-game: {error}",
            ply=ply,
            exit_status=error.exit_status,
        )

    def _hung(self, error: _TransportTimeout, ply: int | None) -> BridgeVoided:
        """Return the M2 void for a deadline breach, killing the process."""
        exit_status = self._connection.kill()
        return self._void(
            mode=M2,
            bucket=VoidBucket.TIMEOUT,
            scope=VoidScope.PAIR,
            detail=(
                f"the v1 process missed the per-pick deadline ({error}); killed. "
                f"Fatal by policy and never retried: retrying a search after a "
                f"timeout is the stale-answer race inference.py:321-328 forbids"
            ),
            ply=ply,
            exit_status=exit_status,
        )

    def _garbled(self, error: ProtocolViolation, ply: int | None) -> BridgeVoided:
        """Return the M4 void for something that is not this protocol."""
        return self._void(
            mode=M4,
            bucket=VoidBucket.PROTOCOL_ERROR,
            scope=VoidScope.PAIR,
            detail=f"the v1 process spoke non-protocol: {error}",
            ply=ply,
        )

    def reset(self, *, pair: int, seat: int) -> None:
        """Start a new game inside the live process."""
        self._pair = pair
        self._seat = seat
        self.evals = []
        self.sims = []
        self.last_win_state = (0, 0, 0)
        try:
            self._connection.reset(pair=pair, seat=seat)
        except _TransportEOF as error:
            raise self._died(error, None) from error
        except _TransportTimeout as error:
            raise self._hung(error, None) from error
        except ProtocolViolation as error:
            raise self._garbled(error, None) from error

    def observe(self, action: int, record: MoveRecord) -> None:
        """Forward one played move to the adapter (§D1's ``observe``)."""
        try:
            self._connection.observe(record)
        except _TransportEOF as error:
            raise self._died(error, None) from error
        except _TransportTimeout as error:
            raise self._hung(error, None) from error
        except ProtocolViolation as error:
            raise self._garbled(error, None) from error

    def pick(self, state: GameState, *, ply: int, seed: int) -> int:
        """Ask for a move and return the action id, or void per §D6.

        Raises:
            BridgeVoided: For any of M1--M5, M7, M9 or M10. Every one of them
                carries the quantities the resolution needs; none of them is a
                score.
        """
        try:
            response = self._connection.pick(ply=ply, seed=seed)
        except _TransportEOF as error:
            raise self._died(error, ply) from error
        except _TransportTimeout as error:
            raise self._hung(error, ply) from error
        except ProtocolViolation as error:
            raise self._garbled(error, ply) from error

        if isinstance(response, PickBudgetBreach):
            # §D6 precedence rule 5: this is M8 even if the process dies now.
            raise self._void(
                mode=M8,
                bucket=VoidBucket.BUDGET_BREACH,
                scope=VoidScope.MATCH,
                invariant="P2",
                detail=(
                    f"the v1 counting evaluator refused evaluation "
                    f"{response.evals_used} within one move (ceiling "
                    f"{self._ceiling}, §D3.3). 257 is the arithmetic maximum of "
                    f"the pre-registered search structure, so a 258th request "
                    f"means the adapter is not the one pre-registered and every "
                    f"move it has already played was searched under an unknown "
                    f"budget"
                ),
                ply=ply,
                evals_used=response.evals_used,
                sims_used=response.sims_used,
                tree_nodes_at_entry=response.tree_nodes_at_entry,
            )

        if isinstance(response, PickNoMove):
            self.last_win_state = response.win_state
            raise self._void(
                mode=M5,
                bucket=VoidBucket.RULES_DIVERGENCE,
                scope=VoidScope.PAIR,
                detail=(
                    f"v1 has no move at a position Komugi has not adjudicated: "
                    f"v1 win_state={response.win_state!r}, reason="
                    f"{response.reason!r}. Expected to occur -- the two engines "
                    f"disagree about legality by construction (game.py:8-12)"
                ),
                ply=ply,
                fen=position_fen(state),
            )

        self.last_win_state = response.win_state
        return self._accept_move(state, response, ply)

    def _accept_move(self, state: GameState, response: PickMove, ply: int) -> int:
        """Run §D6's checks over a ``move`` response and return its action id."""
        if response.tree_nodes_at_entry != 0:
            raise self._void(
                mode=M9,
                bucket=VoidBucket.TREE_REUSE,
                scope=VoidScope.MATCH,
                invariant="P3",
                detail=(
                    f"v1 entered this move's search with "
                    f"{response.tree_nodes_at_entry} nodes already in its tree; "
                    f"§D3.2 clause 2 requires a fresh search object per pick, so "
                    f"the per-move budget was not a per-move budget -- "
                    f"retroactively, for every move already played"
                ),
                ply=ply,
                evals_used=response.evals_used,
                sims_used=response.sims_used,
                tree_nodes_at_entry=response.tree_nodes_at_entry,
                record=response.record,
            )

        breach = budget_invariant_breach(
            response.evals_used, response.sims_used, self._ceiling
        )
        if breach is not None:
            raise self._void(
                mode=M10,
                bucket=VoidBucket.BUDGET_INVARIANT,
                scope=VoidScope.PAIR,
                invariant="P1",
                detail=breach,
                ply=ply,
                evals_used=response.evals_used,
                sims_used=response.sims_used,
                tree_nodes_at_entry=response.tree_nodes_at_entry,
                record=response.record,
            )

        index, sans = _legal_index(state)
        action = index.get(response.record.key())
        if action is None:
            raise self._void(
                mode=M3,
                bucket=VoidBucket.ILLEGAL_MOVE,
                scope=VoidScope.PAIR,
                detail=(
                    f"the proposed record matches no entry of Komugi's "
                    f"generate_all_moves: {response.record.to_wire()!r}. Never a "
                    f"loss -- Komugi's Game is the sole referee (§D6)"
                ),
                ply=ply,
                record=response.record,
                fen=position_fen(state),
            )
        if not bool(state.legal_mask()[action]):
            raise self._void(
                mode=M3,
                bucket=VoidBucket.ILLEGAL_MOVE,
                scope=VoidScope.PAIR,
                detail=(
                    f"the proposed record decodes to action {action}, which is "
                    f"absent from Komugi's valid_moves() mask"
                ),
                ply=ply,
                record=response.record,
                fen=position_fen(state),
            )

        komugi_san = sans.get(response.record.key(), "")
        if response.record.san and komugi_san and response.record.san != komugi_san:
            # A3's redundant cross-check: reported, never voided on (see MoveRecord).
            self.san_mismatches += 1
        self.evals.append(response.evals_used)
        self.sims.append(response.sims_used)
        return action

    def close(self) -> None:
        """Say ``bye`` and release the process."""
        self._connection.close()


def budget_invariant_breach(
    evals_used: int, sims_used: int, ceiling: int = EVAL_CEILING
) -> str | None:
    """Return why P1 is breached for one move, or ``None`` if it holds.

    §D3.6 P1, stated over ``evals_used`` and ``sims_used`` **only**:

    * ``1 <= evals``
    * ``evals <= 1 + sims_used`` -- the live check, and the only *in-match*
      detector of RISK C4's first limb, a v1 simulation that consults the
      network more than once. It breaches at the first such ply, on data the
      harness already reads.
    * ``evals <= ceiling`` -- §D3.3's ceiling.
    * ``sims_used == 0 => evals == 1`` -- the forced-move convention, **one
      direction only**. The reverse is false and false *guaranteed*: at the
      turn-cap final ply of every capped game a non-forced position reports
      ``sims_used == n`` and costs exactly 1 evaluation, because every legal
      move leads to a terminal (``game.py:374-376``, ``gumbel.py:776-783``).

    ``X`` appears nowhere: it is measured on neither side, so ``evals == 1 + X``
    is a tautology and M10's first disjunct would be dead code (re-review F3).
    """
    if evals_used < 1:
        return (
            f"evals_used={evals_used} is below 1; even a forced move issues "
            f"exactly one root evaluation through the counted evaluator "
            f"(§D3.4's forced-move pin, mirroring gumbel.py:858)"
        )
    if sims_used < 0:
        return f"sims_used={sims_used} is negative"
    if evals_used > ceiling:
        return f"evals_used={evals_used} exceeds the ceiling of {ceiling} (§D3.3)"
    if evals_used > 1 + sims_used:
        return (
            f"evals_used={evals_used} exceeds 1 + sims_used={sims_used}: a "
            f"simulation consulted the network more than once (RISK C4's first "
            f"limb), which makes the two engines' searches incommensurable"
        )
    if sims_used == 0 and evals_used != 1:
        return (
            f"sims_used=0 with evals_used={evals_used}: a forced move issues "
            f"exactly one root evaluation (§D3.4), no more and no fewer"
        )
    return None


# ---------------------------------------------------------------------------
# entrants, and the ArenaPlayer that refuses the wrong altitude
# ---------------------------------------------------------------------------

MoverFactory = Callable[[], Mover]
"""Builds one side's :class:`Mover`. Called once per match, and again after a
crash so the next pair gets a live process (§D6 M1's "respawn")."""


def _wrong_altitude_factory(player_id: str) -> Callable[[], Any]:
    """Return an ``evaluator_factory`` that refuses, naming §D1.

    See the module docstring. The refusal is the *product*, not a placeholder:
    it is what makes "seat a v1 entrant at the ``BatchEvaluator`` seam" a loud
    failure instead of a hybrid engine that quietly produces an anchor for a
    question KOM-011 did not ask.
    """

    def refuse() -> Any:
        """Raise :class:`WrongAltitude`."""
        raise WrongAltitude(
            f"{player_id!r} is a cross-engine entrant and cannot be built as a "
            f"BatchEvaluator. That seam is the *network* seam "
            f"(inference.py:283-289), so mounting a foreign engine there would "
            f"drive its weights with Komugi's Gumbel search "
            f"(arena.py:509-524) -- a hybrid nobody trained and nobody ships. "
            f"kom-023 §D1 puts the cross-engine seam at the MOVE: play this "
            f"entrant with komugi.eval.v1bridge.play_bridged_match, not with "
            f"arena.play_match"
        )

    return refuse


@dataclass(frozen=True, slots=True)
class BridgeEntrant:
    """One cross-engine entrant: an identity, a :class:`Mover`, and a budget.

    The counterpart of :class:`~komugi.eval.arena.ArenaPlayer` at §D1's seam.
    The budget fields are carried even though this module does not run a Gumbel
    search: they are part of the anchor's **configuration block** (§D3.6 P5),
    and :meth:`arena_player` puts them where every existing consumer of an
    entrant already looks for them.

    Args:
        player: The rateable identity. For a v1 anchor the family is
            :data:`V1_FAMILY`, constructed directly -- never through
            ``ladder_players`` (§D4, acceptance A6).
        mover_factory: Builds this side's mover, once per match and once more
            per crash-respawn.
        m: Root actions sampled per move, for the configuration block.
        n: Simulation budget per move, likewise.
        gumbel_scale: Root noise scale, likewise.
        proof_mode: The proof layer this entrant's movers search with, likewise
            part of the configuration block. Defaults to ``off``, so an entrant
            built before the layers existed publishes the legacy identity.
        discovery_mode: Its discovery layer, likewise, default ``off``. A v1
            anchor runs neither layer and leaves both at ``off``; a Komugi
            entrant gets them from :func:`komugi.eval.gumbel_mover.komugi_entrant`.

    The modes are carried rather than dropped for the same reason the budget is:
    what a report says a side played must be what it played, and an entrant that
    silently published ``off`` for a mover searching with the proof layer would
    put the wrong algorithm's name on the row.
    """

    player: Player
    mover_factory: MoverFactory
    m: int = DEFAULT_M
    n: int = DEFAULT_N
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE
    proof_mode: str = "off"
    discovery_mode: str = "off"

    @property
    def search_identity(self) -> modes.SearchAlgorithmIdentity:
        """Return the algorithm this entrant's two modes name.

        Derived from the modes on every read, exactly as
        :attr:`komugi.eval.arena.ArenaPlayer.search_identity` is: the identity
        :func:`anchor_configuration` publishes cannot drift from the modes the
        movers are built with, because there is nothing stored to drift.
        """
        return modes.search_algorithm_identity(self.proof_mode, self.discovery_mode)

    def arena_player(self) -> ArenaPlayer:
        """Return the :class:`~komugi.eval.arena.ArenaPlayer` for this entrant.

        Real in every respect an ``ArenaPlayer`` is consulted for -- identity,
        family, and the per-player budget -- so §D4's pairing, the roster audit
        and every report keyed on an entrant work unchanged. Its
        ``evaluator_factory`` raises :class:`WrongAltitude`; see the module
        docstring for why that is the correct answer rather than a gap.

        The two search modes travel with it, so an entrant's arena-facing view
        reports the same algorithm identity the entrant does -- turning them off
        on the way through would make the serialized side disagree with the
        played one.
        """
        return ArenaPlayer(
            player=self.player,
            evaluator_factory=_wrong_altitude_factory(self.player.id),
            m=self.m,
            n=self.n,
            gumbel_scale=self.gumbel_scale,
            proof_mode=self.proof_mode,
            discovery_mode=self.discovery_mode,
        )


def v1_entrant(
    player_id: str,
    config: V1Config,
    *,
    ceiling: int = EVAL_CEILING,
    m: int = DEFAULT_M,
    n: int = DEFAULT_N,
    gumbel_scale: float = DEFAULT_GUMBEL_SCALE,
) -> BridgeEntrant:
    """Build a v1 :class:`BridgeEntrant` that spawns `config`'s adapter.

    The factory spawns on **call**, not here, so a crash-respawn (§D6 M1) gets a
    fresh process from the same configuration.
    """

    def build() -> Mover:
        """Spawn the adapter and wrap it as a mover."""
        return V1RemoteMover(spawn_v1(config), player_id, ceiling)

    return BridgeEntrant(
        player=Player(id=player_id, family=V1_FAMILY),
        mover_factory=build,
        m=m,
        n=n,
        gumbel_scale=gumbel_scale,
    )


# ---------------------------------------------------------------------------
# the game loop: arena.py:489-527 with the search replaced by Mover.pick
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BridgedGame:
    """One bridged game: an outcome **or** a void, never both and never neither.

    §D6's opening rule in the type system: a game that failed carries a
    :class:`BridgeVoid` and no result, so there is no representable way for a
    crash to reach the records as a 0 or a 1.
    """

    outcome: GameOutcome | None = None
    void: BridgeVoid | None = None
    referee_ahead: bool = False
    """§D6 M6: Komugi adjudicated while the mover's last view said otherwise.
    **Not a failure** -- the referee has spoken -- and counted as a diagnostic."""

    @property
    def is_void(self) -> bool:
        """Return whether this game produced a void rather than a result."""
        return self.void is not None


def _result_of(state: GameState) -> float:
    """Return the finished game's result from white's perspective.

    The same three lines as ``arena.py:474-486``, which is private to that
    module. Duplicated rather than imported, and duplicated *exactly*: the
    convention that a result is stated from white's seat is what every consumer
    of a :class:`~komugi.eval.records.GameRecord` assumes.
    """
    outcome = state.absolute_outcome()
    if bool(outcome[_DRAW_INDEX]):
        return DRAW
    if bool(outcome[_WHITE]):
        return WHITE_WIN
    if bool(outcome[1 - _WHITE]):
        return BLACK_WIN
    raise ValueError(
        "the game loop exited on a position the engine has not adjudicated: "
        "win_state() is all zero"
    )


def play_bridged_game(
    white: Mover,
    black: Mover,
    config: ArenaConfig,
    seeds: np.random.SeedSequence,
    *,
    pair: int = 0,
    seat: int = 0,
) -> BridgedGame:
    """Play one game between two movers, voiding rather than scoring on failure.

    ``arena.py:489-527`` with exactly one substitution -- the hardcoded
    ``GumbelSearch`` construction becomes :meth:`Mover.pick` -- and one addition,
    the ``observe`` notification §D1 requires. The three load-bearing properties
    of the original loop are unchanged and are one line each below: the mover is
    ``state.to_play()`` and never ply parity; the per-ply seed is
    ``move_seed(seeds, ply)``, the same derivation self-play uses; and the loop
    condition is the engine's ``is_terminal()`` and nothing else, because
    **Komugi's ``Game`` is the sole referee** (§D6).

    **The epoch book opens the game the same way the opponent's moves reach the
    far side** (kom-011d §D3.4): the pair's prefix is replayed on the referee
    before the loop, through the *arena's own* :func:`~komugi.eval.arena.replay_prefix`
    so the bridged and in-process loops cannot disagree about what a pair's
    opening is, and each prefix action is announced to **both** movers exactly
    as the loop announces a real move -- decode on the board it is played from,
    then ``observe``. That is the one mechanism every bridge already has
    (``v1bridge.py:1159-1169``: the v1 process holds its own engine state and
    has *no other way* to learn what was played), which is why books are applied
    this way rather than by handing an adapter a FEN to start from -- that would
    need a new protocol request, a new capability flag and a new failure mode.

    A mover that mis-tracks the prefix proposes an illegal move at the first
    real ply, which is §D6 **M3**: a pair void with the FEN and the record
    logged, not a silently wrong game. So a book-application bug on the far side
    of a wire is detected by machinery that already exists -- and the replay
    sits **inside** the try below for exactly that reason.

    Args:
        white: The mover holding white.
        black: The mover holding black.
        config: The match config; ``draw_score``, ``repetition_rule`` and the
            pair's epoch-book prefix
            (:meth:`~komugi.eval.arena.ArenaConfig.prefix_for_pair`) are what is
            read here. The referee is built by the arena's own
            :func:`~komugi.eval.arena.new_arena_game`, so a bridged game and an
            in-process one at the same config adjudicate repetition under the
            same key (KOM-141) -- which is the whole point of being able to
            referee under gungi.js's rule at all.
        seeds: This game's seed sequence, from ``game_seeds``.
        pair: Pair index, for ``reset``, the void record, **and** the book's
            pair-to-entry map -- both games of a pair therefore open with the
            same entry, from opposite seats.
        seat: Seat index within the pair, likewise.

    Returns:
        A :class:`BridgedGame` carrying either the outcome or the §D6 void.
    """
    state = GameState(new_arena_game(config), config.draw_score)
    moves: list[int] = []
    ply = 0
    try:
        white.reset(pair=pair, seat=seat)
        black.reset(pair=pair, seat=seat)

        def announce(current: GameState, action: int) -> None:
            """Tell both movers one prefix action, decoded where it is played."""
            opening = record_of_action(current, action)
            white.observe(action, opening)
            black.observe(action, opening)

        moves.extend(
            replay_prefix(state, config.prefix_for_pair(pair), announce=announce)
        )
        ply = len(moves)
        while not state.is_terminal():
            side = white if state.to_play() == _WHITE else black
            action = side.pick(state, ply=ply, seed=move_seed(seeds, ply))
            # Decoded before the move is played: the record names the board it
            # is played FROM (§D2.3, game.py:358-367).
            record = record_of_action(state, action)
            state.play(action)
            moves.append(int(action))
            white.observe(action, record)
            black.observe(action, record)
            ply += 1
    except BridgeVoided as voided:
        return BridgedGame(void=replace(voided.void, pair=pair, seat=seat))
    except BudgetBreach as breach:
        # §D3.5's Komugi side of M8: caught at the arena boundary, resolved as
        # the same match-level void a `budget_breach` response resolves to, and
        # never allowed to propagate as a crash of the arena.
        return BridgedGame(
            void=BridgeVoid(
                mode=M8,
                bucket=VoidBucket.BUDGET_BREACH,
                scope=VoidScope.MATCH,
                invariant="P2",
                detail=str(breach),
                ply=ply,
                evals_used=breach.evals_used,
                player_id=breach.player_id,
                pair=pair,
                seat=seat,
            )
        )
    return BridgedGame(
        outcome=GameOutcome(result=_result_of(state), moves=tuple(moves)),
        referee_ahead=_referee_ahead(white, black),
    )


def _referee_ahead(*movers: Mover) -> bool:
    """Return whether any mover's last view said the game was still running.

    §D6 M6 -- "Komugi thinks the game is over, v1 does not" -- which is **not a
    failure**: the referee has adjudicated, and this is counted in a diagnostic
    and never in a void bucket.

    **What the number is worth, stated rather than implied.** §D6 detects M6 as
    "Komugi's ``is_terminal()`` fires; v1 is simply told ``bye``/``reset``", and
    the protocol has no request that asks the adapter whether *it* would have
    adjudicated the final position -- so the best available proxy is the last
    view the adapter volunteered, one ply earlier, which is non-terminal at
    almost every ply by the loop's own invariant. This counter is therefore
    close to "games Komugi ended", and it is informative only **against M5's
    rate**, which is the comparison §D6 wants it for. It is deliberately not
    reported as a divergence count on its own.
    """
    for mover in movers:
        view = getattr(mover, "last_win_state", None)
        if isinstance(view, tuple) and not any(view):
            return True
    return False


# ---------------------------------------------------------------------------
# the match driver: pair-atomic, with §D6's void accounting
# ---------------------------------------------------------------------------

COMPLETED: Final[str] = "completed"
ABORTED: Final[str] = "aborted"
VOIDED: Final[str] = "voided"

_RESPAWN_BUCKETS: Final[frozenset[VoidBucket]] = frozenset(
    {VoidBucket.CRASH, VoidBucket.TIMEOUT, VoidBucket.PROTOCOL_ERROR}
)
"""Buckets after which the next pair needs a fresh process.

The other pair-level buckets (M3, M5, M10) are answers the adapter gave
correctly-framed and in full: the wire is still synchronised and the process is
still the one that shook hands, so respawning would throw away a live match's
warm process to no purpose.
"""


@dataclass(frozen=True, slots=True)
class VoidTally:
    """§D6's void accounting: counts per bucket, plus the retained detail.

    Voids are **never** folded into ``games``, ``draws`` or ``decisive``
    (``ArenaSummary``); they live here, beside it, in ``BridgedMatch``.
    """

    voids: tuple[BridgeVoid, ...] = ()
    pairs_attempted: int = 0
    pairs_scored: int = 0
    referee_ahead: int = 0

    def by_bucket(self) -> dict[str, int]:
        """Return the count per bucket, sorted by bucket name."""
        counts: dict[str, int] = {}
        for void in self.voids:
            counts[void.bucket.value] = counts.get(void.bucket.value, 0) + 1
        return dict(sorted(counts.items()))

    def by_mode(self) -> dict[str, int]:
        """Return the count per §D6 mode, sorted by mode."""
        counts: dict[str, int] = {}
        for void in self.voids:
            counts[void.mode] = counts.get(void.mode, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def crashes(self) -> int:
        """Return the ``crash``-bucket count -- M1's absolute abort rule."""
        return sum(1 for void in self.voids if void.bucket is VoidBucket.CRASH)

    @property
    def void_rate(self) -> float:
        """Return voided pairs over pairs **attempted** (§D6's denominator)."""
        if self.pairs_attempted == 0:
            return 0.0
        return len(self.voids) / self.pairs_attempted

    @property
    def admissible(self) -> bool:
        """Return whether the void rate clears §D6's 2% admissibility gate.

        The gate is evaluated only on matches that ran to completion; a caller
        reading this on an aborted or match-voided run is asking a question §D6
        does not answer, and :attr:`BridgedMatch.admissible` is where that is
        enforced.
        """
        return self.void_rate <= VOID_RATE_GATE


@dataclass(frozen=True, slots=True)
class BridgedMatch:
    """A finished cross-engine match, or the reason there is no match to score.

    Args:
        status: :data:`COMPLETED`, :data:`ABORTED` (M1's third crash) or
            :data:`VOIDED` (M8/M9). §D6: an aborted match "is scored not at
            all", and rev 3 pre-registers that **a match-void is an abort** --
            so both carry ``outcome=None`` and discard their completed pairs.
        reason: ``""``, ``"crash"``, ``"P2"`` or ``"P3"``.
        outcome: The records and summary; ``None`` unless completed.
        tally: §D6's void accounting.
        pairs_completed: Pairs that had been scored when the run ended -- the
            number ``aborted(reason, pairs_completed)`` reports, and which is
            **not** a set of records anyone may keep.
    """

    status: str
    reason: str
    outcome: MatchOutcome | None
    tally: VoidTally
    pairs_completed: int

    @property
    def admissible(self) -> bool:
        """Return whether this run may be cited as anchor evidence (§D6).

        False for anything that did not run to completion, whatever its rate:
        "voids are non-random, so the surviving pairs of an aborted match are
        exactly the biased sample the 2% gate exists to refuse".
        """
        return self.status == COMPLETED and self.tally.admissible


class _MoverSlot:
    """One side's live mover, rebuildable after a crash (§D6 M1's respawn)."""

    def __init__(self, entrant: BridgeEntrant) -> None:
        self._entrant = entrant
        self._mover: Mover | None = None

    @property
    def mover(self) -> Mover:
        """Return the live mover, building it on first use."""
        if self._mover is None:
            self._mover = self._entrant.mover_factory()
        return self._mover

    def start(self) -> None:
        """Build the mover now, so a handshake refusal happens before any game."""
        if self._mover is None:
            self._mover = self._entrant.mover_factory()

    def renew(self) -> None:
        """Discard the current mover so the next pair gets a fresh one."""
        self.close()
        self._mover = None

    def close(self) -> None:
        """Release the current mover, swallowing a failure to say goodbye."""
        if self._mover is not None:
            try:
                self._mover.close()
            except (BridgeVoided, OSError, ProtocolViolation):
                pass


def _assert_declared_book(config: ArenaConfig) -> None:
    """Assert a rated match declares which openings it was played from.

    KOM-023 §D4 **amendment 2** wrote this as one literal: a rated cross-family
    match carries ``book_id = f"empty-{mode}"``, because ``"empty"`` alone does
    not say *which* empty board two processes started from and both games of
    every pair would share the lie. kom-011d §D3 gives such a match a second
    legal way to start -- a frozen epoch book -- and **amendment 4** is the same
    invariant read over it: the declaration a rated match must carry is the
    **declared epoch book**, not the empty-board id.

    So there are two branches and no third:

    * no :attr:`~komugi.eval.arena.ArenaConfig.book`: unchanged, and the games
      really do start from the empty board;
    * a book: ``book_id`` must be that book's ``epoch_book_id``, and the book's
      ``gungi_mode`` must be the mode this process resolved. The per-game
      ``book_id`` s are the book's **entry** ids, all sharing its ``E<NNN>-``
      prefix, so the stored records name the era and the config names the
      artefact -- and neither can be inferred from the other after the fact
      (``ladder.py:218-241``'s lesson).

    Raises:
        ValueError: If the declaration does not match what will be played.
    """
    mode = resolved_gungi_mode()
    if config.book is None:
        expected_book = bridged_book_id(mode)
        if config.book_id != expected_book:
            raise ValueError(
                f"a rated cross-family match must carry book_id "
                f"{expected_book!r} (§D4 amendment 2), got {config.book_id!r}: "
                f"'empty' alone does not name which start position the games "
                f"were played from, and both games of every pair would share "
                f"the lie"
            )
        return
    if config.book_id != config.book.epoch_book_id:
        # Kept as defence in depth. Since review F10,
        # :class:`~komugi.eval.arena.ArenaConfig` refuses this pairing at its own
        # construction, so a config reaching here undeclared has been mutated
        # past a frozen dataclass -- but the rated path is where an anchor is
        # produced, and it does not take the earlier check on trust.
        raise ValueError(
            f"a rated cross-family match played on an epoch book must declare "
            f"it: book_id is {config.book_id!r} but the book supplied is "
            f"{config.book.epoch_book_id!r} (§D4 amendment 4). The records "
            f"carry entry ids, so the config is the only place the artefact "
            f"they came from is named"
        )
    if config.book.gungi_mode != mode:
        raise ValueError(
            f"epoch book {config.book.epoch_book_id!r} was built under "
            f"GUNGI_MODE={config.book.gungi_mode!r} and this process runs "
            f"{mode!r}: a prefix replayed from a different start position is a "
            f"different line, or illegal -- and the far side would replay it "
            f"under its own mode"
        )


def play_bridged_match(
    a: BridgeEntrant,
    b: BridgeEntrant,
    config: ArenaConfig,
    *,
    rated: bool = False,
) -> BridgedMatch:
    """Play `config.games` games as seat-swapped pairs, pair-atomically.

    ``arena.play_match``'s structure (``arena.py:632-649``), with §D4's
    amendments and §D6's accounting:

    * **Pairs are atomic** (§D4 amendment 1). If either game of a pair fails for
      any reason in §D6, *both* games' records are discarded and the pair is
      recorded in the ``voided`` bucket. Never scored, never half-recorded --
      which is also what stops ``check_paired_books`` reporting a dangling
      ``pair_id`` "correctly, but uselessly late".
    * **The roster is called** (§D4 amendment 3): ``roster_families`` runs at
      match construction, so a mis-declared roster raises here rather than in a
      report weeks later.
    * **Three crashes abort, and an abort keeps nothing** (§D6 precedence). The
      absolute rule fires first and is terminal.
    * **A match-void is an abort** (M8/M9, re-review F8): completed pairs are
      discarded and no ``MatchSet`` is emitted.

    Args:
        a: One entrant; holds white in the even game of every pair.
        b: The other. Must have a different ``player.id``.
        config: Game count, seed, contempt, and the openings -- either a
            declared epoch book or the empty-board ``book_id``.
        rated: When true this is anchor evidence, and the pre-registered
            configuration is enforced: §D3.3's literals (A15) and §D4's
            ``book_id`` invariant, which reads the **declared epoch book** when
            one is supplied and amendment 2's ``empty-<mode>`` when one is not
            (:func:`_assert_declared_book`).

    Returns:
        The :class:`BridgedMatch`.

    Raises:
        ValueError: If the entrants share an id, if the roster is
            contradictory, or -- when `rated` -- if the pinned budget or the
            ``book_id`` is not the pre-registered one.
    """
    if a.player.id == b.player.id:
        raise ValueError(
            f"both entrants are {a.player.id!r}: a game record cannot name one "
            f"player in both seats"
        )
    roster_families([a.player, b.player])
    if rated:
        assert_pinned_budget()
        _assert_declared_book(config)

    slot_a = _MoverSlot(a)
    slot_b = _MoverSlot(b)
    records: list[GameRecord] = []
    plies: list[int] = []
    voids: list[BridgeVoid] = []
    referee_ahead = 0
    pairs_attempted = 0
    pairs_scored = 0
    status = COMPLETED
    reason = ""

    try:
        # Built before the first game so §D1's capability gate refuses a
        # non-conforming adapter at match construction (A13), which is the whole
        # point of the gate: a startup failure rather than a mis-bucketed match.
        slot_a.start()
        slot_b.start()
        for pair in range(config.pairs):
            pairs_attempted += 1
            # The arena's own rule, called rather than re-typed: one format for
            # both construction sites, ``ArenaConfig.pair_namespace`` included
            # (kom-011d §D2.3).
            pair_id = pair_id_for(config, a.player.id, b.player.id, pair)
            # One entry per pair, both seats -- the arena's rule, read through
            # the arena's own accessor rather than re-derived (kom-011d §D3.4).
            book_id = config.book_id_for_pair(pair)
            pair_records: list[GameRecord] = []
            pair_plies: list[int] = []
            pair_void: BridgeVoid | None = None

            for seat in (0, 1):
                white_slot, black_slot = (
                    (slot_a, slot_b) if seat == 0 else (slot_b, slot_a)
                )
                white_id = a.player.id if seat == 0 else b.player.id
                black_id = b.player.id if seat == 0 else a.player.id
                game = play_bridged_game(
                    white_slot.mover,
                    black_slot.mover,
                    config,
                    game_seeds(config, pair, seat),
                    pair=pair,
                    seat=seat,
                )
                if game.void is not None:
                    pair_void = game.void
                    break
                assert game.outcome is not None  # BridgedGame's invariant
                if game.referee_ahead:
                    referee_ahead += 1
                pair_records.append(
                    GameRecord(
                        white=white_id,
                        black=black_id,
                        result=game.outcome.result,
                        book_id=book_id,
                        pair_id=pair_id,
                    )
                )
                # Searched plies, not played ones -- ``arena.play_match``'s own
                # rule (review F8): a pair's replayed prefix is an opening, not
                # a game length, and §D2.4's cost model prices searched plies.
                pair_plies.append(
                    len(game.outcome.moves) - len(config.prefix_for_pair(pair))
                )

            if pair_void is None:
                records.extend(pair_records)
                plies.extend(pair_plies)
                pairs_scored += 1
                continue

            # Pair-atomic: whatever the first game of this pair produced is
            # discarded with it (§D4 amendment 1).
            voids.append(pair_void)
            if pair_void.scope is VoidScope.MATCH:
                status = VOIDED
                reason = pair_void.invariant or pair_void.mode
                break
            if pair_void.bucket in _RESPAWN_BUCKETS:
                # §D6 M1's "respawn the process for the next pair", extended to
                # the two other buckets that leave no usable wire: M2 killed the
                # process, and after M4 the stream may hold the tail of whatever
                # was not this protocol, which would be read as the next pair's
                # answer.
                slot_a.renew()
                slot_b.renew()
            if pair_void.bucket is VoidBucket.CRASH and (
                sum(1 for void in voids if void.bucket is VoidBucket.CRASH)
                >= CRASH_ABORT_THRESHOLD
            ):
                # The absolute rule fires first and is terminal (§D6 precedence 1).
                status = ABORTED
                reason = VoidBucket.CRASH.value
                break
    finally:
        slot_a.close()
        slot_b.close()

    tally = VoidTally(
        voids=tuple(voids),
        pairs_attempted=pairs_attempted,
        pairs_scored=pairs_scored,
        referee_ahead=referee_ahead,
    )
    if status != COMPLETED:
        # An abort and a match-void both keep nothing (§D6 precedence 2 and 4).
        return BridgedMatch(
            status=status,
            reason=reason,
            outcome=None,
            tally=tally,
            pairs_completed=pairs_scored,
        )
    return BridgedMatch(
        status=COMPLETED,
        reason="",
        outcome=MatchOutcome(tuple(records), _summarise(records, plies)),
        tally=tally,
        pairs_completed=pairs_scored,
    )


def _summarise(
    records: Sequence[GameRecord], plies: Sequence[int]
) -> ArenaSummary:
    """Return the :class:`~komugi.eval.arena.ArenaSummary` for scored games.

    ``arena.py:566-578``'s summary over the **scored** records only: a voided
    pair contributes to neither ``games`` nor ``draws`` nor the ply extremes,
    which is §D6's void-accounting rule applied at the one place a void could
    otherwise leak into a rated number.
    """
    draws = sum(1 for record in records if record.is_draw)
    return ArenaSummary(
        games=len(records),
        decisive=len(records) - draws,
        draws=draws,
        total_plies=sum(plies),
        shortest_plies=min(plies, default=0),
        longest_plies=max(plies, default=0),
    )


def anchor_configuration(
    entrant: BridgeEntrant, connection: V1Connection, delta_mean: float | None = None
) -> dict[str, Any]:
    """Return §D3.6 P5's configuration block for the anchor artefact.

    "A reader of KOM-011 six months from now sees ``v1-0015`` and a budget, and
    none of that" -- so the block carries what the *identity* of a frozen anchor
    actually is: the ceiling and both engines' budgets, the root convention that
    fixed ``S₁``, tree reuse off on both sides, the forced-move convention, v1's
    root noise off, ``memo_separable`` as declared in ``hello``, the referee,
    the ``book_id``, the device + determinism block (A21), and ``Δmean``.

    Since KOM-224 the budget is no longer the whole of "what search was played":
    an entrant also has a proof layer and a discovery layer, so the block
    publishes both modes and the ``decision_id``/``target_id`` they name
    (:attr:`BridgeEntrant.search_identity`). A reader who sees ``m``/``n`` and no
    algorithm would attribute a proof-layer row to the legacy search, which is
    exactly the misreading §D3.6 P5 exists to prevent. An entrant built before
    the layers existed publishes ``off``/``off`` and the legacy identity.
    """
    hello = connection.hello
    identity = entrant.search_identity
    return {
        "ceiling": EVAL_CEILING,
        "m": entrant.m,
        "n": entrant.n,
        "gumbel_scale": entrant.gumbel_scale,
        "proof_mode": identity.proof_mode,
        "discovery_mode": identity.discovery_mode,
        "decision_id": identity.decision_id,
        "target_id": identity.target_id,
        "tree_reuse": "off",
        "forced_move_convention": "one root evaluation through the counted evaluator",
        "root_noise": "off",
        "memo_separable": hello.capabilities.memo_separable,
        "memo_citation": hello.capabilities.memo_citation,
        "referee": "komugi",
        "book_id": bridged_book_id(hello.gungi_mode or resolved_gungi_mode()),
        "checkpoint_digest": hello.checkpoint_digest,
        "device": {
            "torch_version": hello.device.torch_version,
            "numpy_version": hello.device.numpy_version,
            "device": hello.device.device,
            "cudnn_benchmark": hello.device.cudnn_benchmark,
            "deterministic_algorithms": hello.device.deterministic_algorithms,
        },
        "delta_mean": delta_mean,
        "player": {"id": entrant.player.id, "family": entrant.player.family},
    }


def roster_map(entrants: Iterable[BridgeEntrant]) -> dict[str, str]:
    """Return the ``{player_id: family}`` map that travels with the anchor.

    §D4 amendment 3: ``GameRecord`` has no ``family`` field
    (``records.py:89-93``), so a v1 anchor in a stored ``MatchSet`` would be
    distinguishable from a Komugi rung only by a ``"v1-"`` prefix convention in
    an id string. The map is persisted **alongside** the ``MatchSet`` and a
    report reads family from it, never from the prefix.
    """
    return roster_families(entrant.player for entrant in entrants)
