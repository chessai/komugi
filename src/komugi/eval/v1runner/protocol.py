"""§D1's wire, **vendored**: the message layer with no komugi imports at all.

Why this is a copy rather than ``from komugi.eval.v1bridge import ...``
----------------------------------------------------------------------
This package runs inside ``nix develop .#v1`` -- a different interpreter
generation, a different numpy, a different torch -- against v1's own compiled
engine. ``komugi.eval.v1bridge`` cannot be imported there, and not for a
stylistic reason: its module body imports ``numpy``, ``komugi.engine.cyboard``,
``komugi.engine.cymovegen``, ``komugi.engine.action_encoding``,
``komugi.engine.game``, ``komugi.eval.arena`` (which pulls torch) and
``komugi.selfplay.adapter``. Three of those are Cython extensions built in
place against **Komugi's** numpy ABI by ``ci/check.sh``, so importing that
module in the v1 process would either fail on the ABI or -- far worse -- load a
second gungi engine into the process whose whole justification (§D1, "Two
compiled engines in one interpreter") is that this must never happen. The task
spec's escape hatch is taken deliberately: **the message layer is vendored, and
this docstring is the "and say so".**

What keeps the copy honest
--------------------------
Nothing, unless it is tested -- so it is. The v1runner unit tests (not part of
this repository) run in the DEFAULT shell, where both modules import, and
assert that every constant here equals its ``v1bridge`` original, that :func:`encode_message`
produces byte-identical lines for the same payloads, and that
:func:`record_from_wire` and :func:`p1_breach_reason` accept and reject exactly
what ``MoveRecord.from_wire`` and ``budget_invariant_breach`` do. A vendored
copy that drifts is a protocol split between the two ends of one wire; the
drift test is the only thing standing between that and a silently mis-scored
anchor.

Scope: stdlib only. Nothing in this package imports numpy, torch, or komugi.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

# ---------------------------------------------------------------------------
# the pre-registered numbers, mirrored from v1bridge.py:93-135
# ---------------------------------------------------------------------------

PROTOCOL_VERSION: Final[int] = 1
"""Harness protocol version (``v1bridge.PROTOCOL_VERSION``)."""

EVAL_CEILING: Final[int] = 257
"""§D3.3's per-move, per-side evaluation ceiling, as a literal.

Written out rather than derived for the same reason ``v1bridge`` writes it out
(§D3.4, review F17): a frozen anchor pinned to a symbol silently redefines
itself when the symbol moves, and no diff anywhere shows it.
"""

DEFAULT_SIMS_ALLOWANCE: Final[int] = 256
"""``S₁`` -- 256 iff v1 evaluates its root **outside** the simulation loop as
Komugi does (``gumbel.py:499``), 257 if the root is expanded inside the first
simulation. Which holds is a §0.2 U-A fact about v1's ``MCTS.py``, which is not
in this worktree, so this is a default the operator overrides with
``--sims-allowance`` once the U-A report states it with a ``file:line``."""

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

OK: Final[str] = "ok"
MOVE: Final[str] = "move"
NO_MOVE: Final[str] = "no_move"
BUDGET_BREACH: Final[str] = "budget_breach"
OBSERVATION: Final[str] = "observation"

PICK_RESPONSES: Final[frozenset[str]] = frozenset({MOVE, NO_MOVE, BUDGET_BREACH})

REFUSED: Final[str] = "refused"
"""Not in §D1's table, and deliberately so.

A refusal to start is written to **stderr**, never to stdout, because §D1 gives
``hello`` exactly one valid answer and a novel type on stdout would reach the
harness as a bare ``ProtocolViolation`` whose message names a missing
``protocol_version`` rather than the actual cause. ``spawn_v1`` appends the
child's captured stderr verbatim to every :class:`HandshakeRefused`
(``v1bridge.py:1048-1055``), so a structured line there *is* protocol-visible --
it lands in the operator's exception -- while leaving stdout free of anything
that is not §D1.
"""

# ---------------------------------------------------------------------------
# §D2.3's wire form
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

Identical field for field in the two engines
(``src/komugi/engine/cymovegen.pyx:481-492`` against
``docs/reference-v1/alphazero-env/cymovegen.pyx:429-440``), which is what makes
the record -- and never an action id, the two spaces being 15390 against 37341
and Komugi -> v1 lossy *and partial* -- the thing that crosses this wire.
"""

_MAX_SQUARE: Final[int] = 8
_MAX_TIER: Final[int] = 3
_MAX_MOVE_TYPE: Final[int] = 4
_MAX_PIECE_TYPE_INDEX: Final[int] = 13
_MAX_CONVERTIBLE_TIER: Final[int] = 2


class ProtocolError(Exception):
    """The wire carried something that is not this protocol -- §D6 M4.

    Raised on the v1 side by the same checks ``v1bridge.ProtocolViolation`` is
    raised by on the harness side. Both ends validate: a runner that trusted
    the harness's records would turn a harness bug into a v1 engine crash, i.e.
    into M1 ``crash`` -- a bucket whose remedy ("three crashes abort the
    match") is the wrong one for a version skew, which is exactly the M3/M4
    distinction §D6 exists to keep.
    """


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------


def encode_message(payload: Mapping[str, Any]) -> str:
    """Return one wire line: JSON, one object, newline-terminated (§D1).

    ``sort_keys`` and the compact separators are not cosmetic: they are what
    makes two conforming ends emit byte-identical lines for the same message,
    which is what makes a recorded transcript comparable at all. The drift test
    asserts this function is byte-identical to ``v1bridge.encode_message``.

    Raises:
        ProtocolError: If `payload` carries no ``type``.
    """
    if "type" not in payload:
        raise ProtocolError(f"a message must carry a 'type', got {dict(payload)!r}")
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def decode_message(line: str) -> dict[str, Any]:
    """Parse one wire line into a message dict -- the M4 boundary.

    Raises:
        ProtocolError: On invalid JSON, a non-object, or a missing or
            non-string ``type``.
    """
    text = line.strip()
    if not text:
        raise ProtocolError("empty line: the protocol is one message per line")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"not JSON: {error} in {text[:200]!r}") from error
    if not isinstance(parsed, dict):
        raise ProtocolError(f"a message must be a JSON object, got {parsed!r}")
    kind = parsed.get("type")
    if not isinstance(kind, str):
        raise ProtocolError(f"a message must carry a string 'type', got {kind!r}")
    return {str(key): value for key, value in parsed.items()}


# ---------------------------------------------------------------------------
# field validation
# ---------------------------------------------------------------------------


def require_int(payload: Mapping[str, Any], field: str) -> int:
    """Return ``payload[field]`` as an ``int``, or raise :class:`ProtocolError`.

    ``bool`` is rejected rather than coerced: ``True`` is a perfectly good
    ``int`` in Python and would otherwise pass silently as square 1, or as a
    ``seed`` of 1.
    """
    if field not in payload:
        raise ProtocolError(f"message is missing required field {field!r}")
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"field {field!r} must be an int, got {value!r}")
    return int(value)


def require_range(value: int, field: str, low: int, high: int) -> int:
    """Return `value`, or raise if it is outside ``[low, high]``."""
    if not low <= value <= high:
        raise ProtocolError(f"field {field!r} is {value}, outside [{low}, {high}]")
    return value


def _parse_captured(value: object) -> list[tuple[int, int]]:
    """Parse the ``captured`` field: a list of ``(type_index, tier)`` pairs."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(
            f"move record field 'captured' must be a list, got {value!r}"
        )
    parsed: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ProtocolError(
                f"captured[{index}] must be a (type_index, tier) pair, got {item!r}"
            )
        type_index, tier = item
        if isinstance(type_index, bool) or not isinstance(type_index, int):
            raise ProtocolError(f"captured[{index}] type index is {type_index!r}")
        if isinstance(tier, bool) or not isinstance(tier, int):
            raise ProtocolError(f"captured[{index}] tier is {tier!r}")
        require_range(
            int(type_index), f"captured[{index}].type", 0, _MAX_PIECE_TYPE_INDEX
        )
        require_range(int(tier), f"captured[{index}].tier", 1, _MAX_CONVERTIBLE_TIER)
        parsed.append((int(type_index), int(tier)))
    return parsed


def record_from_wire(payload: object) -> dict[str, Any]:
    """Validate an incoming record and return it in the **engine's** own shape.

    The return value is handed straight to
    ``docs/reference-v1/alphazero-env/cyengine.pyx:176`` ``apply_move(Board b,
    dict move)``, so it is a plain dict with ``captured`` as a list of
    ``(type_index, tier)`` tuples -- the shape ``_make_move_record`` itself
    emits (``docs/reference-v1/alphazero-env/cymovegen.pyx:429-440``).

    Every failure here is §D6 M4 (``protocol_error``) and never M3
    (``illegal_move``): M3 is a rules question, M4 is a version-skew question,
    and folding them would read a disagreement about a field's *shape* as a
    disagreement about the *rules*.

    Raises:
        ProtocolError: On a missing field, a wrong type, or a square, tier,
            move type or piece index outside the engine's range.
    """
    if not isinstance(payload, dict):
        raise ProtocolError(f"move record must be an object, got {payload!r}")
    for field in STRUCTURAL_FIELDS:
        if field not in payload:
            raise ProtocolError(f"move record is missing {field!r}")
    draft_finished = payload["draft_finished"]
    if not isinstance(draft_finished, bool):
        raise ProtocolError(
            f"move record field 'draft_finished' must be a bool, got "
            f"{draft_finished!r}"
        )
    san = payload.get("san", "")
    if not isinstance(san, str):
        raise ProtocolError(f"move record field 'san' must be a string, got {san!r}")
    # A drop has no origin square and says so with -1 (cymovegen.pyx:429-440).
    from_r = require_range(require_int(payload, "from_r"), "from_r", -1, _MAX_SQUARE)
    from_f = require_range(require_int(payload, "from_f"), "from_f", -1, _MAX_SQUARE)
    return {
        "from_r": from_r,
        "from_f": from_f,
        "to_r": require_range(require_int(payload, "to_r"), "to_r", 0, _MAX_SQUARE),
        "to_f": require_range(require_int(payload, "to_f"), "to_f", 0, _MAX_SQUARE),
        "to_tier": require_range(
            require_int(payload, "to_tier"), "to_tier", 0, _MAX_TIER
        ),
        "move_type": require_range(
            require_int(payload, "move_type"), "move_type", 0, _MAX_MOVE_TYPE
        ),
        "piece_type_index": require_range(
            require_int(payload, "piece_type_index"),
            "piece_type_index",
            0,
            _MAX_PIECE_TYPE_INDEX,
        ),
        "color": require_range(require_int(payload, "color"), "color", 0, 1),
        "captured": _parse_captured(payload["captured"]),
        "draft_finished": draft_finished,
        "san": san,
    }


RULES_DIVERGENCE: Final[str] = "rules_divergence"
"""§D6's name for "the two engines disagree about what is legal here".

Mirrors ``v1bridge``'s void mode of the same spelling. It is not a *response*
type -- ``observe`` is answered ``ok`` or it is not answered at all -- but the
runner puts this word in the refusal's detail so the one failure §D2.3's
generator match exists to catch is greppable in the stderr the harness attaches
to its exception, rather than being indistinguishable from a schema complaint.
"""


def record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the identity of one record: §D2.3's ten structural fields.

    The same shape, in the same order, as ``v1bridge.MoveRecord.key()``, and
    normalised so that a record that crossed the wire (``captured`` as lists of
    ints, from JSON) compares equal to one v1's own ``generate_all_moves`` just
    built (``captured`` as tuples). ``san`` is deliberately **not** in the key:
    it is the two engines' *notation*, not their rules, and §D2.3 pins identity
    on the ten structural fields precisely because the notations may differ.

    Raises:
        ProtocolError: If a structural field is missing or is not the kind of
            value the engines put there.
    """
    try:
        captured = tuple(
            (int(item[0]), int(item[1])) for item in (record.get("captured") or ())
        )
        return (
            int(record["from_r"]),
            int(record["from_f"]),
            int(record["to_r"]),
            int(record["to_f"]),
            int(record["to_tier"]),
            int(record["move_type"]),
            int(record["piece_type_index"]),
            int(record["color"]),
            captured,
            bool(record["draft_finished"]),
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ProtocolError(
            f"a move record must carry {STRUCTURAL_FIELDS!r}; {type(error).__name__}: "
            f"{error} while keying {dict(record)!r}"
        ) from error


def record_to_wire(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON-safe wire form of one engine record: ten fields + ``san``.

    The engine is trusted about its own records -- this is the *encode*
    direction, where the record came out of v1's own ``generate_all_moves`` or
    ``decode_cython_action`` a moment ago. ``captured`` is normalised to a list
    of two-element lists because a tuple is not JSON and ``json.dumps`` would
    silently emit it as an array anyway; doing it here keeps the wire form
    explicit rather than incidental.
    """
    captured = record.get("captured") or ()
    return {
        "from_r": int(record["from_r"]),
        "from_f": int(record["from_f"]),
        "to_r": int(record["to_r"]),
        "to_f": int(record["to_f"]),
        "to_tier": int(record["to_tier"]),
        "move_type": int(record["move_type"]),
        "piece_type_index": int(record["piece_type_index"]),
        "color": int(record["color"]),
        "captured": [[int(item[0]), int(item[1])] for item in captured],
        "draft_finished": bool(record["draft_finished"]),
        "san": str(record.get("san", "")),
    }


def win_state_to_wire(win_state: Sequence[Any]) -> list[int]:
    """Return v1's ``[white, black, draw]`` view as three plain ints.

    ``Game.win_state`` returns ``np.zeros(3, dtype=np.uint8)``
    (``docs/reference-v1/alphazero-env/gungi.py:256-286``) and a ``np.uint8`` is
    not JSON-serialisable, so the conversion is not optional. It is also the
    M5/M6 divergence detector and never an adjudication: Komugi's ``Game`` is
    the sole referee (§D6, "The referee rule").

    Raises:
        ProtocolError: If `win_state` is not exactly three slots.
    """
    values = [int(slot) for slot in win_state]
    if len(values) != 3:
        raise ProtocolError(f"win_state must be three ints, got {win_state!r}")
    return values


# ---------------------------------------------------------------------------
# §D3.6 P1, as a predicate
# ---------------------------------------------------------------------------


def p1_breach_reason(
    evals_used: int, sims_used: int, ceiling: int = EVAL_CEILING
) -> str | None:
    """Return why §D3.6 P1 is breached for one move, or ``None`` if it holds.

    Decision-equivalent to ``v1bridge.budget_invariant_breach``, and the drift
    test pins that equivalence over a grid rather than trusting it. P1, stated
    over the two quantities that are actually on the wire:

    * ``1 <= evals`` -- even a forced move issues exactly one root evaluation
      through the counted evaluator (§D3.4's forced-move pin);
    * ``evals <= 1 + sims_used`` -- the only *in-match* detector of RISK C4's
      first limb, a v1 simulation that consults the network more than once;
    * ``evals <= ceiling`` -- §D3.3's 257;
    * ``sims_used == 0 => evals == 1`` -- one direction only. The reverse is
      false *guaranteed*: at the turn-cap final ply of every capped game a
      non-forced position reports ``sims_used == n`` and costs exactly 1
      evaluation (§D3.4).

    **This predicate never changes a reported number.** The runner uses it as a
    self-diagnostic only, on stderr: if v1's search really does breach P1, the
    harness must *see* the breaching numbers and void the pair as M10. A runner
    that clamped its own report to make P1 hold would delete the detector.
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
