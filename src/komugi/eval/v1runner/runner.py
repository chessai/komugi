"""The v1-side subprocess entry: §D1's remote engine, §D3's ceiling, on stdio.

This is the far end of the wire ``komugi.eval.v1bridge`` speaks. It reads one
JSON request per line on stdin and writes one response per line on stdout;
stderr carries diagnostics and, at startup, the structured refusal
(``reference.py``). It holds v1's own ``Game`` and keeps it in step with the
harness's board through ``observe`` alone -- §D2.2's decision to convert
*moves* rather than *positions*, which is what makes v1's own 117-plane
encoder, and not a re-implementation of it living in Komugi, the thing that
feeds v1's net.

What this module is responsible for, and nothing else
-----------------------------------------------------
* **The protocol**, exactly as §D1's table states it, including all three
  ``pick`` responses. ``no_move`` and ``budget_breach`` are emitted, not merely
  declared: an adapter that declares ``supports_budget_breach`` in ``hello``
  and then dies on a breach re-creates the re-review F1 collision with the
  capability gate defeated.
* **§D3's structural ceiling on the v1 side**: a fresh search per ``pick``
  (clause 2), a counter at the net-call boundary counting every forward
  (clause 1), the count reported per move, and the 258th ask refused and
  reported rather than served (§D3.5).
* **Refusing to start** when v1 is not importable, with a reason the harness
  can read (``reference.py``).

What it deliberately does not do
--------------------------------
* **It does not adjudicate.** ``win_state`` is reported as v1 sees it and is
  the M5/M6 divergence detector only; Komugi's ``Game`` is the sole referee
  (§D6, "The referee rule").
* **It does not repair its own numbers.** :func:`~komugi.eval.v1runner.protocol.p1_breach_reason`
  is used for a stderr warning and never to alter ``evals_used`` or
  ``sims_used``. If v1's search really does breach P1 the harness must see the
  breaching numbers and void the pair as M10; a runner that clamped its report
  to make P1 hold would delete the detector and leave the anchor unfalsifiable.
* **It does not import komugi.** The only komugi modules in scope are this
  package's own, and the package is stdlib-only: no numpy, no torch, no engine.
  That is what lets it run under ``.#v1``'s python 3.11 against v1's numpy
  while the harness runs under 3.12 against Komugi's.

Two failures are routed to process death on purpose, and it is worth being
explicit about which: an exception out of v1's **search** or out of
``apply_move``/``generate_all_moves`` on an incoming record. Both leave this
process with no truthful response to send -- there is no §D6 mode for "v1's
engine rejected a move Komugi's engine generated" -- so the runner writes the
offending record and the traceback to stderr and exits non-zero, which the
harness books as M1 with that stderr attached verbatim. That is a **named gap
in the pre-registration**, not a design: §D6 has no mode for the second case,
and KOM-023c is where the first real occurrence decides whether it needs one.

One exception to that, and it is the point of :meth:`Runner._pick`'s shape: an
exception out of the search whose *cause* was this move's own budget refusal is
**not** a search failure. It is P2, and P2's remedy is M8. Which of the two it
is, is decided by the counter's own record and never by the exception's type --
so a v1 search that swallows :class:`BudgetRefused` and returns a move anyway,
or one that re-raises it wrapped in its own type, both reach the harness as
``budget_breach`` and neither can produce a move. Nothing truncated is ever
scored, and that is now a property of this file rather than of v1's exception
discipline.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import IO, Any, Final, NoReturn

from komugi.eval.v1runner.counter import (
    DEFAULT_OBSERVATION_RANK,
    BudgetRefused,
    CountedNet,
    EvalCounter,
)
from komugi.eval.v1runner.protocol import (
    BUDGET_BREACH,
    BYE,
    EVAL_CEILING,
    HELLO,
    MOVE,
    NO_MOVE,
    OBSERVATION,
    OBSERVE,
    OK,
    PICK,
    PROBE,
    PROTOCOL_VERSION,
    REFUSED,
    RESET,
    RULES_DIVERGENCE,
    DEFAULT_SIMS_ALLOWANCE,
    ProtocolError,
    encode_message,
    decode_message,
    p1_breach_reason,
    record_from_wire,
    record_key,
    record_to_wire,
    require_int,
    win_state_to_wire,
)
from komugi.eval.v1runner.reference import (
    DEFAULT_GAME_FACTORY,
    LoadedReference,
    ReferenceUnavailable,
    default_reference_root,
    load_reference,
    observation_shape,
)

PROTOCOL_ERROR: Final[str] = "protocol_error"
"""Answer to a request §D1's table does not contain.

Not ``ok``: a runner that answered ``ok`` to an unknown request would let a
harness-side typo pass as a served request, and the divergence would surface
plies later as a desynchronised board. Any non-``ok`` type reaches the harness
as :class:`ProtocolViolation` -> §D6 M4, which is the right bucket for a
version skew.
"""

EXIT_REFUSED: Final[int] = 2
"""Exit status for a startup refusal. Non-zero so ``spawn_v1``'s handshake
branch reports it, and distinct from 1 so an operator can tell a refusal from a
crash without parsing."""


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Everything the runner was told on its command line.

    Args:
        reference_root: Where v1 lives. Defaults to ``docs/reference-v1``, or
            to ``$KOMUGI_V1_REFERENCE_ROOT`` when that is set.
        extra_paths: Further ``sys.path`` entries, relative to the root -- a
            built extension tree usually lands in one.
        game_factory: ``MODULE:ATTR`` of v1's ``GameState``.
        net_factory: ``MODULE:ATTR`` of ``(checkpoint, device) -> net``, where
            ``net.predict(observation) -> (policy, value)``. No default: U-A.
        search_factory: ``MODULE:ATTR`` of ``(net, *, sims, seed) ->
            SearchBackend`` with ``node_count()``, ``sims_used()`` and
            ``pick(game)``. No default: U-A.
        checkpoint: v1's weights. Hashed into ``hello``'s digest.
        device: Reported in ``hello``'s device block and passed to the net
            factory.
        ceiling: §D3.3's per-move ceiling.
        sims_allowance: ``S₁``.
        memo_separable: §D3.2 clause 1's shape question, **declared, not
            detected**. Defaults to ``False`` -- the conservative branch, under
            which acceptance A7e is recorded *vacuous for v1* rather than
            reported as passed. A runner that guessed ``True`` would report a
            witness it does not have.
        memo_citation: The ``file:line`` the U-A report cites for that answer.
        check_imports: Load v1, report, and exit without serving.
    """

    reference_root: Path
    game_factory: str = DEFAULT_GAME_FACTORY
    net_factory: str = ""
    search_factory: str = ""
    checkpoint: str = ""
    device: str = "cpu"
    ceiling: int = EVAL_CEILING
    sims_allowance: int = DEFAULT_SIMS_ALLOWANCE
    memo_separable: bool = False
    memo_citation: str = ""
    check_imports: bool = False
    extra_paths: tuple[str, ...] = field(default_factory=tuple)


class ArgumentError(Exception):
    """A bad command line. Raised instead of ``argparse``'s ``SystemExit``.

    ``argparse`` exits the process and writes to the real ``sys.stderr``, which
    would put a usage message on the operator's terminal and **nothing** on the
    stderr the harness captures. Raising instead routes a bad command line
    through the same structured refusal every other startup failure takes.
    """


class HelpRequested(ArgumentError):
    """``-h``/``--help``: not an error, but still not a §D1 exit.

    argparse's own help action writes usage to the **real ``sys.stdout``** and
    then raises ``SystemExit``, which puts non-protocol text on the one stream
    the harness parses as ``hello`` and bypasses the structured refusal every
    other startup exit takes (review v2 F10). Carrying the usage text as the
    exception's message routes it to stderr, inside the refusal line, where the
    module's claim -- "stdout carries §D1 and nothing else, even when nothing
    can start" -- stays true.
    """


class _Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` that raises :class:`ArgumentError` instead of exiting."""

    def error(self, message: str) -> NoReturn:
        """Raise rather than exit."""
        raise ArgumentError(message)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        """Raise rather than exit, on every path argparse has, not just ``error``."""
        raise ArgumentError(
            message
            if message
            else f"argparse asked to exit with status {status} and no message"
        )


def build_parser() -> argparse.ArgumentParser:
    """Return the runner's command-line parser.

    ``add_help=False`` and a plain flag in its place: the built-in help action
    prints to ``sys.stdout`` from inside ``parse_args``, before anything this
    module could intercept.
    """
    parser = _Parser(
        prog="python -m komugi.eval.v1runner",
        add_help=False,
        description=(
            "The v1 side of kom-023 §D1's cross-engine protocol. Runs inside "
            "`nix develop .#v1`; speaks line-delimited JSON on stdio."
        ),
    )
    parser.add_argument("-h", "--help", action="store_true", dest="help")
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--reference-path", action="append", default=[])
    parser.add_argument("--game-factory", default=DEFAULT_GAME_FACTORY)
    parser.add_argument("--net-factory", default="")
    parser.add_argument("--search-factory", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ceiling", type=int, default=EVAL_CEILING)
    parser.add_argument("--sims-allowance", type=int, default=DEFAULT_SIMS_ALLOWANCE)
    parser.add_argument("--memo-separable", action="store_true")
    parser.add_argument("--memo-citation", default="")
    parser.add_argument("--check-imports", action="store_true")
    return parser


def parse_args(argv: Sequence[str]) -> RunnerConfig:
    """Parse `argv` into a :class:`RunnerConfig`.

    Raises:
        HelpRequested: On ``-h``/``--help``, carrying the usage text.
        ArgumentError: On an unknown flag, a missing value, a non-integer
            ``--ceiling``/``--sims-allowance``, a ceiling below 1, or a ceiling
            above §D3.3's 257. The bounds are checked here rather than at first
            use because a ceiling of 0 makes §D3.4's forced-move root
            evaluation a breach and P1's ``evals >= 1`` floor unsatisfiable --
            every ply would void -- and because a ceiling ABOVE 257 inverts the
            pair/match precedence rev 3 was rewritten to get right (review v2
            F5): the counter would *serve* a 258th evaluation, which P2 forbids
            in terms, and the harness would see ``evals_used`` past the ceiling
            on a ``move`` response and book M10 -- a pair void -- where the
            event is an M8 match void.
    """
    parser = build_parser()
    namespace = parser.parse_args(list(argv))
    if bool(getattr(namespace, "help", False)):
        raise HelpRequested(parser.format_help())
    if int(namespace.ceiling) < 1:
        raise ArgumentError(
            f"--ceiling {namespace.ceiling} is below 1; §D3.4 pins exactly one "
            f"root evaluation on a forced move, so P1's evals >= 1 floor could "
            f"never be met"
        )
    if int(namespace.ceiling) > EVAL_CEILING:
        raise ArgumentError(
            f"--ceiling {namespace.ceiling} is above the pre-registered "
            f"{EVAL_CEILING} (§D3.3). P2 says no move on either side is SERVED "
            f"a {EVAL_CEILING + 1}th evaluation, so a higher ceiling does not "
            f"widen the budget -- it deletes the detector: the breach would "
            f"reach the harness as evals_used on a 'move' response and be "
            f"booked M10, a pair void, instead of M8, the match void that "
            f"exists solely for the side that can trip P2"
        )
    if int(namespace.sims_allowance) < 0:
        raise ArgumentError(f"--sims-allowance {namespace.sims_allowance} is negative")
    root = (
        Path(str(namespace.reference_root))
        if namespace.reference_root is not None
        else default_reference_root()
    )
    return RunnerConfig(
        reference_root=root,
        extra_paths=tuple(str(item) for item in namespace.reference_path),
        game_factory=str(namespace.game_factory),
        net_factory=str(namespace.net_factory),
        search_factory=str(namespace.search_factory),
        checkpoint=str(namespace.checkpoint),
        device=str(namespace.device),
        ceiling=int(namespace.ceiling),
        sims_allowance=int(namespace.sims_allowance),
        memo_separable=bool(namespace.memo_separable),
        memo_citation=str(namespace.memo_citation),
        check_imports=bool(namespace.check_imports),
    )


# ---------------------------------------------------------------------------
# hello's two blocks
# ---------------------------------------------------------------------------


def checkpoint_digest(path: str) -> str:
    """Return ``sha256:<hex>`` of the checkpoint, or ``""`` if none was given.

    ``_checkpoint_evaluator`` refuses to rate unverified weights under a
    checkpoint's name (``arena.py:673-677``); §D1 requires the v1 side to make
    the same refusal possible, and a digest is what makes it possible.
    """
    if not path:
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _optional_module(name: str) -> ModuleType | None:
    """Return an imported module, or ``None`` if it is not installed.

    Used for ``torch`` and ``numpy``, which this package needs only to *report*
    on. Keeping them optional is what lets the whole runner -- and its unit
    suite -- run in the default shell, where v1's torch is not present.
    """
    try:
        return __import__(name)
    except ImportError:
        return None


def pin_determinism(torch: ModuleType | None) -> dict[str, Any]:
    """Set what determinism this process can, and report what actually took.

    ``arena.py:165-192``'s ``pin_determinism`` is process-global torch state
    called by ``play_match``, and the v1 subprocess is a **different process**,
    so none of it reaches here (§D5, review F19). Without this, an A9
    determinism failure caused by cuDNN autotuning in the v1 process is
    indistinguishable from the RNG-discipline failure U-D predicts, and the
    degraded determinism claim ships where two lines would have earned the
    strong one. Those are the two lines.

    Everything is read back after being set rather than assumed: a torch build
    without cuDNN, or one that refuses ``use_deterministic_algorithms`` for an
    op, must be reported as it is, because the block is evidence and not a
    declaration of intent.
    """
    if torch is None:
        return {
            "torch_version": "",
            "cudnn_benchmark": False,
            "deterministic_algorithms": False,
        }
    try:
        torch.backends.cudnn.benchmark = False
    except (AttributeError, RuntimeError):
        pass
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError, ValueError):
        pass
    try:
        benchmark = bool(torch.backends.cudnn.benchmark)
    except (AttributeError, RuntimeError):
        benchmark = False
    try:
        deterministic = bool(torch.are_deterministic_algorithms_enabled())
    except (AttributeError, RuntimeError):
        deterministic = False
    return {
        "torch_version": str(getattr(torch, "__version__", "")),
        "cudnn_benchmark": benchmark,
        "deterministic_algorithms": deterministic,
    }


def device_block(device: str, torch: ModuleType | None, numpy: ModuleType | None) -> dict[str, Any]:
    """Return ``hello``'s device + determinism block (§D1, §D5)."""
    block = pin_determinism(torch)
    block["numpy_version"] = str(getattr(numpy, "__version__", "")) if numpy else ""
    block["device"] = device
    return block


# ---------------------------------------------------------------------------
# the serve loop
# ---------------------------------------------------------------------------


class Runner:
    """One v1 process, one match: §D1's adapter, holding v1's own board.

    Args:
        reference: The loaded v1 pieces.
        config: The command line this was started with.
        counter: §D3.5's counted evaluator. Injected so the unit suite can
            drive the arithmetic directly.
        torch: v1's torch, or ``None``. Reporting only.
        numpy: v1's numpy, or ``None``. Reporting only.
        warn: Where P1 self-diagnostics go. Never stdout -- a diagnostic on
            stdout is a protocol violation.
    """

    def __init__(
        self,
        *,
        reference: LoadedReference,
        config: RunnerConfig,
        counter: EvalCounter | None = None,
        torch: ModuleType | None = None,
        numpy: ModuleType | None = None,
        warn: IO[str] | None = None,
    ) -> None:
        self._reference = reference
        self._config = config
        self._counter = counter if counter is not None else EvalCounter(config.ceiling)
        self._observation_shape = observation_shape(reference.game_cls)
        self._net = CountedNet(
            reference.net,
            self._counter,
            observation_rank=(
                len(self._observation_shape) or DEFAULT_OBSERVATION_RANK
            ),
        )
        self._torch = torch
        self._numpy = numpy
        self._warn = warn
        self._game: Any = None
        self._stopped = False

    @property
    def counter(self) -> EvalCounter:
        """Return the counter, for diagnostics and the unit suite."""
        return self._counter

    @property
    def stopped(self) -> bool:
        """Return whether ``bye`` has been answered."""
        return self._stopped

    @property
    def game(self) -> Any:
        """Return v1's current ``Game``, or ``None`` before the first ``reset``."""
        return self._game

    def _note(self, text: str) -> None:
        """Write one diagnostic line to stderr, if a stream was given."""
        if self._warn is not None:
            self._warn.write(text + "\n")
            self._warn.flush()

    # -- §D1's requests ----------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        """Answer one request. Every §D1 row, and a typed answer for the rest."""
        kind = message["type"]
        if kind == HELLO:
            return self._hello(message)
        if kind == RESET:
            return self._reset(message)
        if kind == OBSERVE:
            return self._observe(message)
        if kind == PICK:
            return self._pick(message)
        if kind == PROBE:
            return self._probe()
        if kind == BYE:
            self._stopped = True
            return {"type": OK}
        return {
            "type": PROTOCOL_ERROR,
            "detail": (
                f"{kind!r} is not a request in kom-023 §D1's protocol table; "
                f"this adapter speaks hello, reset, observe, pick, probe, bye"
            ),
        }

    def _hello(self, message: dict[str, Any]) -> dict[str, Any]:
        """Answer the handshake, echoing the **resolved** start mode.

        Every field ``spawn_v1`` sends is consumed, not just the mode:
        ``protocol_version``, ``gungi_mode``, ``checkpoint``, ``ceiling``,
        ``sims_allowance`` and ``seed_policy`` (``v1bridge.py:1036-1044``). Three
        of them used to be dropped on the floor, which let the two ends disagree
        about the ceiling and about ``S₁`` -- the number that decides what 257
        *means* -- with nothing anywhere to show it (review v2 F5). The response
        reports the **resolved** value of each, so a harness-side gate can bite
        later, and any disagreement is on stderr now.

        ``observation_size`` is here for the same reason: §D1's table row 1
        lists it, §D2.1 proves v1 is 117 planes against Komugi's 128, and
        ``Game.observation_size()`` is free to call at handshake time -- so "v1's
        encoder is not the one this design was written against" can be a
        handshake fact rather than a mid-match A4 probe failure (review v2 F6).

        §D4 amendment 2 makes a ``GUNGI_MODE`` mismatch fatal at startup, and
        the echo is what the harness gates on. So the mode the harness names is
        *applied* here -- v1 exposes ``Game.set_start_mode``
        (``docs/reference-v1/alphazero-env/gungi.py:192-197``) -- and then read
        back off the class rather than echoed from the request. Applying it is
        right because both engines must start from the same position; reading
        it back is what keeps the gate real, because a mode v1 does not know
        raises inside ``set_start_mode``, the class keeps its old value, and
        the echo then differs and the harness refuses (``spawn_v1``'s
        ``gungi_mode`` branch). An adapter that echoed the request would pass
        the gate while playing a different opening.
        """
        mode = str(message.get("gungi_mode", ""))
        mode_error = ""
        if mode:
            try:
                self._reference.game_cls.set_start_mode(mode)
            except Exception as error:  # noqa: BLE001 - v1 raises ValueError here,
                # but a stale checkout may raise anything, and a mode this
                # adapter cannot honour must reach the harness as a mismatched
                # echo rather than as a crash of the handshake.
                mode_error = f"{type(error).__name__}: {error}"
                self._note(
                    f"v1runner: the harness asked for GUNGI_MODE {mode!r} and v1 "
                    f"refused it ({mode_error}); echoing v1's own resolved mode "
                    f"so the §D4 amendment 2 gate fires at startup"
                )
        resolved = str(getattr(self._reference.game_cls, "_START_MODE", ""))
        self._consume_harness_hello(message)
        return {
            "type": HELLO,
            "protocol_version": PROTOCOL_VERSION,
            "checkpoint_digest": checkpoint_digest(self._config.checkpoint),
            "action_size": int(self._reference.game_cls.action_size()),
            "observation_size": [int(dim) for dim in self._observation_shape],
            "gungi_mode": resolved,
            "ceiling": int(self._counter.ceiling),
            "sims_allowance": int(self._config.sims_allowance),
            "seed_policy": str(message.get("seed_policy", "")),
            "seed_source": "harness_per_pick",
            "capabilities": {
                "supports_no_move": True,
                "supports_budget_breach": True,
                "memo_separable": bool(self._config.memo_separable),
                "memo_citation": self._config.memo_citation,
            },
            "device": device_block(self._config.device, self._torch, self._numpy),
        }

    def _consume_harness_hello(self, message: dict[str, Any]) -> None:
        """Read every field the harness sent, and say so when one disagrees.

        Nothing here *adopts* a harness value. Adopting the ceiling would move
        the budget under a counter that is already constructed, and adopting
        ``S₁`` would change what the factory was probed with at startup; both
        would hide the disagreement instead of reporting it. The resolved values
        go back in the response, and the mismatch goes to stderr, which
        ``spawn_v1`` attaches verbatim to any refusal (``v1bridge.py:1048-1055``).
        """
        version = message.get("protocol_version")
        if isinstance(version, int) and not isinstance(version, bool):
            if int(version) != PROTOCOL_VERSION:
                self._note(
                    f"v1runner: the harness speaks protocol version {version} "
                    f"and this adapter speaks {PROTOCOL_VERSION}; answering "
                    f"with my own so spawn_v1's version gate fires"
                )
        ceiling = message.get("ceiling")
        if isinstance(ceiling, int) and not isinstance(ceiling, bool):
            if int(ceiling) != self._counter.ceiling:
                self._note(
                    f"v1runner: the harness enforces a per-move ceiling of "
                    f"{ceiling} and this adapter was started with "
                    f"{self._counter.ceiling} (§D3.3). Reporting mine: a "
                    f"disagreement here decides whether a breach reaches the "
                    f"harness as M8 (match void) or M10 (pair void)"
                )
        allowance = message.get("sims_allowance")
        if isinstance(allowance, int) and not isinstance(allowance, bool):
            if int(allowance) != self._config.sims_allowance:
                self._note(
                    f"v1runner: the harness's S1 is {allowance} and this "
                    f"adapter passes {self._config.sims_allowance} to its "
                    f"search factory. S1 is the number §D3.3 says fixes the "
                    f"ceiling, so a mismatch changes what 257 means"
                )
        checkpoint = str(message.get("checkpoint", ""))
        if checkpoint and checkpoint != self._config.checkpoint:
            self._note(
                f"v1runner: the harness believes v1 plays {checkpoint!r} and "
                f"this adapter loaded {self._config.checkpoint!r}; the digest "
                f"in this hello is of MINE (arena.py:673-677 refuses to rate "
                f"unverified weights under a checkpoint's name)"
            )
        if not self._observation_shape:
            self._note(
                "v1runner: this game class does not expose observation_size(), "
                "so hello reports an empty observation_size and §D2.1's plane "
                "count cannot be checked until the first A4 probe"
            )

    def _reset(self, message: dict[str, Any]) -> dict[str, Any]:
        """Start a new game inside the live process (§D1's ``reset``).

        A fresh ``Game`` and nothing carried over: the pair/seat are read and
        validated so a malformed ``reset`` is caught here rather than becoming
        an unexplained desync, but they are not otherwise used -- v1's board is
        seeded from the start mode, and the seat is the harness's bookkeeping.
        """
        require_int(message, "pair")
        require_int(message, "seat")
        self._game = self._reference.game_cls()
        return {"type": OK}

    def _observe(self, message: dict[str, Any]) -> dict[str, Any]:
        """Apply one played move -- either side's -- to v1's own board.

        §D2.2: this is the **only** way the v1 process learns what the opponent
        played. The record is applied directly rather than re-encoded to a v1
        action id and passed to ``play_action``, and that is forced rather than
        chosen: §D2.3 proves no v1 action id names a partial betrayal or a
        decline-the-betrayal tsuke, so an id round trip would fail on exactly
        the position class Komugi's encoding was widened to reach. This mirrors
        ``docs/reference-v1/alphazero-env/gungi.py:248-254`` with the decode
        removed.

        ``last_action`` is set to -1 for the same reason: there is no id to
        record. It is read by v1's ``clone``/``__reduce__`` (``gungi.py:295``,
        ``:303``) and by nothing this runner calls, but leaving it stale would
        make a pickled v1 ``Game`` claim a move it did not play.
        """
        if self._game is None:
            raise ProtocolError("observe before reset: this adapter has no game")
        record = record_from_wire(message.get("record"))
        board = self._game._board
        self._require_v1_generates(record, board)
        try:
            self._reference.cyengine.apply_move(board, record)
        except Exception:
            self._note(
                f"v1runner: v1's engine refused a record Komugi's engine "
                f"generated: {record!r}\n{traceback.format_exc()}"
            )
            raise
        self._game._history.append(self._reference.cyengine.placement_key(board))
        self._game._player = board.turn
        self._game._turns += 1
        self._game.last_action = -1
        return {"type": OK}

    def _require_v1_generates(self, record: dict[str, Any], board: Any) -> None:
        """Refuse an incoming record v1's own generator does not contain (§D2.3).

        §D2.3 pre-registers **both** halves of the contract -- "each side decodes
        its own action id to a record before sending, *and matches an incoming
        record against its own ``generate_all_moves`` output before applying*" --
        and only the first half was implemented (review v2 F4).
        ``cyengine.apply_move`` is a mutator, not a validator: it writes the
        record's fields onto the board. So a record that is schema-valid but not
        legal *in v1's position* -- exactly the class §D2.3 spends a page on,
        where the two engines' legal sets genuinely differ -- used to be applied
        silently, after which v1's 117-plane observation is computed from a
        position Komugi never reached, v1's search plays from a fiction, and no
        §D6 mode fires: not M3 (Komugi is the referee and Komugi generated the
        move), not M4 (the schema passed), not M5 (v1's mask is non-empty, on
        the wrong board).

        The refusal is :class:`ProtocolError` -- M4 at the harness, since
        ``V1Connection.observe`` requires ``ok`` -- and the detail names
        :data:`RULES_DIVERGENCE` so the one event this check exists for is
        greppable rather than indistinguishable from a schema complaint. It is
        raised **before** ``apply_move``, so the board is untouched: a refusal
        that had already mutated the board would leave the process with no
        truthful state to keep serving from.
        """
        generator = getattr(self._reference.cymovegen, "generate_all_moves", None)
        if not callable(generator):
            raise ProtocolError(
                "v1's cymovegen has no callable generate_all_moves, so §D2.3's "
                "incoming-record match cannot be made and an illegal-under-v1 "
                "record would be applied silently"
            )
        try:
            candidates = list(generator(board))
        except Exception:
            # v1's own generator failing on v1's own board is a v1 failure, not
            # a protocol one; it takes the route the module docstring names --
            # stderr and process death, booked M1 with this stderr attached.
            self._note(
                f"v1runner: v1's generate_all_moves raised on v1's own board "
                f"while checking an incoming record: {record!r}\n"
                f"{traceback.format_exc()}"
            )
            raise
        wanted = record_key(record)
        for candidate in candidates:
            try:
                if record_key(candidate) == wanted:
                    return
            except ProtocolError:
                # A candidate this runner cannot key cannot be the match; v1's
                # own records are not this runner's to validate.
                continue
        raise ProtocolError(
            f"{RULES_DIVERGENCE}: the harness sent a schema-valid record that "
            f"v1's own generate_all_moves does not contain at this position, so "
            f"applying it would desynchronise the two boards with no §D6 mode "
            f"firing (§D2.3's incoming-record match). Record {record!r}; v1 "
            f"generated {len(candidates)} moves here"
        )

    def _pick(self, message: dict[str, Any]) -> dict[str, Any]:
        """Search one move and answer with one of §D1's three ``pick`` types."""
        if self._game is None:
            raise ProtocolError("pick before reset: this adapter has no game")
        ply = require_int(message, "ply")
        seed = require_int(message, "seed")
        game = self._game
        win_state = win_state_to_wire(game.win_state())

        mask = game.valid_moves()
        legal = int(mask.sum())
        if legal == 0:
            # §D6 M5, and it is *expected to occur*: v1's valid_moves returns an
            # all-zero mask where its cells-only fourfold key has fired and
            # Komugi's repaired 274-byte key has not (rules-audit F1). A
            # distinct response type is the only way this reaches the harness as
            # itself -- with an all-zero mask a PUCT raises, divides by zero, or
            # returns an arbitrary index, and the harness would observe M1, M4
            # or M3 instead.
            return {
                "type": NO_MOVE,
                "win_state": win_state,
                "reason": (
                    "v1's valid_moves is all-zero at a position Komugi has not "
                    "adjudicated (gungi.py:234-246)"
                ),
            }

        search: Any = None
        nodes = 0
        action = -1
        sims = 0
        # A refusal that escaped, in case it was raised by something other than
        # this move's counter -- the counter stays the primary authority.
        escaped: int | None = None
        self._counter.begin_move()
        try:
            # §D3.2 clause 2: a FRESH search per pick. Constructed here, inside
            # the per-move tally, and never stored on self -- there is nothing
            # for a later move to reuse even by accident.
            search = self._reference.search_factory(
                self._net, sims=self._config.sims_allowance, seed=seed
            )
            # P3 is enforced by the harness on this number, so it is *measured*
            # right after construction rather than reported as the 0 the
            # construction is supposed to guarantee. A runner that hard-coded 0
            # would make P3 vacuous exactly where it is load-bearing.
            nodes = int(search.node_count())
            if legal == 1:
                # §D3.4's forced-move pin: exactly ONE root evaluation through
                # the counted evaluator, mirroring gumbel.py:858, and
                # sims_used == 0. Not zero evaluations: P1's
                # `sims_used == 0 => evals == 1` would then fire at every v1
                # forced ply and void every pair containing one, into a bucket
                # that counts against the 2% admissibility gate. That
                # evaluation *is* part of this move's search for clause 3.
                self._net.predict(game.observation())
                action = int(mask.nonzero()[0][0])
                sims = 0
            else:
                action = int(search.pick(game))
                sims = int(search.sims_used())
        except BudgetRefused as refusal:
            escaped = refusal.attempted
            # §D3.5's report-and-refuse. The search is unwound, the move is
            # discarded, and THE PROCESS STAYS ALIVE: a raise that reached the
            # harness as a process death would be M1 `crash` -- a pair void
            # counting toward "three crashes abort" -- and M8, the match-level
            # remedy that exists solely for the only side that can trip P2,
            # would be unreachable (re-review F1). The numbers are read off the
            # counter below rather than off this exception, so that this arm and
            # the two arms beneath it report identically.
        except Exception as error:  # noqa: BLE001 - see the comment
            # A search that catches BudgetRefused and re-raises it as its own
            # type -- `raise SearchError(...) from BudgetRefused`, or a Cython
            # layer that converts it -- is a *masked budget refusal*, not a
            # search failure, and the two are distinguishable by a flag the
            # counter already set. Masked or not, a breach is M8. A genuine
            # search failure (no refusal recorded) still takes the route the
            # module docstring names: out of here, out of serve, process death,
            # M1 with this stderr attached.
            if self._counter.refused_at is None:
                raise
            self._note(
                f"v1runner: v1's search converted the counted evaluator's "
                f"refusal into {type(error).__name__}: {error} (ply {ply}); the "
                f"counter recorded the breach, so this is M8 -- a match void -- "
                f"and not M1"
            )
        finally:
            self._counter.end_move()

        # §D3.5's exclusivity, made STRUCTURAL rather than left to v1's
        # exception discipline (review v2 F1). The counter -- not the presence
        # of an exception -- decides, and it is consulted on every path out of
        # the search, so a breaching ply CANNOT also yield a move. The path that
        # made it possible: a v1 MCTS descending inside `try/except Exception`,
        # or one evaluating leaves through a thread pool where the refusal is
        # captured into a future, swallows the 258th ask, finishes its
        # simulations on what it already has, and returns an action. `served` is
        # capped at the ceiling by construction, so the old success path
        # reported a perfectly P1-conformant {"evals_used": 257, "sims_used":
        # 256} and the harness scored a truncated move -- "nothing truncated is
        # ever scored" being, until now, a property of v1's code and not of this
        # runner's.
        refused_at = self._counter.refused_at
        if refused_at is None:
            refused_at = escaped
        if refused_at is not None:
            self._note(
                f"v1runner: the counted evaluator refused evaluation "
                f"{refused_at} within one move (ceiling "
                f"{self._counter.ceiling}, §D3.3/§D3.5) at ply {ply}, seed "
                f"{seed}; answering budget_breach, discarding the move and "
                f"staying alive"
            )
            return {
                "type": BUDGET_BREACH,
                "evals_used": refused_at,
                "sims_used": self._sims_after_breach(search),
                "tree_nodes_at_entry": nodes,
            }

        evals = self._counter.served
        # The COUNTER's ceiling, never the config's: the counter is what
        # actually enforced the budget, and a self-diagnostic that disagreed
        # with the enforcement is the one thing a self-diagnostic must not do
        # (review v2 F11).
        breach = p1_breach_reason(evals, sims, self._counter.ceiling)
        if breach is not None:
            # Reported, never repaired. The harness voids the pair as M10 on
            # these very numbers; changing them here would delete the detector.
            self._note(
                f"v1runner: this move's own numbers breach §D3.6 P1 at ply "
                f"{ply}: {breach}. Reporting them unchanged -- the harness "
                f"voids the pair as M10, which is the correct outcome"
            )
        record = self._reference.action_encoding.decode_cython_action(
            action, self._game._board
        )
        return {
            "type": MOVE,
            "record": record_to_wire(record),
            "evals_used": evals,
            "sims_used": sims,
            "tree_nodes_at_entry": nodes,
            "win_state": win_state,
        }

    def _sims_after_breach(self, search: Any) -> int:
        """Return ``sims_used`` from a search that was just unwound, or 0.

        Defensive on purpose: the breach happened *inside* the search, so the
        backend's own bookkeeping may be mid-update, and a second exception
        raised while building the breach report would convert the M8 the design
        went to some trouble to make reachable back into the M1 it was rescued
        from.
        """
        if search is None:
            return 0
        try:
            return int(search.sims_used())
        except Exception:  # noqa: BLE001 - see the docstring
            return 0

    def _probe(self) -> dict[str, Any]:
        """Return v1's current observation -- §D2.4's differential probe (A4).

        Issued **outside** any move's tally: §D3.2 clause 3 excludes anything
        that is not this move's search, and ``observation()`` does not touch
        the network at all (``gungi.py:288-293``).
        """
        if self._game is None:
            raise ProtocolError("probe before reset: this adapter has no game")
        observation = self._game.observation()
        return {
            "type": OBSERVATION,
            "shape": [int(dim) for dim in observation.shape],
            "data": base64.b64encode(
                observation.astype("float32").tobytes()
            ).decode("ascii"),
        }

    # -- the loop ----------------------------------------------------------

    def serve(self, stdin: IO[str], stdout: IO[str]) -> int:
        """Read requests until ``bye`` or EOF. Returns the process exit status.

        A blank line is skipped rather than treated as a message: the framing
        is one message per line, and a stray newline from a shell pipeline is
        not a protocol error worth voiding a match over. Anything else that is
        not one protocol message is answered with :data:`PROTOCOL_ERROR`, which
        the harness books as M4.
        """
        for line in stdin:
            if not line.strip():
                continue
            try:
                message = decode_message(line)
            except ProtocolError as error:
                stdout.write(
                    encode_message({"type": PROTOCOL_ERROR, "detail": str(error)})
                )
                stdout.flush()
                continue
            try:
                response = self.handle(message)
            except ProtocolError as error:
                response = {"type": PROTOCOL_ERROR, "detail": str(error)}
            stdout.write(encode_message(response))
            stdout.flush()
            if self._stopped:
                return 0
        return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def write_refusal(stream: IO[str], payload: dict[str, Any]) -> None:
    """Write one structured refusal line to stderr, and a human line beside it.

    Both, deliberately. The JSON line is what an operator or KOM-023c greps;
    the prose line is what a human reads in the middle of a
    :class:`HandshakeRefused` traceback, where ``spawn_v1`` has appended this
    stream verbatim under ``-- v1 stderr --`` (``v1bridge.py:1048-1055``).
    """
    body = dict(payload)
    body["type"] = REFUSED
    stream.write(encode_message(body))
    stream.write(
        f"v1runner: refusing to start -- {body.get('reason')}: "
        f"{body.get('detail')}\n"
    )
    stream.flush()


def main(
    argv: Sequence[str],
    *,
    stdin: IO[str],
    stdout: IO[str],
    stderr: IO[str],
) -> int:
    """Run the adapter. Returns the process exit status.

    The streams are parameters rather than ``sys.*`` so the unit suite can
    drive a whole handshake through :func:`main` in-process, in the default
    shell, with no subprocess and no v1.
    """
    try:
        config = parse_args(argv)
    except HelpRequested as usage:
        # Not an error, but not §D1 either: usage text belongs on stderr,
        # inside the refusal line, and never on the stream the harness parses
        # as `hello` (review v2 F10). The status is EXIT_REFUSED for the same
        # reason every other non-protocol exit is: this process served no move.
        write_refusal(
            stderr,
            {"reason": "help_requested", "detail": str(usage), "missing": []},
        )
        return EXIT_REFUSED
    except ArgumentError as error:
        write_refusal(
            stderr,
            {"reason": "bad_arguments", "detail": str(error), "missing": []},
        )
        return EXIT_REFUSED

    torch = _optional_module("torch")
    numpy = _optional_module("numpy")
    try:
        reference = load_reference(
            root=config.reference_root,
            game_factory=config.game_factory,
            net_factory=config.net_factory,
            search_factory=config.search_factory,
            checkpoint=config.checkpoint,
            device=config.device,
            sims_allowance=config.sims_allowance,
            extra_paths=config.extra_paths,
        )
    except ReferenceUnavailable as error:
        write_refusal(stderr, error.payload())
        return EXIT_REFUSED

    if config.check_imports:
        stdout.write(
            encode_message(
                {
                    "type": OK,
                    "reference_root": str(config.reference_root),
                    "package": reference.package,
                    "searched": list(reference.searched),
                    "action_size": int(reference.game_cls.action_size()),
                    "observation_size": [
                        int(dim) for dim in observation_shape(reference.game_cls)
                    ],
                    "gungi_mode": str(
                        getattr(reference.game_cls, "_START_MODE", "")
                    ),
                    "checkpoint_digest": checkpoint_digest(config.checkpoint),
                    "device": device_block(config.device, torch, numpy),
                }
            )
        )
        stdout.flush()
        return 0

    runner = Runner(
        reference=reference,
        config=config,
        torch=torch,
        numpy=numpy,
        warn=stderr,
    )
    return runner.serve(stdin, stdout)
