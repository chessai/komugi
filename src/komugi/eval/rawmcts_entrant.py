"""§D6.3's seam: a RawMCTS ruler, seated as a cross-engine :class:`BridgeEntrant`.

The other half of the module split ``docs/design/kom-011d-ladder-pool.md``
§D6.3 makes, and the reason for the split is one import: a
:class:`~komugi.eval.v1bridge.BridgeEntrant` comes from
:mod:`komugi.eval.v1bridge`, which imports :mod:`komugi.eval.arena`
(``v1bridge.py:67``), which imports torch (``arena.py:81``). **This module is
torch-tolerant and claims nothing otherwise.** :mod:`komugi.eval.rawmcts` --
the search itself -- imports none, at import time or in a function body, and §6
node 29 pins both halves of that boundary in spawned interpreters.

Rev 1 of the design asserted a single no-torch constraint over both; it is
mechanically impossible against this import graph, and pretending otherwise
would have made node 29 red by construction.

The identity, and why it is minted here
---------------------------------------
§D1.2 gives the family the member form ``n<sims>@<sha8>`` -- *"sim count **and**
the digest of §D1.3's config block"*. A ``rawmcts`` entrant is **code, not
weights**, so "frozen" has to be constructed: :func:`rawmcts_digest` hashes a
canonical JSON block naming every input that could change a move, and
:func:`rawmcts_entrant_id` puts eight hex characters of it in the id. An
entrant whose configuration drifts therefore gets a **new row** rather than
silently becoming a different opponent under an old one -- which is the
property §D1.2 property 2 exists for, and the failure
``ab_endpoint.py:326-351`` already had to be written against once.

The block is §D1.3's, with one field carrying a different value than the design
text's illustrative literal: ``rollout`` is ``"random"``, because §D6.3 defines
RawMCTS as *"value-by-random-rollout"* and §D1.3's example block (which is
shared with the not-yet-built ``ab`` family) prints ``"none"``. The digest is
opaque either way; a block that described this mover's rollout as absent would
be a false statement in a published manifest, which is the one thing it may not
be.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from komugi.engine.action_encoding import ACTION_SIZE
from komugi.eval.rawmcts import DEFAULT_C_PUCT, DEFAULT_SIMS, RawMctsMover
from komugi.eval.records import Player
from komugi.eval.v1bridge import BridgeEntrant, Mover, resolved_gungi_mode

RAWMCTS_FAMILY: Final[str] = "rawmcts"
"""The entrant family, one of :data:`~komugi.eval.records.KNOWN_FAMILIES`."""

ENGINE_RULES_VERSION: Final[int] = 1
"""Which revision of the engine's *rules* this entrant is frozen against.

Added by §D1.3, and it is not decoration: the repetition repair
(``docs/design/rules-audit.md:361-411``) changes which games are drawn, so a
``rawmcts`` entrant measured before and after it is not the same opponent, and
no weights digest would notice because there are no weights. Bumping this
integer is a **deliberate act** that mints a new entrant id and therefore a new
row in the pool -- which is the correct outcome and not a loss.
"""

DIGEST_CHARS: Final[int] = 8
"""Hex characters of the config digest that reach the id (§D1.2's ``@sha8``)."""

NO_GUMBEL_ROOT: Final[int] = 0
"""What :class:`BridgeEntrant`'s Gumbel fields carry for a net-free ruler.

``m`` and ``gumbel_scale`` describe the *arena's* root sampling, and this player
has no root sampling: it consults no network and runs no Gumbel round.
:func:`~komugi.eval.v1bridge.anchor_configuration` publishes those two fields
verbatim (``v1bridge.py:2300-2302``), so carrying the arena's ``32`` and ``1.0``
here would put a configuration in a published block that this player never ran.
``0`` declares the absence instead.
"""


def rawmcts_config_block(
    *,
    sims: int,
    c_puct: float,
    draw_score: float,
    gungi_mode: str | None = None,
) -> dict[str, object]:
    """Return §D1.3's config block: every input that could change a move.

    ``action_size`` and ``gungi_mode`` are the two guards KOM-014's book loader
    already found necessary (``book.py:1160-1181``): an action id *"means
    nothing across an encoding change"*, and a line played under a different
    start mode *"is a different line, or illegal"*.

    Args:
        sims: The per-move simulation budget.
        c_puct: The PUCT constant the mover selects with.
        draw_score: The contempt the games are declared to be played under.
        gungi_mode: The start mode. Defaults to the one **this process's engine
            resolved** (:func:`~komugi.eval.v1bridge.resolved_gungi_mode`),
            read from the engine rather than the environment because
            ``game.py:250`` reads ``GUNGI_MODE`` at import time.

    Returns:
        The block, ready for :func:`rawmcts_digest`.
    """
    return {
        "family": RAWMCTS_FAMILY,
        "sims": int(sims),
        "rollout": "random",
        "prior": "uniform",
        "c_puct": float(c_puct),
        "draw_score": float(draw_score),
        "action_size": int(ACTION_SIZE),
        "gungi_mode": gungi_mode if gungi_mode is not None else resolved_gungi_mode(),
        "engine_rules_version": ENGINE_RULES_VERSION,
    }


def rawmcts_digest(block: dict[str, object]) -> str:
    """Return the ``sha8`` of `block`, over a canonical JSON serialisation.

    Canonical means sorted keys and no insignificant whitespace, so the digest
    is a function of the *configuration* and not of the order a caller happened
    to build a dict in.
    """
    canonical = json.dumps(block, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:DIGEST_CHARS]


def rawmcts_entrant_id(
    *,
    sims: int = DEFAULT_SIMS,
    c_puct: float = DEFAULT_C_PUCT,
    draw_score: float,
    gungi_mode: str | None = None,
) -> str:
    """Return §D1.2's id for this configuration: ``rawmcts/n<sims>@<sha8>``.

    Public and separate from :func:`rawmcts_entrant` because the pool mints ids
    into its epoch manifest before it builds anything to play with them (§D1.2
    property 4: *"minted once, by the pool … not recomputed at read time"*),
    and because a reader checking that a stored id belongs to a stated
    configuration must be able to recompute it without an engine.

    No ``:`` appears anywhere in the result, which §D1.2 property 1 requires:
    ``pair_id`` is ``:``-joined (``arena.py:691``, ``v1bridge.py:2171``).
    """
    block = rawmcts_config_block(
        sims=sims, c_puct=c_puct, draw_score=draw_score, gungi_mode=gungi_mode
    )
    return f"{RAWMCTS_FAMILY}/n{int(sims)}@{rawmcts_digest(block)}"


def rawmcts_entrant(
    *,
    sims: int = DEFAULT_SIMS,
    draw_score: float,
    c_puct: float = DEFAULT_C_PUCT,
    gungi_mode: str | None = None,
    player_id: str = "",
) -> BridgeEntrant:
    """Return a §D6.3 RawMCTS :class:`~komugi.eval.v1bridge.BridgeEntrant`.

    :func:`komugi.eval.gumbel_mover.komugi_entrant`'s shape, and the same three
    load-bearing details:

    * the family is :data:`RAWMCTS_FAMILY`, so the roster map that travels with
      a match (§D4 amendment 3, ``v1bridge.py:2323-2332``) says which engine
      this row is without anyone parsing an id prefix;
    * the factory builds a **fresh** :class:`~komugi.eval.rawmcts.RawMctsMover`
      per call -- once per match, and once more per §D6 M1 respawn -- rather
      than closing over a live one;
    * the configuration that identifies this entrant is carried *into* the id
      as well as into the mover, so the published identity is the configuration
      that was played.

    This player consults no network, so it is **exempt** from KOM-023's
    257-evaluation ceiling (§D7.4) -- there is nothing to count -- and the
    exemption is a declaration the manifest carries, not something inferred
    from the absence of a counter here.

    Args:
        sims: The per-move simulation budget; §D1.1 pre-registers
            ``{100, 800, 6400}``.
        draw_score: The contempt the games are played under; §D7.3 pins ``0.0``
            for every pool game. Required, with no default, for
            :class:`~komugi.eval.arena.ArenaConfig`'s own reason. **The mover
            never reads it**: a RawMCTS rollout scores a terminal position
            through the referee's own
            :meth:`komugi.selfplay.adapter.GameState.terminal_value`, which
            carries the ``ArenaConfig``'s ``draw_score``. It is required here
            because it is one of §D1.3's inputs that *could change a move*, so
            it belongs in the identity digest -- and passing one value here
            while playing under another would publish an id for a
            configuration that was not played. The seam cannot check that, for
            the reason ``gumbel_mover.py:211-216`` states: ``pick`` is handed a
            position, not a config.
        c_puct: The PUCT constant; part of the identity, not a tuning knob.
        gungi_mode: The start mode for the digest; defaults to the engine's own.
        player_id: An explicit id, for a pool that mints ids itself. Defaults
            to :func:`rawmcts_entrant_id` for this configuration.

    Returns:
        The entrant, ready for
        :func:`~komugi.eval.v1bridge.play_bridged_match`.
    """
    entrant_id = player_id or rawmcts_entrant_id(
        sims=sims, c_puct=c_puct, draw_score=draw_score, gungi_mode=gungi_mode
    )

    def build() -> Mover:
        """Build this side's mover, fresh, on the call."""
        return RawMctsMover(sims=sims, player_id=entrant_id, c_puct=c_puct)

    return BridgeEntrant(
        player=Player(id=entrant_id, family=RAWMCTS_FAMILY),
        mover_factory=build,
        m=NO_GUMBEL_ROOT,
        n=int(sims),
        gumbel_scale=float(NO_GUMBEL_ROOT),
    )
