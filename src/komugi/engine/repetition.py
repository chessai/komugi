"""Which repetition KEY a :class:`komugi.engine.game.Game` counts (KOM-141).

Two rules, and the difference between them is the whole of the gungi.js parity
question:

* :attr:`RepetitionRule.FULL` — Komugi's own key, the KOM-027 repair of
  ``docs/design/rules-audit.md`` findings F1-F4: cells + side to move + draft
  flags + hands, 274 bytes (``cyengine.repetition_key``).  It is the key every
  training game, every self-play row and every arena game is refereed under,
  and it is the DEFAULT everywhere in this repo.
* :attr:`RepetitionRule.PLACEMENT` — gungi.js@3.4.11's key, the one its
  ``isFourfoldRepetition`` builds: **FEN field 1 only**, i.e. the cells and
  nothing else, 243 bytes (``cyengine.placement_key``).  It exists for exactly
  one reason: so a REFEREE can apply the site's rule and the Elo cost of the
  mismatch can be measured.  By construction it re-introduces the aliasing the
  audit documents — Scenario A (same cells, opposite side to move) and
  Scenario B (same cells, hands spent) collapse into one key again — so it
  must never be the default anywhere, and reaching it always takes an explicit
  ``repetition_rule=`` keyword.  ``tests/engine/test_repetition_rule_default.py``
  is the guard that keeps that true (it is rules-audit Q13's successor: the
  raising ``placement_key`` alias used to be).

This module is **stdlib only, and deliberately a leaf**: it imports nothing
from ``komugi`` and nothing compiled.  That is what lets
:class:`komugi.config.SelfPlayConfig` name a rule without dragging the Cython
engine or torch into a config import.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class RepetitionRule(str, Enum):
    """The key a game's fourfold-repetition counter is keyed on.

    A ``str`` enum so a rule round-trips through a CLI flag, a JSON report and
    a config file as its own value (``"full"`` / ``"placement"``) with no
    translation table: ``RepetitionRule("full") is RepetitionRule.FULL``, and
    ``json.dumps`` writes the string rather than a repr.
    """

    #: Komugi's repaired 274-byte key (rules-audit F1-F4 / KOM-027).
    FULL = "full"
    #: gungi.js@3.4.11's cells-only 243-byte key.  Referee use only.
    PLACEMENT = "placement"


FULL_KEY_BYTES: Final[int] = 274
"""Length of a :attr:`RepetitionRule.FULL` key: 243 cells + 3 scalars + 28 hand."""

PLACEMENT_KEY_BYTES: Final[int] = 243
"""Length of a :attr:`RepetitionRule.PLACEMENT` key: ``int8[9][9][3]`` cells."""


def key_bytes_for(rule: RepetitionRule) -> int:
    """Return the exact byte length of the key *rule* builds.

    The one place the two lengths are mapped, so a history validator and a key
    mirror cannot disagree about how long a key of a given rule is.

    Raises:
        ValueError: If *rule* is not a :class:`RepetitionRule`.  A new member
            added without a length here fails loudly rather than defaulting to
            one of the two and silently validating the wrong thing.
    """
    if rule is RepetitionRule.FULL:
        return FULL_KEY_BYTES
    if rule is RepetitionRule.PLACEMENT:
        return PLACEMENT_KEY_BYTES
    raise ValueError(
        f"{rule!r} is not a RepetitionRule; the known rules are "
        f"{[member.value for member in RepetitionRule]}"
    )
