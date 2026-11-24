"""The tactical-provider seam: a rules-level generator the search consults.

KOM-222, unit U2 of the Gumbel tactical-coverage decomposition. The precedent
is Teytaud & Teytaud (2010)'s *decisive moves*: a class of moves a rules layer
can enumerate exactly, and which a Monte-Carlo tree search must not be allowed
to miss merely because its policy prior gave them no mass. A
:class:`TacticalProvider` is consulted **regardless of the prior**, so the
protocol below is deliberately narrow, read-only, and frozen: it is the whole
of what the search is allowed to learn from the rules layer.

Nothing here imports from ``komugi.selfplay`` or ``komugi.engine``. The
protocol is generic over :class:`~komugi.search.state.SearchState`; the
engine-backed implementation lives in ``komugi.selfplay.tactical``.

**The contracts** (restated on :class:`TacticalProvider`'s methods, and
enforced by ``tests/selfplay/unit/test_tactical_provider.py``):

*Candidates*
  ``candidate_actions`` returns action ids that are **ascending**, **unique**,
  plain ``int``\\ s, and drawn **only** from the supplied legal mask -- a
  provider never widens the mask, so an action the caller has masked off can
  never come back through this seam. The set is **complete** for the class of
  moves the provider declares (for the shipped one: ordinary A0 marshal-removing
  captures); it is allowed to be **conservative**, i.e. to return a superset
  containing false positives, and a candidate is therefore **never a proof of a
  win** -- it is a move the search is obliged to look at, nothing more.

*Purity*
  Enumeration and validation **never clone, play, or evaluate**, and never
  mutate either the state or the mask they are handed. A provider is a read of
  the position, so consulting one is free of search-visible side effects and
  costs no engine transitions.

*Terminal identification*
  ``terminal_winner`` requires an **actually terminal** state and a
  **categorical one-hot** outcome; it answers with the winning player's index or
  ``None`` for a real draw, and it never repairs a malformed outcome into an
  answer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from komugi.search.state import SearchState


@runtime_checkable
class TacticalProvider(Protocol):
    """A read-only, rules-level source of must-look-at actions. **Frozen.**

    Frozen means the three methods and the identity attribute below are the
    entire seam: a richer provider (one that scored its candidates, say, or one
    that could prove a win) would let the rules layer reach into the search's
    value accounting, which is precisely what this decomposition keeps apart.
    """

    #: Non-empty, stable, declarative identity of this provider -- what goes in
    #: a config, a run manifest and a row of recorded search diagnostics. Stable
    #: means a change of behaviour is a change of ``provider_id`` (hence the
    #: ``-v1`` suffix convention), so a recorded run names the generator that
    #: produced it rather than "whatever that name meant at the time".
    provider_id: str

    def validate_state(self, state: SearchState) -> None:
        """Raise ``TypeError`` if *state* is not a kind this provider supports.

        Called by the search once, at wiring time, so an incompatible pairing
        of provider and state fails where it is configured rather than at some
        interior node mid-search. Every other method of this protocol validates
        again, first thing, before touching its argument.

        Never clones, plays, evaluates or mutates.
        """
        ...

    def candidate_actions(
        self, state: SearchState, legal_mask: npt.NDArray[np.bool_]
    ) -> tuple[int, ...]:
        """Return the must-look-at action ids of *state*, ascending and unique.

        Args:
            state: The position to enumerate. Not mutated.
            legal_mask: The caller's ``(A,)`` boolean legality mask for *state*.
                Not mutated, and never widened: every returned id ``i``
                satisfies ``legal_mask[i]``.

        Returns:
            A strictly ascending tuple of distinct ``int`` ids, complete for the
            class of moves this provider declares and possibly conservative
            (false positives are permitted, so a candidate never proves a win).
            Empty when the position offers none.

        Raises:
            TypeError: *state* is not supported (same rule as
                :meth:`validate_state`, checked first).
            ValueError: *legal_mask* is not the expected shape or dtype.
        """
        ...

    def terminal_winner(self, terminal_state: SearchState) -> int | None:
        """Return the winner of an **already terminal** *terminal_state*.

        Args:
            terminal_state: A position whose ``is_terminal()`` is ``True``.

        Returns:
            ``0`` (white) or ``1`` (black) for a win, and ``None`` for a **real
            draw** -- a draw the rules adjudicated, not a draw score and not
            "no result yet".

        Raises:
            TypeError: *terminal_state* is not supported.
            ValueError: *terminal_state* is not terminal, or its outcome is not
                a categorical one-hot. A malformed outcome is an error, never
                something to repair by taking an ``argmax``.
        """
        ...
