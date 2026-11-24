"""The subtree-value-bias table of design §A2.

**What it corrects.** A move played in similar local circumstances tends to
have a *systematically* mis-estimated network value -- the leaf net is wrong in
the same direction each time. This table estimates that systematic component
per "kind of move" and adds it to direct neural mass, avoiding the same
error at every occurrence.

Two properties are the whole of the implementation, and both are corrections
the design makes to the obvious version:

* **Storage is in WDL space, never in utility space.** A bias stored as a
  utility error would be contempt-dependent and would poison a reused tree at
  exactly the moment contempt changed (design §S2). Each entry holds a ``(3,)``
  error vector summing to zero and a weight, and the correction is applied by
  the reader.

  WDL space is *mover-relative*, so the frame is part of the contract and the
  caller owns it: **every contribution to a key is stored in the perspective of
  the mover that key names**, which is the mover at the parent of the edge
  (design §A2). That is the frame the reader's own statistics are in, and it is
  the same for every contributor to one entry, so the average this module takes
  is an average of comparable quantities. A caller that stored each node's own
  perspective instead would pool two opposite frames under one key and hand
  back a correction with its win and loss slots transposed.
* **A node's contribution is removed in the frame the entry has decayed to.**
  An entry decays by ``rho`` on *every* update, including updates made by
  other nodes sharing the key -- and this key is deliberately coarser than
  KataGo's, so other nodes sharing it is the normal case. Removing a node's
  contribution at its **undecayed** magnitude takes out mass the entry does
  not contain, and ``weight_sum`` then walks down through zero and goes
  **negative**, at which point ``b(k)`` either blows up or flips sign. So each
  node records the generation its contribution was written at, and removal
  scales by ``rho ** (entry.gen - node.bias_gen)``.

The exactness identity that follows, stated over every node that has
contributed and not since been replaced -- **live or freed**, since nothing
here removes a contribution when a node is dropped by a rehome::

    entry.weight_sum == sum_i rho ** (entry.gen - bias_gen_i) * s_i   >= 0

Under tree reuse the left side is an **upper bound** on the sum over live
nodes alone, with the gap exactly the decayed mass of freed contributors; the
``>= 0`` half is unaffected, because every dropped term is non-negative.

The table is an attribute of one :class:`~komugi.search.match.tree.MatchTree`
and never a module-level object (design P-D3): that is what makes two
concurrent match searches independent, and two players in one process is the
normal case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import numpy.typing as npt

#: Length of a WDL triple. Spelled out here rather than imported, because
#: design §M3 gives this module numpy and nothing else.
_WDL: Final[int] = 3

#: Below this, a negative ``weight_sum`` cannot be float64 round-off and the
#: generation bookkeeping is wrong, so it raises instead of clamping.
_NEGATIVE_TOLERANCE: Final[float] = -1e-9


class BiasTableInvariant(Exception):
    """A bias entry's ``weight_sum`` went meaningfully negative.

    Raised rather than clamped, because a negative weight means the
    generation bookkeeping is wrong and every later ``b(k)`` on that key is
    arbitrary -- clamping would convert a bookkeeping bug into a silently
    wrong correction, which is strictly harder to diagnose.

    Args:
        key: The bias key whose entry broke.
        entry_gen: The entry's generation counter at the moment of the break.
        node_gen: The generation the removed contribution claimed.
        weight_sum: The offending value.
    """

    def __init__(
        self, key: int, entry_gen: int, node_gen: int, weight_sum: float
    ) -> None:
        super().__init__(
            f"subtree-value-bias entry {key} reached weight_sum={weight_sum!r} "
            f"after removing a contribution written at generation {node_gen} "
            f"from an entry now at generation {entry_gen}; a negative weight "
            f"means the generation-scaled removal of design §A2 is wrong, and "
            f"every later b(k) on this key is arbitrary"
        )
        self.key = key
        self.entry_gen = entry_gen
        self.node_gen = node_gen
        self.weight_sum = weight_sum


@dataclass
class BiasEntry:
    """One key's accumulated WDL error, its weight, and its decay counter.

    Attributes:
        err_sum: ``(3,)`` float64 sum of weighted WDL errors. Each summand is
            a difference of two probability vectors, so it sums to zero and
            so does the total.
        weight_sum: The summed weight behind `err_sum`.
        gen: How many times this entry has been decayed. It is what makes a
            removal exact, and it is contempt-free like everything else here.
    """

    err_sum: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(_WDL, dtype=np.float64)
    )
    weight_sum: float = 0.0
    gen: int = 0


class SubtreeValueBias:
    """Design §A2's table: ``key -> (err_sum, weight_sum, gen)``.

    Args:
        free_prop: ``rho``, the per-update decay applied to an entry. Also
            applied once per rehome to *every* entry, through
            :meth:`decay_all`.
        weight_floor: ``epsilon_w``. An entry whose weight is below it reads
            as **empty**, not as a large correction.
    """

    def __init__(self, *, free_prop: float, weight_floor: float) -> None:
        self._free_prop = float(free_prop)
        self._weight_floor = float(weight_floor)
        self._entries: dict[int, BiasEntry] = {}

    @property
    def free_prop(self) -> float:
        """Return ``rho``, the per-update decay."""
        return self._free_prop

    @property
    def weight_floor(self) -> float:
        """Return ``epsilon_w``, below which an entry reads as empty."""
        return self._weight_floor

    def keys(self) -> tuple[int, ...]:
        """Return every key with an entry, ascending.

        Sorted, and never used by the search itself: design §D1 requires that
        no iteration over an unordered container can influence a result, so
        the table is read **by key only** during a search and this exists for
        tests and diagnostics.
        """
        return tuple(sorted(self._entries))

    def entry(self, key: int) -> BiasEntry | None:
        """Return the entry stored under `key`, or ``None``."""
        return self._entries.get(key)

    def generation(self, key: int) -> int:
        """Return `key`'s generation, creating the entry empty if need be.

        Called at expansion, so that a node's ``bias_gen`` is always the
        generation of a *real* entry: a node initialised against a generation
        that does not exist yet would have its first removal scaled by the
        wrong power of ``rho``.
        """
        stored = self._entries.get(key)
        if stored is None:
            stored = BiasEntry()
            self._entries[key] = stored
        return stored.gen

    def read(self, key: int) -> npt.NDArray[np.float64]:
        """Return ``b(k)``: the mean WDL error under `key`, or zeros.

        An entry below :attr:`weight_floor` reads as ``(0, 0, 0)`` rather than
        as ``err_sum`` divided by nearly nothing. **This is the only place
        ``epsilon_w`` is applied**: storage keeps a legitimately small positive
        ``weight_sum`` verbatim, so an entry whose mass decays below the floor
        and is then contributed to again resumes from where it actually was
        rather than from zero. The result sums to zero,
        because it is an average of differences of probability vectors --
        which is what guarantees the correction cannot change the total
        probability mass a node claims.
        """
        stored = self._entries.get(key)
        if stored is None or stored.weight_sum < self._weight_floor:
            return np.zeros(_WDL, dtype=np.float64)
        return np.asarray(stored.err_sum / stored.weight_sum, dtype=np.float64)

    def update(
        self,
        key: int,
        *,
        contribution: npt.NDArray[np.float64],
        weight: float,
        gen: int,
        delta: npt.NDArray[np.float64],
        new_weight: float,
    ) -> tuple[npt.NDArray[np.float64], float, int]:
        """Replace one node's contribution to `key` and return the new one.

        An edge contributes **once**, at its current best estimate, so the
        update removes what the node put in before -- scaled to what that mass
        has since decayed to -- rather than accumulating a second time.

        Args:
            key: The node's bias key.
            contribution: The node's current ``bias_contribution``, ``(3,)``.
            weight: The node's current ``bias_weight``.
            gen: The node's current ``bias_gen``.
            delta: ``x_sub(n) - x_net(n)``, the node's observed WDL error, in
                the perspective of the mover `key` names -- **not** necessarily
                the node's own; the caller swaps it there. The caller must not
                call this while ``W_tot(n) < 1``, the weighted confidence
                threshold. Only zero total makes the subtree mean ``0/0``;
                positive totals below one remain mathematically meaningful.
            new_weight: ``W_tot(n) ** beta``, the weight of the new
                contribution.

        Returns:
            The node's new ``(bias_contribution, bias_weight, bias_gen)``, to
            be written back onto the node by the caller. Returning them
            instead of mutating the node is what keeps this module free of
            any dependency on the node schema.

        Raises:
            ValueError: If `delta` or `new_weight` is not finite. This is the
                first-backup case made unmissable: on the backup that first
                passes *into* a node ``W_tot(n)`` is zero, so the subtree mean
                is ``0/0`` and ``delta`` is ``nan`` -- and a ``nan`` written
                into ``err_sum`` is permanent and poisons every later read of
                the key. The caller's rule is to skip the update entirely
                while ``W_tot(n) < 1``; this refuses rather than trusting it.
            BiasTableInvariant: If the removal drives ``weight_sum`` below
                ``-1e-9``, which is not round-off.
        """
        observed = np.asarray(delta, dtype=np.float64)
        if not bool(np.isfinite(observed).all()) or not np.isfinite(new_weight):
            raise ValueError(
                f"a non-finite subtree-value-bias update for key {key}: "
                f"delta={observed!r}, new_weight={new_weight!r}. The update is "
                f"skipped entirely while the node has no child weight, where "
                f"the subtree mean is 0/0 (design §A2/B3b)"
            )
        stored = self._entries.get(key)
        if stored is None:
            stored = BiasEntry()
            self._entries[key] = stored

        scale = self._free_prop ** (stored.gen - gen)
        stored.err_sum = stored.err_sum - scale * np.asarray(
            contribution, dtype=np.float64
        )
        stored.weight_sum -= scale * weight
        stored.err_sum = stored.err_sum * self._free_prop
        stored.weight_sum *= self._free_prop
        stored.gen += 1

        if stored.weight_sum < _NEGATIVE_TOLERANCE:
            raise BiasTableInvariant(key, stored.gen, gen, stored.weight_sum)
        if stored.weight_sum < 0.0:
            # Only the round-off band ``[-1e-9, 0)`` is flattened, and only so
            # that the module's ``weight_sum >= 0`` guarantee survives float64
            # cancellation. A *positive* weight below ``epsilon_w`` is left
            # exactly where it is: it is real decayed mass, and clamping it in
            # **storage** destroys the exactness identity for every node still
            # holding a contribution to this key. ``epsilon_w`` is a statement
            # about what may be *read* -- an entry with nearly no weight behind
            # it reads as empty rather than as a huge quotient -- and
            # :meth:`read` is where that statement belongs and is applied.
            stored.weight_sum = 0.0

        new_contribution = np.asarray(new_weight * observed, dtype=np.float64)
        stored.err_sum = stored.err_sum + new_contribution
        stored.weight_sum += float(new_weight)
        return new_contribution, float(new_weight), stored.gen

    def decay_all(self) -> None:
        """Age every entry by one decay step -- design §A4's rehome decay.

        The bias table is the one structure whose entries genuinely go stale
        on rehoming: its keys are ``(mover, move, previous move)`` triples
        whose local circumstances have moved on, and unlike the WDL sums an
        entry is a *systematic-error* estimate rather than an average of leaf
        values.

        It must increment the same counter the per-update decay does, for
        every entry including entries with no live node. Otherwise a retained
        node's ``bias_gen`` under-counts the decays its contribution has
        actually suffered, and the next removal takes out mass the entry no
        longer contains -- the un-scaled removal bug, re-entering through the
        reuse path.
        """
        for stored in self._entries.values():
            stored.err_sum = stored.err_sum * self._free_prop
            stored.weight_sum *= self._free_prop
            stored.gen += 1

    def clear(self) -> None:
        """Drop every entry. Called when the tree itself is cleared."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
