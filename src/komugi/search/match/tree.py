"""The match tree: playouts, the WDL backup, and rehoming across moves.

Three things live here, and each is somewhere a duplicated copy of a familiar
algorithm could drift into a defect no shape assertion catches.

**The playout loop.** One playout descends by PUCT until it reaches either a
terminal position or an edge whose child does not exist yet, expands that one
child, and backs the leaf's value up the path it came down. One playout
therefore asks the network **at most once**, which is what makes design §B1's
evaluation identity an identity rather than an estimate.

**The backup is a WDL swap, not a negation.** Walking up, at each edge where
the mover changes, ``(w, d, l)`` becomes ``(l, d, w)``. Check the arithmetic:
with ``u(x) = w - l + c*d``, ``u(swap(x)) = l - w + c*d = -(w - l) + c*d``.
The win/loss component negates, exactly as a value should across an edge; the
draw component does **not**, so a draw is worth ``+c`` to *whoever is to move*,
at every depth, with no oscillation. Scalar negamax cannot express that,
because it has already added ``c*d`` into a number it can only negate
wholesale -- which is the documented P0a defect this backup is the
pre-registered fix for.

**Rehoming is a subtree move, not a graph operation.** ``descend`` promotes the
played child to root and drops every sibling; nodes carry no parent pointer, so
the siblings become unreachable and collectable rather than being kept alive
through a back-edge for the rest of the game. Everything on the retained
subtree survives verbatim -- ``N``, ``W``, ``S``, ``M``, the priors, the
expanded children -- and nothing is rescaled, because the statistics are
negamax averages of leaf evaluations, which are path-independent, and the
evaluator is the same object for the whole match.

Design §A2's subtree-value-bias table is constructed here, per tree rather than
per module (design P-D3), and this revision **wires its three call sites**:

* at expansion, a node's ``bias_gen`` is the generation of a *real* entry, so
  its first removal is scaled by the right power of ``rho``;
* on the way up the backup, the edge into each node on the path replaces that
  node's contribution -- but only once the node has child weight, since the
  subtree mean is ``0/0`` before that and a ``nan`` in the table is permanent,
  and **in the frame of the mover the key names**, i.e. the parent's: the
  reader adds the direct-neural fraction of the entry to statistics in the parent's
  perspective, one negamax swap away from the node's own (§A6);
* on ``descend``, every entry ages by one decay step (design §A4).

The read side is :func:`~komugi.search.match.select.child_bias_correction`,
which the descent passes this table to.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

from komugi.search.match.bias import SubtreeValueBias
from komugi.search.match.config import MatchConfig
from komugi.search.match.node import (
    NO_MOVER,
    NO_PREVIOUS_ACTION,
    WDL,
    MatchNode,
    bias_key_of,
    legal_hash_of,
    pack_outer,
)
from komugi.search.match.select import select_child
from komugi.search.match.state import MatchState, WdlEval, WdlEvaluator


#: How far an evaluator or terminal state's ``(w, d, l)`` may sit from summing
#: to 1 before the tree refuses it (KOM-179/KOM-218). Loose enough that a
#: float32 head's softmax, re-summed in float64, never trips it; tight enough
#: that an unnormalised answer cannot pass as a distribution.
WDL_SIMPLEX_TOL: Final[float] = 1e-6


def _validate_wdl_distribution(
    value: npt.NDArray[np.float64], *, source: str
) -> None:
    """Refuse `value` unless it is a finite nonnegative WDL simplex.

    The caller owns conversion and, where needed, copying. In particular the
    terminal caller validates the same owned array it installs in the node.
    The mass test is deliberately absolute: this is the shared evaluator and
    terminal ingress contract, not ``np.isclose`` with a relative tolerance.
    """
    if value.shape != (WDL,):
        raise ValueError(f"{source} WDL must have shape ({WDL},), got {value.shape!r}")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{source} returned a non-finite wdl: {value.tolist()}")
    if bool((value < 0.0).any()):
        raise ValueError(
            f"{source} returned a wdl with a negative entry: {value.tolist()}"
        )
    total = float(value.sum())
    if abs(total - 1.0) > WDL_SIMPLEX_TOL:
        raise ValueError(
            f"{source} returned a wdl that is not a distribution: "
            f"{value.tolist()} sums to "
            f"{total}, which is further than {WDL_SIMPLEX_TOL} from 1"
        )


class MatchTreeDesync(Exception):
    """A rehomed root does not describe the position the search was handed.

    Raised rather than silently clearing the tree. A rehoming bug -- an
    off-by-one on the action id, a stale clone, an ``observe`` the player
    missed -- otherwise manifests as a search over a tree describing a
    *different* position, with correct-looking numbers and no other symptom;
    and silently clearing would turn that correctness bug into a performance
    mystery.

    Args:
        ply: The ply the desync was detected at, or ``-1`` if the caller does
            not track one.
        stored_hash: The retained root's stored ``legal_hash``.
        actual_hash: The hash of the legal set the state actually reports.
    """

    def __init__(self, ply: int, stored_hash: int, actual_hash: int) -> None:
        super().__init__(
            f"the retained root's legal set does not match the position at ply "
            f"{ply}: stored legal_hash {stored_hash}, actual {actual_hash}. The "
            f"tree describes a different position than the one being searched"
        )
        self.ply = ply
        self.stored_hash = stored_hash
        self.actual_hash = actual_hash


def swap(wdl: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ``(w, d, l)`` seen from the other side: ``(l, d, w)``.

    The negamax step of design §A6, in WDL space. It is a *permutation*, not a
    negation, and that is the entire content of the fix: the draw mass stays
    where it is, so a constant draw utility no longer alternates in sign on
    the way up the tree.

    The result is a **reversed view** of `wdl`, not a copy: ``np.asarray`` of a
    float64 slice is a no-op. Every caller here only reads it, but a caller
    that wrote into it would be writing into whatever the argument was -- which
    on the backup path is a node's ``wdl_net``.
    """
    return np.asarray(wdl[::-1], dtype=np.float64)


class MatchTree:
    """One retained search tree, its evaluator, and its bias table.

    Args:
        config: The search's configuration. Rebindable through
            :attr:`config`, because contempt is a property of the reader:
            installing a new config against the same tree is exactly what a
            mid-game contempt change is, and every number the search reads is
            re-derived from raw WDL under the new one.
        evaluator: The WDL-carrying leaf evaluator. Held for the life of the
            tree, which is what makes a retained statistic "produced by a
            shallower search" rather than "produced by a different network".
    """

    def __init__(self, config: MatchConfig, evaluator: WdlEvaluator) -> None:
        self._config = config
        self._evaluator = evaluator
        self._root: MatchNode | None = None
        self._root_action = NO_PREVIOUS_ACTION
        # The mover at the **parent** of the edge that reached the current
        # root, or `NO_MOVER` when no such edge is known. Design §A2 keys an
        # edge on the mover who played it, which at the root is not the root's
        # own mover: see `_root_bias_key`.
        self._root_parent_to_play = NO_MOVER
        self._evaluations = 0
        # Per tree, never per module (design P-D3): a module-level table would
        # silently couple the two players an arena wires into one process.
        self._bias = SubtreeValueBias(
            free_prop=config.bias_free_prop,
            weight_floor=config.bias_weight_floor,
        )

    @property
    def config(self) -> MatchConfig:
        """Return the configuration every read path is currently using."""
        return self._config

    @config.setter
    def config(self, config: MatchConfig) -> None:
        """Install a new configuration against this same tree.

        Nothing stored is touched, and nothing is recomputed: no node holds a
        number that depends on a config field, so a new contempt takes effect
        at the next read and the tree stays valid (design §S2).
        """
        self._config = config

    @property
    def evaluator(self) -> WdlEvaluator:
        """Return the leaf evaluator this tree was built with."""
        return self._evaluator

    @property
    def bias(self) -> SubtreeValueBias:
        """Return this tree's subtree-value-bias table (design §A2).

        Read by the descent through
        :func:`~komugi.search.match.select.select_child`, written by
        :meth:`_backup`, and aged by :meth:`descend`.
        """
        return self._bias

    @property
    def root(self) -> MatchNode | None:
        """Return the current root, or ``None`` if the tree is empty."""
        return self._root

    @property
    def root_action(self) -> int:
        """Return the action played into the root, or ``-1`` at a fresh root.

        Design §A2's key includes the move that preceded the one being made,
        so the tree has to remember one action across a rehome to keep the
        keys of the retained subtree's *children* the same as they were.
        """
        return self._root_action

    @property
    def evaluations(self) -> int:
        """Return how many positions this tree has asked the network about.

        Cumulative over the tree's life; a caller takes a difference to get a
        per-move count, which is what design §B1's identity is stated over.
        """
        return self._evaluations

    @property
    def node_count(self) -> int:
        """Return how many nodes are live in this tree.

        Walked iteratively rather than recursively: a reused tree's principal
        variation is as deep as the game is long, and a 200-ply recursion is a
        stack overflow waiting for the one game that goes the distance.
        """
        if self._root is None:
            return 0
        seen = 0
        stack: list[MatchNode] = [self._root]
        while stack:
            node = stack.pop()
            seen += 1
            stack.extend(node.children.values())
        return seen

    def clear(self) -> None:
        """Drop the tree. The next search starts empty and pays its own root.

        The bias table is deliberately **not** cleared: an entry is a
        systematic-error estimate keyed on ``(mover, move, previous move)``,
        not a property of any particular node, and design §A2's exactness
        identity is stated over every node that has contributed *and not been
        replaced* -- live or freed -- precisely so that dropping nodes does
        not have to touch the table.
        """
        self._root = None
        self._root_action = NO_PREVIOUS_ACTION
        self._root_parent_to_play = NO_MOVER

    def descend(self, action: int) -> None:
        """Rehome the tree onto the child reached by `action`.

        Called once per **played move, by either side**: the match player
        learns of both through ``Mover.observe``. It is deliberately not
        called per ply parity -- during the draft the same player can move
        twice in a row, and a rehoming rule keyed on parity would rehome to
        the wrong node in exactly those positions.

        If the played action has no expanded child -- it was never selected,
        or the tree is already empty -- the tree is cleared and the next
        search pays for its own root, though the *edge* just played is
        remembered so that root still gets design §A2's real key. Otherwise
        that child becomes the root,
        every sibling subtree is released, and the new root's ``is_root`` flag
        switches FPU from the interior constant to the root one. Nothing else
        about the retained node changes.

        **Every bias entry ages by one decay step here**, whichever branch is
        taken (design §A4). It is one factor of ``rho`` on the mass *and* one
        increment of the generation counter, for every entry including entries
        with no live node: a decay that moved the mass without the counter
        would leave every retained node's ``bias_gen`` under-counting the
        decays its contribution has actually suffered, and the next removal
        would take out mass the entry no longer contains -- the un-scaled
        removal bug, re-entering through the reuse path.

        The consistency check that guards a mis-rehome cannot run here:
        ``observe`` carries no state to check against. It runs at the head of
        the next search instead (:meth:`check_consistency`).

        Args:
            action: The action id actually played in the game.
        """
        self._bias.decay_all()
        root = self._root
        if root is None:
            self.clear()
            return
        index = root.child_index(action)
        child = None if index is None else root.children.get(index)
        if child is None:
            # The *tree* goes; the **edge** is a fact about the game and does
            # not. Remembering it is what lets the root the next search expands
            # carry design §A2's real key -- mover at the parent, the action
            # played -- instead of the "no edge" key a cleared tree would give
            # it, which would file the same edge under two different keys
            # depending on whether the child happened to be expanded.
            self.clear()
            self._root_action = int(action)
            self._root_parent_to_play = int(root.to_play)
            return
        child.is_root = True
        self._root = child
        self._root_action = int(action)
        self._root_parent_to_play = int(root.to_play)
        # `root` -- and with it the dict holding every sibling subtree -- is
        # now unreferenced. Nodes carry no parent pointer, so nothing keeps
        # the siblings alive: this is a release, not a hint to the collector.

    def check_consistency(self, state: MatchState, *, ply: int = -1) -> None:
        """Assert that the root's stored legal set is the state's legal set.

        An **equality of sets, checked in both directions**, over design §S1
        fields only. ``legal_hash`` is the O(1) filter and the ascending id
        arrays are the exact test; comparing the two arrays catches a stored
        action that is no longer legal *and* a legal action the node does not
        have. The second direction is the one that matters most and is the
        easiest to omit: a position whose legal set has grown passes any
        "every stored action is still legal" test silently, and the search
        then runs over a truncated legal set and can never select the missing
        move.

        Args:
            state: The position the next search will run on.
            ply: Reported in the exception; the tree does not track one.

        Raises:
            MatchTreeDesync: If the two legal sets differ in either direction.
        """
        root = self._root
        if root is None:
            return
        current = np.flatnonzero(state.legal_mask()).astype(np.int32)
        actual = legal_hash_of(current)
        if root.legal_hash != actual or not np.array_equal(root.actions, current):
            raise MatchTreeDesync(ply, root.legal_hash, actual)

    def ensure_root(self, state: MatchState) -> None:
        """Expand the root from `state` if the tree is empty.

        Costs exactly one evaluation on an empty tree and zero on a retained
        one, which is the ``[1 if the root was not already expanded else 0]``
        term of design §B1.
        """
        if self._root is not None:
            return
        node, _ = self._expand(state, bias_key=self._root_bias_key(state))
        node.is_root = True
        self._root = node

    def playout(self, state: MatchState) -> None:
        """Run one playout from `state`: descend, expand at most one node, back up.

        `state` is never mutated -- the descent walks a clone -- and the
        network is asked at most once, when the descent reaches an edge whose
        child does not exist and the position behind it is not terminal.

        Raises:
            ValueError: If the tree has no root; :meth:`ensure_root` runs
                first.
        """
        root = self._root
        if root is None:
            raise ValueError("playout on an empty tree: expand the root first")

        simulation = state.clone()
        path: list[tuple[MatchNode, int]] = []
        previous_action = self._root_action
        node = root
        evaluation: WdlEval | None = None

        while not node.is_terminal:
            index = select_child(node, self._config, bias=self._bias)
            action = int(node.actions[index])
            path.append((node, index))
            simulation.play(action)
            child = node.children.get(index)
            if child is None:
                key = bias_key_of(
                    node.to_play, action, previous_action, self._config.num_actions
                )
                child, evaluation = self._expand(simulation, bias_key=key)
                node.children[index] = child
                node = child
                break
            previous_action = action
            node = child

        if evaluation is None:
            evaluation = _leaf_evaluation(node)
        self._backup(path, node, evaluation)

    def _root_bias_key(self, state: MatchState) -> int:
        """Return the bias key of the edge that reached the current root.

        Design §A2's mover slot is **the mover at the parent of the edge**, not
        the mover at the node the edge reaches -- which is what ``playout``
        passes for an interior edge (``node.to_play``, the parent's). The root
        is the one node whose parent is gone, so the mover is remembered across
        the rehome that dropped it: `_root_parent_to_play` is the old root's
        mover, recorded by :meth:`descend`.

        With no edge at all -- a fresh tree, or one reset at a game boundary --
        there is no parent and nothing to remember, and the key degenerates to
        the "no edge" key ``(to_play, -1, -1)``. The mover slot there is the
        state's own, which is the only mover in evidence; that convention
        matters only in that it is written down, since an edgeless key can
        never collide with an edged one.
        """
        parent_to_play = (
            state.to_play()
            if self._root_parent_to_play == NO_MOVER
            else self._root_parent_to_play
        )
        return bias_key_of(
            parent_to_play,
            self._root_action,
            NO_PREVIOUS_ACTION,
            self._config.num_actions,
        )

    def _expand(
        self, state: MatchState, *, bias_key: int
    ) -> tuple[MatchNode, WdlEval | None]:
        """Create the node for `state`, asking the network unless it is terminal.

        Args:
            state: The position to expand. Not mutated.
            bias_key: Design §A2's key for the edge that reaches it.

        Returns:
            The new node, and the evaluation it was built from -- ``None`` at
            a terminal position, which is exactly the case that costs no
            evaluation.

        Raises:
            ValueError: If a terminal WDL or evaluator WDL is not a
                distribution -- wrongly shaped, non-finite, negative anywhere,
                or off unit mass by more than :data:`WDL_SIMPLEX_TOL` -- if a
                non-terminal position has no legal action, or if the evaluator
                returns a wrongly shaped policy vector.
        """
        config = self._config
        if state.is_terminal():
            # Read once, then copy before inspection. The state owns the
            # returned object and may reuse or mutate its buffer; this exact
            # owned copy is both validated and installed in the node.
            result = np.array(state.terminal_wdl(), dtype=np.float64, copy=True)
            _validate_wdl_distribution(result, source="terminal ingress")
            return (
                MatchNode(
                    to_play=state.to_play(),
                    is_terminal=True,
                    actions=np.zeros(0, dtype=np.int32),
                    logits=np.zeros(0, dtype=np.float32),
                    wdl_net=result,
                    bias_key=bias_key,
                    bias_gen=self._bias.generation(bias_key),
                    moment_dtype=config.moment_dtype,
                ),
                None,
            )

        actions = np.flatnonzero(state.legal_mask()).astype(np.int32)
        if int(actions.shape[0]) == 0:
            raise ValueError(
                "a non-terminal position reported no legal action: the search "
                "has nothing to expand and no value to back up"
            )
        evaluation = self._evaluate(state)
        logits = np.asarray(evaluation.policy_logits, dtype=np.float32)
        if logits.shape != (config.num_actions,):
            raise ValueError(
                f"policy_logits must be ({config.num_actions},), got "
                f"{logits.shape!r}"
            )
        value = np.asarray(evaluation.wdl, dtype=np.float64)
        # KOM-179, and the message names the EVALUATOR deliberately. A `nan`
        # leaf is backed up into `wdl_sum`, `moment` and -- once an edge has
        # unit weight -- design §A2's bias table, where it is permanent. The
        # first ValueError it produced came from the table's own guard at the
        # second playout, naming the table, so the seam that actually returned
        # it was two subsystems away from the only thing that complained.
        _validate_wdl_distribution(value, source="evaluator")
        return (
            MatchNode(
                to_play=state.to_play(),
                is_terminal=False,
                actions=actions,
                logits=logits[actions],
                wdl_net=value,
                bias_key=bias_key,
                # Design §A2: a node's generation is that of a *real* entry,
                # so the entry is created empty here if this key has never
                # been seen. A node initialised against a generation that does
                # not exist yet would have its first removal scaled by the
                # wrong power of `rho`.
                bias_gen=self._bias.generation(bias_key),
                moment_dtype=config.moment_dtype,
            ),
            # `value` is the array the node now stores as `wdl_net`, and this
            # evaluation is handed to the leaf weighter: a copy, for the reason
            # `_leaf_evaluation` gives. `logits` is the caller's own full-width
            # array and is *not* a node buffer -- the node holds `logits[actions]`,
            # which numpy's fancy indexing has already copied -- so it is passed
            # through rather than duplicated on every evaluation.
            WdlEval(logits, value.copy()),
        )

    def _evaluate(self, state: MatchState) -> WdlEval:
        """Ask the network about `state`, counting the question."""
        self._evaluations += 1
        return self._evaluator(state)

    def _backup(
        self, path: list[tuple[MatchNode, int]], leaf: MatchNode, evaluation: WdlEval
    ) -> None:
        """Accumulate `leaf`'s WDL up `path`, swapping at every mover change.

        The weight comes from the design §A5 seam rather than being hardcoded
        to ``1.0``, and it is applied to **all three** accumulators, so that
        ``W(a)``, ``S(a)`` and ``M(a)`` stay mutually consistent when the
        uncertainty head eventually makes the weights differ. With the default
        weighter every weight is exactly ``1.0`` and ``W(a) == float(N(a))``.

        Design §A2's table update rides on this walk: every edge the backup
        passes through is an edge *into* ``below``, which is exactly the
        occasion on which ``below`` replaces its contribution to its key. The
        root is not one of those nodes -- the edge that reached it belongs to a
        move already played, not to this playout -- and the leaf is skipped by
        the ``W_tot < 1`` rule, having no child weight yet.

        Args:
            path: The ``(parent, edge index)`` sequence from the root down to
                `leaf`, root first.
            leaf: The node whose value enters the tree.
            evaluation: What the leaf weighter is shown.

        Raises:
            ValueError: If the §A5 seam returns a weight that is not finite and
                strictly positive. Checked before any statistic is touched, so
                a refused playout leaves the tree exactly as it found it.
        """
        weight = float(self._config.leaf_weighter.weight(evaluation))
        # Checked BEFORE a single statistic moves (KOM-179): the three
        # accumulators are mutated in one pass up the path, so a weight refused
        # halfway would leave `N`, `W`, `S` and `M` inconsistent with each other
        # on the edges already visited -- and `W(a) == float(N(a))` is the
        # invariant the whole §A5 seam is checked by. A zero weight is refused
        # rather than ignored: a playout that backs up nothing still costs a
        # visit, which is a silent divergence between the budget spent and the
        # weight the readouts divide by.
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"leaf weighter "
                f"{type(self._config.leaf_weighter).__name__} returned "
                f"{weight}: a leaf weight must be finite and > 0"
            )
        value = np.asarray(leaf.wdl_net, dtype=np.float64)
        below = leaf
        for parent, index in reversed(path):
            if below.to_play != parent.to_play:
                value = swap(value)
            # `below`'s own statistics are already final for this playout: it
            # was the `parent` of the previous iteration, or it is the leaf.
            # `parent` is passed because the contribution is stored in *its*
            # frame -- the mover design §A2's key names.
            self._update_bias(below, parent)
            parent.visits[index] += 1
            parent.weight[index] += weight
            parent.wdl_sum[index] += weight * value
            parent.moment[index] += weight * pack_outer(value)
            below = parent

    def _update_bias(self, node: MatchNode, parent: MatchNode) -> None:
        """Replace `node`'s contribution to its key -- design §A2's update.

        Skipped entirely while ``W_tot(node) < 1``, a weighted confidence
        threshold. Only zero total makes the subtree mean ``0/0``; positive
        totals below one have a mean but deliberately do not contribute yet.
        A terminal node, which has no edges at all, is skipped by
        the same test rather than by a second one.

        **The contribution is stored in `parent`'s frame, not in `node`'s**,
        and that swap is what makes the correction a correction rather than an
        anti-correction. ``delta`` as design §A2 writes it is ``x_sub(n) -
        x_net(n)`` with both terms in ``n``'s own perspective, but the only
        reader of the entry is
        :func:`~komugi.search.match.select.child_bias_correction`, which
        adds its direct-neural fraction to ``S(a)/W(a)`` -- §S1 stores ``S(a)`` in
        the **parent's** perspective, one negamax swap away (design §A6). Left
        unswapped the win and loss slots are transposed at every edge where the
        mover changes, so the collapsed correction enters selection and LCB
        with its sign reversed: an edge whose subtree has come in *below* the
        value the net promised is then rewarded for it rather than discounted,
        which is how the shipped ``lambda`` turned a perfect-value evaluator
        into a losing game.

        The frame is well defined per key rather than per node, which is why
        this is a normalisation and not a patch: design §A2's key carries the
        mover at the parent of the edge, so every contributor to one entry is
        stored in that one mover's perspective, and pooling them is an average
        of comparable quantities. Storing ``n``'s own perspective would pool
        two opposite frames under one key wherever the same side moves twice in
        a row, which the draft rules make routine.

        Every number here comes from contempt-free storage and none of it is
        collapsed, so what is written back onto the node -- itself design §S1
        state -- carries no draw score (acceptance A10b); the swap is a
        permutation, so the ``(3,)`` error still sums to zero.

        Args:
            node: The node the backup has just passed into.
            parent: The node above it on the path -- the mover the key names,
                and therefore the frame the contribution is stored in.
        """
        total = node.total_weight()
        if total < 1.0:
            return
        delta: npt.NDArray[np.float64] = np.asarray(
            node.subtree_wdl_sum() / total - node.wdl_net, dtype=np.float64
        )
        if node.to_play != parent.to_play:
            delta = swap(delta)
        contribution, node_weight, generation = self._bias.update(
            node.bias_key,
            contribution=node.bias_contribution,
            weight=node.bias_weight,
            gen=node.bias_gen,
            delta=delta,
            new_weight=total**self._config.bias_weight_exponent,
        )
        node.bias_contribution = contribution
        node.bias_weight = node_weight
        node.bias_gen = generation


def _leaf_evaluation(node: MatchNode) -> WdlEval:
    """Return a terminal node's value shaped as an evaluation.

    A decided position has no legal actions, so its policy over legal actions
    is the empty vector -- which is exactly what ``node.logits`` holds there --
    and its value is the accepted terminal WDL distribution. Ordinary game
    adapters produce one-hot outcomes, while custom states may provide a valid
    fractional simplex. The leaf weighter is defined over evaluations, so the
    terminal path hands it one instead of hardcoding the weight to ``1.0`` and
    silently exempting terminal leaves from a weighting scheme.

    **Both arrays are copies.** :class:`~komugi.search.match.state.LeafWeighter`
    is a seam for code this package does not own, and the obvious
    implementation of a future one -- normalise the WDL, take a softmax of the
    logits -- is an in-place ``/=`` away from writing through into design §S1
    storage. That write would happen *during* the search, so A10a's buffers
    would never see it move; handing over a copy makes the seam unable to
    corrupt a node rather than asking it not to.
    """
    return WdlEval(node.logits.copy(), node.wdl_net.copy())
