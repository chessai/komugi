"""T7 -- π and ρ, the bridge that keeps the v1 goldens alive.

The design's "How old goldens survive" defines two functions in a test-only
``komugi.engine.compat``:

* **π**, a pure total ``v2_id -> v1_id`` projection that forgets the selector.
  Mask goldens survive as ``{π(x) for x in mask_v2(P)} == mask_v1_golden(P)``:
  ``⊆`` because every v2 bit comes from a branch that also fired in v1, ``⊇``
  because wherever v1's A1/B/C bit fired, v2 emits at least the ``sel = 0`` /
  ``bsel = 0`` id.
* **ρ**, the state-dependent ``(position, v1_id) -> v2_id`` lift: identity on
  A0, ``sel = Q`` on A1, ``bsel = 1`` iff ``Q != 0`` on B/C.  That is exactly
  v1's ``max(betrays, key=len(captured))`` rule, because the full qualifying
  set is the unique longest combination.

**Scope note (T7 (b)).** The frozen v1 DECODE goldens live under
``tests/engine/goldens`` and belong to that suite; KOM-128 owns
``tests/encoding`` only and may not import their fixtures.  The v1 MASK side
of T7 (a) is therefore reconstructed here by collapsing the reference dumps --
which is the same set by
``test_reference_dumps_sanity.test_our_engine_matches_the_reference_when_v1_collapsed``,
and has the advantage of being reference truth rather than a recording of the
engine under test.  T7 (b) proper stays with whoever owns those goldens.

Gated on KOM-028 -- and on ``ACTION_SIZE`` ALONE.  The compat module is
deliberately not part of the gate: ``helpers.compat_module`` soft-imports it
only while the extension has not landed, and hard-fails once it has, so a
KOM-028 that ships without ``compat.py`` turns this module red instead of
skipping it into permanent silence.
"""

from __future__ import annotations

from typing import Any

import pytest

ae: Any = pytest.importorskip("komugi.engine.action_encoding")

if ae.ACTION_SIZE != 37341:
    pytest.skip("KOM-028 extension not yet implemented", allow_module_level=True)

from komugi.engine.action_encoding import decode_cython_action  # noqa: E402

from helpers import (  # noqa: E402
    V1_ACTION_SIZE,
    V1_A1_START,
    V2_A1_START,
    V2_ACTION_SIZE,
    board_of_fen,
    compat_module,
    compat_pi,
    compat_rho,
    corpus_fen,
    corpus_names,
    dumped_moves,
    is_draft_fen,
    legal_mask,
    masked_ids,
    normalize_dump_move,
    project_v1,
    qualifying_mask,
    sq_index,
    v1_id,
    v2_block,
)

CORPUS: tuple[str, ...] = corpus_names()


def _v1_reachable(name: str) -> set[int]:
    """The v1 mask for fixture *name*, rebuilt by collapsing the dump."""
    keys = [normalize_dump_move(move) for move in dumped_moves(name)]
    if is_draft_fen(corpus_fen(name)):
        keys = [key for key in keys if key.kind == "arata"]
    return {v1_id(key) for key in keys}


# ---------------------------------------------------------------------------
# π -- the projection itself
# ---------------------------------------------------------------------------


def test_pi_agrees_with_the_designs_arithmetic_on_every_id() -> None:
    """The shipped π equals this suite's transcription of the design's table."""
    pi = compat_pi(compat_module())
    for action_id in range(V2_ACTION_SIZE):
        assert pi(action_id) == project_v1(action_id), action_id


def test_pi_lands_inside_the_v1_space_and_forgets_only_the_selector() -> None:
    """π is total onto ``[0, 15390)``, and the four selectors share an image."""
    pi = compat_pi(compat_module())
    for action_id in range(V2_ACTION_SIZE):
        assert 0 <= pi(action_id) < V1_ACTION_SIZE

    for from_sq in (0, 40, 80):
        for to_sq in (0, 32, 80):
            images = {
                pi(V2_A1_START + (from_sq * 4 + sel) * 81 + to_sq)
                for sel in range(4)
            }
            assert images == {V1_A1_START + from_sq * 81 + to_sq}


# ---------------------------------------------------------------------------
# T7 (a) -- mask goldens survive as an equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS)
def test_t7a_the_projected_v2_mask_equals_the_v1_mask(name: str) -> None:
    """``{π(x) for x in mask_v2(P)} == mask_v1(P)`` -- no golden needs re-recording."""
    pi = compat_pi(compat_module())
    board = board_of_fen(corpus_fen(name))

    projected = {int(pi(action_id)) for action_id in masked_ids(legal_mask(board))}
    assert projected == _v1_reachable(name), (
        f"{name}: projected-only {sorted(projected - _v1_reachable(name))}, "
        f"v1-only {sorted(_v1_reachable(name) - projected)}"
    )


@pytest.mark.parametrize("name", CORPUS)
def test_t7a_the_selector_zero_id_is_always_present(name: str) -> None:
    """The ``⊇`` half, stated directly: v1's bit always has a v2 witness.

    Wherever the mask sets any A1 id at a square pair, it also sets the
    ``sel = 0`` id there -- the plain tsuke is pushed unconditionally ahead of
    the combos (``move_gen.ts:154``).  Same for ``bsel = 0`` in B/C.
    """
    board = board_of_fen(corpus_fen(name))
    ids = masked_ids(legal_mask(board))

    for action_id in ids:
        block = v2_block(action_id)
        if block == "A1":
            plane, to_sq = divmod(action_id - V2_A1_START, 81)
            base = V2_A1_START + (plane // 4 * 4) * 81 + to_sq
        elif block in ("B", "C"):
            start = ae.B_START if block == "B" else ae.C_START
            plane, to_sq = divmod(action_id - start, 81)
            base = start + (plane // 2 * 2) * 81 + to_sq
        else:
            continue
        assert base in ids, f"{name}: {action_id} set without its selector-0 id"


# ---------------------------------------------------------------------------
# T7 (c) -- ρ is a section of π
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS)
def test_t7c_pi_of_rho_is_the_identity_on_every_v1_reachable_id(name: str) -> None:
    """``π(ρ(P, id)) == id`` -- the lift never leaves its own fibre."""
    module = compat_module()
    pi = compat_pi(module)
    rho = compat_rho(module)
    board = board_of_fen(corpus_fen(name))

    for action_id in sorted(_v1_reachable(name)):
        lifted = rho(board, action_id)
        assert 0 <= lifted < V2_ACTION_SIZE
        assert pi(lifted) == action_id, (
            f"{name}: ρ({action_id}) = {lifted}, which projects to {pi(lifted)}"
        )


@pytest.mark.parametrize("name", CORPUS)
def test_t7c_every_lifted_id_is_legal_in_the_position_it_was_lifted_at(
    name: str,
) -> None:
    """ρ names a move that exists: the lifted id is set in the v2 mask.

    A lift that produced an unmasked id would make every re-asserted decode
    golden raise instead of comparing.
    """
    rho = compat_rho(compat_module())
    board = board_of_fen(corpus_fen(name))
    legal = masked_ids(legal_mask(board))

    for action_id in sorted(_v1_reachable(name)):
        assert int(rho(board, action_id)) in legal


# ---------------------------------------------------------------------------
# T7 (d) -- ρ reproduces the v1 canonicalization
# ---------------------------------------------------------------------------


def test_t7d_rho_picks_the_maximal_conversion_on_the_three_combo_fixture() -> None:
    """``ρ`` on ``mixed_pair_hand_both`` selects ``sel = Q = 0b11``.

    v1's decoder returned ``max(betrays, key=len(captured))``
    (``action_encoding.py:181``, ``:307``); the full qualifying set is the
    unique longest combination, so the two rules coincide.  Three combos exist
    here, which is what makes the choice observable at all.
    """
    rho = compat_rho(compat_module())
    name = "mixed_pair_hand_both"
    board = board_of_fen(corpus_fen(name))

    from_sq, to_sq = sq_index(5, 5), sq_index(4, 4)
    assert qualifying_mask(board, 4, 4, 0) == 0b11

    v1_stack = V1_A1_START + from_sq * 81 + to_sq
    assert v1_stack == 9833
    assert v1_stack in _v1_reachable(name)

    lifted = int(rho(board, v1_stack))
    assert lifted == V2_A1_START + (from_sq * 4 + 0b11) * 81 + to_sq

    decoded = decode_cython_action(lifted, board)
    assert sorted(tier for _, tier in decoded["captured"]) == [1, 2]
    assert decoded["san"] == "謀(5-5-2)(4-4-3)返兵侍"


def test_t7d_rho_degrades_to_the_plain_tsuke_when_nothing_qualifies() -> None:
    """``Q == 0`` lifts to ``sel = 0``, matching v1's ``candidates[0]`` fallback."""
    rho = compat_rho(compat_module())
    name = "same_type_pair_hand_one"
    board = board_of_fen(corpus_fen(name))

    from_sq, to_sq = sq_index(5, 5), sq_index(4, 4)
    assert qualifying_mask(board, 4, 4, 0) == 0b00

    lifted = int(rho(board, V1_A1_START + from_sq * 81 + to_sq))
    assert lifted == V2_A1_START + (from_sq * 4) * 81 + to_sq
    assert decode_cython_action(lifted, board)["captured"] == []


def test_t7d_rho_sets_the_arata_bit_exactly_where_a_conversion_exists() -> None:
    """``bsel = 1`` iff ``Q != 0`` at the drop square (arata_conversion)."""
    rho = compat_rho(compat_module())
    board = board_of_fen(corpus_fen("arata_conversion"))

    converting = [
        normalize_dump_move(move)
        for move in dumped_moves("arata_conversion")
        if move["kind"] == "arata" and move["capturedTiers"]
    ]
    assert len(converting) == 1
    key = converting[0]

    lifted = int(rho(board, v1_id(key)))
    decoded = decode_cython_action(lifted, board)
    assert [tier for _, tier in decoded["captured"]] == [1]
    assert decoded["san"] == "新謀(7-5-3)返兵"


def test_t7d_rho_is_the_identity_on_block_a0() -> None:
    """Route and capture ids carry no selector, so the lift cannot move them."""
    rho = compat_rho(compat_module())
    board = board_of_fen(corpus_fen("mixed_pair_hand_both"))

    for action_id in sorted(_v1_reachable("mixed_pair_hand_both")):
        if action_id < V1_A1_START:
            assert int(rho(board, action_id)) == action_id
