"""KOM-141: FULL is the default EVERYWHERE, and a new site cannot change that.

This module is the successor to rules-audit **Q13**. Q13 asked that no call site
silently keep the aliased cells-only key, and until now it was enforced by
``cyengine.placement_key`` raising: the key was unreachable, so no site could
keep it. KOM-141 makes the key real -- a referee has to be able to apply
gungi.js@3.4.11's rule so the Elo cost of the mismatch can be measured -- and
the guard has to move somewhere it still bites.

It bites here, in two halves:

1. **The defaults themselves.** ``Game()``, ``ArenaConfig`` and
   ``SelfPlayConfig`` each default to :attr:`RepetitionRule.FULL`, Komugi's
   repaired 274-byte key (rules-audit F1-F4 / KOM-027). Nothing flips silently.
2. **A source scan.** Every ``Game(`` call in ``src/komugi`` must either pass an
   explicit ``repetition_rule=`` keyword or be one of the sites listed in
   :data:`DEFAULT_FULL_SITES` below, each with the reason it is correct at FULL.
   A new default-FULL construction added later fails this test until it is
   either threaded or listed -- which is the point: listing it is a decision
   somebody wrote down, and threading it is a decision the caller now makes.

Why the allowlist is a COUNT PER FILE rather than a set of line numbers: line
numbers rot on the first unrelated edit, and a bare set of filenames would let a
second, unexamined ``Game()`` be added to an already-listed file for free.

The listed sites, and why FULL is right for each:

* ``eval/pool_books.py`` -- replays a training book's prefix to check it reaches
  a live position. The prefix was PLAYED under FULL by self-play, so replaying
  it under anything else would be replaying a different game.
* ``selfplay/book.py`` -- the same replay on the self-play side, for the same
  reason: a draft book is a self-play artefact and self-play is FULL.
* ``campaign/loop.py`` -- the campaign's own replay of a recorded game. Those
  games were played by the workers, under FULL.

None of the three is a REFEREE of a new game; all three re-walk moves that a
FULL-refereed game already produced. ``engine/game.py`` is deliberately NOT
listed: ``clone()``, the mirror symmetry and ``_unpickle_game`` must each carry
the rule of the game they are copying, so they pass it explicitly and this test
is what says so.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Final

import pytest

import komugi
from komugi.engine.game import Game
from komugi.engine.repetition import RepetitionRule

#: The installed package root -- ``<repo>/src/komugi`` in a dev checkout.
PACKAGE_ROOT: Final[Path] = Path(komugi.__file__).resolve().parent

#: The keyword a threaded construction site must pass.
RULE_KEYWORD: Final[str] = "repetition_rule"

#: Package-relative path -> how many default-FULL ``Game(`` calls it may hold.
#: Every entry is justified in this module's docstring. Adding one here is a
#: decision; forgetting to is a failure.
DEFAULT_FULL_SITES: Final[dict[str, int]] = {
    "eval/pool_books.py": 1,
    "selfplay/book.py": 1,
    "campaign/loop.py": 1,
}


def _names_the_rule(call: ast.Call) -> bool:
    """True iff `call` passes ``repetition_rule=`` explicitly.

    A ``**kwargs`` splat (``arg is None``) does NOT count: what the mapping
    holds is not visible here, so a site that splatted its way to a rule would
    be indistinguishable from one that never set it.
    """
    return any(keyword.arg == RULE_KEYWORD for keyword in call.keywords)


def _is_game_call(call: ast.Call) -> bool:
    """True iff `call` constructs the engine's ``Game``.

    Both spellings the codebase can produce: ``Game(...)`` after a
    ``from ... import Game`` (an ``ast.Name``) and ``game.Game(...)`` after a
    module import (an ``ast.Attribute``). Names that merely END in ``Game`` --
    ``BridgedGame``, ``R5Game`` -- are different classes and are not matched.
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "Game"
    if isinstance(func, ast.Attribute):
        return func.attr == "Game"
    return False


def default_rule_game_sites() -> list[tuple[str, int]]:
    """Return every ``Game(`` call in the package that names no rule.

    ``(package-relative path, line number)`` pairs, in file order. Parsed with
    :mod:`ast` rather than grepped so a mention inside a docstring, a comment or
    a string literal is not a site -- ``eval/v1runner/reference.py`` discusses
    ``class Game(GameState)`` in prose, and a grep-based guard would have to
    carve it out by hand.
    """
    sites: list[tuple[str, int]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_game_call(node):
                if not _names_the_rule(node):
                    relative = path.relative_to(PACKAGE_ROOT).as_posix()
                    sites.append((relative, node.lineno))
    return sites


# ---------------------------------------------------------------------------
# (i) The defaults
# ---------------------------------------------------------------------------


def test_a_bare_game_is_refereed_under_the_full_rule() -> None:
    """The engine's own default: Komugi's repaired 274-byte key."""
    assert Game().repetition_rule is RepetitionRule.FULL


def test_an_arena_config_defaults_to_the_full_rule() -> None:
    """A match referees under FULL unless a caller asks for site parity."""
    pytest.importorskip("torch")
    from komugi.eval.arena import ArenaConfig

    config = ArenaConfig(games=2, seed=0, draw_score=0.0)
    assert config.repetition_rule is RepetitionRule.FULL


def test_a_self_play_config_defaults_to_the_full_rule() -> None:
    """Training is FULL, and PLACEMENT is not a training rule at all."""
    from komugi.config import SelfPlayConfig

    config = SelfPlayConfig(workers=1, process_batch_size=1)
    assert config.repetition_rule is RepetitionRule.FULL


# ---------------------------------------------------------------------------
# (ii) The source scan -- Q13's successor
# ---------------------------------------------------------------------------


def test_every_game_construction_site_names_its_rule_or_is_listed() -> None:
    """No new default-FULL ``Game(`` may appear unexamined.

    The failure message names the offending file and line, because the fix is a
    choice between two specific edits: thread the caller's rule through, or add
    the site to :data:`DEFAULT_FULL_SITES` with its reason in the module
    docstring.
    """
    found = Counter(path for path, _ in default_rule_game_sites())
    unexpected = [
        (path, line)
        for path, line in default_rule_game_sites()
        if found[path] > DEFAULT_FULL_SITES.get(path, 0)
    ]

    assert dict(found) == DEFAULT_FULL_SITES, (
        f"the default-FULL Game() sites in {PACKAGE_ROOT} are "
        f"{dict(found)}, expected {DEFAULT_FULL_SITES}. Sites involved: "
        f"{unexpected or [(p, l) for p, l in default_rule_game_sites()]}. "
        f"Either pass repetition_rule= at the new site, or list it in "
        f"DEFAULT_FULL_SITES with the reason FULL is correct there "
        f"(KOM-141 / rules-audit Q13)"
    )


def test_the_engine_itself_threads_the_rule_through_every_copy() -> None:
    """``engine/game.py`` is not allowlisted, and must not need to be.

    ``clone()``, the mirror symmetry and ``_unpickle_game`` each rebuild a Game
    from its parts and each must carry the rule of the game it is copying. A
    default there would silently re-referee a placement game under Komugi's key
    at the first clone -- which a search does constantly.
    """
    assert "engine/game.py" not in DEFAULT_FULL_SITES
    assert all(
        path != "engine/game.py" for path, _ in default_rule_game_sites()
    ), "a Game( in engine/game.py stopped passing repetition_rule="


def test_the_scan_can_actually_see_a_default_site() -> None:
    """Non-vacuity: the scanner is not returning nothing by construction.

    A guard that found no sites would pass the allowlist check the day the
    allowlist emptied and every day after, including the day somebody added an
    unthreaded referee. So the two helpers are exercised against source this
    test writes itself, one site of each kind.
    """
    module = ast.parse(
        "Game()\n"
        "Game(repetition_rule=rule)\n"
        "game.Game()\n"
        "BridgedGame()\n"
    )
    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    matched = [call for call in calls if _is_game_call(call)]

    assert len(matched) == 3, "Game(), Game(repetition_rule=...) and game.Game()"
    assert [_names_the_rule(call) for call in matched] == [False, True, False]

    # And the real scan finds exactly the listed sites, which is only a claim
    # if the scan finds something at all.
    assert default_rule_game_sites(), "the scan found no Game( call anywhere"
