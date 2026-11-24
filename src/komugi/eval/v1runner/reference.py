"""Loading v1, and refusing to start when v1 is not there.

§D1 makes the adapter's failure to come up a **startup** failure rather than a
per-move surprise, and the harness's half of that is already written: a child
that never answers ``hello`` becomes a :class:`HandshakeRefused` carrying the
child's stderr verbatim (``v1bridge.py:1048-1055``). This module is the other
half. Every way v1 can fail to be importable is a named code with a structured
payload, written to stderr as one JSON line, and the process exits without
answering ``hello``.

Why that is worth a module rather than a ``try: import``
--------------------------------------------------------
Because in this worktree v1 is **not** importable, and the reasons are specific
and worth naming individually rather than collapsing into "ImportError":

* ``docs/reference-v1/`` is a *verbatim copy*, not a package layout. Its
  ``alphazero-env/`` directory holds the contents of ``alphazero/envs/gungi/``
  and its ``gungi-py/`` directory holds the contents of ``gungi/``; neither
  directory name is a legal module name, so no ``sys.path`` entry makes
  ``import alphazero.envs.gungi.gungi`` work against the copy as it stands
  (``docs/reference-v1/README.md:1-5``).
* The three engine extensions are shipped as ``.pyx``/``.pxd`` sources
  (``alphazero-env/cyboard.pyx``, ``cymovegen.pyx``, ``cyengine.pyx``) and
  nothing in the tree builds them -- there is no ``setup.py`` and no
  ``pyproject.toml`` anywhere under ``docs/reference-v1``.
* v1's **network and MCTS are absent entirely**: kom-023 §0.2 U-A, "there is no
  ``NNet.py``, no ``MCTS.py``, no model definition". That is why
  :data:`DEFAULT_GAME_FACTORY` has a default derived from the tree
  (``alphazero-env/gungi.py:187`` defines ``class Game(GameState)``) while
  ``--net-factory`` and ``--search-factory`` have **none**: a default there
  would be an invented name asserting a fact about a file nobody in this
  repository has read.

So the honest behaviour is to say which of those it is. ``--check-imports``
runs exactly this path and prints the answer, which is what KOM-023c uses to
find out how far the reference tree is from runnable before a single game is
attempted.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from komugi.eval.v1runner.counter import (
    DEFAULT_OBSERVATION_RANK,
    CountedNet,
    EvalCounter,
)

# ---------------------------------------------------------------------------
# where v1 is, and what it is called
# ---------------------------------------------------------------------------

REFERENCE_ROOT_ENV: Final[str] = "KOMUGI_V1_REFERENCE_ROOT"
"""Environment override for the reference root, exported by ``.#v1``'s shellHook.

Read only as a **default**: an explicit ``--reference-root`` always wins, so a
stale export in an operator's shell can never silently redirect a rated run.
"""

DEFAULT_GAME_FACTORY: Final[str] = "alphazero.envs.gungi.gungi:Game"
"""``MODULE:ATTR`` of v1's ``GameState``.

Derived from the tree, not guessed: ``docs/reference-v1/alphazero-env/gungi.py:187``
is ``class Game(GameState)``, and its own siblings import it as
``alphazero.envs.gungi`` (``gungi.py:17-20``).
"""

ENGINE_SIBLINGS: Final[tuple[str, ...]] = (
    "action_encoding",
    "cyengine",
    "cymovegen",
)
"""The three modules beside ``Game`` the runner itself calls.

``action_encoding.decode_cython_action`` turns v1's chosen action id into the
record that crosses the wire (§D2.3, "Each side decodes its own action id to a
record before sending"); ``cyengine.apply_move``/``placement_key`` are how an
incoming record is applied to v1's own board, mirroring
``docs/reference-v1/alphazero-env/gungi.py:248-254`` with the decode removed;
``cymovegen.generate_all_moves`` is §D2.3's **other** half -- "and matches an
incoming record against its own ``generate_all_moves`` output before applying"
-- without which a record that is schema-valid but illegal in v1's position is
applied silently and the two boards diverge with no §D6 mode firing (review v2
F4).
"""

ASSEMBLED_ROOT_PARTS: Final[tuple[str, ...]] = ("build", "v1-reference")
"""Where the **assembled, importable** v1 tree is expected: ``build/v1-reference``.

Not ``docs/reference-v1``. That directory is a verbatim read-only copy whose own
README says "NEVER edit these; NEVER import them into komugi code", and §D2.2
pre-registers that v1 is run from a real checkout and never from the copy. A
default pointing at the copy is harmless only for as long as nobody assembles a
package layout underneath it, and becomes a pre-registration violation silently
on the day someone does (review v2 F9). ``build/`` is this repository's own
throwaway build directory, so the assembled tree is a build artefact rather than
an edit to the frozen copy.
"""

ASSEMBLY_INSTRUCTIONS: Final[str] = (
    "docs/reference-v1 is a VERBATIM copy, not an importable package layout "
    "(README.md:1-5), so it is assembled elsewhere -- by default "
    "<repo>/build/v1-reference, which is what KOMUGI_V1_REFERENCE_ROOT points "
    "at inside `nix develop .#v1`. The layout that actually imports, derived "
    "from the tree's own imports rather than guessed: (1) copy "
    "alphazero-env/{gungi.py,action_encoding.py,observation.py,*.pyx,*.pxd} to "
    "<root>/alphazero/envs/gungi/; (2) copy alphazero-env/Game.py to "
    "<root>/alphazero/Game.py -- NOT into envs/gungi/, because "
    "alphazero-env/gungi.py:16 is `from alphazero.Game import GameState`, one "
    "level up and out of the envs subtree; (3) copy gungi-py/ to <root>/gungi/ "
    "(it already carries its own __init__.py), because gungi.py:22-27 is `from "
    "gungi.fen import ...`; (4) create empty __init__.py in <root>/alphazero/ "
    "and <root>/alphazero/envs/ and <root>/alphazero/envs/gungi/; (5) build the "
    "three Cython extensions IN PLACE under <root>/alphazero/envs/gungi/ "
    "against this shell's numpy, since the tree ships no setup.py and no "
    "pyproject.toml; (6) point --reference-root at <root>."
)
"""The reassembly recipe, in the one place both refusals quote it from.

KOM-023c reads this message before it reads anything else, so following it
exactly has to produce a tree that imports. The previous wording sent
``Game.py`` to ``alphazero/envs/gungi/Game.py``, which leaves
``from alphazero.Game import GameState`` unsatisfiable -- an instruction that
was pinned by a test asserting three substrings, none of them the wrong part
(review v2 F8).
"""

# refusal codes -- stable identifiers, because an operator greps these
ROOT_MISSING: Final[str] = "reference_root_missing"
ENGINE_IMPORT_FAILED: Final[str] = "reference_import_failed"
FACTORY_UNSPECIFIED: Final[str] = "factory_unspecified"
FACTORY_IMPORT_FAILED: Final[str] = "factory_import_failed"
CHECKPOINT_MISSING: Final[str] = "checkpoint_missing"
NET_INCOMPLETE: Final[str] = "net_incomplete"
BACKEND_INCOMPLETE: Final[str] = "backend_incomplete"

REFUSAL_CODES: Final[frozenset[str]] = frozenset(
    {
        ROOT_MISSING,
        ENGINE_IMPORT_FAILED,
        FACTORY_UNSPECIFIED,
        FACTORY_IMPORT_FAILED,
        CHECKPOINT_MISSING,
        NET_INCOMPLETE,
        BACKEND_INCOMPLETE,
    }
)


class ReferenceUnavailable(Exception):
    """v1 could not be loaded. Carries a code, not just a message.

    Args:
        code: One of :data:`REFUSAL_CODES`.
        detail: What went wrong, always naming the observed value.
        missing: The module names, attributes or paths that were not found.
        root: The reference root that was searched.
        searched: The ``sys.path`` entries the load added.
    """

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        missing: Sequence[str] = (),
        root: str = "",
        searched: Sequence[str] = (),
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.missing = tuple(missing)
        self.root = root
        self.searched = tuple(searched)

    def payload(self) -> dict[str, Any]:
        """Return the structured refusal line's body (see ``protocol.REFUSED``)."""
        return {
            "reason": self.code,
            "detail": self.detail,
            "missing": list(self.missing),
            "reference_root": self.root,
            "searched": list(self.searched),
        }


def repo_root() -> Path:
    """Return the komugi checkout this module was imported from."""
    return Path(__file__).resolve().parents[4]


def verbatim_copy_root() -> Path:
    """Return ``docs/reference-v1`` -- the frozen copy, which is never imported."""
    return repo_root() / "docs" / "reference-v1"


def default_reference_root() -> Path:
    """Return the reference root to use when ``--reference-root`` is absent.

    The **assembled** tree (:data:`ASSEMBLED_ROOT_PARTS`), never the verbatim
    copy: see that constant for why defaulting to ``docs/reference-v1`` is a
    pre-registration violation waiting for someone to assemble a package layout
    under it (review v2 F9).
    """
    override = os.environ.get(REFERENCE_ROOT_ENV, "").strip()
    if override:
        return Path(override)
    return repo_root().joinpath(*ASSEMBLED_ROOT_PARTS)


def observation_shape(game_cls: Any) -> tuple[int, ...]:
    """Return v1's observation shape, or ``()`` if the game class will not say.

    ``Game.observation_size()`` is a static part of alphazero-general's
    ``GameState`` contract (``docs/reference-v1/alphazero-env/gungi.py:220-222``)
    and is free to call at handshake time. Two things need it: §D1's ``hello``
    lists ``observation_size`` among the facts the adapter reports (review v2
    F6), and :class:`~komugi.eval.v1runner.counter.CountedNet` needs to know how
    many axes ONE position has before it can tell a batch from a position
    (review v2 F3).

    ``()`` rather than a raise: a game class without the classmethod is a v1
    this design has not seen, and the honest report of that is an empty shape on
    the wire plus the fallback rank, not a startup crash.
    """
    getter = getattr(game_cls, "observation_size", None)
    if not callable(getter):
        return ()
    try:
        size = getter()
    except Exception:  # noqa: BLE001 - a v1 method can fail any way
        return ()
    try:
        return tuple(int(dim) for dim in size)
    except (TypeError, ValueError):
        return ()


def describe_root(root: Path) -> dict[str, Any]:
    """Return what is actually at `root`, for a refusal an operator can act on.

    A bare ``ModuleNotFoundError: alphazero`` tells nobody whether the root is
    wrong, the layout is wrong, or the extensions are unbuilt. This says which.
    """
    entries = sorted(child.name for child in root.iterdir()) if root.is_dir() else []
    unbuilt = (
        sorted(str(path.relative_to(root)) for path in root.rglob("*.pyx"))
        if root.is_dir()
        else []
    )
    return {
        "exists": root.is_dir(),
        "entries": entries,
        "unbuilt_cython_sources": unbuilt,
        "verbatim_copy_layout": sorted(
            name for name in entries if name in {"alphazero-env", "gungi-py"}
        ),
    }


def install_reference_paths(root: Path, extra: Sequence[str] = ()) -> list[str]:
    """Put `root` and each `extra` on ``sys.path``, front first, and return them.

    Paths are resolved against `root` when relative, so ``--reference-path
    build/lib`` means what an operator expects. Prepended rather than appended:
    the v1 process must reach v1's ``gungi`` package and not any same-named
    thing already installed in the shell.
    """
    added: list[str] = []
    for candidate in [root, *(root / item for item in extra)]:
        entry = str(Path(candidate).resolve())
        if entry not in sys.path:
            sys.path.insert(0, entry)
        added.append(entry)
    return added


def _split_spec(spec: str, what: str) -> tuple[str, str]:
    """Split ``MODULE:ATTR``, raising a :class:`ReferenceUnavailable` on junk."""
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name.strip() or not attribute.strip():
        raise ReferenceUnavailable(
            FACTORY_IMPORT_FAILED,
            f"{what} must be given as MODULE:ATTR, got {spec!r}",
            missing=[spec],
        )
    return module_name.strip(), attribute.strip()


def load_object(spec: str, what: str, *, root: str = "", searched: Sequence[str] = ()) -> Any:
    """Import ``MODULE:ATTR`` and return the attribute.

    Raises:
        ReferenceUnavailable: With :data:`FACTORY_IMPORT_FAILED` if the module
            does not import or the attribute is absent.
    """
    module_name, attribute = _split_spec(spec, what)
    try:
        module = importlib.import_module(module_name)
    except Exception as error:  # noqa: BLE001 - a v1 import can fail any way
        raise ReferenceUnavailable(
            FACTORY_IMPORT_FAILED,
            f"{what} {spec!r}: importing {module_name!r} raised "
            f"{type(error).__name__}: {error}",
            missing=[module_name],
            root=root,
            searched=searched,
        ) from error
    if not hasattr(module, attribute):
        raise ReferenceUnavailable(
            FACTORY_IMPORT_FAILED,
            f"{what} {spec!r}: module {module_name!r} has no attribute "
            f"{attribute!r}",
            missing=[f"{module_name}.{attribute}"],
            root=root,
            searched=searched,
        )
    return getattr(module, attribute)


@dataclass(frozen=True, slots=True)
class LoadedReference:
    """Everything the serve loop needs from v1, resolved and checked.

    Args:
        game_cls: v1's ``GameState`` subclass (``gungi.py:187``).
        action_encoding: v1's ``action_encoding`` module.
        cyengine: v1's ``cyengine`` module.
        cymovegen: v1's ``cymovegen`` module -- §D2.3's incoming-record check.
        net: The counted-net's collaborator -- whatever ``--net-factory`` built.
        search_factory: ``(net, *, sims, seed) -> SearchBackend``, called
            **fresh on every pick** (§D3.2 clause 2).
        package: The engine package the three above came from.
        searched: The ``sys.path`` entries that were added.
    """

    game_cls: Any
    action_encoding: Any
    cyengine: Any
    cymovegen: Any
    net: Any
    search_factory: Any
    package: str
    searched: tuple[str, ...]


def load_engine(
    root: Path, game_factory: str, extra_paths: Sequence[str] = ()
) -> tuple[Any, Any, Any, Any, str, tuple[str, ...]]:
    """Import v1's ``Game`` and the three engine modules the runner itself calls.

    Returns:
        ``(game_cls, action_encoding, cyengine, cymovegen, package, searched)``.

    Raises:
        ReferenceUnavailable: :data:`ROOT_MISSING` if `root` is not a
            directory, :data:`ENGINE_IMPORT_FAILED` if any of the three does
            not import. The latter's detail carries :func:`describe_root`'s
            findings, because "no module named alphazero" and "you pointed me
            at a verbatim copy whose directories are called ``alphazero-env``
            and ``gungi-py``" are the same exception and very different bugs.
    """
    if not root.is_dir():
        raise ReferenceUnavailable(
            ROOT_MISSING,
            f"the reference root {str(root)!r} is not a directory. "
            f"{ASSEMBLY_INSTRUCTIONS}",
            missing=[str(root)],
            root=str(root),
        )
    searched = tuple(install_reference_paths(root, extra_paths))
    module_name, _ = _split_spec(game_factory, "--game-factory")
    package = module_name.rpartition(".")[0] or module_name
    try:
        game_cls = load_object(
            game_factory, "--game-factory", root=str(root), searched=searched
        )
        siblings = [
            importlib.import_module(f"{package}.{name}") for name in ENGINE_SIBLINGS
        ]
    except ReferenceUnavailable as error:
        found = describe_root(root)
        raise ReferenceUnavailable(
            ENGINE_IMPORT_FAILED,
            f"{error.detail}. What is actually at the reference root: {found}. "
            f"If 'verbatim_copy_layout' is non-empty, this root IS "
            f"docs/reference-v1 -- a read-only copy whose directories are not "
            f"importable package names. If 'unbuilt_cython_sources' is "
            f"non-empty, the three engine extensions still need building "
            f"against this shell's numpy. {ASSEMBLY_INSTRUCTIONS}",
            missing=list(error.missing),
            root=str(root),
            searched=searched,
        ) from error
    except Exception as error:  # noqa: BLE001 - a v1 import can fail any way
        raise ReferenceUnavailable(
            ENGINE_IMPORT_FAILED,
            f"importing {package}.{{{','.join(ENGINE_SIBLINGS)}}} raised "
            f"{type(error).__name__}: {error}. What is actually at the "
            f"reference root: {describe_root(root)}",
            missing=[f"{package}.{name}" for name in ENGINE_SIBLINGS],
            root=str(root),
            searched=searched,
        ) from error
    return game_cls, siblings[0], siblings[1], siblings[2], package, searched


def _check_net(net: Any, spec: str, root: str, searched: Sequence[str]) -> None:
    """Refuse a net that has no callable ``predict`` -- the counted boundary.

    The refusal text carries the two U-A obligations the factory's author --
    who has never read this module -- must satisfy, because this string is
    where they will read them (review v2 F2, F3): the search must reach the
    network *only* through the object this package hands it, and ``predict``
    must be about one position, or its batch is charged in full.
    """
    if not callable(getattr(net, "predict", None)):
        raise ReferenceUnavailable(
            NET_INCOMPLETE,
            f"--net-factory {spec!r} produced {type(net).__name__} with no "
            f"callable 'predict'. predict(observation) -> (policy, value) is "
            f"the net-call boundary §D3.5 places the counter at; without it "
            f"there is nowhere to count and evals_used would be fiction. Two "
            f"further obligations on --search-factory, which this package "
            f"cannot check for you: the search must reach the network ONLY "
            f"through the CountedNet it is handed on each call (never a "
            f"module-global, an lru_cache'd or a class-attribute net captured "
            f"at startup -- this package hands a CountedNet to every call "
            f"including its own probe, so a captured one is a captured counted "
            f"one, but a *second* wrapper around .inner is uncounted); and "
            f"predict must be about ONE position, since §D3.2 counts positions "
            f"and a stacked batch is charged its full length",
            missing=[f"{type(net).__name__}.predict"],
            root=root,
            searched=searched,
        )


def _check_backend(backend: Any, spec: str, root: str, searched: Sequence[str]) -> None:
    """Refuse a search backend that cannot answer §D3's three wire questions.

    ``node_count`` is checked hardest, and it is checked *at startup* with an
    assertion that a freshly built backend reports **0**. §D3.6 P3 is enforced
    by the harness on a number this process reports, so a backend that reported
    a constant 0 -- or that had no notion of tree size at all -- would make P3
    vacuous exactly where it is load-bearing, and the harness could not tell.
    """
    missing = [
        name
        for name in ("node_count", "sims_used", "pick")
        if not callable(getattr(backend, name, None))
    ]
    if missing:
        raise ReferenceUnavailable(
            BACKEND_INCOMPLETE,
            f"--search-factory {spec!r} produced {type(backend).__name__} "
            f"without {missing}. node_count() is what §D3.6 P3 "
            f"(tree_nodes_at_entry == 0 on every pick) is enforced on, "
            f"sims_used() is P1's second term, and pick() is the search",
            missing=missing,
            root=root,
            searched=searched,
        )
    nodes = int(backend.node_count())
    if nodes != 0:
        raise ReferenceUnavailable(
            BACKEND_INCOMPLETE,
            f"--search-factory {spec!r} built a search whose tree already holds "
            f"{nodes} nodes before any simulation. §D3.2 clause 2 requires a "
            f"fresh search object per pick, and a factory that starts non-empty "
            f"cannot satisfy it at any ply",
            missing=["node_count() == 0"],
            root=root,
            searched=searched,
        )


def load_reference(
    *,
    root: Path,
    game_factory: str,
    net_factory: str,
    search_factory: str,
    checkpoint: str,
    device: str,
    sims_allowance: int,
    extra_paths: Sequence[str] = (),
) -> LoadedReference:
    """Import and check every piece of v1 the serve loop needs.

    Raises:
        ReferenceUnavailable: On any of :data:`REFUSAL_CODES`. Nothing is
            partially initialised on the way out: this runs before ``hello`` is
            read, so a refusal here is a startup failure the harness reports as
            :class:`HandshakeRefused` with this process's stderr attached.
    """
    game_cls, action_encoding, cyengine, cymovegen, package, searched = load_engine(
        root, game_factory, extra_paths
    )
    for name, spec in (("--net-factory", net_factory), ("--search-factory", search_factory)):
        if not spec.strip():
            raise ReferenceUnavailable(
                FACTORY_UNSPECIFIED,
                f"{name} was not given, and there is no default to fall back "
                f"on: v1's network and MCTS are kom-023 §0.2 U-A -- absent "
                f"from docs/reference-v1 entirely (no NNet.py, no MCTS.py, no "
                f"model definition), so any default here would be an invented "
                f"name asserting a fact about a file nobody has read. Pass "
                f"{name} MODULE:ATTR naming v1's own",
                missing=[name],
                root=str(root),
                searched=searched,
            )
    if checkpoint and not Path(checkpoint).is_file():
        raise ReferenceUnavailable(
            CHECKPOINT_MISSING,
            f"--checkpoint {checkpoint!r} is not a file. The harness refuses to "
            f"rate unverified weights under a checkpoint's name "
            f"(arena.py:673-677) and the v1 side must make the same refusal "
            f"possible, so hello's checkpoint_digest is never faked",
            missing=[checkpoint],
            root=str(root),
            searched=searched,
        )

    build_net = load_object(
        net_factory, "--net-factory", root=str(root), searched=searched
    )
    build_search = load_object(
        search_factory, "--search-factory", root=str(root), searched=searched
    )
    try:
        net = build_net(checkpoint=checkpoint, device=device)
    except Exception as error:  # noqa: BLE001 - a v1 net can fail any way
        raise ReferenceUnavailable(
            NET_INCOMPLETE,
            f"--net-factory {net_factory!r}(checkpoint=..., device={device!r}) "
            f"raised {type(error).__name__}: {error}",
            missing=[net_factory],
            root=str(root),
            searched=searched,
        ) from error
    _check_net(net, net_factory, str(root), searched)

    # The probe gets a CountedNet, not the raw net (review v2 F2). The three
    # most natural shapes of a `--search-factory` -- a module-global `_NET`, an
    # lru_cache on the factory, a backend that keeps the net as a CLASS
    # attribute -- all make the object handed over at STARTUP the one that
    # plays every rated move. Handing the raw net here would make that object
    # uncounted, and the partial-bypass case (counted leaves, startup net for
    # the root) under-reports by exactly one evaluation per move with P1 still
    # holding and nothing anywhere to show it. So this package never gives a
    # raw net to anyone: the scratch counter is opened for the call, because a
    # factory that does a warm-up forward at construction must be served (and
    # charged) rather than meeting a bare "no move in progress".
    probe_counter = EvalCounter()
    probe_counter.begin_move()
    probe_net = CountedNet(
        net,
        probe_counter,
        observation_rank=len(observation_shape(game_cls)) or DEFAULT_OBSERVATION_RANK,
    )
    try:
        probe = build_search(probe_net, sims=sims_allowance, seed=0)
    except Exception as error:  # noqa: BLE001 - a v1 search can fail any way
        raise ReferenceUnavailable(
            BACKEND_INCOMPLETE,
            f"--search-factory {search_factory!r}(CountedNet, "
            f"sims={sims_allowance}, seed=0) raised {type(error).__name__}: "
            f"{error}",
            missing=[search_factory],
            root=str(root),
            searched=searched,
        ) from error
    finally:
        probe_counter.end_move()
    _check_backend(probe, search_factory, str(root), searched)

    return LoadedReference(
        game_cls=game_cls,
        action_encoding=action_encoding,
        cyengine=cyengine,
        cymovegen=cymovegen,
        net=net,
        search_factory=build_search,
        package=package,
        searched=searched,
    )
