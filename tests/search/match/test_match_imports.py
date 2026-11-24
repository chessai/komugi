"""Static gates on the match package: disjointness, imports, and taint.

Three acceptances, all of them ``ast`` walks over source rather than
observations of a running process, and each of them guarding a failure whose
runtime symptom is either invisible or arrives far from its cause.

* **A7 (T5 / P-D2 / P-D3)** -- no self-play module imports the match package,
  and the match package holds no module-level mutable state. The import walk
  is by ``ast`` and not by ``sys.modules`` because an import-time check passes
  vacuously if the offending module was never imported.
* **A8** -- the import allowlist, with exactly one named exception, plus the
  ban on the scalar-result identifier. The check is AST-**identifier**-scoped
  rather than textual, which is what lets the design's own pre-registered
  docstrings name the method in prose without failing.
* **A10b** -- the static half of the contempt invariant: no contempt-derived
  expression is ever *written* into a design §S1 field, with the taint closed
  over the intra-package call graph so a helper cannot launder it by being
  called instead of inlined. Returns, attributes and parameters are all
  followed, and across module boundaries and import aliases, because every one
  of those is a one-line change to mutant M6b that would otherwise walk past
  the gate. The analysis is itself **tested**, over laundered snippets that it
  must flag (review F2): a checker run only over code with nothing to find
  reports green whatever its reach.

A static analysis can always be walked around by one more indirection, so the
during-search window A10b guards is *also* guarded by bytes, in
``test_match_contempt.py``'s forced-descent clause: two searches whose descent
contempt cannot steer must store identical buffers.

Deliberately torch-free: everything below parses files, and nothing here
imports the package under test. A static gate that skipped because an
unrelated dependency was missing would be a gate that reports green for the
wrong reason.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
SRC_ROOT: Final[Path] = REPO_ROOT / "src"
PACKAGE_DIR: Final[Path] = SRC_ROOT / "komugi" / "search" / "match"

#: Design §M3's ten modules. The count is asserted, not assumed: an earlier
#: revision of the design said "nine" over a ten-row table, so a test written
#: to the prose failed against the table.
EXPECTED_MODULES: Final[frozenset[str]] = frozenset(
    {
        "__init__",
        "state",
        "config",
        "node",
        "bias",
        "select",
        "tree",
        "search",
        "player",
        "adapter",
    }
)

#: The **four** names design §M3 permits from ``komugi.selfplay.inference``.
INFERENCE_ALLOWED: Final[frozenset[str]] = frozenset(
    {"BatchEvaluator", "NetEval", "WDL_SIZE", "wdl_utility"}
)

#: ``wdl_utility`` is for A9's cross-check helper and nothing else, so it may
#: appear in exactly one module.
WDL_UTILITY_HOME: Final[str] = "node"

#: Modules nothing in the package may import from, at all.
FORBIDDEN_MODULES: Final[tuple[str, ...]] = (
    "komugi.selfplay.worker",
    "komugi.selfplay.orchestrate",
    "komugi.search.gumbel",
    "komugi.protocols",
)

#: A8's single named exception, as the triple the test encodes it as: the
#: name, the file, and the requirement that it be ``TYPE_CHECKING``-guarded.
ADAPTER_EXCEPTION: Final[tuple[str, str]] = ("GameState", "adapter")

#: Design §M3's "May import" column, transcribed verbatim from the table at
#: ``kom-007-match-play-search.md:411-422``. The design is the authority here,
#: so this dict is what the table says and **not** what the package happens to
#: import: where the two differ the drift is recorded below rather than
#: legislated away by widening this one.
EXTERNAL_ALLOWLIST: Final[dict[str, frozenset[str]]] = {
    "node": frozenset({"komugi.selfplay.inference"}),
    "adapter": frozenset({"komugi.selfplay.inference"}),
    "player": frozenset({"komugi.eval.v1bridge"}),
}

#: Names §M3 permits from each allowlisted module, where it names any. §M3
#: gives ``player.py`` ``komugi.eval.v1bridge`` "(``Mover`` only)".
EXTERNAL_NAME_ALLOWLIST: Final[dict[str, frozenset[str]]] = {
    "komugi.eval.v1bridge": frozenset({"Mover"}),
    "komugi.selfplay.inference": INFERENCE_ALLOWED,
}

#: **What ships but §M3 does not grant** (review F8b). Two deviations, both in
#: ``player.py``, both real:
#:
#: * ``komugi.eval.v1runner.counter`` -- §B2 makes ``player.py`` the owner of
#:   the move lifecycle and of the ``BudgetRefused -> BudgetBreach``
#:   translation, so it needs ``EvalCounter`` and ``BudgetRefused``; §M3's row
#:   names only ``v1bridge``. The *design table* is what is stale.
#: * ``BudgetBreach`` from ``v1bridge``, where §M3 says "(``Mover`` only)" --
#:   and ``player.py`` does not import ``Mover`` at all, since it satisfies it
#:   structurally.
#:
#: Recorded as data, not as a widened allowlist, so that
#: :func:`test_the_shipped_imports_are_within_design_m3` can fail on exactly
#: this drift until §M3 is amended -- at which point this dict empties and
#: that test flips to a hard failure until it is un-marked.
M3_DEVIATIONS: Final[dict[str, frozenset[str]]] = {
    "player": frozenset({"komugi.eval.v1runner.counter"})
}

#: The same, at name granularity: module -> imported module -> extra names.
M3_NAME_DEVIATIONS: Final[dict[str, dict[str, frozenset[str]]]] = {
    "player": {"komugi.eval.v1bridge": frozenset({"BudgetBreach"})}
}

#: The roots of the self-play import closure P-D2 walks.
SELFPLAY_ROOTS: Final[tuple[str, ...]] = (
    "komugi.selfplay.worker",
    "komugi.search.gumbel",
    "komugi.train.wiring",
)

MATCH_PACKAGE: Final[str] = "komugi.search.match"


def _module_source(name: str) -> Path | None:
    """Return the file backing dotted module `name`, or ``None``."""
    candidate = SRC_ROOT.joinpath(*name.split("."))
    if candidate.is_dir() and (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    flat = candidate.with_suffix(".py")
    return flat if flat.is_file() else None


def _parse(path: Path) -> ast.Module:
    """Parse `path`, naming it in any syntax error it raises."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _package_files() -> list[Path]:
    """Return every module of the match package, sorted by name."""
    return sorted(PACKAGE_DIR.glob("*.py"))


def _imported_modules(tree: ast.Module, package: str) -> set[str]:
    """Return every dotted module name `tree` imports directly.

    ``from a.b import c`` contributes both ``a.b`` and ``a.b.c``: the second
    is what a ``from komugi.search import match`` would look like, and a check
    that only recorded the first would miss it.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                # A relative import, resolved against the importing module's
                # own package. The repo uses absolute imports throughout, so
                # this branch exists so the walk cannot silently stop at one
                # rather than because one is expected.
                parts = package.split(".")[: -node.level]
                base = ".".join([*parts, base] if base else parts)
            if base:
                found.add(base)
                found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def test_the_package_has_exactly_the_ten_modules() -> None:
    """A1: design §M3's module table, asserted as a set rather than a count."""
    found = {path.stem for path in _package_files()}
    assert found == EXPECTED_MODULES
    assert len(found) == 10


def test_no_selfplay_module_imports_the_match_package() -> None:
    """A7 / P-D2: the self-play closure never names ``komugi.search.match``.

    Walked with ``ast`` from three roots rather than checked against
    ``sys.modules``: an import-time check passes vacuously whenever the
    offending module happens not to have been imported, which is exactly the
    condition under which a reverse edge would go unnoticed.
    """
    seen: set[str] = set()
    queue = list(SELFPLAY_ROOTS)
    offenders: list[str] = []
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        source = _module_source(name)
        if source is None:
            continue
        for imported in sorted(_imported_modules(_parse(source), name)):
            if imported == MATCH_PACKAGE or imported.startswith(f"{MATCH_PACKAGE}."):
                offenders.append(f"{name} -> {imported}")
            elif imported.startswith("komugi"):
                queue.append(imported)
    assert offenders == []
    # The walk has to have gone somewhere, or an empty result proves nothing.
    assert "komugi.search.gumbel" in seen
    assert len(seen) > 5


def _module_level_statements(tree: ast.Module) -> Iterator[ast.stmt]:
    """Yield module-level statements, descending into ``if`` bodies."""
    for statement in tree.body:
        if isinstance(statement, ast.If):
            yield from statement.body
            yield from statement.orelse
        else:
            yield statement


def _is_immutable_literal(value: ast.expr) -> bool:
    """Return whether `value` is a constant, or a tuple of constants."""
    if isinstance(value, ast.Constant):
        return True
    if isinstance(value, ast.UnaryOp):
        return _is_immutable_literal(value.operand)
    if isinstance(value, ast.BinOp):
        return _is_immutable_literal(value.left) and _is_immutable_literal(value.right)
    if isinstance(value, ast.Tuple):
        return all(_is_immutable_literal(item) for item in value.elts)
    if isinstance(value, ast.Set):
        # `frozenset({...})` is handled by the Call branch; a bare set is not.
        return False
    if isinstance(value, ast.Call):
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return name in {"frozenset", "Callable"}
    if isinstance(value, ast.Subscript):
        # A typing alias such as `Callable[[MatchState], WdlEval]`.
        return True
    if isinstance(value, ast.Name):
        return True
    return False


@pytest.mark.parametrize("path", _package_files(), ids=lambda path: path.stem)
def test_no_module_level_mutable_state(path: Path) -> None:
    """A7 / P-D3: every module-level binding is a constant, a class or a def.

    No cache dict, no table, no counter, no ``lru_cache``, no singleton. This
    is what makes two concurrent match searches independent and a player
    disposable at ``reset`` -- and two players in one process is the normal
    case, so a table declared at module scope would silently couple both sides
    of every game an arena runs.
    """
    offenders: list[str] = []
    for statement in _module_level_statements(_parse(path)):
        if isinstance(
            statement,
            (
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Expr,
                ast.Pass,
            ),
        ):
            continue
        if isinstance(statement, ast.AnnAssign):
            annotation = ast.unparse(statement.annotation)
            immutable = "Final" in annotation or "TypeAlias" in annotation
            if not immutable or (
                statement.value is not None
                and not _is_immutable_literal(statement.value)
            ):
                offenders.append(ast.unparse(statement))
            continue
        if isinstance(statement, ast.Assign):
            targets = {ast.unparse(target) for target in statement.targets}
            # `__all__` is a re-export declaration, not state: nothing reads it
            # during a search and rebinding it changes no behaviour.
            if targets != {"__all__"}:
                offenders.append(ast.unparse(statement))
            continue
        offenders.append(ast.unparse(statement))
    assert offenders == [], f"{path.name}: module-level state {offenders}"


def test_the_import_allowlist_holds() -> None:
    """A8: only the four inference names, and nothing from the banned modules.

    The limit of this check is stated rather than discovered: it is a
    **direct**-import check. ``player.py`` imports ``komugi.eval.v1bridge``,
    which at import time executes ``komugi.eval.arena`` and
    ``komugi.selfplay.adapter``, so both are reachable at runtime through an
    attribute chain no AST walk can see. A8 guards *naming*, one level deep;
    what guards the behaviour it was aimed at is A10b's taint check and the
    identifier ban below.

    The allowlist consulted here is §M3's column **plus**
    :data:`M3_DEVIATIONS`, the two shipped imports the design table does not
    grant. Keeping the deviations in their own constant is the point: this
    node stays a real check on every other module, and the drift itself is
    asserted against, by name, in
    :func:`test_the_shipped_imports_are_within_design_m3`.
    """
    problems: list[str] = []
    for path in _package_files():
        module = path.stem
        tree = _parse(path)
        allowed = EXTERNAL_ALLOWLIST.get(module, frozenset()) | M3_DEVIATIONS.get(
            module, frozenset()
        )
        for statement in ast.walk(tree):
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    if alias.name.startswith("komugi"):
                        problems.append(f"{module}: plain `import {alias.name}`")
                continue
            if not isinstance(statement, ast.ImportFrom):
                continue
            base = statement.module or ""
            if not base.startswith("komugi"):
                continue
            if base.startswith("komugi.search.match"):
                continue
            names = [alias.name for alias in statement.names]
            if base in FORBIDDEN_MODULES:
                problems.append(f"{module}: imports {names} from banned {base}")
            elif base == "komugi.selfplay.inference":
                extra = set(names) - INFERENCE_ALLOWED
                if extra:
                    problems.append(f"{module}: {sorted(extra)} from inference")
                if "wdl_utility" in names and module != WDL_UTILITY_HOME:
                    problems.append(f"{module}: wdl_utility outside {WDL_UTILITY_HOME}")
            elif base == "komugi.selfplay.adapter":
                # A8's single named exception has its own test below, which
                # checks the guard as well as the name and the file; repeating
                # a weaker version of it here would only add a second place to
                # keep in step.
                continue
            elif base not in allowed:
                problems.append(f"{module}: {base} is not on its allowlist")
    assert problems == []


@pytest.mark.xfail(
    strict=True,
    reason=(
        "design erratum owed: §M3's player.py row grants only "
        "komugi.eval.v1bridge (Mover only), while §B2 requires the eval "
        "counter and the BudgetRefused -> BudgetBreach translation (F8b)"
    ),
)
def test_the_shipped_imports_are_within_design_m3() -> None:
    """A8, against §M3's table **exactly** -- the drift, stated as a failure.

    The design is the authority on what this package may import, so the honest
    encoding of a package that imports more than its table grants is a red
    test naming the erratum, not an allowlist quietly widened to match the
    code (review F8b). It is marked ``xfail(strict=True)`` so the suite stays
    green while the erratum is owed *and* so amending §M3 -- adding
    ``komugi.eval.v1runner.counter`` to ``player.py``'s row and replacing the
    "(``Mover`` only)" restriction with the names actually used -- turns this
    node into a hard failure until :data:`M3_DEVIATIONS`,
    :data:`M3_NAME_DEVIATIONS` and this marker are removed together.

    Both deviations are asserted to still be exactly what is recorded, so this
    cannot start passing for a reason nobody looked at: a *third* deviation
    would be caught by :func:`test_the_import_allowlist_holds` above, which
    consults the recorded ones and nothing more.
    """
    assert M3_DEVIATIONS == {"player": frozenset({"komugi.eval.v1runner.counter"})}
    assert M3_NAME_DEVIATIONS == {
        "player": {"komugi.eval.v1bridge": frozenset({"BudgetBreach"})}
    }

    problems: list[str] = []
    for path in _package_files():
        module = path.stem
        granted = EXTERNAL_ALLOWLIST.get(module, frozenset())
        for statement in ast.walk(_parse(path)):
            if not isinstance(statement, ast.ImportFrom):
                continue
            base = statement.module or ""
            if not base.startswith("komugi") or base.startswith("komugi.search.match"):
                continue
            if base == "komugi.selfplay.adapter":
                # §M3 grants this one under TYPE_CHECKING, and the triple it is
                # granted as has its own test below.
                continue
            names = frozenset(alias.name for alias in statement.names)
            if base not in granted:
                problems.append(f"{module}: §M3 does not grant {base}")
                continue
            permitted = EXTERNAL_NAME_ALLOWLIST.get(base)
            if permitted is not None and not names <= permitted:
                problems.append(
                    f"{module}: §M3 grants {base} only for "
                    f"{sorted(permitted)}, got {sorted(names)}"
                )
    assert problems == []


def test_the_one_permitted_adapter_import_is_type_checking_guarded() -> None:
    """A8: the single named exception, encoded as a (name, file, guard) triple.

    Design §M2/B5 narrows the total runtime ban on ``komugi.selfplay.adapter``
    to exactly one site, so that A1 and A8 cannot both be unsatisfiable if a
    future ``mypy`` refuses the structural match. Any other name, any other
    file, or an unguarded import still fails.
    """
    name, home = ADAPTER_EXCEPTION
    sightings: list[tuple[str, bool, tuple[str, ...]]] = []
    for path in _package_files():
        tree = _parse(path)
        guarded: set[int] = set()
        for statement in tree.body:
            if isinstance(statement, ast.If) and "TYPE_CHECKING" in ast.unparse(
                statement.test
            ):
                for inner in ast.walk(statement):
                    guarded.add(id(inner))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "komugi.selfplay.adapter"
            ):
                sightings.append(
                    (
                        path.stem,
                        id(node) in guarded,
                        tuple(alias.name for alias in node.names),
                    )
                )
    assert sightings == [(home, True, (name,))]


def test_the_scalar_result_identifier_appears_nowhere() -> None:
    """A8: the banned identifier is absent from the package's parsed source.

    Scoped to ``Name`` and ``Attribute`` nodes, and therefore blind to
    docstrings and comments **by construction**: the method is named in prose
    all over this design, and a grep-shaped check would fail on the design's
    own pre-registered text while catching nothing a reader could not see.
    What it does catch is a call, which is the thing that would fold a draw
    score into a stored value and void the contempt invariant.
    """
    banned = "terminal_value"
    offenders: list[str] = []
    for path in _package_files():
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Name) and node.id == banned:
                offenders.append(f"{path.name}:{node.lineno} name")
            elif isinstance(node, ast.Attribute) and node.attr == banned:
                offenders.append(f"{path.name}:{node.lineno} attribute")
    assert offenders == []


def _s1_fields() -> tuple[str, ...]:
    """Return design §S1's field list, read out of ``node.py``'s own literal.

    Parsed rather than imported, so this file stays free of the package's
    runtime dependencies -- and so that the "declared once" property is itself
    checked: if the list stopped being a module-level literal in ``node.py``,
    this would fail rather than silently fall back to a copy.
    """
    tree = _parse(PACKAGE_DIR / "node.py")
    for statement in tree.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "S1_FIELDS"
            and isinstance(statement.value, ast.Tuple)
        ):
            return tuple(
                item.value
                for item in statement.value.elts
                # `ast.Constant.value` is any literal, so the string narrowing
                # is what makes this a `tuple[str, ...]`; a non-string entry in
                # `S1_FIELDS` is not a field name and has no business here.
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    raise AssertionError("node.py no longer declares S1_FIELDS as a literal tuple")


def test_s1_fields_is_the_designs_field_list() -> None:
    """The taint check below is only as good as the field list it reads."""
    fields = _s1_fields()
    assert fields[0] == "to_play"
    assert "wdl_sum" in fields and "moment" in fields and "children" in fields
    assert len(fields) == 17
    for absent in ("value", "value_sum", "q", "utility"):
        assert absent not in fields


def _returns_a_position(function: ast.FunctionDef) -> bool:
    """Return whether `function` is annotated as returning an ``int``.

    The taint below is over **values**, and an ``int`` in this package is never
    one: it is an edge index, a bias key, a mover id, a count or an action id
    -- a *position*, not a quantity that could carry a draw score. Contempt is
    supposed to decide **which** edge a descent takes; that is the entire point
    of the parameter, and a taint that flowed through
    ``select_child() -> int`` would mark every downstream consequence of a
    selection as contempt-derived, including the child object stored under the
    selected index. What A10b forbids is a contempt-derived *number* reaching
    storage, and every helper that returns one -- :func:`utility`,
    ``node_utility``, ``quadratic_form``, ``child_utilities`` -- is annotated
    ``float`` or ``NDArray[float64]`` and stays tainted.

    The escape this rule opens is narrow and is covered elsewhere: a helper
    that laundered contempt into an ``int`` could only reach ``visits``,
    ``bias_gen``, ``legal_hash``, ``bias_key`` or ``to_play``, and a stored
    integer that moved when contempt changed is exactly what A10a's buffer
    snapshot catches.
    """
    annotation = function.returns
    if annotation is None:
        return False
    return ast.unparse(annotation).replace(" ", "") in {"int", "int|None", "None|int"}


def _qualified_functions(
    node: ast.AST, prefix: str
) -> Iterator[tuple[str, ast.FunctionDef]]:
    """Yield ``(qualified name, def)`` for every function under `node`.

    The qualified name is ``module.Class.method`` -- the *whole* path, because
    a bare name is not a key: this package has six ``config`` functions, two
    ``clear``s and two ``weight``s, and keying on the bare name means the last
    one parsed silently replaces every other (review F3). A tainted function
    shadowed by an untainted namesake loses its taint at every call site,
    which is a hole in the analysis rather than in the code it reads.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified = f"{prefix}.{child.name}"
            if isinstance(child, ast.FunctionDef):
                yield qualified, child
            yield from _qualified_functions(child, qualified)
        else:
            yield from _qualified_functions(child, prefix)


def _import_aliases(trees: dict[str, ast.Module]) -> dict[str, str]:
    """Return ``local alias -> original bare name`` for every aliased import.

    ``from komugi.search.match.select import _q_vector as _q`` renames a
    helper at its call site, and a call resolved by bare name alone looks for
    ``_q``, finds no function of that name in the package, and reports the
    write as clean -- the M6b laundering route with one extra keyword on the
    import line (review F2). Aliases are collected package-wide, which
    over-approximates in the direction a gate should: if two modules alias the
    same short name to different helpers, a call to it is tainted if either is.
    """
    aliases: dict[str, str] = {}
    for tree in trees.values():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name.rsplit(".", 1)[-1]
    return aliases


def _assigned_attribute(target: ast.expr) -> str | None:
    """Return the attribute name `target` assigns into, through subscripts.

    ``self._cached_q[1] = ...`` assigns into ``_cached_q``; a bare local does
    not name an attribute at all and yields ``None``.
    """
    while isinstance(target, ast.Subscript):
        target = target.value
    return target.attr if isinstance(target, ast.Attribute) else None


def _assignments(
    function: ast.FunctionDef,
) -> Iterator[tuple[list[ast.expr], ast.expr]]:
    """Yield ``(targets, value)`` for every value-carrying assignment in `function`."""
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.expr):
                yield list(node.targets), node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.value, ast.expr):
                yield [node.target], node.value


def _positional_parameters(function: ast.FunctionDef) -> list[str]:
    """Return `function`'s positional parameter names, ``self``/``cls`` dropped.

    A call site never passes the receiver, so the argument at index ``i`` of a
    method call is the parameter at index ``i`` of *this* list.
    """
    positional = [
        argument.arg for argument in [*function.args.posonlyargs, *function.args.args]
    ]
    if positional and positional[0] in ("self", "cls"):
        return positional[1:]
    return positional


class _Taint:
    """Contempt taint over one package, closed over its own call graph.

    Three flows are followed, because a value that reaches a design §S1 field
    can travel by any of them and all three are cheap to launder through
    (review F2, the M6b family):

    * **return values** -- a helper whose return expression is tainted taints
      every call to it, under its own name or under an import alias;
    * **attributes** -- a value parked on ``self`` in one method and written to
      a node field in another, which no purely intra-procedural walk can see;
    * **parameters** -- a call that passes a tainted argument taints the
      callee's parameter, so a two-line "write this vector into that node"
      helper cannot launder the value by being the one holding the pen.
    """

    def __init__(self, trees: dict[str, ast.Module]) -> None:
        # A list per key, not one def: a property's getter and its setter share
        # a qualified name, and "the last one parsed wins" is the very failure
        # this keying exists to remove.
        self._functions: dict[str, list[ast.FunctionDef]] = {}
        for module, tree in trees.items():
            for qualified, function in _qualified_functions(tree, module):
                self._functions.setdefault(qualified, []).append(function)
        #: Bare name -> every qualified name it could refer to. A call site
        #: names a function without its class, so resolution is by bare name
        #: and a call is tainted if **any** candidate is: over-approximating
        #: is the safe direction for a gate, and it is what stops one
        #: untainted namesake from laundering the rest.
        self._by_bare_name: dict[str, list[str]] = {}
        for qualified in self._functions:
            bare = qualified.rsplit(".", 1)[1]
            self._by_bare_name.setdefault(bare, []).append(qualified)
        self._import_aliases = _import_aliases(trees)
        self.tainted_returns: set[str] = set()
        #: Attribute names anywhere in the package that are assigned a tainted
        #: value. Read of such an attribute is tainted wherever it happens.
        self.tainted_attributes: set[str] = set()
        #: ``id(FunctionDef) -> parameter names`` reached by a tainted argument
        #: at some call site. Keyed by identity because the same qualified name
        #: can hold several defs and each has its own parameters.
        self._tainted_parameters: dict[int, set[str]] = {}
        self._close()

    def taints_call(self, name: str) -> bool:
        """Return whether a call to bare `name` may return a tainted value.

        The name is looked up as written **and** as what it was imported as,
        so an aliased import cannot rename a tainted helper into an unknown.
        """
        candidates = {name, self._import_aliases.get(name, name)}
        return any(
            qualified in self.tainted_returns
            for candidate in candidates
            for qualified in self._by_bare_name.get(candidate, ())
        )

    def _callees(self, call: ast.Call) -> Iterator[ast.FunctionDef]:
        """Yield every package function `call` could be calling."""
        func = call.func
        called = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        )
        for candidate in {called, self._import_aliases.get(called, called)}:
            for qualified in self._by_bare_name.get(candidate, ()):
                yield from self._functions[qualified]

    def _close(self) -> None:
        """Iterate the call graph to a fixed point.

        Index-returning functions are excluded from the *return* closure for
        the reason :func:`_returns_a_position` gives: they return a choice, not
        a value. Their parameters are still tainted by a tainted argument,
        because a function that writes what it is handed is not made safe by
        also returning an index.
        """
        for _ in range(len(self._functions) + 2):
            grew = self._grow_attribute_taint()
            grew = self._grow_parameter_taint() or grew
            for qualified, functions in self._functions.items():
                if qualified in self.tainted_returns:
                    continue
                for function in functions:
                    if _returns_a_position(function):
                        continue
                    if self._returns_tainted(function):
                        self.tainted_returns.add(qualified)
                        grew = True
                        break
            if not grew:
                return

    def _grow_attribute_taint(self) -> bool:
        """Taint every attribute assigned a tainted value; report whether any is new.

        This is what closes the *stash* route: ``self._cached_q[1] =
        cfg.contempt`` in one method and ``parent.bias_contribution =
        self._cached_q * d`` in another is M6b with the helper's return value
        replaced by an instance attribute, and every step of it is invisible to
        a walk that only tracks local names.
        """
        grew = False
        for functions in self._functions.values():
            for function in functions:
                names = self.local_taint(function)
                for targets, value in _assignments(function):
                    if not self.expression_tainted(value, names):
                        continue
                    for target in targets:
                        attribute = _assigned_attribute(target)
                        if attribute is not None and (
                            attribute not in self.tainted_attributes
                        ):
                            self.tainted_attributes.add(attribute)
                            grew = True
        return grew

    def _grow_parameter_taint(self) -> bool:
        """Taint callee parameters reached by a tainted argument; report new ones."""
        grew = False
        for functions in self._functions.values():
            for caller in functions:
                names = self.local_taint(caller)
                for node in ast.walk(caller):
                    if not isinstance(node, ast.Call):
                        continue
                    grew = self._pass_arguments(node, names) or grew
        return grew

    def _pass_arguments(self, call: ast.Call, names: set[str]) -> bool:
        """Mark every parameter `call` hands a tainted argument to."""
        grew = False
        tainted_positions = [
            index
            for index, argument in enumerate(call.args)
            if self.expression_tainted(argument, names)
        ]
        tainted_keywords: set[str] = set()
        for keyword in call.keywords:
            if keyword.arg is None:  # `**kwargs`, which names no parameter
                continue
            if self.expression_tainted(keyword.value, names):
                tainted_keywords.add(keyword.arg)
        if not tainted_positions and not tainted_keywords:
            return False
        for callee in self._callees(call):
            reached = self._tainted_parameters.setdefault(id(callee), set())
            positional = _positional_parameters(callee)
            named = {
                argument.arg
                for argument in [
                    *callee.args.posonlyargs,
                    *callee.args.args,
                    *callee.args.kwonlyargs,
                ]
            }
            for index in tainted_positions:
                if index < len(positional) and positional[index] not in reached:
                    reached.add(positional[index])
                    grew = True
            for parameter in sorted(tainted_keywords & named):
                if parameter not in reached:
                    reached.add(parameter)
                    grew = True
        return grew

    def _returns_tainted(self, function: ast.FunctionDef) -> bool:
        names = self.local_taint(function)
        return any(
            isinstance(node, ast.Return)
            and node.value is not None
            and self.expression_tainted(node.value, names)
            for node in ast.walk(function)
        )

    def local_taint(self, function: ast.FunctionDef) -> set[str]:
        """Return the tainted local names of `function`.

        Seeded with any parameter named ``contempt``, plus any parameter some
        call site hands a tainted argument, and closed under assignment: a name
        is tainted if a tainted name, a ``.contempt`` attribute access, a read
        of a tainted attribute, or a call to a tainted-return function appears
        in its right-hand side.

        The target of such an assignment taints its **base name**, through any
        subscripts and attributes. Without that, a helper that builds ``q(c)``
        by item assignment --

            vector = np.zeros(3); vector[1] = cfg.contempt; return vector

        -- launders contempt completely: the only assignment target is a
        ``Subscript``, so ``vector`` never becomes tainted, the helper never
        enters :attr:`tainted_returns`, and its call sites contain neither a
        ``.contempt`` attribute nor a tainted name (review F2, mutant M6b).
        """
        arguments = function.args
        names = {
            argument.arg
            for argument in [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            ]
            if argument.arg == "contempt"
        }
        names |= self._tainted_parameters.get(id(function), set())
        for _ in range(3):  # a fixed point over statement order
            for targets, value in _assignments(function):
                if not self.expression_tainted(value, names):
                    continue
                for target in targets:
                    base = _base_name(target)
                    if base is not None:
                        names.add(base)
        return names

    def expression_tainted(self, expression: ast.expr, names: set[str]) -> bool:
        """Return whether `expression` reads contempt, directly or through a call."""
        for node in ast.walk(expression):
            if isinstance(node, ast.Attribute) and (
                node.attr == "contempt" or node.attr in self.tainted_attributes
            ):
                return True
            if isinstance(node, ast.Name) and node.id in names:
                return True
            if isinstance(node, ast.Call):
                func = node.func
                called = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else getattr(func, "id", "")
                )
                if self.taints_call(called):
                    return True
        return False


def _write_targets(statement: ast.stmt) -> Iterator[ast.expr]:
    """Yield the assignment targets of `statement`, if it writes anything."""
    if isinstance(statement, ast.Assign):
        yield from statement.targets
    elif isinstance(statement, (ast.AugAssign, ast.AnnAssign)):
        yield statement.target


def _base_name(target: ast.expr) -> str | None:
    """Return the name `target` is rooted at, through subscripts and attributes."""
    while isinstance(target, (ast.Subscript, ast.Attribute, ast.Starred)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _attribute_of(expression: ast.expr) -> str | None:
    """Return the attribute `expression` reads, through any subscripts."""
    while isinstance(expression, ast.Subscript):
        expression = expression.value
    if isinstance(expression, ast.Attribute):
        return expression.attr
    return None


def _array_aliases(function: ast.FunctionDef, fields: set[str]) -> dict[str, str]:
    """Return local names bound to a node array, mapped to the field's name.

    ``buffer = node.wdl_sum`` makes ``buffer`` another name for a design §S1
    buffer, and ``buffer[i] = <tainted>`` is then a write into that field
    which a target-shaped check cannot see: the stripped target is an
    ``ast.Name``, not an ``ast.Attribute`` (review F2, hole (ii)). Rebinding is
    not tracked -- an alias, once made, is assumed to hold for the function,
    which over-approximates in the direction a gate should.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        field = _attribute_of(value)
        if field not in fields:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and field is not None:
                aliases[target.id] = field
    return aliases


def _written_field(target: ast.expr, aliases: dict[str, str]) -> str | None:
    """Return the node field `target` writes, or ``None``.

    Through any subscripts, and through a local alias of a node array.
    """
    while isinstance(target, ast.Subscript):
        target = target.value
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return aliases.get(target.id)
    return None


def _call_writes(call: ast.Call) -> tuple[list[ast.expr], list[ast.expr]]:
    """Split `call` into ``(what it writes into, what it writes)``.

    Two numpy shapes write through an argument rather than through an
    assignment target: the destination-first ``np.copyto`` / ``np.add.at``,
    and **any** ufunc call with an ``out=`` keyword. Only the first was
    inspected before, and only positionally, so
    ``np.multiply(x, config.contempt, out=node.weight)`` was invisible
    (review F2, hole (iii)).
    """
    name = ast.unparse(call.func)
    written: list[ast.expr] = []
    values: list[ast.expr] = list(call.args)
    if name in ("np.copyto", "np.add.at") and call.args:
        written.append(call.args[0])
        values = list(call.args[1:])
    for keyword in call.keywords:
        if keyword.arg == "out":
            written.append(keyword.value)
        else:
            values.append(keyword.value)
    return written, values


def _taint_offenders(
    trees: dict[str, ast.Module], fields: set[str]
) -> tuple[list[str], _Taint]:
    """Return every contempt-derived write into a §S1 field, and the analysis.

    Extracted from A10b's test so that the *checker* can be tested rather than
    trusted: the self-tests below run it over deliberately laundered snippets
    and assert that each one is caught.
    """
    taint = _Taint(trees)
    offenders: list[str] = []
    for module, tree in trees.items():
        for function in [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]:
            names = taint.local_taint(function)
            aliases = _array_aliases(function, fields)
            for statement in ast.walk(function):
                value = getattr(statement, "value", None)
                if not isinstance(statement, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    continue
                if not isinstance(value, ast.expr):
                    continue
                for target in _write_targets(statement):
                    if _written_field(target, aliases) not in fields:
                        continue
                    if taint.expression_tainted(value, names):
                        offenders.append(
                            f"{module}.{function.name}:{statement.lineno} "
                            f"{ast.unparse(statement)}"
                        )
            for call in [
                node for node in ast.walk(function) if isinstance(node, ast.Call)
            ]:
                written, arguments = _call_writes(call)
                if not any(
                    _written_field(target, aliases) in fields for target in written
                ):
                    continue
                if any(
                    taint.expression_tainted(argument, names) for argument in arguments
                ):
                    offenders.append(
                        f"{module}.{function.name}:{call.lineno} "
                        f"{ast.unparse(call.func)}"
                    )
    return offenders, taint


def test_nothing_contempt_derived_is_written_into_a_node_field() -> None:
    """A10b: the static half of design §S1's contempt invariant.

    Its partner A10a catches a *lazily* cached utility, which is written after
    the snapshot and moves the bytes; this half catches one written during the
    search, which A10a's buffers would never see move. An implementation that
    stores contempt-poisoned values has to write a contempt-derived expression
    at one of exactly those two times, so the pair is discriminating -- and a
    correct implementation, which stores raw WDL and re-derives every scalar
    from ``q(c)`` at call time, writes nothing at either and passes both.

    The taint is closed over the intra-package call graph: a helper whose
    return expression is tainted taints every name bound from a call to it, so
    ``def _q(self): return np.array([1.0, self.cfg.contempt, -1.0])`` cannot
    launder the taint by being called from the write site instead of inlined
    into it. Calls out of the package are not walked and are not a laundering
    path, since numpy cannot itself read a ``MatchConfig``.

    The taint stops at an **index**, per :func:`_returns_a_position`. Contempt
    decides which edge the descent selects -- that is what the parameter is
    for -- so ``select_child`` returns a contempt-*dependent* position, and a
    taint that followed it would flag ``node.children[index] = child``: the
    creation of a child under a selected edge, which stores a node object and
    no number at all. A10b is about what a stored field *contains*, not about
    which edge the search chose to expand.
    """
    fields = set(_s1_fields())
    trees = {path.stem: _parse(path) for path in _package_files()}
    offenders, taint = _taint_offenders(trees, fields)

    # The analysis must actually find contempt somewhere, or it is asserting
    # the absence of a thing it cannot see. What it can *reach* is asserted
    # separately, over laundered snippets, in the self-tests below.
    assert taint.tainted_returns, "no contempt-tainted function found at all"
    assert taint.taints_call("utility")
    assert "node.utility" in taint.tainted_returns

    assert offenders == []


#: Mutant M6b, reduced: a helper that builds ``q(c)`` by **item assignment**
#: and is called from a write site. Nothing here names ``contempt`` at the
#: write, and before review F2's fix nothing in the analysis followed a
#: subscript target, so this escaped A10b -- and, being written during the
#: search, A10a's after-the-fact snapshot too.
LAUNDERED_ITEM_ASSIGNMENT = '''
import numpy as np

def _q_vector(cfg):
    """Build q(c) by item assignment."""
    vector = np.zeros(3, dtype=np.float64)
    vector[0] = 1.0
    vector[1] = cfg.contempt
    vector[2] = -1.0
    return vector

def _backup(self, parent, index, value):
    parent.bias_contribution = _q_vector(self._config) * float(value[1])
'''

#: The same write, laundered through a **local alias** of the node's array.
LAUNDERED_LOCAL_ALIAS = '''
def _backup(node, index, config):
    buffer = node.wdl_sum
    buffer[index] = config.contempt
'''

#: And through a numpy ``out=`` keyword, which is neither an assignment target
#: nor a positional argument of ``np.copyto``/``np.add.at``.
LAUNDERED_OUT_KEYWORD = '''
import numpy as np

def _rescale(node, config):
    np.multiply(node.weight, config.contempt, out=node.weight)
'''

#: The control: a backup that stores raw WDL, which is what the package
#: actually does. It contains a genuinely tainted function, so a checker that
#: reported nothing here because it found no contempt at all would not pass.
CLEAN_BACKUP = '''
def utility(wdl, contempt):
    return float(wdl[0]) - float(wdl[2]) + contempt * float(wdl[1])

def _backup(node, index, value, weight):
    node.wdl_sum[index] += weight * value
    node.weight[index] += weight
'''

#: **Mutant M6b itself, in the shape it takes in the package**: the helper in
#: ``select.py`` and the write in ``tree.py``'s ``_backup``, transcribed from
#: the review's own diff hunk. The one-module reduction above cannot pin this,
#: because a taint that stopped at a module boundary would still flag it.
M6B_SELECT_MODULE = '''
import numpy as np
import numpy.typing as npt

from komugi.search.match.config import MatchConfig
from komugi.search.match.node import WDL


def _q_vector(cfg: MatchConfig) -> npt.NDArray[np.float64]:
    """Build q(c) by item assignment, which the taint walk does not follow."""
    vector = np.zeros(WDL, dtype=np.float64)
    vector[0] = 1.0
    vector[1] = cfg.contempt
    vector[2] = -1.0
    return vector
'''

M6B_TREE_MODULE = '''
import numpy as np

from komugi.search.match.node import pack_outer
from komugi.search.match.select import _q_vector


class MatchTree:
    def _backup(self, path, leaf, evaluation):
        weight = float(self._config.leaf_weighter.weight(evaluation))
        value = np.asarray(leaf.wdl_net, dtype=np.float64)
        for parent, index in reversed(path):
            parent.visits[index] += 1
            parent.weight[index] += weight
            parent.wdl_sum[index] += weight * value
            parent.bias_contribution = _q_vector(self._config) * float(value[1])
            parent.moment[index] += weight * pack_outer(value)
'''

#: The same pair with the helper renamed on the import line, so that the bare
#: name at the write site is one no function in the package answers to.
M6B_ALIASED_TREE_MODULE = '''
import numpy as np

from komugi.search.match.select import _q_vector as _q


class MatchTree:
    def _backup(self, parent, index, value):
        parent.bias_contribution = _q(self._config) * float(value[1])
'''

#: M6b with the helper's *return value* replaced by an instance attribute:
#: contempt is parked on ``self`` in one method and written into a design §S1
#: field in another, so no single function contains both halves.
LAUNDERED_INSTANCE_ATTRIBUTE = '''
import numpy as np


class MatchTree:
    def _prepare(self, config):
        self._cached_q = np.zeros(3, dtype=np.float64)
        self._cached_q[1] = config.contempt

    def _backup(self, parent, index, value):
        parent.bias_contribution = self._cached_q * float(value[1])
'''

#: M6b with the pen in the other hand: the tainted vector is *passed* to a
#: two-line helper, and the function that names the node field never sees
#: contempt at all.
LAUNDERED_WRITE_HELPER = '''
import numpy as np


def _stash(node, vector):
    node.bias_contribution = vector


def _backup(parent, index, value, config):
    vector = np.zeros(3, dtype=np.float64)
    vector[1] = config.contempt
    _stash(parent, vector)
'''

#: Review F3: two same-named methods in different classes, the tainted one
#: **first**, so that a dict keyed on the bare name keeps the untainted
#: namesake and loses the taint at every call site.
SHADOWED_NAMESAKES = '''
import numpy as np

class Loud:
    def reading(self, config):
        return np.full(3, config.contempt, dtype=np.float64)

class Quiet:
    def reading(self, config):
        return np.zeros(3, dtype=np.float64)

def _backup(node, loud, config):
    node.bias_contribution = loud.reading(config)
'''


def _snippet_offenders(source: str) -> list[str]:
    """Return what the A10b analysis flags in a one-module `source`."""
    return _snippet_analysis(source)[0]


def _snippet_analysis(source: str) -> tuple[list[str], _Taint]:
    """Run the A10b analysis over `source` against the real §S1 field list."""
    return _package_analysis({"snippet": source})


def _package_analysis(sources: dict[str, str]) -> tuple[list[str], _Taint]:
    """Run the A10b analysis over a whole made-up package of `sources`.

    Keyed by module name, because the laundering that matters most puts the
    helper and the write in **different files**: that is the shape M6b takes in
    the package, and a taint that stopped at a module boundary would pass a
    one-module reduction of it.
    """
    return _taint_offenders(
        {name: ast.parse(source) for name, source in sources.items()},
        set(_s1_fields()),
    )


def test_the_taint_analysis_catches_each_laundering_route() -> None:
    """A10b's checker is tested, not trusted (review F2).

    A static gate is only as good as its reach, and this one's reach was
    asserted nowhere: the acceptance ran it over a package that has nothing to
    find, so three laundering routes -- one of which, M6b, defeats *both*
    halves of A10 and therefore the design's own "the pair is discriminating"
    argument -- passed it silently. Each snippet below writes a
    contempt-derived value into a design §S1 field by one of those routes, and
    each must be flagged; the fourth stores raw WDL and must not be.
    """
    item_offenders, item_taint = _snippet_analysis(LAUNDERED_ITEM_ASSIGNMENT)
    # The whole of M6b's trick: the helper has to become tainted despite the
    # only assignment to it being a subscript.
    assert "snippet._q_vector" in item_taint.tainted_returns
    assert item_offenders, "M6b escaped A10b"

    assert _snippet_offenders(LAUNDERED_LOCAL_ALIAS), "a node-array alias escaped"
    assert _snippet_offenders(LAUNDERED_OUT_KEYWORD), "an out= target escaped"

    clean_offenders, clean_taint = _snippet_analysis(CLEAN_BACKUP)
    assert clean_offenders == []
    # ...and the control is not clean because the checker found no contempt.
    assert clean_taint.taints_call("utility")


def test_the_analysis_catches_m6b_across_the_two_modules_it_lives_in() -> None:
    """M6b as the review wrote it: helper in ``select``, write in ``tree``.

    The reduction above puts both halves in one module, which cannot tell a
    taint that crosses files from one that does not -- and M6b's whole point is
    that the write site names no contempt, because the arithmetic that reads it
    is a file away. Each variant below is the same mutant with one more
    laundering step, and every one of them writes a contempt-derived value into
    ``bias_contribution``, a design §S1 field, **during** the search, where
    A10a's after-the-fact snapshot cannot see it move.
    """
    offenders, taint = _package_analysis(
        {"select": M6B_SELECT_MODULE, "tree": M6B_TREE_MODULE}
    )
    assert "select._q_vector" in taint.tainted_returns
    assert taint.taints_call("_q_vector")
    assert offenders, "M6b escaped A10b across the module boundary"
    assert any("bias_contribution" in offender for offender in offenders)
    # The clean writes beside it are not swept up: `visits`, `weight`,
    # `wdl_sum` and `moment` take the same untainted `weight * value` they
    # always did, and an analysis that flagged those would be a gate nobody
    # could keep green.
    assert not any("wdl_sum" in offender for offender in offenders)

    aliased, aliased_taint = _package_analysis(
        {"select": M6B_SELECT_MODULE, "tree": M6B_ALIASED_TREE_MODULE}
    )
    assert aliased_taint.taints_call("_q")
    assert aliased, "M6b escaped by renaming the helper on the import line"


def test_the_analysis_catches_contempt_parked_on_an_attribute() -> None:
    """M6b with an instance attribute in place of the helper's return value.

    ``self._cached_q`` is written in one method and read in another, so no
    single function holds both the contempt and the node field. The attribute
    itself is the carrier and is what has to be tainted.
    """
    offenders, taint = _snippet_analysis(LAUNDERED_INSTANCE_ATTRIBUTE)
    assert "_cached_q" in taint.tainted_attributes
    assert offenders, "a contempt-derived instance attribute escaped"


def test_the_analysis_catches_a_helper_that_holds_the_pen() -> None:
    """M6b with the write delegated: the writer never names contempt.

    ``_stash(node, vector)`` is handed the poisoned vector and does the
    assignment, so the function containing the §S1 field name has a completely
    clean body. Taint has to travel **into** a parameter for this to be seen.
    """
    offenders, _ = _snippet_analysis(LAUNDERED_WRITE_HELPER)
    assert offenders, "a write helper laundered contempt through its parameter"


def test_the_taint_is_not_lost_to_a_same_named_function() -> None:
    """F3: functions are keyed by module, class and name -- not by name.

    ``Loud.reading`` reads contempt and ``Quiet.reading`` does not. Keyed on
    the bare name the second silently replaces the first, and the write below
    -- which calls the tainted one -- is passed as clean. The same collision
    is live in the package itself: six ``config`` functions, two ``clear``s,
    two ``weight``s.
    """
    offenders, taint = _snippet_analysis(SHADOWED_NAMESAKES)
    assert "snippet.Loud.reading" in taint.tainted_returns
    assert "snippet.Quiet.reading" not in taint.tainted_returns
    assert taint.taints_call("reading")
    assert offenders, "a tainted function was shadowed by its untainted namesake"


def test_the_package_really_has_same_named_functions() -> None:
    """The premise of F3's fix, checked rather than asserted in prose."""
    trees = {path.stem: _parse(path) for path in _package_files()}
    bare: dict[str, int] = {}
    for module, tree in trees.items():
        for qualified, _ in _qualified_functions(tree, module):
            name = qualified.rsplit(".", 1)[1]
            bare[name] = bare.get(name, 0) + 1
    collisions = {name for name, count in bare.items() if count > 1}
    assert {"config", "clear", "weight"} <= collisions
