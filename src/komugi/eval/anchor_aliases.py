"""Backward compatibility for data written before the external-anchor rename.

The external-anchor family and kind constants, and the pool ids built from
them, were renamed to a generic ``external`` (see ``komugi.eval.pool``). Data
written before that rename -- stored ``pool.json``/``records.jsonl`` carrying
the old family value as a literal string, or a pool id with the old family as
its prefix -- still needs to load.

``KOMUGI_ANCHOR_ALIASES``, when set, names a JSON file mapping old tokens to
their new form, e.g. ``{"<old codename>": "external"}``. Every reader of a
stored family, kind or pool id goes through :func:`resolve_family` or
:func:`resolve_pool_id`, so that data loads under the new names without this
module -- or any caller that leaves the variable unset -- ever naming the old
token itself. With the variable unset, both functions are the identity.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Final

ALIASES_ENV_VAR: Final[str] = "KOMUGI_ANCHOR_ALIASES"
"""Names a JSON file of ``{old_token: new_token}``; unset means no aliases."""


class AnchorAliasError(ValueError):
    """``KOMUGI_ANCHOR_ALIASES`` names a file that is not a valid alias table."""


@lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    """Return the alias table named by :data:`ALIASES_ENV_VAR`, or ``{}``."""
    path = os.environ.get(ALIASES_ENV_VAR)
    if not path:
        return {}
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise AnchorAliasError(
            f"{ALIASES_ENV_VAR}={path!r} could not be read: {exc}"
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnchorAliasError(
            f"{ALIASES_ENV_VAR}={path!r} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AnchorAliasError(
            f"{ALIASES_ENV_VAR}={path!r} must decode to a JSON object mapping "
            f"strings to strings, got {type(payload).__name__}"
        )
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise AnchorAliasError(
                f"{ALIASES_ENV_VAR}={path!r} has a non-string key or value at "
                f"{key!r}: {value!r} -- every entry must map a string to a string"
            )
    return dict(payload)


def reset_cache() -> None:
    """Drop the cached alias table.

    Tests that change :data:`ALIASES_ENV_VAR` between reads call this; the
    table is cached because every reader in a hot loop (`family_of` included)
    would otherwise re-read and re-parse the file per call.
    """
    _aliases.cache_clear()


def resolve_family(family: str) -> str:
    """Return `family` mapped through the alias table, or `family` unchanged.

    For a bare family or kind value -- the exact string an old record stored,
    e.g. an old codename -- never a pool id (see :func:`resolve_pool_id`).
    """
    return _aliases().get(family, family)


def resolve_pool_id(pool_id: str) -> str:
    """Return `pool_id` with an aliased family prefix, if it has one.

    `pool_id` is ``<family>/<member>`` (``komugi.eval.pool.FAMILY_SEPARATOR``);
    only the prefix before the first ``/`` is looked up, so a member name that
    happens to collide with an old token is never touched.
    """
    aliases = _aliases()
    if not aliases:
        return pool_id
    prefix, separator, rest = pool_id.partition("/")
    if not separator:
        return aliases.get(pool_id, pool_id)
    return f"{aliases.get(prefix, prefix)}{separator}{rest}"
