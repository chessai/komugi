"""Unit 1 acceptance items 5 (AC5: schema validation) and 9 (F8: action
encoding version pin) — see docs/design/kom1188-model-releases.md section 6
and 7. The design places these in tools/test_release_model.py; the advisor's
unit 1a/1b split moves them here since tools/release_model.py (unit 1b) is
out of scope for this task.
"""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

import pytest

import schema_check

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CARD_SCHEMA = _REPO_ROOT / "schemas" / "model-card.schema.json"
_INDEX_SCHEMA = _REPO_ROOT / "schemas" / "models-index.schema.json"


def _valid_card() -> dict:
    return {
        "id": "komugi-c6-s580800-9f3a2b7c",
        "campaign": "c6", "iteration": 1936, "step": 580800,
        "compatibilityKey": {
            "planeMapVersion": 3, "actionEncodingVersion": 2,
            "rulesProfile": "komugi", "architectureTag": "b10c128",
        },
        "modesTrained": ["intro", "beginner", "intermediate", "advanced"],
        "assets": {
            "pt": {"path": "komugi-c6-s580800-9f3a2b7c.pt", "sha256": "a" * 64},
            "onnx": {"path": "komugi-c6-s580800-9f3a2b7c.onnx", "sha256": "b" * 64},
        },
        "sourceCheckpoint": {
            "name": "checkpoint-000580800-0123456789abcdef.pt", "sha256": "c" * 64,
        },
        "exportOpset": 17,
        "exportParity": {"n": 4, "maxAbsDelta": 6.2e-6, "atol": 3.9e-4, "rtol": 1.1e-4},
        "evalPointers": ["evals/c6-s580800-gauntlet.json"],
        "license": "MIT, Copyright (c) 2026 chessai",
        "createdAt": "2026-11-22T00:00:00Z",
    }


def _valid_index() -> dict:
    return {
        "schemaVersion": 1,
        "recommended": {
            "intro": "komugi-c6-s580800-9f3a2b7c",
            "beginner": "komugi-c6-s580800-9f3a2b7c",
            "intermediate": "komugi-c6-s580800-9f3a2b7c",
            "advanced": "komugi-c6-s580800-9f3a2b7c",
        },
        "models": {
            "komugi-c6-s580800-9f3a2b7c": {
                "card": "komugi-c6-s580800-9f3a2b7c.card.json",
                "onnxSha256": "b" * 64,
            },
        },
    }


def test_schema_files_accept_valid_instances():
    card_schema = schema_check.load_schema(_CARD_SCHEMA)
    index_schema = schema_check.load_schema(_INDEX_SCHEMA)
    schema_check.validate(_valid_card(), card_schema)
    schema_check.validate(_valid_index(), index_schema)


def test_card_rejects_relative_but_wrong_created_at_yesterday():
    card_schema = schema_check.load_schema(_CARD_SCHEMA)
    card = _valid_card()
    card["createdAt"] = "yesterday"
    with pytest.raises(schema_check.ValidationError):
        schema_check.validate(card, card_schema)


def test_card_rejects_bare_date_created_at():
    card_schema = schema_check.load_schema(_CARD_SCHEMA)
    card = _valid_card()
    card["createdAt"] = "2026-11-22"
    with pytest.raises(schema_check.ValidationError):
        schema_check.validate(card, card_schema)


def test_card_rejects_missing_license():
    card_schema = schema_check.load_schema(_CARD_SCHEMA)
    card = _valid_card()
    del card["license"]
    with pytest.raises(schema_check.ValidationError):
        schema_check.validate(card, card_schema)


def test_card_rejects_absolute_asset_path():
    card_schema = schema_check.load_schema(_CARD_SCHEMA)
    card = _valid_card()
    card["assets"]["pt"]["path"] = "/tmp/out/komugi-c6-s580800-9f3a2b7c.pt"
    with pytest.raises(schema_check.ValidationError):
        schema_check.validate(card, card_schema)


def test_card_accepts_missing_asset_bytes():
    card_schema = schema_check.load_schema(_CARD_SCHEMA)
    card = _valid_card()
    assert "bytes" not in card["assets"]["pt"]
    assert "bytes" not in card["assets"]["onnx"]
    schema_check.validate(card, card_schema)


def test_card_accepts_positive_asset_bytes():
    card_schema = schema_check.load_schema(_CARD_SCHEMA)
    card = _valid_card()
    card["assets"]["pt"]["bytes"] = 1234
    card["assets"]["onnx"]["bytes"] = 5678
    schema_check.validate(card, card_schema)


def test_card_rejects_nonpositive_asset_bytes():
    card_schema = schema_check.load_schema(_CARD_SCHEMA)
    card = _valid_card()
    card["assets"]["onnx"]["bytes"] = 0
    with pytest.raises(schema_check.ValidationError, match="minimum"):
        schema_check.validate(card, card_schema)


def _assert_recommended_subset_of_models(index: dict) -> None:
    """The referential 'every recommended value names a models key' check.

    Section 4.4: a JSON Schema can't assert this without $dynamicRef
    gymnastics, so it's a plain assertion here, never part of the schema.
    """
    for mode, model_id in index["recommended"].items():
        assert model_id in index["models"], (
            f"recommended[{mode!r}] = {model_id!r} is not a key of models"
        )


def test_recommended_subset_of_models_holds_for_valid_index():
    _assert_recommended_subset_of_models(_valid_index())


def test_recommended_subset_of_models_catches_dangling_reference():
    index = copy.deepcopy(_valid_index())
    index["recommended"]["intro"] = "komugi-c6-s999999-deadbeef"
    with pytest.raises(AssertionError):
        _assert_recommended_subset_of_models(index)


def test_validator_raises_on_unsupported_keyword():
    bad_schema = {"type": "object", "patternProperties": {"^x-": {"type": "string"}}}
    with pytest.raises(schema_check.SchemaError):
        schema_check.check_schema_keywords(bad_schema)
    with pytest.raises(schema_check.SchemaError):
        schema_check.validate({"x-foo": "bar"}, bad_schema)


def test_validator_raises_on_unsupported_keyword_nested_in_properties():
    bad_schema = {
        "type": "object",
        "properties": {"inner": {"type": "string", "contentEncoding": "base64"}},
    }
    with pytest.raises(schema_check.SchemaError):
        schema_check.check_schema_keywords(bad_schema)


# ---------------------------------------------------------------------------
# F8: test_action_encoding_version_pinned
# ---------------------------------------------------------------------------

_CONSTANTS_TS = _REPO_ROOT / "packages" / "engine" / "src" / "constants.ts"

# Pinned digest of (ACTION_SIZE, A0_START, A1_START, B_START, C_START) for
# ACTION_ENCODING_VERSION == 2 (review F8). Changing either constant, or any
# block start, without bumping the version, turns this test red.
_PINNED_DIGESTS = {
    2: "892ab84a261b14a182c3f2413326f24acd244402ede783d0c067692c71ab5711",
}


def _digest_of(action_size: int, a0: int, a1: int, b: int, c: int) -> str:
    payload = f"{action_size}:{a0}:{a1}:{b}:{c}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _parse_ts_int_constant(source: str, name: str) -> int:
    match = re.search(rf"export const {re.escape(name)} = (\d+);", source)
    assert match is not None, f"{name} not found in constants.ts"
    return int(match.group(1))


def test_action_encoding_version_pinned():
    from komugi.engine import action_encoding

    ts_source = _CONSTANTS_TS.read_text()
    ts_version = _parse_ts_int_constant(ts_source, "ACTION_ENCODING_VERSION")
    ts_action_size = _parse_ts_int_constant(ts_source, "ACTION_SIZE")
    ts_a0 = _parse_ts_int_constant(ts_source, "A0_START")
    ts_a1 = _parse_ts_int_constant(ts_source, "A1_START")
    ts_b = _parse_ts_int_constant(ts_source, "B_START")
    ts_c = _parse_ts_int_constant(ts_source, "C_START")

    assert action_encoding.ACTION_ENCODING_VERSION == ts_version
    assert action_encoding.ACTION_SIZE == ts_action_size
    assert action_encoding.A0_START == ts_a0
    assert action_encoding.A1_START == ts_a1
    assert action_encoding.B_START == ts_b
    assert action_encoding.C_START == ts_c

    digest = _digest_of(
        action_encoding.ACTION_SIZE,
        action_encoding.A0_START,
        action_encoding.A1_START,
        action_encoding.B_START,
        action_encoding.C_START,
    )
    assert action_encoding.ACTION_ENCODING_VERSION in _PINNED_DIGESTS
    assert digest == _PINNED_DIGESTS[action_encoding.ACTION_ENCODING_VERSION]
