"""Stdlib-only JSON Schema validator for schemas/model-card.schema.json and
schemas/models-index.schema.json.

No ``jsonschema`` package is available in any dev shell (review F2), so this
implements exactly the keywords those two schemas use: ``$schema``, ``$id``
(annotations, ignored), ``type``, ``required``, ``properties``,
``additionalProperties`` (boolean or schema-valued), ``propertyNames``,
``items``, ``enum``, ``const``, ``pattern``, ``minLength``, ``minItems``,
``minimum``, ``format`` (``date-time`` only), ``$ref``/``$defs``. Any other
keyword raises ``SchemaError`` — this validator must never silently pass a
schema it does not fully understand.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

SUPPORTED_KEYWORDS = frozenset({
    "$schema", "$id", "type", "required", "properties", "additionalProperties",
    "propertyNames", "items", "enum", "const", "pattern", "minLength",
    "minItems", "minimum", "format", "$ref", "$defs",
})

SUPPORTED_FORMATS = frozenset({"date-time"})

# RFC 3339 date-time (review N5: datetime.fromisoformat alone accepts a bare
# date, so the regex requires the time-of-day and offset first).
_DATE_TIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
)


class SchemaError(Exception):
    """A schema uses a keyword, format, or $ref this validator doesn't implement."""


class ValidationError(Exception):
    """An instance does not satisfy the schema."""


def check_schema_keywords(schema: Any, path: str = "$") -> None:
    """Recursively assert every keyword in *schema* is one this module implements.

    Walks every schema position reachable structurally (``properties``,
    ``items``, ``$defs``, ``additionalProperties`` when schema-valued,
    ``propertyNames``), independent of any instance, so an unsupported
    keyword is caught even if no test instance happens to exercise it.
    """
    if not isinstance(schema, dict):
        return
    unsupported = set(schema) - SUPPORTED_KEYWORDS
    if unsupported:
        raise SchemaError(f"{path}: unsupported schema keyword(s): {sorted(unsupported)}")

    fmt = schema.get("format")
    if fmt is not None and fmt not in SUPPORTED_FORMATS:
        raise SchemaError(f"{path}: unsupported format {fmt!r}")

    properties = schema.get("properties")
    if properties is not None:
        for key, subschema in properties.items():
            check_schema_keywords(subschema, f"{path}.properties.{key}")

    items = schema.get("items")
    if items is not None:
        check_schema_keywords(items, f"{path}.items")

    defs = schema.get("$defs")
    if defs is not None:
        for key, subschema in defs.items():
            check_schema_keywords(subschema, f"{path}.$defs.{key}")

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        check_schema_keywords(additional, f"{path}.additionalProperties")

    property_names = schema.get("propertyNames")
    if property_names is not None:
        check_schema_keywords(property_names, f"{path}.propertyNames")


def _resolve_ref(ref: str, root: dict) -> dict:
    if not ref.startswith("#/"):
        raise SchemaError(f"unsupported $ref target: {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _matches_type(instance: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(instance, dict)
    if type_name == "array":
        return isinstance(instance, list)
    if type_name == "string":
        return isinstance(instance, str)
    if type_name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if type_name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if type_name == "boolean":
        return isinstance(instance, bool)
    if type_name == "null":
        return instance is None
    raise SchemaError(f"unsupported type {type_name!r}")


def _check_date_time(instance: Any, path: str) -> None:
    if not isinstance(instance, str) or not _DATE_TIME_RE.match(instance):
        raise ValidationError(f"{path}: {instance!r} is not a valid date-time")
    try:
        datetime.fromisoformat(instance.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{path}: {instance!r} is not a valid date-time") from exc


def validate(instance: Any, schema: dict, root: dict | None = None, path: str = "$") -> None:
    """Validate *instance* against *schema*, raising ValidationError on mismatch
    and SchemaError if the schema (or a $ref target it reaches) uses an
    unsupported keyword."""
    if root is None:
        root = schema
        check_schema_keywords(schema, path)

    if "$ref" in schema:
        target = _resolve_ref(schema["$ref"], root)
        validate(instance, target, root, path)
        return

    if "const" in schema and instance != schema["const"]:
        raise ValidationError(f"{path}: expected const {schema['const']!r}, got {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        raise ValidationError(f"{path}: {instance!r} not in enum {schema['enum']!r}")

    if "type" in schema:
        type_ = schema["type"]
        types = type_ if isinstance(type_, list) else [type_]
        if not any(_matches_type(instance, t) for t in types):
            raise ValidationError(f"{path}: {instance!r} is not of type {type_!r}")

    if "pattern" in schema:
        if not isinstance(instance, str) or not re.search(schema["pattern"], instance):
            raise ValidationError(
                f"{path}: {instance!r} does not match pattern {schema['pattern']!r}"
            )

    if "minLength" in schema:
        if not isinstance(instance, str) or len(instance) < schema["minLength"]:
            raise ValidationError(f"{path}: {instance!r} shorter than minLength {schema['minLength']}")

    if "minItems" in schema:
        if not isinstance(instance, list) or len(instance) < schema["minItems"]:
            raise ValidationError(f"{path}: fewer than minItems {schema['minItems']}")

    if "minimum" in schema:
        if (not isinstance(instance, (int, float)) or isinstance(instance, bool)
                or instance < schema["minimum"]):
            raise ValidationError(f"{path}: {instance!r} below minimum {schema['minimum']}")

    if "format" in schema:
        if schema["format"] == "date-time":
            _check_date_time(instance, path)

    if "required" in schema:
        if not isinstance(instance, dict):
            raise ValidationError(f"{path}: required needs an object")
        missing = [k for k in schema["required"] if k not in instance]
        if missing:
            raise ValidationError(f"{path}: missing required properties {missing}")

    properties = schema.get("properties")
    if properties is not None and isinstance(instance, dict):
        for key, subschema in properties.items():
            if key in instance:
                validate(instance[key], subschema, root, f"{path}.{key}")

    if "propertyNames" in schema and isinstance(instance, dict):
        name_schema = schema["propertyNames"]
        for key in instance:
            validate(key, name_schema, root, f"{path}:propertyNames")

    if "additionalProperties" in schema and isinstance(instance, dict):
        additional = schema["additionalProperties"]
        known = set(properties) if properties else set()
        extra = [k for k in instance if k not in known]
        if additional is False:
            if extra:
                raise ValidationError(f"{path}: additional properties not allowed: {extra}")
        elif isinstance(additional, dict):
            for key in extra:
                validate(instance[key], additional, root, f"{path}.{key}")
        else:
            raise SchemaError(f"{path}: additionalProperties must be a bool or a schema")

    if "items" in schema and isinstance(instance, list):
        items_schema = schema["items"]
        for i, item in enumerate(instance):
            validate(item, items_schema, root, f"{path}[{i}]")


def load_schema(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def validate_file(instance_path: Path, schema_path: Path) -> None:
    instance = json.loads(Path(instance_path).read_text())
    schema = load_schema(schema_path)
    validate(instance, schema)
