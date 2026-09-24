"""Safe construction of GoodMem metadata filter expressions.

GoodMem filters are expression strings evaluated server-side. Building one by
interpolating caller data straight into the string is an injection risk, so
these helpers quote values and refuse field names that cannot be expressed
safely.

``val()`` returns JSON, so a text comparison needs an explicit cast:
``CAST(val('$.category') AS TEXT) = 'feat'``.

Escaping was established against a live server (v1.0.320), not assumed. The
grammar escapes with a backslash: SQL-style ``''`` doubling and double-quoted
strings are both rejected with HTTP 400, and a raw newline inside a literal is
rejected outright.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


# A JSONPath member we are willing to build without escaping games.
_SAFE_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The filter grammar has no encoding for these inside a literal.
_FORBIDDEN_IN_LITERAL = re.compile(r"[\x00-\x1f\x7f]")


def _quote(value: str) -> str:
    """Single-quote a literal, backslash-escaping backslashes and quotes.

    Order matters: backslashes are escaped first so the backslash introduced
    for a quote is not escaped a second time.
    """
    if _FORBIDDEN_IN_LITERAL.search(value):
        raise ValueError(
            "Metadata filter values cannot contain control characters; the "
            "GoodMem filter grammar rejects them."
        )
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _check_field(field: str) -> str:
    if not _SAFE_FIELD.match(field):
        raise ValueError(
            f"Unsupported metadata field name {field!r}. Use letters, digits "
            "and underscores, or pass a filter expression directly."
        )
    return field


def text_equals(field: str, value: str) -> str:
    """Build an equality comparison against a text metadata field."""
    return f"CAST(val('$.{_check_field(field)}') AS TEXT) = {_quote(value)}"


def from_mapping(metadata_filter: Mapping[str, Any]) -> str | None:
    """AND-join a mapping of field/value pairs into one filter expression.

    Values are compared as text, which is how GoodMem stores JSON scalars.
    Returns ``None`` for an empty mapping so callers can skip the filter.
    """
    clauses = [
        text_equals(field, str(value)) for field, value in metadata_filter.items()
    ]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return " AND ".join(f"({clause})" for clause in clauses)


def combine(*expressions: str | None) -> str | None:
    """AND-join already-built expressions, ignoring ``None``."""
    present = [e for e in expressions if e]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return " AND ".join(f"({e})" for e in present)
