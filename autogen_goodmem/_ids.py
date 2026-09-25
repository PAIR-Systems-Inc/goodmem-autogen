"""GoodMem IDs: an ID that reaches a URL path must be a canonical UUID.

The goodmem SDK interpolates IDs into request paths unescaped
(``f"/v1/memories/{id}"``) and httpx resolves dot segments before sending, so
``memory_id="../spaces/<id>"`` goes out as ``DELETE /v1/spaces/<id>``: a
delete-memory call deletes a whole space. ``?`` and ``#`` cut the path short
in the same way, and the server normalises ``%2e%2e`` too, so neither end can
be relied on to stop it.

Every GoodMem ID (space, memory, embedder, reranker, LLM) is a UUID, so the
rule is an allow-list rather than an escape: ``8-4-4-4-12`` hex is accepted
and lowercased, and anything else is refused before a request is built.
:func:`require_uuid` is the one check. It is called right before each SDK
call that takes an ID, whatever the ID's source -- a model's tool call, a
developer's argument, configuration, or a value the server returned.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    PlainValidator,
    WithJsonSchema,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema


UUID_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

_UUID = re.compile(UUID_PATTERN)

_UUID_JSON_SCHEMA: JsonSchemaValue = {"type": "string", "format": "uuid", "pattern": UUID_PATTERN}


def require_uuid(value: object, field: str) -> str:
    """Return ``value`` as a lowercase UUID, or raise ``ValueError`` naming ``field``.

    ``fullmatch``, not ``match``: ``$`` also matches before a trailing
    newline, so ``match`` would let ``"<uuid>\\n"`` through.
    """
    if isinstance(value, str) and _UUID.fullmatch(value):
        return value.lower()
    raise ValueError(
        f"{field} must be a UUID, like 123e4567-e89b-12d3-a456-426614174000. "
        "GoodMem IDs are UUIDs; nothing was sent."
    )


def _validate(value: object, info: core_schema.ValidationInfo) -> str:
    return require_uuid(value, info.field_name or "id")


class GoodMemId(str):
    """Annotation for a tool argument that is a GoodMem ID.

    The tool's JSON schema then declares ``format: uuid`` and the pattern, so
    the model is told what an ID looks like, and AutoGen checks the argument
    with :func:`require_uuid` before the tool runs. The value the tool
    receives is a plain lowercase ``str``.

    A class rather than ``Annotated``: AutoGen builds a tool's argument model
    from the parameter annotations and reads ``Annotated`` metadata only as a
    description string, so a constraint has to travel on the type itself.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.with_info_plain_validator_function(_validate)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return dict(_UUID_JSON_SCHEMA)


UuidStr = Annotated[str, PlainValidator(_validate), WithJsonSchema(_UUID_JSON_SCHEMA)]
"""A config field that is a GoodMem ID; typed as ``str`` for callers."""


__all__ = ["UUID_PATTERN", "GoodMemId", "UuidStr", "require_uuid"]
