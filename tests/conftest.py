"""Test support: drive the real SDK over a mock HTTP transport.

Tests exercise the SDK's own request building and NDJSON parsing. Nothing
inside the integration or the SDK is monkeypatched, so a passing test means
the wire behaviour is right rather than that a stub was called.

Event shapes come from ``fixtures/retrieve_real.ndjson``, captured from a live
GoodMem server (v1.0.320), not written from a guess at the schema.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from goodmem import AsyncGoodmem
import httpx
import pytest


FIXTURES = Path(__file__).parent / "fixtures"

# A real vector relevance score straight from the capture. It is negative:
# GoodMem vector scores are opaque similarities, not 0-1 relevance.
REAL_VECTOR_SCORE = -0.5345187187194824


def real_events() -> list[dict[str, Any]]:
    text = (FIXTURES / "retrieve_real.ndjson").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _template(kind: str) -> dict[str, Any]:
    for event in real_events():
        if kind in event:
            return event
    raise AssertionError(f"fixture has no {kind} event")


def chunk_event(
    chunk_id: str, text: str, memory_id: str, score: float = REAL_VECTOR_SCORE
) -> dict[str, Any]:
    event = json.loads(json.dumps(_template("retrievedItem")))
    reference = event["retrievedItem"]["chunk"]
    reference["chunk"]["chunkId"] = chunk_id
    reference["chunk"]["chunkText"] = text
    reference["chunk"]["memoryId"] = memory_id
    reference["relevanceScore"] = score
    return event


def memory_event(memory_id: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    event = json.loads(json.dumps(_template("memoryDefinition")))
    event["memoryDefinition"]["memoryId"] = memory_id
    event["memoryDefinition"]["metadata"] = metadata or {}
    return event


def status_event(code: str | None, message: str, **details: str) -> dict[str, Any]:
    """A status event. ``code=None`` omits the field; an invented code
    simulates a newer server, which the SDK decodes as ``None``."""
    status: dict[str, Any] = {"message": message}
    if code is not None:
        status["code"] = code
    if details:
        status["details"] = details
    return {"status": status}


INFO_STATUS = status_event(
    "FEATURE_DISABLED",
    "Abstract reply generation disabled",
    feature="summarization",
    required_param="llm_id",
)


def ndjson(*objects: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        text="\n".join(json.dumps(o) for o in objects),
        headers={"content-type": "application/x-ndjson"},
    )


def space_json(space_id: str, name: str, embedder_id: str = "emb-1") -> dict[str, Any]:
    """A Space with the fields the server really returns."""
    return {
        "spaceId": space_id,
        "name": name,
        "labels": {},
        "spaceEmbedders": [
            {
                "spaceId": space_id,
                "embedderId": embedder_id,
                "defaultRetrievalWeight": 1.0,
                "createdAt": 1789607045012,
                "updatedAt": 1789607045012,
                "createdById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
                "updatedById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
            }
        ],
        "createdAt": 1789607045012,
        "updatedAt": 1789607045012,
        "ownerId": "019cfcff-37c5-76d0-bd46-8525e29a9c82",
        "createdById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
        "updatedById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
        "defaultChunkingConfig": {
            "recursive": {
                "chunkSize": 512,
                "chunkOverlap": 64,
                "separators": [],
                "keepStrategy": "KEEP_END",
                "separatorIsRegex": False,
                "lengthMeasurement": "CHARACTER_COUNT",
            }
        },
    }


def memory_json(
    memory_id: str,
    *,
    content_b64: str | None = None,
    content_type: str = "text/plain",
    status: str = "COMPLETED",
) -> dict[str, Any]:
    """A Memory as the server returns it. ``originalContent`` is base64."""
    payload: dict[str, Any] = {
        "memoryId": memory_id,
        "spaceId": "space-1",
        "originalContentLength": 11,
        "originalContentSha256": "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        "contentType": content_type,
        "processingStatus": status,
        "pageImageStatus": "PENDING",
        "pageImageCount": 0,
        "metadata": {"title": "t"},
        "createdAt": 1789607045060,
        "updatedAt": 1789607047968,
        "createdById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
        "updatedById": "019cfcff-37c7-75ef-be71-06c83dae99c3",
    }
    if content_b64 is not None:
        payload["originalContent"] = content_b64
    return payload


async def _resolve(response: Any, request: httpx.Request) -> httpx.Response:
    """A route may be a Response, a function, or an async function."""
    if callable(response):
        result = response(request)
        if inspect.isawaitable(result):
            return await result
        return result
    return response


class Recorder:
    """Records outgoing requests and serves queued responses."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: list[tuple[str, str, Any]] = []
        self.default: Any = None

    def route(self, method: str, path_suffix: str, response: Any) -> None:
        self.routes.append((method.upper(), path_suffix, response))

    async def handler(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        for method, suffix, response in self.routes:
            if request.method == method and request.url.path.endswith(suffix):
                return await _resolve(response, request)
        if self.default is not None:
            return await _resolve(self.default, request)
        return httpx.Response(
            404, json={"error": f"unrouted {request.method} {request.url.path}"}
        )

    def body(self, index: int = -1) -> dict[str, Any]:
        content = self.requests[index].content
        return json.loads(content) if content else {}

    def paths(self) -> list[str]:
        return [f"{r.method} {r.url.path}" for r in self.requests]


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def client(recorder: Recorder) -> AsyncGoodmem:
    """An async SDK client whose transport is the recorder.

    base_url and the API key live on the httpx client: the SDK refuses to
    take them alongside an injected http_client.
    """
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(recorder.handler),
        base_url="https://goodmem.test",
        headers={"x-api-key": "test-key"},
    )
    return AsyncGoodmem(http_client=http_client)
