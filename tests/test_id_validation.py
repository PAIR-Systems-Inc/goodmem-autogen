"""IDs that reach a URL path are canonical UUIDs, refused before any request.

mock: reproduced against 0.2.0 with a local HTTP server. The goodmem SDK
interpolates IDs into paths unescaped (``f"/v1/memories/{id}"``) and httpx
resolves dot segments before sending, so
``goodmem_delete_memory(memory_id="../spaces/<U>")`` went out as
``DELETE /v1/spaces/<U>`` and the tool reported ``{"deleted": true}``. On
0.2.0, 357 of the 396 hostile cases below reached the server; the other 39
were stopped by accident rather than by an ID check (httpx rejects a
newline in a URL; an empty ID trips the SDK's or the config's emptiness
check).

These tests run the real SDK and httpx over a real socket to a local server
that records every request line exactly as it arrived. Every entry point
that takes an ID -- model-facing tools, developer arguments, configuration,
and IDs the server hands back -- is driven with each hostile value; it must
be refused and the server must have received nothing. A valid UUID must
still reach exactly the intended path.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from typing import Any
from urllib.parse import urlsplit

from autogen_core import CancellationToken
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.tools import FunctionTool, StaticWorkbench
from autogen_goodmem import (
    GoodMemContextProvider,
    GoodMemMemoryConfig,
    PostProcessorConfig,
    create_goodmem_admin_tools,
    create_goodmem_search_tool,
)
from goodmem import AsyncGoodmem
import pytest

from .conftest import memory_json, space_json


# The ID a traversal is aimed at, and a legitimate one.
U = "3f1c2a4e-9b7d-4e2f-8a61-0c5d7e9b1a23"
OK = "7d9e4b12-5c3a-4f8e-b1d6-2a0e9c8f7b45"
EMBEDDER = "0b6f2d8e-1a4c-4e7b-9d3f-5c8a2e6b4f10"
MEMORY = "c4e8a2f6-3b7d-4a1e-8f5c-9d2b6e0a7c31"

UUID_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

HOSTILE = [
    f"../spaces/{U}",
    f"a/../../spaces/{U}",
    f"%2e%2e/spaces/{U}",
    f"..%2Fspaces%2F{U}",
    f"{U}/../../spaces/{U}",
    "",
    f" {U}",
    f"{U}?x=1",
    f"{U}#frag",
    # `$` also matches before a trailing newline; a regex check that uses
    # match() instead of fullmatch() lets this one through.
    f"{U}\n",
    f"{U}/",
    "..",
]


# ------------------------------------------------------------ the server
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _RecordingServer

    def _serve(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        with self.server.lock:
            self.server.requests.append((self.command, self.path, body))
        status, payload = self.server.answer(self.command, urlsplit(self.path).path)
        if payload is None:
            data, content_type = b"", "application/json"
        elif isinstance(payload, bytes):
            data, content_type = payload, "application/x-ndjson"
        else:
            data, content_type = json.dumps(payload).encode(), "application/json"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_PUT = do_DELETE = _serve

    def log_message(self, *args: Any) -> None:
        pass


class _RecordingServer(ThreadingHTTPServer):
    """Answers like GoodMem would and records each request line verbatim."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.lock = threading.Lock()
        self.requests: list[tuple[str, str, bytes]] = []
        self.overrides: dict[tuple[str, str], tuple[int, Any]] = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def lines(self) -> list[str]:
        """Request lines as received: method and raw target, query included."""
        return [f"{method} {target}" for method, target, _ in self.requests]

    def paths(self) -> list[str]:
        return [f"{method} {urlsplit(target).path}" for method, target, _ in self.requests]

    def body(self, index: int = -1) -> Any:
        return json.loads(self.requests[index][2] or b"{}")

    def answer(self, method: str, path: str) -> tuple[int, Any]:
        if (method, path) in self.overrides:
            return self.overrides[(method, path)]
        if method == "DELETE":
            return 204, None
        if method == "POST" and path == "/v1/memories:retrieve":
            return 200, b""
        if method == "POST" and path == "/v1/memories":
            return 200, memory_json(MEMORY)
        if method == "POST" and path == "/v1/spaces":
            return 200, space_json(OK, "kb", embedder_id=EMBEDDER)
        if method == "GET" and path == "/v1/spaces":
            return 200, {"spaces": []}
        if method == "GET" and re.fullmatch(r"/v1/spaces/[^/]+/memories", path):
            return 200, {"memories": []}
        if match := re.fullmatch(r"/v1/spaces/([^/]+)", path):
            return 200, space_json(match[1], "kb", embedder_id=EMBEDDER)
        if method == "GET" and (match := re.fullmatch(r"/v1/memories/([^/]+)", path)):
            return 200, memory_json(match[1])
        return 404, {"error": f"unrouted {method} {path}"}


@pytest.fixture
def server() -> Iterator[_RecordingServer]:
    srv = _RecordingServer()
    # A short poll interval: shutdown() otherwise waits up to 0.5s per test.
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


@dataclass
class Ctx:
    server: _RecordingServer
    client: AsyncGoodmem
    upload_dir: Path
    closers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)

    def config(self, **overrides: Any) -> GoodMemMemoryConfig:
        values: dict[str, Any] = {"base_url": self.server.url, "api_key": "gm_test_key"}
        values.update(overrides)
        return GoodMemMemoryConfig(**values)

    def provider(self, config: GoodMemMemoryConfig) -> GoodMemContextProvider:
        # No injected client: the provider builds its own from the config,
        # which is the path a deployed agent takes.
        provider = GoodMemContextProvider(config=config)
        self.closers.append(provider.close)
        return provider


@pytest.fixture
async def ctx(server: _RecordingServer, tmp_path: Path) -> Any:
    client = AsyncGoodmem(base_url=server.url, api_key="gm_test_key")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "note.txt").write_text("hello")
    context = Ctx(server=server, client=client, upload_dir=upload_dir)
    try:
        yield context
    finally:
        for close in context.closers:
            await close()
        await client.close()


# ------------------------------------------------------- the entry points
def _tools(ctx: Ctx) -> dict[str, FunctionTool]:
    tools = create_goodmem_admin_tools(ctx.client, upload_dir=str(ctx.upload_dir))
    return {t.name: t for t in tools}


def _tool(name: str, args: Callable[[str], dict[str, Any]], *, schema: bool) -> Callable[[Ctx, str], Awaitable[Any]]:
    """Call an admin tool the way a model does.

    ``schema=True`` goes through ``run_json``, which validates arguments
    against the tool's schema first. ``schema=False`` builds the arguments
    without validation, so only the check at the SDK call stands between
    the value and the request.
    """

    async def call(ctx: Ctx, value: str) -> Any:
        tool = _tools(ctx)[name]
        if schema:
            return await tool.run_json(args(value), CancellationToken())
        unchecked = tool.args_type().model_construct(**args(value))
        return await tool.run(unchecked, CancellationToken())

    return call


async def _search_space_ids(ctx: Ctx, value: str) -> Any:
    tool = create_goodmem_search_tool(ctx.client, space_ids=[value])
    return await tool.run_json({"query": "q"}, CancellationToken())


async def _search_reranker(ctx: Ctx, value: str) -> Any:
    tool = create_goodmem_search_tool(ctx.client, space_ids=[OK], reranker_id=value)
    return await tool.run_json({"query": "q"}, CancellationToken())


async def _query_space_ids(ctx: Ctx, value: str) -> Any:
    return await ctx.provider(ctx.config(space_id=OK)).query("q", space_ids=[value])


async def _config_space_clear(ctx: Ctx, value: str) -> Any:
    await ctx.provider(ctx.config(space_id=value, allow_clear=True)).clear()


async def _config_space_query(ctx: Ctx, value: str) -> Any:
    return await ctx.provider(ctx.config(space_id=value)).query("q")


async def _config_space_add(ctx: Ctx, value: str) -> Any:
    provider = ctx.provider(ctx.config(space_id=value))
    await provider.add(MemoryContent(content="hello", mime_type=MemoryMimeType.TEXT))


async def _config_space_add_file(ctx: Ctx, value: str) -> Any:
    provider = ctx.provider(ctx.config(space_id=value, upload_dir=str(ctx.upload_dir)))
    return await provider.add_file("note.txt")


async def _config_embedder(ctx: Ctx, value: str) -> Any:
    provider = ctx.provider(ctx.config(space_name="kb", embedder_id=value))
    await provider.add(MemoryContent(content="hello", mime_type=MemoryMimeType.TEXT))


async def _config_reranker(ctx: Ctx, value: str) -> Any:
    config = ctx.config(space_id=OK, post_processor=PostProcessorConfig(reranker_id=value))
    return await ctx.provider(config).query("q")


async def _config_llm(ctx: Ctx, value: str) -> Any:
    config = ctx.config(space_id=OK, post_processor=PostProcessorConfig(llm_id=value))
    return await ctx.provider(config).query("q")


async def _component_json(ctx: Ctx, value: str) -> Any:
    """A component config as it arrives from a file or a web request."""
    provider = GoodMemContextProvider.load_component(
        {
            "provider": "autogen_goodmem.GoodMemContextProvider",
            "component_type": "memory",
            "config": {
                "base_url": ctx.server.url,
                "api_key": "gm_test_key",
                "space_id": value,
                "allow_clear": True,
            },
        }
    )
    ctx.closers.append(provider.close)
    await provider.clear()


async def _config_changed_after_validation(ctx: Ctx, value: str) -> Any:
    """Pydantic does not re-validate on assignment; the call must still check."""
    config = ctx.config(space_id=OK, allow_clear=True)
    config.space_id = value
    await ctx.provider(config).clear()


async def _server_listed_memory(ctx: Ctx, value: str) -> Any:
    ctx.server.overrides[("GET", f"/v1/spaces/{OK}/memories")] = (
        200,
        {"memories": [memory_json(value)]},
    )
    await ctx.provider(ctx.config(space_id=OK, allow_clear=True)).clear()


async def _server_created_memory(ctx: Ctx, value: str) -> Any:
    ctx.server.overrides[("POST", "/v1/memories")] = (200, memory_json(value))
    provider = ctx.provider(ctx.config(space_id=OK))
    await provider.add(MemoryContent(content="hello", mime_type=MemoryMimeType.TEXT))


async def _server_named_space(ctx: Ctx, value: str) -> Any:
    ctx.server.overrides[("GET", "/v1/spaces")] = (
        200,
        {"spaces": [space_json(value, "kb", embedder_id=EMBEDDER)]},
    )
    config = ctx.config(space_name="kb", embedder_id=EMBEDDER, allow_clear=True)
    await ctx.provider(config).clear()


def _body_space_key(ctx: Ctx) -> str:
    return str(ctx.server.body()["spaceKeys"][0]["spaceId"])


def _body_space_id(ctx: Ctx) -> str:
    return str(ctx.server.body()["spaceId"])


@dataclass
class Entry:
    name: str
    source: str
    call: Callable[[Ctx, str], Awaitable[Any]]
    # Request lines for a valid ID; "{id}" is where it must land.
    expect: list[str]
    # For a body ID, where to read it back from the last request.
    in_body: Callable[[Ctx], str] | None = None
    # Requests that legitimately come first because they return the ID.
    prelude: list[str] = field(default_factory=list)


def _admin_entries(schema: bool) -> list[Entry]:
    label = "tool" if schema else "tool-unvalidated"

    def entry(name: str, args: Callable[[str], dict[str, Any]], expect: list[str], in_body: Any = None) -> Entry:
        return Entry(f"{label}:{name}", "model", _tool(name, args, schema=schema), expect, in_body)

    return [
        entry("goodmem_get_space", lambda v: {"space_id": v}, ["GET /v1/spaces/{id}"]),
        entry("goodmem_update_space", lambda v: {"space_id": v, "name": "renamed"}, ["PUT /v1/spaces/{id}"]),
        entry("goodmem_delete_space", lambda v: {"space_id": v}, ["DELETE /v1/spaces/{id}"]),
        entry("goodmem_list_memories", lambda v: {"space_id": v}, ["GET /v1/spaces/{id}/memories"]),
        entry("goodmem_get_memory", lambda v: {"memory_id": v}, ["GET /v1/memories/{id}"]),
        entry("goodmem_delete_memory", lambda v: {"memory_id": v}, ["DELETE /v1/memories/{id}"]),
        entry(
            "goodmem_create_memory",
            lambda v: {"space_id": v, "content": "hello"},
            ["POST /v1/memories"],
            _body_space_id,
        ),
        entry(
            "goodmem_create_space",
            lambda v: {"name": "kb", "embedder_id": v},
            ["POST /v1/spaces"],
            lambda c: str(c.server.body()["spaceEmbedders"][0]["embedderId"]),
        ),
        entry(
            "goodmem_upload_file",
            lambda v: {"space_id": v, "file_name": "note.txt"},
            ["POST /v1/memories"],
            _body_space_id,
        ),
    ]


ENTRIES: list[Entry] = [
    *_admin_entries(schema=True),
    *_admin_entries(schema=False),
    Entry("search_tool:space_ids", "developer", _search_space_ids, ["POST /v1/memories:retrieve"], _body_space_key),
    Entry(
        "search_tool:reranker_id",
        "developer",
        _search_reranker,
        ["POST /v1/memories:retrieve"],
        lambda c: str(c.server.body()["postProcessor"]["config"]["reranker_id"]),
    ),
    Entry("provider.query:space_ids", "developer", _query_space_ids, ["POST /v1/memories:retrieve"], _body_space_key),
    Entry("config.space_id:clear", "config", _config_space_clear, ["GET /v1/spaces/{id}/memories"]),
    Entry("config.space_id:query", "config", _config_space_query, ["POST /v1/memories:retrieve"], _body_space_key),
    Entry(
        "config.space_id:add",
        "config",
        _config_space_add,
        ["POST /v1/memories", f"GET /v1/memories/{MEMORY}"],
        lambda c: str(c.server.body(0)["spaceId"]),
    ),
    Entry(
        "config.space_id:add_file",
        "config",
        _config_space_add_file,
        ["POST /v1/memories", f"GET /v1/memories/{MEMORY}"],
        lambda c: str(c.server.body(0)["spaceId"]),
    ),
    Entry(
        "config.embedder_id:add",
        "config",
        _config_embedder,
        ["GET /v1/spaces", "POST /v1/spaces", "POST /v1/memories", f"GET /v1/memories/{MEMORY}"],
        lambda c: str(c.server.body(1)["spaceEmbedders"][0]["embedderId"]),
    ),
    Entry(
        "config.post_processor.reranker_id:query",
        "config",
        _config_reranker,
        ["POST /v1/memories:retrieve"],
        lambda c: str(c.server.body()["postProcessor"]["config"]["reranker_id"]),
    ),
    Entry(
        "config.post_processor.llm_id:query",
        "config",
        _config_llm,
        ["POST /v1/memories:retrieve"],
        lambda c: str(c.server.body()["postProcessor"]["config"]["llm_id"]),
    ),
    Entry("load_component(json).space_id:clear", "config", _component_json, ["GET /v1/spaces/{id}/memories"]),
    Entry(
        "config.space_id-assigned-later:clear",
        "config",
        _config_changed_after_validation,
        ["GET /v1/spaces/{id}/memories"],
    ),
    Entry(
        "server-listed memory_id:clear",
        "internal",
        _server_listed_memory,
        [f"GET /v1/spaces/{OK}/memories", "DELETE /v1/memories/{id}"],
        prelude=[f"GET /v1/spaces/{OK}/memories"],
    ),
    Entry(
        "server-created memory_id:add",
        "internal",
        _server_created_memory,
        ["POST /v1/memories", "GET /v1/memories/{id}"],
        prelude=["POST /v1/memories"],
    ),
    Entry(
        "server-named space_id:clear",
        "internal",
        _server_named_space,
        ["GET /v1/spaces", "GET /v1/spaces/{id}/memories"],
        prelude=["GET /v1/spaces"],
    ),
]


# ------------------------------------------------------------------ tests
@pytest.mark.parametrize("value", HOSTILE, ids=repr)
@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
async def test_a_non_uuid_id_is_refused_before_any_request(ctx: Ctx, entry: Entry, value: str) -> None:
    outcome: Any
    try:
        outcome = await entry.call(ctx, value)
    except Exception as exc:  # noqa: BLE001 - kept for the message; the asserts judge it
        outcome = exc
    # The server first: what reached it is the defect, whatever came back.
    assert ctx.server.paths() == entry.prelude, (
        f"{entry.name}({value!r}): server received {ctx.server.lines()}; caller got {outcome!r}"
    )
    assert isinstance(outcome, ValueError), f"{entry.name}({value!r}) was not refused: {outcome!r}"
    assert "must be a UUID" in str(outcome)


@pytest.mark.parametrize("value", [OK, OK.upper()], ids=["lower", "upper"])
@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
async def test_a_valid_uuid_reaches_exactly_the_intended_path(ctx: Ctx, entry: Entry, value: str) -> None:
    await entry.call(ctx, value)
    assert ctx.server.paths() == [line.replace("{id}", OK) for line in entry.expect]
    if entry.in_body is not None:
        assert entry.in_body(ctx) == OK


async def test_the_reported_delete_memory_traversal_is_an_error_to_the_model(ctx: Ctx) -> None:
    """The reproduction from the report, through AutoGen's own workbench:
    0.2.0 deleted the space and told the model {"deleted": true}."""
    workbench = StaticWorkbench(list(_tools(ctx).values()))
    result = await workbench.call_tool(
        "goodmem_delete_memory", {"memory_id": f"../spaces/{U}"}, CancellationToken()
    )
    text = result.to_text()
    assert result.is_error, text
    assert "memory_id" in text
    assert "UUID" in text
    assert "deleted" not in text
    assert ctx.server.lines() == []


def test_id_arguments_are_declared_as_uuids_in_the_tool_schema(ctx: Ctx) -> None:
    """The model is told what an ID looks like; the check at the call is
    what enforces it."""
    id_params = {
        "goodmem_get_space": ["space_id"],
        "goodmem_update_space": ["space_id"],
        "goodmem_delete_space": ["space_id"],
        "goodmem_create_space": ["embedder_id"],
        "goodmem_create_memory": ["space_id"],
        "goodmem_list_memories": ["space_id"],
        "goodmem_get_memory": ["memory_id"],
        "goodmem_delete_memory": ["memory_id"],
        "goodmem_upload_file": ["space_id"],
    }
    tools = _tools(ctx)
    for tool_name, params in id_params.items():
        props = tools[tool_name].schema["parameters"]["properties"]
        for param in params:
            assert props[param].get("format") == "uuid", (tool_name, param, props[param])
            assert props[param].get("pattern") == UUID_PATTERN, (tool_name, param, props[param])
    # Every other string argument is left alone.
    assert "pattern" not in tools["goodmem_create_space"].schema["parameters"]["properties"]["name"]


def test_config_id_fields_are_declared_as_uuids() -> None:
    def declared(schema: dict[str, Any]) -> bool:
        options = schema.get("anyOf", [schema])
        return any(o.get("format") == "uuid" and o.get("pattern") == UUID_PATTERN for o in options)

    memory = GoodMemMemoryConfig.model_json_schema()
    for name in ("space_id", "embedder_id"):
        assert declared(memory["properties"][name]), memory["properties"][name]
    post = PostProcessorConfig.model_json_schema()
    for name in ("reranker_id", "llm_id"):
        assert declared(post["properties"][name]), post["properties"][name]


def test_require_uuid_is_an_allow_list() -> None:
    from autogen_goodmem._ids import UUID_PATTERN as package_pattern, require_uuid

    assert package_pattern == UUID_PATTERN
    assert require_uuid(OK.upper(), "space_id") == OK
    for value in [*HOSTILE, None, 42, OK.encode(), OK.replace("-", ""), f"{{{OK}}}", f"urn:uuid:{OK}"]:
        with pytest.raises(ValueError, match=r"^space_id must be a UUID"):
            require_uuid(value, "space_id")
