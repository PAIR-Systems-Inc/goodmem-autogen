"""Every Python snippet in the README runs as written.

mock: on 0.2.1 the "As tools" quickstart -- which is also the PyPI long
description -- raised ``NameError: name 'client' is not defined``: it passed
``client`` to ``create_goodmem_search_tool`` without ever creating one or
naming ``goodmem.AsyncGoodmem``.

Each ```python block is executed, top-level ``await`` included, against a
local HTTP server that answers like GoodMem (retrieval replays a live
capture). The only substitutions are the ones a reader makes: the example
server URL, the API key, and the ``<...-uuid>`` placeholders.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import json
from pathlib import Path
import re
import threading
from typing import Any
from urllib.parse import urlsplit

import pytest

from .conftest import (
    EMBEDDER_ID,
    FIXTURES,
    MEMORY_ID,
    RERANKER_ID,
    SPACE_ID,
    memory_json,
    space_json,
)


README = Path(__file__).resolve().parent.parent / "README.md"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
PLACEHOLDERS = {
    "<space-uuid>": SPACE_ID,
    "<embedder-uuid>": EMBEDDER_ID,
    "<reranker-uuid>": RERANKER_ID,
}


class _Handler(BaseHTTPRequestHandler):
    server: _GoodMemLike

    def _reply(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        self.rfile.read(length)
        path = urlsplit(self.path).path
        self.server.requests.append(f"{self.command} {path}")
        status, body, content_type = self.server.answer(self.command, path)
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_PUT = do_DELETE = _reply

    def log_message(self, *args: Any) -> None:
        pass


class _GoodMemLike(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[str] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    def answer(self, method: str, path: str) -> tuple[int, Any, str]:
        js = "application/json"
        if method == "POST" and path == "/v1/memories:retrieve":
            return 200, (FIXTURES / "retrieve_real.ndjson").read_bytes(), "application/x-ndjson"
        if method == "GET" and path == "/v1/spaces":
            return 200, {"spaces": []}, js
        if method == "POST" and path == "/v1/spaces":
            return 200, space_json(SPACE_ID, "handbook"), js
        if method == "POST" and path == "/v1/memories":
            return 200, memory_json(MEMORY_ID, status="PENDING"), js
        if method == "GET" and path == f"/v1/memories/{MEMORY_ID}":
            return 200, memory_json(MEMORY_ID), js
        return 404, {"error": f"unrouted {method} {path}"}, js


@pytest.fixture
def server() -> Iterator[_GoodMemLike]:
    srv = _GoodMemLike()
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def _blocks() -> list[str]:
    return BLOCK.findall(README.read_text())


def _as_run(source: str, url: str) -> str:
    source = source.replace("https://goodmem.example.com", url).replace('"gm_..."', '"gm_test_key"')
    for placeholder, value in PLACEHOLDERS.items():
        source = source.replace(placeholder, value)
    left = re.findall(r"<[a-z-]+>", source)
    assert not left, f"unknown placeholders {left}"
    return source


def test_the_readme_has_the_snippets_this_file_expects() -> None:
    blocks = _blocks()
    assert len(blocks) == 3
    assert "GoodMemContextProvider(" in blocks[0]
    assert "create_goodmem_search_tool(" in blocks[2]


@pytest.mark.parametrize("index", range(3), ids=["memory", "result-metadata", "tools"])
async def test_readme_snippet_runs(index: int, server: _GoodMemLike, capsys: Any) -> None:
    code = compile(
        _as_run(_blocks()[index], server.url),
        f"README.md[python block {index}]",
        "exec",
        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
    )
    namespace: dict[str, Any] = {"__name__": "__readme__"}
    result = eval(code, namespace)
    if inspect.iscoroutine(result):
        await result

    if index == 0:
        assert "POST /v1/memories:retrieve" in server.requests
        assert namespace["results"].results, "the query found nothing"
    if index == 2:
        assert server.requests == ["POST /v1/memories:retrieve"]
        printed = json.loads(capsys.readouterr().out)
        assert printed["results"], printed
