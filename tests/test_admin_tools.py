"""Every admin tool, called the way an agent calls it, against the real SDK.

mock: on 0.2.1 ``goodmem_list_embedders`` and ``goodmem_list_rerankers``
raised ``TypeError: AsyncEmbeddersAPI.list() got an unexpected keyword
argument 'max_items'`` on every call, with goodmem 0.1.34 and 0.1.35 alike:
only ``spaces.list`` and ``memories.list`` paginate, and ``embedders.list()``
/ ``rerankers.list()`` take no ``max_items`` and return a plain list. An
agent given the admin toolset could not discover an embedder ID for
``goodmem_create_space``. No test called either tool, so nothing noticed.

Each case here runs one tool through ``FunctionTool.run_json`` -- the call
AutoGen's workbench makes -- over the real SDK and a mock transport, and
checks the request that went out and the JSON that came back. The coverage
test fails if a tool is added without a case.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from autogen_core import CancellationToken
from autogen_core.tools import StaticWorkbench
from autogen_goodmem import ADMIN_TOOL_NAMES, create_goodmem_admin_tools
import httpx
import pytest

from .conftest import (
    EMBEDDER_ID,
    MEMORY_ID,
    OTHER_EMBEDDER_ID,
    RERANKER_ID,
    SPACE_ID,
    Recorder,
    embedder_json,
    memory_json,
    reranker_json,
    space_json,
)


OTHER_RERANKER_ID = "d2a7c5e1-8f3b-4a9d-b6e2-1c4f8a3d7e59"


@dataclass
class Case:
    tool: str
    args: dict[str, Any]
    request: str
    routes: list[tuple[str, str, Any]]
    check: Callable[[dict[str, Any], Recorder], None]


def _ok(status: int, body: Any) -> httpx.Response:
    return httpx.Response(status, json=body) if body is not None else httpx.Response(status)


def _check_embedders(out: dict[str, Any], rec: Recorder) -> None:
    assert [e["embedder_id"] for e in out["embedders"]] == [EMBEDDER_ID, OTHER_EMBEDDER_ID]
    assert out["returned"] == 2
    assert out["truncated"] is False


def _check_rerankers(out: dict[str, Any], rec: Recorder) -> None:
    assert [r["reranker_id"] for r in out["rerankers"]] == [RERANKER_ID, OTHER_RERANKER_ID]
    assert out["returned"] == 2
    assert out["truncated"] is False


def _check_spaces(out: dict[str, Any], rec: Recorder) -> None:
    assert [s["space_id"] for s in out["spaces"]] == [SPACE_ID]
    assert rec.requests[0].url.params.get("name_filter") == "kb*"


def _check_space(out: dict[str, Any], rec: Recorder) -> None:
    assert out["space_id"] == SPACE_ID


def _check_create_space(out: dict[str, Any], rec: Recorder) -> None:
    assert out["space_id"] == SPACE_ID
    body = rec.body()
    assert body["name"] == "kb"
    assert body["spaceEmbedders"] == [{"embedderId": EMBEDDER_ID}]


def _check_update_space(out: dict[str, Any], rec: Recorder) -> None:
    assert out["space_id"] == SPACE_ID
    assert rec.body() == {"name": "renamed", "mergeLabels": {"team": "a"}}


def _check_delete_space(out: dict[str, Any], rec: Recorder) -> None:
    assert out == {"deleted": True, "space_id": SPACE_ID}


def _check_create_memory(out: dict[str, Any], rec: Recorder) -> None:
    assert out["memory_id"] == MEMORY_ID
    body = rec.body()
    assert body["spaceId"] == SPACE_ID
    assert body["originalContent"] == "hello world"
    assert body["contentType"] == "text/plain"
    assert body["metadata"] == {"category": "policy"}


def _check_list_memories(out: dict[str, Any], rec: Recorder) -> None:
    assert [m["memory_id"] for m in out["memories"]] == [MEMORY_ID]
    assert out["truncated"] is False


def _check_get_memory(out: dict[str, Any], rec: Recorder) -> None:
    assert out["memory_id"] == MEMORY_ID
    assert out["content"] == "hello world"


def _check_delete_memory(out: dict[str, Any], rec: Recorder) -> None:
    assert out == {"deleted": True, "memory_id": MEMORY_ID}


def _check_upload(out: dict[str, Any], rec: Recorder) -> None:
    assert out["memory_id"] == MEMORY_ID
    body = rec.body()
    assert body["spaceId"] == SPACE_ID
    assert body["originalContent"] == "hello from a file"
    assert body["metadata"] == {"title": "note.txt"}


CASES = [
    Case(
        "goodmem_list_embedders", {}, "GET /v1/embedders",
        [("GET", "/v1/embedders", {"embedders": [embedder_json(), embedder_json(OTHER_EMBEDDER_ID, "e2")]})],
        _check_embedders,
    ),
    Case(
        "goodmem_list_rerankers", {}, "GET /v1/rerankers",
        [("GET", "/v1/rerankers", {"rerankers": [reranker_json(), reranker_json(OTHER_RERANKER_ID, "r2")]})],
        _check_rerankers,
    ),
    Case(
        "goodmem_list_spaces", {"name_filter": "kb*"}, "GET /v1/spaces",
        [("GET", "/v1/spaces", {"spaces": [space_json(SPACE_ID, "kb")]})],
        _check_spaces,
    ),
    Case(
        "goodmem_get_space", {"space_id": SPACE_ID}, f"GET /v1/spaces/{SPACE_ID}",
        [("GET", f"/v1/spaces/{SPACE_ID}", space_json(SPACE_ID, "kb"))],
        _check_space,
    ),
    Case(
        "goodmem_create_space", {"name": "kb", "embedder_id": EMBEDDER_ID}, "POST /v1/spaces",
        [("POST", "/v1/spaces", space_json(SPACE_ID, "kb"))],
        _check_create_space,
    ),
    Case(
        "goodmem_update_space",
        {"space_id": SPACE_ID, "name": "renamed", "merge_labels": {"team": "a"}},
        f"PUT /v1/spaces/{SPACE_ID}",
        [("PUT", f"/v1/spaces/{SPACE_ID}", space_json(SPACE_ID, "renamed"))],
        _check_update_space,
    ),
    Case(
        "goodmem_delete_space", {"space_id": SPACE_ID}, f"DELETE /v1/spaces/{SPACE_ID}",
        [("DELETE", f"/v1/spaces/{SPACE_ID}", None)],
        _check_delete_space,
    ),
    Case(
        "goodmem_create_memory",
        {"space_id": SPACE_ID, "content": "hello world", "metadata": {"category": "policy"}},
        "POST /v1/memories",
        [("POST", "/v1/memories", memory_json(MEMORY_ID, status="PENDING"))],
        _check_create_memory,
    ),
    Case(
        "goodmem_list_memories", {"space_id": SPACE_ID}, f"GET /v1/spaces/{SPACE_ID}/memories",
        [("GET", f"/v1/spaces/{SPACE_ID}/memories", {"memories": [memory_json(MEMORY_ID)]})],
        _check_list_memories,
    ),
    Case(
        "goodmem_get_memory", {"memory_id": MEMORY_ID}, f"GET /v1/memories/{MEMORY_ID}",
        [("GET", f"/v1/memories/{MEMORY_ID}", memory_json(MEMORY_ID, content_b64="aGVsbG8gd29ybGQ="))],
        _check_get_memory,
    ),
    Case(
        "goodmem_delete_memory", {"memory_id": MEMORY_ID}, f"DELETE /v1/memories/{MEMORY_ID}",
        [("DELETE", f"/v1/memories/{MEMORY_ID}", None)],
        _check_delete_memory,
    ),
    Case(
        "goodmem_upload_file", {"space_id": SPACE_ID, "file_name": "note.txt"}, "POST /v1/memories",
        [("POST", "/v1/memories", memory_json(MEMORY_ID, status="PENDING"))],
        _check_upload,
    ),
]


@pytest.fixture
def upload_dir(tmp_path: Path) -> Path:
    (tmp_path / "note.txt").write_text("hello from a file")
    return tmp_path


def test_every_admin_tool_has_a_case(client: Any, upload_dir: Path) -> None:
    built = {t.name for t in create_goodmem_admin_tools(client, upload_dir=str(upload_dir))}
    assert built == set(ADMIN_TOOL_NAMES) | {"goodmem_upload_file"}
    assert {c.tool for c in CASES} == built


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.tool)
async def test_admin_tool_works_against_the_sdk(
    case: Case, client: Any, recorder: Recorder, upload_dir: Path
) -> None:
    for method, path, body in case.routes:
        recorder.route(method, path, _ok(200 if body is not None else 204, body))
    tools = {t.name: t for t in create_goodmem_admin_tools(client, upload_dir=str(upload_dir))}

    raw = await tools[case.tool].run_json(case.args, CancellationToken())

    assert recorder.paths() == [case.request]
    case.check(json.loads(str(raw)), recorder)


@pytest.mark.parametrize(
    ("tool", "path", "key", "factory"),
    [
        ("goodmem_list_embedders", "/v1/embedders", "embedders", embedder_json),
        ("goodmem_list_rerankers", "/v1/rerankers", "rerankers", reranker_json),
    ],
)
async def test_model_listing_honours_max_items(
    client: Any, recorder: Recorder, tool: str, path: str, key: str, factory: Callable[..., dict[str, Any]]
) -> None:
    """The SDK returns the whole list; ``max_items`` caps what the model sees."""
    ids = [f"00000000-0000-4000-8000-00000000000{i}" for i in range(3)]
    recorder.route("GET", path, httpx.Response(200, json={key: [factory(i) for i in ids]}))
    tools = {t.name: t for t in create_goodmem_admin_tools(client, max_items=2)}

    out = json.loads(str(await tools[tool].run_json({}, CancellationToken())))

    assert out["returned"] == 2
    assert out["truncated"] is True
    assert len(out[key]) == 2


async def test_list_embedders_through_the_workbench_is_not_an_error(client: Any, recorder: Recorder) -> None:
    """What the reported agent saw: an admin agent asking for embedders."""
    recorder.route("GET", "/v1/embedders", httpx.Response(200, json={"embedders": [embedder_json()]}))
    workbench = StaticWorkbench(create_goodmem_admin_tools(client))

    result = await workbench.call_tool("goodmem_list_embedders", {})

    assert result.is_error is False, result.to_text()
    assert EMBEDDER_ID in result.to_text()
