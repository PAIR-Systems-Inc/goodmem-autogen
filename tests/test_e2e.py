"""Live tests against a running GoodMem server.

Opt in with GOODMEM_BASE_URL and GOODMEM_API_KEY. There is no default key:
an unset environment skips rather than reaching for a committed credential.

Every space created here is registered for deletion the moment it is created,
deletion is verified, and teardown failures are reported rather than swallowed.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

from autogen_core import CancellationToken
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_goodmem import (
    GoodMemContextProvider,
    GoodMemMemoryConfig,
    GoodMemUploadError,
    PostProcessorConfig,
    create_goodmem_admin_tools,
    create_goodmem_search_tool,
)
from goodmem import AsyncGoodmem
import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.environ.get("GOODMEM_BASE_URL") and os.environ.get("GOODMEM_API_KEY")),
        reason="GOODMEM_BASE_URL / GOODMEM_API_KEY not set",
    ),
]

BASE = os.environ.get("GOODMEM_BASE_URL", "")
KEY = os.environ.get("GOODMEM_API_KEY", "")
EMBEDDER = os.environ.get("GOODMEM_EMBEDDER_ID", "")
VERIFY = os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() != "false"


@pytest.fixture
async def client():
    c = AsyncGoodmem(base_url=BASE, api_key=KEY, verify=VERIFY)
    try:
        yield c
    finally:
        await c.close()


@pytest.fixture
async def space(client):
    """A real space, deleted and verified gone afterwards."""
    created = await client.spaces.create(
        name=f"autogen-e2e-{uuid.uuid4().hex[:10]}",
        space_embedders=[{"embedderId": EMBEDDER}],
    )
    space_id = created.space_id
    try:
        yield space_id
    finally:
        await client.spaces.delete(id=space_id)
        remaining = [s.space_id async for s in await client.spaces.list(max_items=1000)]
        assert space_id not in remaining, f"cleanup failed: {space_id} still present"


@pytest.fixture
async def provider(client, space):
    p = GoodMemContextProvider(
        config=GoodMemMemoryConfig(
            base_url=BASE, api_key=KEY, space_id=space, verify_ssl=VERIFY
        ),
        client=client,
    )
    yield p
    await p.close()


@pytest.fixture
async def indexed(provider):
    await provider.add(
        MemoryContent(
            content="The internal audit codeword is ZEPHYR-7. Revenue rose in the north.",
            mime_type=MemoryMimeType.TEXT,
            metadata={"title": "audit note", "category": "audit"},
        )
    )
    return provider


async def test_add_waits_so_query_finds_it_immediately(indexed):
    """v0.1.0 polled the search for up to 60s hoping indexing had finished."""
    started = asyncio.get_running_loop().time()
    result = await indexed.query("what is the audit codeword?")
    elapsed = asyncio.get_running_loop().time() - started

    assert result.results, "the memory was written and waited for; it must be findable"
    assert "ZEPHYR-7" in str(result.results[0].content)
    assert elapsed < 5, f"query should be one request, took {elapsed:.1f}s"


async def test_query_carries_stored_metadata_and_score(indexed):
    result = await indexed.query("audit codeword")
    meta = result.results[0].metadata
    assert meta["title"] == "audit note"
    assert meta["category"] == "audit"
    assert meta["score_kind"] == "vector"
    assert meta["partial"] is False
    assert meta["chunk_id"] and meta["memory_id"]


async def test_empty_space_answers_immediately(provider):
    """v0.1.0 spent 64.2s polling a space that was simply empty."""
    started = asyncio.get_running_loop().time()
    result = await provider.query("nothing has been stored here")
    elapsed = asyncio.get_running_loop().time() - started

    assert result.results == []
    assert elapsed < 5, f"empty is an answer; took {elapsed:.1f}s"


async def test_invalid_reranker_is_reported_not_hidden(client, space, indexed):
    """v0.1.0 returned unreranked chunks and reported nothing wrong."""
    provider = GoodMemContextProvider(
        config=GoodMemMemoryConfig(
            base_url=BASE, api_key=KEY, space_id=space, verify_ssl=VERIFY,
            post_processor=PostProcessorConfig(
                reranker_id="00000000-0000-0000-0000-000000000000"
            ),
        ),
        client=client,
    )
    result = await provider.query("audit codeword")
    meta = result.results[0].metadata
    assert meta["partial"] is True
    codes = {s.get("code") for s in meta["statuses"]}
    assert codes & {"RERANKING_FAILED", "NOT_FOUND"}, codes


async def test_update_space_succeeds_without_public_read(client, space):
    tools = {t.name: t for t in create_goodmem_admin_tools(client)}
    raw = await tools["goodmem_update_space"].run_json(
        {"space_id": space, "merge_labels": {"audited": "yes"}}, CancellationToken()
    )
    assert json.loads(str(raw))["labels"]["audited"] == "yes"


async def test_search_tool_returns_passages(client, space, indexed):
    tool = create_goodmem_search_tool(client, space_ids=[space], limit=3)
    payload = json.loads(str(await tool.run_json({"query": "codeword"}, CancellationToken())))
    assert payload["total_results"] >= 1
    assert payload["partial"] is False
    assert "ZEPHYR-7" in payload["results"][0]["chunk_text"]


async def test_metadata_filter_is_applied_server_side(client, space, indexed):
    hit = create_goodmem_search_tool(client, space_ids=[space], metadata_filter={"category": "audit"})
    miss = create_goodmem_search_tool(client, space_ids=[space], metadata_filter={"category": "nope"})
    assert json.loads(str(await hit.run_json({"query": "codeword"}, CancellationToken())))["total_results"] >= 1
    assert json.loads(str(await miss.run_json({"query": "codeword"}, CancellationToken())))["total_results"] == 0


async def test_filter_value_with_an_apostrophe_is_escaped(client, space, indexed):
    """Live proof that the escaping we emit is the one the grammar accepts."""
    tool = create_goodmem_search_tool(client, space_ids=[space], metadata_filter={"category": "o'brien"})
    payload = json.loads(str(await tool.run_json({"query": "codeword"}, CancellationToken())))
    assert payload["total_results"] == 0


async def test_get_memory_returns_readable_text(client, space, indexed):
    result = await indexed.query("codeword")
    memory_id = result.results[0].metadata["memory_id"]
    tools = {t.name: t for t in create_goodmem_admin_tools(client)}
    payload = json.loads(str(await tools["goodmem_get_memory"].run_json({"memory_id": memory_id}, CancellationToken())))
    assert "ZEPHYR-7" in payload["content"]


async def test_upload_refuses_outside_dir_and_accepts_inside(client, space, tmp_path):
    allowed = tmp_path / "ok"
    allowed.mkdir()
    (allowed / "note.txt").write_text("Uploaded through the approved directory.")
    provider = GoodMemContextProvider(
        config=GoodMemMemoryConfig(
            base_url=BASE, api_key=KEY, space_id=space, verify_ssl=VERIFY,
            upload_dir=str(allowed),
        ),
        client=client,
    )
    with pytest.raises(GoodMemUploadError):
        await provider.add_file("/etc/passwd")
    result = await provider.add_file("note.txt")
    assert result["status"] == "COMPLETED"
    await provider.close()


async def test_space_reuse_requires_matching_embedder(client, space):
    """Attach by name to an existing space whose embedder differs."""
    name = (await client.spaces.get(id=space)).name
    provider = GoodMemContextProvider(
        config=GoodMemMemoryConfig(
            base_url=BASE, api_key=KEY, space_name=name,
            embedder_id="00000000-0000-0000-0000-000000000000", verify_ssl=VERIFY,
        ),
        client=client,
    )
    with pytest.raises(ValueError, match="different embedder"):
        await provider._ensure_space()
    await provider.close()


async def test_clear_requires_opt_in_then_empties_the_space(client, space, indexed):
    with pytest.raises(PermissionError):
        await indexed.clear()

    allowed = GoodMemContextProvider(
        config=GoodMemMemoryConfig(
            base_url=BASE, api_key=KEY, space_id=space, verify_ssl=VERIFY, allow_clear=True
        ),
        client=client,
    )
    await allowed.clear()
    remaining = [m async for m in await client.memories.list(space_id=space)]
    assert remaining == [], "clear() must actually empty the space"
    await allowed.close()
