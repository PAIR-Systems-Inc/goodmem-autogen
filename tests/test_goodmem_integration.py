"""End-to-end integration tests for autogen-goodmem against a live server.

These tests require a reachable GoodMem instance. They are gated by the
``integration`` marker — run with ``pytest -m integration``.

Two test classes, 15 tests total:

- :class:`TestGoodMemClient`  — 13 tests, one per public client method, run in
  numeric order and sharing state through a module-level dict.
- :class:`TestGoodMemTools`   — 2 tests covering the tool factory and a full
  end-to-end round-trip exercising every tool.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

# Allow running this file directly out of the source tree without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autogen_goodmem import (  # noqa: E402
    GoodMemClient,
    GoodMemContextProvider,
    GoodMemMemoryConfig,
    TOOL_NAMES,
    create_goodmem_tools,
)


# ── Configuration ──────────────────────────────────────────────────────

GOODMEM_API_KEY = os.environ.get("GOODMEM_API_KEY", "GOODMEM_API_KEY_REDACTED")
GOODMEM_BASE_URL = os.environ.get("GOODMEM_BASE_URL", "https://localhost:8080")
EMBEDDER_ID = os.environ.get("GOODMEM_EMBEDDER_ID", "019cfd1c-c033-7517-b7de-f73941a0464b")
RERANKER_ID = os.environ.get("GOODMEM_RERANKER_ID", "019cfda4-7e2f-743c-9edb-e469a97b95c6")
LLM_ID = os.environ.get("GOODMEM_LLM_ID", "019cfd9f-0963-76f9-b069-4cde19a64ba8")
PDF_FILE_PATH = os.environ.get(
    "GOODMEM_PDF_PATH",
    "/home/bashar/Downloads/New Quran.com Search Analysis (Nov 26, 2025)-1.pdf",
)

# Distinct space names per run so concurrent test runs don't collide.
CLIENT_SPACE_NAME = f"autogen-goodmem-client-{uuid.uuid4().hex[:8]}"
TOOLS_SPACE_NAME = f"autogen-goodmem-tools-{uuid.uuid4().hex[:8]}"

# Pin every async test in this module to the same event loop so the shared
# module-scoped client fixture can be reused across tests (pytest-asyncio v1+).
pytestmark = pytest.mark.asyncio(loop_scope="module")


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client():
    c = GoodMemClient(
        base_url=GOODMEM_BASE_URL,
        api_key=GOODMEM_API_KEY,
        verify_ssl=False,
    )
    try:
        yield c
    finally:
        await c.close()


# ── Module-level shared state ──────────────────────────────────────────

state: dict = {}


# =====================================================================
# TestGoodMemClient: one test per public GoodMemClient method
# =====================================================================


@pytest.mark.integration
class TestGoodMemClient:
    """One test per public method on :class:`GoodMemClient`."""

    async def test_01_list_embedders(self, client: GoodMemClient) -> None:
        embedders = await client.list_embedders()
        assert isinstance(embedders, list)
        assert len(embedders) > 0
        first = embedders[0]
        assert "embedderId" in first
        print(f"\n  Found {len(embedders)} embedders. First: {first.get('displayName')} ({first.get('embedderId')})")

    async def test_02_create_space(self, client: GoodMemClient) -> None:
        result = await client.create_space(CLIENT_SPACE_NAME, EMBEDDER_ID)
        assert result["spaceId"]
        assert result["reused"] is False
        state["client_space_id"] = result["spaceId"]
        print(f"\n  create_space -> {result}")

        # Idempotency check
        again = await client.create_space(CLIENT_SPACE_NAME, EMBEDDER_ID)
        assert again["spaceId"] == result["spaceId"]
        assert again["reused"] is True
        print(f"  Idempotent reuse -> reused={again['reused']}")

    async def test_03_list_spaces(self, client: GoodMemClient) -> None:
        spaces = await client.list_spaces()
        assert isinstance(spaces, list)
        ours = [s for s in spaces if s.get("name") == CLIENT_SPACE_NAME]
        assert len(ours) == 1
        print(f"\n  Total spaces: {len(spaces)}; test space found.")

    async def test_04_get_space(self, client: GoodMemClient) -> None:
        space_id = state["client_space_id"]
        space = await client.get_space(space_id)
        assert space.get("spaceId") == space_id or space.get("id") == space_id
        assert space.get("name") == CLIENT_SPACE_NAME
        print(f"\n  get_space -> id={space_id} name={space.get('name')}")

    async def test_05_update_space(self, client: GoodMemClient) -> None:
        space_id = state["client_space_id"]
        new_name = CLIENT_SPACE_NAME + "-renamed"
        result = await client.update_space(
            space_id,
            name=new_name,
            merge_labels={"environment": "test", "framework": "autogen"},
        )
        assert result.get("name") == new_name
        print(f"\n  update_space -> renamed to {new_name}, labels merged.")
        # Rename back to keep cleanup simple
        await client.update_space(space_id, name=CLIENT_SPACE_NAME)

    async def test_06_create_memory_text(self, client: GoodMemClient) -> None:
        space_id = state["client_space_id"]
        result = await client.create_memory(
            space_id=space_id,
            content=(
                "AutoGen is a Microsoft Research framework for building multi-agent "
                "AI applications with conversational patterns and tool use."
            ),
            metadata={"category": "framework", "topic": "autogen"},
        )
        assert result.get("memoryId")
        state["text_memory_id"] = result["memoryId"]
        print(f"\n  create_memory (text) -> memoryId={result['memoryId']}")

    async def test_07_create_memory_pdf(self, client: GoodMemClient) -> None:
        if not os.path.exists(PDF_FILE_PATH):
            pytest.skip(f"PDF file not found at {PDF_FILE_PATH}")
        space_id = state["client_space_id"]
        result = await client.create_memory(space_id=space_id, file_path=PDF_FILE_PATH)
        assert result.get("memoryId")
        state["pdf_memory_id"] = result["memoryId"]
        print(
            f"\n  create_memory (pdf) -> memoryId={result['memoryId']}, "
            f"contentType={result.get('contentType')}, "
            f"status={result.get('processingStatus')}"
        )

    async def test_08_list_memories(self, client: GoodMemClient) -> None:
        space_id = state["client_space_id"]
        # Indexing takes a moment; the listing endpoint may still report 0
        # immediately after create_memory. Poll briefly.
        memories = []
        for _ in range(12):
            page = await client.list_memories(space_id)
            memories = page["memories"]
            if memories:
                break
            await asyncio.sleep(2)
        assert len(memories) > 0
        ids = {m.get("memoryId") for m in memories}
        assert state["text_memory_id"] in ids
        print(f"\n  list_memories -> {len(memories)} memories; nextToken={page.get('nextToken')}")

    async def test_09_retrieve_memories(self, client: GoodMemClient) -> None:
        space_id = state["client_space_id"]
        result = await client.retrieve_memories(
            message="What is AutoGen used for?",
            space_ids=[space_id],
            max_results=5,
            wait_for_indexing=True,
        )
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0
        top = result["results"][0]
        print(
            f"\n  retrieve_memories -> {len(result['results'])} results; "
            f"top score={top.get('relevanceScore')}; "
            f"chunk={str(top.get('chunkText'))[:120]!r}"
        )

    async def test_10_retrieve_with_reranker_and_llm(self, client: GoodMemClient) -> None:
        space_id = state["client_space_id"]
        result = await client.retrieve_memories(
            message="Summarize what AutoGen is for in one sentence.",
            space_ids=[space_id],
            max_results=5,
            wait_for_indexing=True,
            reranker_id=RERANKER_ID,
            llm_id=LLM_ID,
            relevance_threshold=0.0,
            llm_temperature=0.2,
        )
        abstract = result.get("abstractReply")
        print(f"\n  abstract={abstract is not None and bool(abstract)}")
        if isinstance(abstract, dict):
            print(f"  abstract text: {str(abstract.get('text') or abstract.get('reply') or abstract)[:200]!r}")
        else:
            print(f"  abstract: {str(abstract)[:200]!r}")
        assert abstract, "Expected an abstractReply when llm_id is provided"

    async def test_11_get_memory(self, client: GoodMemClient) -> None:
        memory_id = state.get("text_memory_id")
        assert memory_id
        result = await client.get_memory(memory_id, include_content=True)
        assert "memory" in result
        assert result["memory"].get("memoryId") == memory_id
        has_content = "content" in result
        print(
            f"\n  get_memory -> id={memory_id}, "
            f"status={result['memory'].get('processingStatus')}, "
            f"content_returned={has_content}"
        )

    async def test_12_delete_memory(self, client: GoodMemClient) -> None:
        memory_id = state["text_memory_id"]
        result = await client.delete_memory(memory_id)
        assert result["success"] is True
        assert result["memoryId"] == memory_id
        print(f"\n  delete_memory -> {memory_id} deleted")

    async def test_13_delete_space(self, client: GoodMemClient) -> None:
        # Delete the PDF memory first if it was created — keeps the server
        # tidy for the next test run.
        pdf_id = state.get("pdf_memory_id")
        if pdf_id:
            try:
                await client.delete_memory(pdf_id)
            except Exception as exc:  # noqa: BLE001
                print(f"  (non-fatal) failed to delete pdf memory: {exc}")
        space_id = state["client_space_id"]
        result = await client.delete_space(space_id)
        assert result["success"] is True
        print(f"\n  delete_space -> {space_id} deleted")


# =====================================================================
# TestGoodMemTools: tool surface + end-to-end round trip
# =====================================================================


@pytest.mark.integration
class TestGoodMemTools:
    async def test_all_tools_present(self, client: GoodMemClient) -> None:
        tools = create_goodmem_tools(client)
        names = [t.name for t in tools]
        print(f"\n  Tool names: {names}")
        assert names == TOOL_NAMES, f"Expected {TOOL_NAMES}, got {names}"
        assert len(tools) == 11

    async def test_tools_roundtrip(self, client: GoodMemClient) -> None:
        tools = {t.name: t for t in create_goodmem_tools(client)}

        # ``tool_name`` (not ``name``) so it doesn't collide with tools whose
        # own signature has a ``name`` argument (e.g. goodmem_create_space).
        async def call(tool_name: str, **kwargs):
            from autogen_core import CancellationToken

            tool = tools[tool_name]
            args = tool.args_type().model_validate(kwargs)
            return await tool.run(args, CancellationToken())

        space_id = None
        memory_id = None
        try:
            # create space
            created = await call(
                "goodmem_create_space",
                name=TOOLS_SPACE_NAME,
                embedder_id=EMBEDDER_ID,
            )
            assert created["spaceId"]
            space_id = created["spaceId"]
            print(f"\n  tool create_space -> {created}")

            # list spaces -> must contain our space
            all_spaces = await call("goodmem_list_spaces")
            assert any(s.get("name") == TOOLS_SPACE_NAME for s in all_spaces)

            # get_space
            got_space = await call("goodmem_get_space", space_id=space_id)
            assert got_space.get("name") == TOOLS_SPACE_NAME

            # update_space (rename + back)
            renamed = TOOLS_SPACE_NAME + "-r"
            await call("goodmem_update_space", space_id=space_id, name=renamed)
            await call("goodmem_update_space", space_id=space_id, name=TOOLS_SPACE_NAME)

            # create_memory (text)
            mem = await call(
                "goodmem_create_memory",
                space_id=space_id,
                content="The Eiffel Tower is in Paris and was completed in 1889.",
                metadata={"topic": "landmarks"},
            )
            assert mem.get("memoryId")
            memory_id = mem["memoryId"]
            print(f"  tool create_memory -> memoryId={memory_id}")

            # list_memories
            page = None
            for _ in range(12):
                page = await call("goodmem_list_memories", space_id=space_id)
                if page["memories"]:
                    break
                await asyncio.sleep(2)
            assert page and page["memories"], "Expected at least one memory after create"
            print(f"  tool list_memories -> {len(page['memories'])} memories")

            # retrieve_memories with reranker+LLM
            ret = await call(
                "goodmem_retrieve_memories",
                message="Where is the Eiffel Tower?",
                space_ids=[space_id],
                max_results=3,
                wait_for_indexing=True,
                reranker_id=RERANKER_ID,
                llm_id=LLM_ID,
                relevance_threshold=0.0,
                llm_temperature=0.2,
            )
            print(
                f"  tool retrieve_memories -> {len(ret['results'])} results; "
                f"abstract={ret.get('abstractReply') is not None and bool(ret.get('abstractReply'))}"
            )
            assert ret["results"], "Expected retrieval to return at least one chunk"
            assert ret.get("abstractReply"), "Expected an abstractReply with LLM enabled"

            # get_memory
            got = await call("goodmem_get_memory", memory_id=memory_id, include_content=True)
            assert got["memory"].get("memoryId") == memory_id

            # delete_memory
            del_m = await call("goodmem_delete_memory", memory_id=memory_id)
            assert del_m["success"] is True
            memory_id = None
            print("  tool delete_memory -> ok")

        finally:
            # Always clean up the space (and dangling memory if delete failed).
            if memory_id:
                try:
                    await client.delete_memory(memory_id)
                except Exception:
                    pass
            if space_id:
                try:
                    await client.delete_space(space_id)
                    print(f"  tool delete_space (cleanup) -> {space_id} deleted")
                except Exception as exc:  # noqa: BLE001
                    print(f"  cleanup delete_space failed: {exc}")
