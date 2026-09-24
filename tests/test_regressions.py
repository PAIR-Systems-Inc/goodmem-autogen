"""Regression tests. Each one pins a defect reproduced against v0.1.0.

The reproduction class is named in each docstring: "live" means reproduced
against a running GoodMem server, "mock" against a local HTTP server,
"local" with no server at all.
"""

from __future__ import annotations

import asyncio
import json

from autogen_core import CancellationToken
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import UserMessage
from autogen_goodmem import (
    GoodMemContextProvider,
    GoodMemMemoryConfig,
    GoodMemUploadError,
    PostProcessorConfig,
    create_goodmem_admin_tools,
    create_goodmem_search_tool,
)
import httpx
import pytest

from .conftest import (
    INFO_STATUS,
    REAL_VECTOR_SCORE,
    chunk_event,
    memory_event,
    memory_json,
    ndjson,
    space_json,
    status_event,
)


def make_provider(client, **overrides):
    config = GoodMemMemoryConfig(
        base_url="https://goodmem.test",
        api_key="gm_test_key",
        space_id="space-1",
        **overrides,
    )
    return GoodMemContextProvider(config=config, client=client)


# ------------------------------------------------------------------ P21
def test_api_key_is_not_serialized(client):
    """local: v0.1.0 wrote the key verbatim into dump_component(), which
    AssistantAgent._to_config() calls for every attached memory."""
    provider = make_provider(client)
    dumped = provider.dump_component().model_dump_json()
    assert "gm_test_key" not in dumped
    assert "**********" in dumped
    assert provider._config.api_key.get_secret_value() == "gm_test_key"


def test_api_key_is_not_in_repr(client):
    assert "gm_test_key" not in repr(make_provider(client)._config)


# ------------------------------------------------------------------ P13
async def test_clear_refuses_without_opt_in(client, recorder):
    """local: v0.1.0's clear() logged a warning and returned, having deleted
    nothing, while satisfying an @abstractmethod of the Memory ABC."""
    provider = make_provider(client)
    with pytest.raises(PermissionError, match="allow_clear=True"):
        await provider.clear()
    assert recorder.requests == [], "a refused clear must not touch the server"


async def test_clear_actually_deletes_when_allowed(client, recorder):
    recorder.route("GET", "/space-1/memories", httpx.Response(200, json={"memories": [memory_json("m1"), memory_json("m2")]}))
    recorder.route("DELETE", "/memories/m1", httpx.Response(204))
    recorder.route("DELETE", "/memories/m2", httpx.Response(204))
    provider = make_provider(client, allow_clear=True)
    await provider.clear()
    deletes = [p for p in recorder.paths() if p.startswith("DELETE")]
    assert len(deletes) == 2


# ------------------------------------------------------------------ P14
async def test_cancelled_token_prevents_the_request(client, recorder):
    """local: v0.1.0 accepted a cancellation_token and ignored it; an already
    cancelled token still issued the HTTP call."""
    token = CancellationToken()
    token.cancel()
    provider = make_provider(client)
    with pytest.raises(asyncio.CancelledError):
        await provider.query("anything", cancellation_token=token)
    assert recorder.requests == [], "nothing should go out once cancelled"


async def test_token_cancels_an_in_flight_request(client, recorder):
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return ndjson()

    recorder.route("POST", ":retrieve", slow)
    token = CancellationToken()
    provider = make_provider(client)
    task = asyncio.create_task(provider.query("q", cancellation_token=token))
    await asyncio.sleep(0.05)
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ------------------------------------------------------------------ P4/P3
async def test_failed_reranking_is_reported_not_hidden(client, recorder):
    """live: server sent NOT_FOUND + RERANKING_FAILED; v0.1.0 returned the
    fallback chunks with no trace of either."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            status_event("NOT_FOUND", "Reranker not found"),
            INFO_STATUS,
            status_event("RERANKING_FAILED", "Failed to create reranker client"),
            memory_event("m1", {"title": "doc"}),
            chunk_event("c1", "fallback text", "m1"),
        ),
    )
    provider = make_provider(client, post_processor=PostProcessorConfig(reranker_id="r-missing"))
    result = await provider.query("q")

    assert len(result.results) == 1
    meta = result.results[0].metadata
    assert meta["partial"] is True
    assert [s["code"] for s in meta["statuses"]] == ["NOT_FOUND", "RERANKING_FAILED"]


async def test_total_failure_is_empty_and_warns_not_raises(client, recorder):
    """Contract Q4b. MemoryQueryResult has nowhere to carry a flag on an empty
    list, so the failure surfaces as a warning carrying the statuses."""
    recorder.route("POST", ":retrieve", ndjson(status_event("VECTOR_SEARCH_FAILED", "backend down")))
    with pytest.warns(UserWarning, match="VECTOR_SEARCH_FAILED"):
        result = await make_provider(client).query("q")
    assert result.results == []


async def test_feature_disabled_is_informational_whatever_its_details(client, recorder):
    """Contract Q1: the code alone decides. The server defines FEATURE_DISABLED
    as 'disabled due to missing configuration', so no details check."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            status_event("FEATURE_DISABLED", "Reranking disabled: no reranker configured.",
                         feature="reranking", required_param="reranker_id"),
            memory_event("m1"),
            chunk_event("c1", "text", "m1"),
        ),
    )
    result = await make_provider(client).query("q")
    assert result.results[0].metadata["partial"] is False
    assert "statuses" not in result.results[0].metadata


async def test_informational_notice_is_not_treated_as_a_failure(client, recorder):
    recorder.route("POST", ":retrieve", ndjson(INFO_STATUS, memory_event("m1"), chunk_event("c1", "text", "m1")))
    result = await make_provider(client).query("q")
    assert result.results[0].metadata["partial"] is False
    assert "statuses" not in result.results[0].metadata


async def test_unknown_future_status_does_not_discard_results(client, recorder):
    """A code from a newer server decodes to None; it must be surfaced without
    throwing away the chunks that did arrive."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(memory_event("m1"), chunk_event("c1", "still useful", "m1"),
               status_event("A_CODE_FROM_THE_FUTURE", "hello")),
    )
    result = await make_provider(client).query("q")
    assert len(result.results) == 1
    meta = result.results[0].metadata
    assert meta["partial"] is True
    assert meta["statuses"][0]["code"] == "UNKNOWN"
    assert meta["statuses"][0]["unrecognized"] is True


# ------------------------------------------------------------------ P30
async def test_query_carries_the_memorys_own_metadata(client, recorder):
    """live: v0.1.0 parsed memoryDefinition then dropped it, so stored
    title/category never reached the agent."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(chunk_event("c1", "text", "m1"),
               memory_event("m1", {"title": "Quarterly report", "category": "finance"})),
    )
    result = await make_provider(client).query("q")
    meta = result.results[0].metadata
    assert meta["title"] == "Quarterly report"
    assert meta["category"] == "finance"
    assert meta["memory_id"] == "m1"
    assert meta["chunk_id"] == "c1"


async def test_two_chunks_of_one_memory_are_two_results(client, recorder):
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(memory_event("m1"), chunk_event("c1", "first", "m1"), chunk_event("c2", "second", "m1")),
    )
    result = await make_provider(client).query("q")
    assert {str(r.content) for r in result.results} == {"first", "second"}


# ------------------------------------------------------------------ P29
async def test_real_vector_scores_are_negative_and_preserved(client, recorder):
    """live capture: a real vector relevanceScore is -0.5345."""
    recorder.route("POST", ":retrieve", ndjson(memory_event("m1"), chunk_event("c1", "text", "m1")))
    result = await make_provider(client).query("q")
    meta = result.results[0].metadata
    assert meta["score"] == REAL_VECTOR_SCORE
    assert meta["score_kind"] == "vector"


async def test_relevance_threshold_requires_a_reranker(client, recorder):
    provider = make_provider(client, post_processor=PostProcessorConfig(relevance_threshold=0.5))
    with pytest.raises(ValueError, match="needs a reranker_id"):
        await provider.query("q")
    assert recorder.requests == []


# ------------------------------------------------------------------ P5
async def test_query_makes_exactly_one_request(client, recorder):
    """live: the provider default wait_for_indexing=True made an empty-space
    query poll for 64.2s."""
    recorder.route("POST", ":retrieve", ndjson())
    result = await make_provider(client).query("nothing here")
    assert result.results == []
    assert len([p for p in recorder.paths() if p.endswith(":retrieve")]) == 1


async def test_add_waits_for_its_own_memory_never_by_searching(client, recorder):
    polls = {"n": 0}

    def get_memory(request: httpx.Request) -> httpx.Response:
        polls["n"] += 1
        status = "PENDING" if polls["n"] < 3 else "COMPLETED"
        return httpx.Response(200, json=memory_json("m-new", status=status))

    recorder.route("POST", "/v1/memories", httpx.Response(201, json=memory_json("m-new", status="PENDING")))
    recorder.route("GET", "/memories/m-new", get_memory)
    provider = make_provider(client)
    await provider.add(MemoryContent(content="hello", mime_type=MemoryMimeType.TEXT))

    assert polls["n"] == 3
    assert not any(p.endswith(":retrieve") for p in recorder.paths()), "waiting never searches"


# ------------------------------------------------------------------ P32/P6
async def test_space_reuse_requires_a_matching_embedder(client, recorder):
    """live: v0.1.0 reused a same-named space and reported back the embedder
    that was *asked for*, not the one the space actually uses."""
    recorder.route("GET", "/v1/spaces", httpx.Response(200, json={"spaces": [space_json("s-existing", "shared", embedder_id="emb-OTHER")]}))
    provider = GoodMemContextProvider(
        config=GoodMemMemoryConfig(base_url="https://goodmem.test", api_key="k",
                                   space_name="shared", embedder_id="emb-WANTED"),
        client=client,
    )
    with pytest.raises(ValueError, match="different embedder"):
        await provider._ensure_space()


async def test_space_reuse_succeeds_when_the_embedder_matches(client, recorder):
    recorder.route("GET", "/v1/spaces", httpx.Response(200, json={"spaces": [space_json("s-existing", "shared", embedder_id="emb-1")]}))
    provider = GoodMemContextProvider(
        config=GoodMemMemoryConfig(base_url="https://goodmem.test", api_key="k",
                                   space_name="shared", embedder_id="emb-1"),
        client=client,
    )
    assert await provider._ensure_space() == "s-existing"


async def test_ambiguous_space_name_is_an_error(client, recorder):
    recorder.route("GET", "/v1/spaces", httpx.Response(200, json={"spaces": [space_json("a", "dup"), space_json("b", "dup")]}))
    provider = GoodMemContextProvider(
        config=GoodMemMemoryConfig(base_url="https://goodmem.test", api_key="k",
                                   space_name="dup", embedder_id="emb-1"),
        client=client,
    )
    with pytest.raises(ValueError, match="accessible spaces are named"):
        await provider._ensure_space()


async def test_space_lookup_uses_a_server_side_name_filter(client, recorder):
    """v0.1.0 listed page one and scanned it in Python."""
    recorder.route("GET", "/v1/spaces", httpx.Response(200, json={"spaces": [space_json("s1", "wanted")]}))
    provider = GoodMemContextProvider(
        config=GoodMemMemoryConfig(base_url="https://goodmem.test", api_key="k",
                                   space_name="wanted", embedder_id="emb-1"),
        client=client,
    )
    await provider._ensure_space()
    assert "name_filter=wanted" in str(recorder.requests[0].url)


# ------------------------------------------------------------------ P10
async def test_upload_refuses_paths_outside_the_configured_directory(client, tmp_path):
    """local: v0.1.0's file_path was a model-facing tool argument with no
    restriction; it read /etc/passwd and the caller's goodmem config."""
    allowed = tmp_path / "uploads"
    allowed.mkdir()
    (allowed / "ok.txt").write_text("fine")
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE")

    provider = make_provider(client, upload_dir=str(allowed))
    for escape in ("../secret.txt", str(secret), "/etc/passwd"):
        with pytest.raises(GoodMemUploadError):
            await provider.add_file(escape)


async def test_upload_is_disabled_until_a_directory_is_configured(client):
    with pytest.raises(GoodMemUploadError, match="upload_dir"):
        await make_provider(client).add_file("anything.txt")


def test_admin_tools_omit_file_upload_unless_a_directory_is_given(client):
    names = {t.name for t in create_goodmem_admin_tools(client)}
    assert "goodmem_upload_file" not in names
    with_dir = {t.name for t in create_goodmem_admin_tools(client, upload_dir="/tmp")}
    assert "goodmem_upload_file" in with_dir


def test_create_memory_tool_takes_no_file_path(client):
    tools = {t.name: t for t in create_goodmem_admin_tools(client)}
    props = tools["goodmem_create_memory"].schema["parameters"]["properties"]
    assert "file_path" not in props


# ------------------------------------------------------------------ P2
def test_update_space_tool_has_no_public_read(client):
    """live: sending publicRead fails the whole update with HTTP 400."""
    tools = {t.name: t for t in create_goodmem_admin_tools(client)}
    props = tools["goodmem_update_space"].schema["parameters"]["properties"]
    assert "public_read" not in props
    assert set(props) <= {"space_id", "name", "merge_labels", "replace_labels"}


async def test_update_space_sends_only_known_fields(client, recorder):
    recorder.route("PUT", "/spaces/s1", httpx.Response(200, json=space_json("s1", "n")))
    tools = {t.name: t for t in create_goodmem_admin_tools(client)}
    await tools["goodmem_update_space"].run_json({"space_id": "s1", "name": "n"}, CancellationToken())
    assert "publicRead" not in recorder.body()


# ------------------------------------------------------------------ P28
def test_search_tool_exposes_only_the_query(client):
    """v0.1.0's retrieve tool took ten arguments, including wait_for_indexing
    and llm_temperature, all model-chosen."""
    tool = create_goodmem_search_tool(client, space_ids=["s1"])
    assert list(tool.schema["parameters"]["properties"]) == ["query"]


async def test_search_tool_configuration_is_not_model_reachable(client, recorder):
    recorder.route("POST", ":retrieve", ndjson())
    tool = create_goodmem_search_tool(
        client, space_ids=["configured"], filter="CAST(val('$.a') AS TEXT) = 'b'"
    )
    await tool.run_json({"query": "q"}, CancellationToken())
    body = recorder.body()
    assert body["spaceKeys"][0]["spaceId"] == "configured"
    assert body["spaceKeys"][0]["filter"] == "CAST(val('$.a') AS TEXT) = 'b'"


# ------------------------------------------------------------------ P16
async def test_get_memory_returns_readable_text_in_one_request(client, recorder):
    """live: v0.1.0 made a second call to /content and reported success with
    the content missing when it failed."""
    recorder.route("GET", "/memories/m1", httpx.Response(200, json=memory_json("m1", content_b64="aGVsbG8gd29ybGQ=")))
    tools = {t.name: t for t in create_goodmem_admin_tools(client)}
    raw = await tools["goodmem_get_memory"].run_json({"memory_id": "m1"}, CancellationToken())
    payload = json.loads(str(raw))
    assert payload["content"] == "hello world"
    assert "original_content" not in payload
    assert len(recorder.requests) == 1


async def test_get_memory_describes_binary_rather_than_dumping_base64(client, recorder):
    recorder.route("GET", "/memories/m2", httpx.Response(200, json=memory_json("m2", content_b64="JVBERi0xLjQK", content_type="application/pdf")))
    tools = {t.name: t for t in create_goodmem_admin_tools(client)}
    payload = json.loads(str(await tools["goodmem_get_memory"].run_json({"memory_id": "m2"}, CancellationToken())))
    assert "content" not in payload
    assert "application/pdf" in payload["content_omitted"]


# ------------------------------------------------------------- update_context
async def test_update_context_injects_a_system_message(client, recorder):
    """Mutating the passed context is the documented AutoGen pattern, which
    ListMemory and every bundled ext memory also follow."""
    recorder.route("POST", ":retrieve", ndjson(memory_event("m1"), chunk_event("c1", "the answer is 42", "m1")))
    context = UnboundedChatCompletionContext()
    await context.add_message(UserMessage(content="what is the answer?", source="user"))
    result = await make_provider(client).update_context(context)

    assert len(result.memories.results) == 1
    messages = await context.get_messages()
    assert any("the answer is 42" in str(m.content) for m in messages)


async def test_update_context_ignores_a_non_text_turn(client, recorder):
    """v0.1.0 fell back to str(message), searching for the repr of a message
    object rather than any real query."""
    context = UnboundedChatCompletionContext()
    # A structured (list) turn: valid for UserMessage, but not a plain string.
    await context.add_message(UserMessage(content=["look at this", "and this"], source="user"))
    result = await make_provider(client).update_context(context)
    assert result.memories.results == []
    assert recorder.requests == [], "no search should be issued for a non-text turn"


async def test_a_threshold_that_may_have_emptied_the_result_warns(client, recorder):
    """The server applies relevance_threshold, so the dropped scores are not
    visible here. Measured live 2026-09-24: Voyage rerank-2.5 0.27..0.93 and
    Jina jina-reranker-v3 -0.14..0.43 on the same documents, so a threshold
    tuned for one empties the other. Nothing back + nothing reported must
    not pass as a plain miss."""
    recorder.route("POST", ":retrieve", ndjson(memory_event("m1")))
    provider = make_provider(
        client, post_processor=PostProcessorConfig(reranker_id="jina", relevance_threshold=0.6)
    )
    with pytest.warns(UserWarning, match="relevance_threshold=0.6 may have removed"):
        result = await provider.query("q")
    assert result.results == []
