"""A failed reranker leaves vector-scored hits, and they are labelled so.

mock: on 0.2.1 ``score_kind`` was decided from configuration
(``reranked = bool(reranker_id)``). When the reranker fails the server
still returns the vector-search hits -- the fixture is what a live server
(v1.0.320) sent for a missing reranker: ``NOT_FOUND``, ``FEATURE_DISABLED``,
``RERANKING_FAILED``, then two hits scored -0.5689 and -0.5608. Both the
search tool and ``query()`` returned those hits labelled
``score_kind: "reranker"``, telling a caller that vector similarities were
reranker relevance scores.

Contract Q4a: problem + hits returns the hits with ``partial`` and the
statuses. Whether they were reranked is read from the response.
"""

from __future__ import annotations

import json
from typing import Any
import warnings

from autogen_core import CancellationToken
from autogen_goodmem import (
    GoodMemContextProvider,
    GoodMemMemoryConfig,
    PostProcessorConfig,
    create_goodmem_search_tool,
)
import pytest

from .conftest import (
    RERANKER_ID,
    SPACE_ID,
    Recorder,
    chunk_event,
    fixture_events,
    memory_event,
    ndjson,
    status_event,
)


FALLBACK_SCORES = [-0.5688997507095337, -0.5608013272285461]
MEMORY_A = "01a0d101-ea63-73b5-972e-335499daffc3"


def _provider(client: Any, **post: Any) -> GoodMemContextProvider:
    return GoodMemContextProvider(
        config=GoodMemMemoryConfig(
            base_url="https://goodmem.test",
            api_key="gm_test_key",
            space_id=SPACE_ID,
            post_processor=PostProcessorConfig(reranker_id=RERANKER_ID, **post),
        ),
        client=client,
    )


async def test_search_tool_labels_fallback_hits_as_vector(client: Any, recorder: Recorder) -> None:
    recorder.route("POST", ":retrieve", fixture_events("retrieve_broken_reranker.ndjson"))
    tool = create_goodmem_search_tool(client, space_ids=[SPACE_ID], reranker_id=RERANKER_ID)

    out = json.loads(str(await tool.run_json({"query": "noodle soup"}, CancellationToken())))

    assert [h["score"] for h in out["results"]] == FALLBACK_SCORES
    assert [h["score_kind"] for h in out["results"]] == ["vector", "vector"]
    assert out["partial"] is True
    assert [s["code"] for s in out["statuses"]] == ["NOT_FOUND", "RERANKING_FAILED"]
    # The reranker was still requested; only the label follows the response.
    assert recorder.body()["postProcessor"]["config"]["reranker_id"] == RERANKER_ID


async def test_query_labels_fallback_hits_as_vector_and_keeps_them(client: Any, recorder: Recorder) -> None:
    recorder.route("POST", ":retrieve", fixture_events("retrieve_broken_reranker.ndjson"))
    provider = _provider(client, relevance_threshold=0.5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await provider.query("noodle soup")

    metas = [r.metadata or {} for r in result.results]
    assert [m["score"] for m in metas] == FALLBACK_SCORES
    assert [m["score_kind"] for m in metas] == ["vector", "vector"]
    assert all(m["partial"] is True for m in metas)
    assert [s["code"] for s in metas[0]["statuses"]] == ["NOT_FOUND", "RERANKING_FAILED"]
    # Nothing about a reranker threshold: no reranker scores exist to judge.
    assert not [w for w in caught if "relevance_threshold" in str(w.message)]


@pytest.mark.parametrize(
    "status",
    [
        status_event("RERANKING_FAILED", "Reranker timed out"),
        status_event("NOT_FOUND", "Reranker not found: x"),
        status_event("NOT_FOUND", "not found", reranker_id=RERANKER_ID),
    ],
    ids=["reranking-failed", "not-found-by-message", "not-found-by-detail"],
)
async def test_either_reranker_failure_status_means_vector(
    client: Any, recorder: Recorder, status: dict[str, Any]
) -> None:
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(status, memory_event(MEMORY_A), chunk_event("c1", "text", MEMORY_A, score=-0.2715)),
    )
    result = await _provider(client).query("q")

    meta = result.results[0].metadata or {}
    assert (meta["score"], meta["score_kind"], meta["partial"]) == (-0.2715, "vector", True)


async def test_a_reranker_that_ran_is_still_labelled_reranker(client: Any, recorder: Recorder) -> None:
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(memory_event(MEMORY_A), chunk_event("c1", "text", MEMORY_A, score=0.82)),
    )
    tool = create_goodmem_search_tool(client, space_ids=[SPACE_ID], reranker_id=RERANKER_ID)

    out = json.loads(str(await tool.run_json({"query": "q"}, CancellationToken())))

    assert out["results"][0]["score_kind"] == "reranker"
    assert out["partial"] is False


async def test_an_unrelated_not_found_does_not_relabel(client: Any, recorder: Recorder) -> None:
    """A space that is missing is not a reranker that failed."""
    recorder.route(
        "POST",
        ":retrieve",
        ndjson(
            status_event("NOT_FOUND", "Space not found", space_id=SPACE_ID),
            memory_event(MEMORY_A),
            chunk_event("c1", "text", MEMORY_A, score=0.82),
        ),
    )
    result = await _provider(client).query("q")

    meta = result.results[0].metadata or {}
    assert (meta["score_kind"], meta["partial"]) == ("reranker", True)
