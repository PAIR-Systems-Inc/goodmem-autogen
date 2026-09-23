"""Retrieval event handling: statuses, chunk/memory joining, scores."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from goodmem.models.good_mem_status import GoodMemStatus
from goodmem.models.retrieve_memory_event import RetrieveMemoryEvent


# Notices that carry no loss of results.
#
# FEATURE_DISABLED is informational unconditionally. The server defines it as
# "feature disabled due to missing configuration" (common.proto, under
# "Informational status messages (non-error)"): the caller did not configure
# an optional feature, so nothing the caller asked for is missing. A feature
# that was requested and could not be delivered arrives as a different code
# (NOT_FOUND, RERANKING_FAILED, ...). Retrieval status contract, Q1.
_INFORMATIONAL_CODES = frozenset({"LLM_CAPABILITY_INFERRED", "FEATURE_DISABLED"})


def is_informational(status: GoodMemStatus) -> bool:
    """True for notices that do not indicate incomplete retrieval.

    An unrecognized code is deliberately not informational: the SDK decodes
    codes it does not know as ``None``, and a status from a newer server must
    be surfaced rather than assumed harmless.
    """
    return status.code is not None and status.code in _INFORMATIONAL_CODES


def classify(events: Sequence[RetrieveMemoryEvent]) -> tuple[list[dict[str, Any]], bool]:
    """Split statuses into what to report and whether results are incomplete.

    Known informational notices are dropped. Known failures mark the result
    degraded. Codes this SDK does not recognize are surfaced as ``UNKNOWN``
    and mark the result degraded, but never discard chunks and never raise:
    a newer server must not be able to break retrieval.
    """
    surfaced: list[dict[str, Any]] = []
    degraded = False
    for event in events:
        status = event.status
        if status is None or is_informational(status):
            continue
        entry = status.model_dump(exclude_none=True)
        if status.code is None:
            entry["code"] = "UNKNOWN"
            entry["unrecognized"] = True
        surfaced.append(entry)
        degraded = True
    return surfaced, degraded


def hits_from_events(
    events: Iterable[RetrieveMemoryEvent], *, reranked: bool
) -> list[dict[str, Any]]:
    """Join chunks to their memory definitions by UUID, ignoring event order.

    Deduplicates by ``chunk_id``: two chunks of one memory are two distinct
    results, and collapsing them by ``memory_id`` would drop matching content.

    Server ordering and raw scores are preserved. ``score_kind`` records where
    a score came from, because a reranker score and a vector score are not on
    the same scale and must not be compared or thresholded together.
    """
    events = list(events)
    memories = {
        event.memory_definition.memory_id: event.memory_definition
        for event in events
        if event.memory_definition is not None
    }

    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        item = event.retrieved_item
        if item is None or item.chunk is None:
            continue
        reference = item.chunk
        chunk = reference.chunk
        if chunk is None or not chunk.chunk_text:
            continue
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)

        memory = memories.get(chunk.memory_id) or item.memory
        metadata: dict[str, Any] = dict(getattr(memory, "metadata", None) or {})
        hits.append(
            {
                "chunk_text": chunk.chunk_text,
                "chunk_id": chunk.chunk_id,
                "memory_id": chunk.memory_id,
                "space_id": getattr(memory, "space_id", None),
                "source": getattr(memory, "original_content_ref", None) or chunk.memory_id,
                "score": reference.relevance_score,
                "score_kind": "reranker" if reranked else "vector",
                # The memory's own metadata, which is what a caller stored.
                "metadata": metadata,
            }
        )
    return hits


def abstract_reply(events: Iterable[RetrieveMemoryEvent]) -> dict[str, Any] | None:
    for event in events:
        if event.abstract_reply is not None:
            return event.abstract_reply.model_dump(exclude_none=True)
    return None
