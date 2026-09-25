"""AutoGen ``FunctionTool`` factories for GoodMem.

Two factories, because the two jobs have very different blast radii:

* :func:`create_goodmem_search_tool` — one read-only search tool. The model
  supplies a query; spaces, reranking and filters are set by the developer,
  so a model cannot redirect the search or widen it mid-run.
* :func:`create_goodmem_admin_tools` — space and memory management. These
  carry the authority of the configured API key. Give them only to agents
  that need them.

AutoGen feeds a tool exception's ``str()`` straight back to the model
(``StaticWorkbench._format_errors``), so messages here stay free of
credentials and internal detail.

Every ID argument is a :class:`~autogen_goodmem._ids.GoodMemId`, declared as a
UUID in the tool schema, and is checked again with ``require_uuid`` at the SDK
call: the SDK puts IDs into request paths unescaped, so ``"../spaces/<id>"``
passed as a memory ID would otherwise address a space.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

from autogen_core.tools import FunctionTool

from ._ids import GoodMemId, require_uuid
from ._results import abstract_reply, classify, hits_from_events
from ._typing import AsyncGoodmemClient
from ._uploads import resolve_upload_path
from .filters import combine, from_mapping


SEARCH_TOOL_NAME = "goodmem_search"

ADMIN_TOOL_NAMES: list[str] = [
    "goodmem_list_embedders",
    "goodmem_list_rerankers",
    "goodmem_list_spaces",
    "goodmem_get_space",
    "goodmem_create_space",
    "goodmem_update_space",
    "goodmem_delete_space",
    "goodmem_create_memory",
    "goodmem_list_memories",
    "goodmem_get_memory",
    "goodmem_delete_memory",
]


def create_goodmem_search_tool(
    client: AsyncGoodmemClient,
    *,
    space_ids: list[str],
    limit: int = 5,
    fetch_k: int | None = None,
    reranker_id: str | None = None,
    filter: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    name: str = SEARCH_TOOL_NAME,
    description: str | None = None,
) -> FunctionTool:
    """A search tool whose only model-supplied argument is the query."""
    if not space_ids:
        raise ValueError("space_ids must not be empty.")
    # Body fields, not paths, but checked the same way: refused here rather
    # than found out at the first search.
    space_ids = [require_uuid(s, "space_ids") for s in space_ids]
    if reranker_id is not None:
        reranker_id = require_uuid(reranker_id, "reranker_id")
    expression = combine(filter, from_mapping(metadata_filter) if metadata_filter else None)
    reranked = bool(reranker_id)

    async def goodmem_search(query: str) -> str:
        """Search stored knowledge for passages relevant to a question."""
        if not query.strip():
            raise ValueError("query must not be empty")

        call: dict[str, Any] = {
            "message": query,
            "requested_size": fetch_k or limit,
            "fetch_memory": True,
            "stream": False,
        }
        if expression is None:
            call["space_ids"] = list(space_ids)
        else:
            call["space_keys"] = [{"spaceId": s, "filter": expression} for s in space_ids]
        if reranked:
            call["reranker_id"] = reranker_id
            call["max_results"] = limit

        events = list(await client.memories.retrieve(**call))
        statuses, degraded = classify(events)
        hits = hits_from_events(events, reranked=reranked)[:limit]

        payload: dict[str, Any] = {
            "query": query,
            "results": hits,
            "total_results": len(hits),
            # True when the server reported a real problem during the search.
            # With passages, they are usable but may be incomplete; with none,
            # the search failed rather than found nothing. Never raised: the
            # model reads `statuses` and decides. (Retrieval status contract.)
            "partial": degraded,
        }
        if statuses:
            payload["statuses"] = statuses
        if reply := abstract_reply(events):
            payload["abstract_reply"] = reply
        return json.dumps(payload, default=str)

    return FunctionTool(
        goodmem_search,
        name=name,
        description=description
        or "Search stored knowledge for passages relevant to a natural-language query.",
    )


def create_goodmem_admin_tools(
    client: AsyncGoodmemClient,
    *,
    upload_dir: str | None = None,
    max_items: int = 100,
) -> list[FunctionTool]:
    """Space and memory management tools.

    ``upload_dir`` enables ``goodmem_upload_file``; without it that tool is
    not created at all, so no agent can name a path on the host.
    """

    async def goodmem_list_embedders() -> str:
        """List embedders available for creating spaces."""
        items = [e.model_dump(exclude_none=True) async for e in await client.embedders.list(max_items=max_items)]
        return json.dumps({"embedders": items, "returned": len(items)}, default=str)

    async def goodmem_list_rerankers() -> str:
        """List rerankers available to improve search result ordering."""
        items = [r.model_dump(exclude_none=True) async for r in await client.rerankers.list(max_items=max_items)]
        return json.dumps({"rerankers": items, "returned": len(items)}, default=str)

    async def goodmem_list_spaces(name_filter: str | None = None) -> str:
        """List GoodMem spaces, optionally filtered by a name glob."""
        page = await client.spaces.list(
            max_items=max_items, **({"name_filter": name_filter} if name_filter else {})
        )
        items = [s.model_dump(exclude_none=True) async for s in page]
        return json.dumps(
            {"spaces": items, "returned": len(items), "truncated": len(items) >= max_items},
            default=str,
        )

    async def goodmem_get_space(space_id: GoodMemId) -> str:
        """Fetch one space by ID, with its embedders and labels."""
        space = await client.spaces.get(id=require_uuid(space_id, "space_id"))
        return json.dumps(space.model_dump(exclude_none=True), default=str)

    async def goodmem_create_space(name: str, embedder_id: GoodMemId) -> str:
        """Create a new space. Fails if a space with that name already exists."""
        space = await client.spaces.create(
            name=name, space_embedders=[{"embedderId": require_uuid(embedder_id, "embedder_id")}]
        )
        return json.dumps(space.model_dump(exclude_none=True), default=str)

    async def goodmem_update_space(
        space_id: GoodMemId,
        name: str | None = None,
        merge_labels: dict[str, str] | None = None,
        replace_labels: dict[str, str] | None = None,
    ) -> str:
        """Rename a space or change its labels."""
        request: dict[str, Any] = {}
        if name is not None:
            request["name"] = name
        if merge_labels:
            request["mergeLabels"] = merge_labels
        if replace_labels:
            request["replaceLabels"] = replace_labels
        if not request:
            raise ValueError("Nothing to update: pass name, merge_labels or replace_labels.")
        space = await client.spaces.update(id=require_uuid(space_id, "space_id"), request=request)
        return json.dumps(space.model_dump(exclude_none=True), default=str)

    async def goodmem_delete_space(space_id: GoodMemId) -> str:
        """Permanently delete a space and every memory in it."""
        checked = require_uuid(space_id, "space_id")
        await client.spaces.delete(id=checked)
        return json.dumps({"deleted": True, "space_id": checked})

    async def goodmem_create_memory(
        space_id: GoodMemId,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store text in a space so later searches can find it."""
        if not content.strip():
            raise ValueError("content must not be empty")
        memory = await client.memories.create(
            space_id=require_uuid(space_id, "space_id"),
            original_content=content,
            content_type="text/plain",
            metadata=metadata or None,
        )
        return json.dumps(memory.model_dump(exclude_none=True), default=str)

    async def goodmem_list_memories(space_id: GoodMemId) -> str:
        """List the memories stored in a space."""
        page = await client.memories.list(
            space_id=require_uuid(space_id, "space_id"), max_items=max_items
        )
        items = [m.model_dump(exclude_none=True) async for m in page]
        return json.dumps(
            {"memories": items, "returned": len(items), "truncated": len(items) >= max_items},
            default=str,
        )

    async def goodmem_get_memory(memory_id: GoodMemId, include_content: bool = True) -> str:
        """Fetch a memory by ID, with its metadata and readable content."""
        memory = await client.memories.get(
            id=require_uuid(memory_id, "memory_id"), include_content=include_content or None
        )
        payload = memory.model_dump(exclude_none=True)
        # model_dump re-encodes content as base64; the attribute holds bytes.
        payload.pop("original_content", None)
        raw = memory.original_content
        if include_content and raw is not None:
            content_type = str(memory.content_type or "")
            if isinstance(raw, str):
                raw = raw.encode()
            if content_type.startswith("text/") or "json" in content_type:
                try:
                    payload["content"] = raw.decode("utf-8")
                except UnicodeDecodeError:
                    payload["content_error"] = "Content is not valid UTF-8 text."
            else:
                payload["content_omitted"] = (
                    f"{len(raw)} bytes of {content_type or 'binary'} content is not "
                    "included; fetch it through the SDK if you need the bytes."
                )
        return json.dumps(payload, default=str)

    async def goodmem_delete_memory(memory_id: GoodMemId) -> str:
        """Permanently delete a memory."""
        checked = require_uuid(memory_id, "memory_id")
        await client.memories.delete(id=checked)
        return json.dumps({"deleted": True, "memory_id": checked})

    funcs: list[Callable[..., Any]] = [
        goodmem_list_embedders,
        goodmem_list_rerankers,
        goodmem_list_spaces,
        goodmem_get_space,
        goodmem_create_space,
        goodmem_update_space,
        goodmem_delete_space,
        goodmem_create_memory,
        goodmem_list_memories,
        goodmem_get_memory,
        goodmem_delete_memory,
    ]

    if upload_dir is not None:

        async def goodmem_upload_file(
            space_id: GoodMemId,
            file_name: str,
            metadata: dict[str, Any] | None = None,
        ) -> str:
            """Store a file from the approved upload directory."""
            import base64
            import mimetypes

            checked = require_uuid(space_id, "space_id")
            path = resolve_upload_path(file_name, upload_dir)
            raw = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            merged = dict(metadata or {})
            merged.setdefault("title", path.name)
            kwargs: dict[str, Any] = {
                "space_id": checked,
                "content_type": content_type,
                "metadata": merged,
            }
            if content_type.startswith("text/"):
                kwargs["original_content"] = raw.decode("utf-8", errors="replace")
            else:
                kwargs["original_content_b64"] = base64.b64encode(raw).decode("ascii")
            memory = await client.memories.create(**kwargs)
            return json.dumps(memory.model_dump(exclude_none=True), default=str)

        funcs.append(goodmem_upload_file)

    return [
        FunctionTool(f, name=f.__name__, description=(f.__doc__ or f.__name__).strip().splitlines()[0])
        for f in funcs
    ]


__all__ = [
    "ADMIN_TOOL_NAMES",
    "SEARCH_TOOL_NAME",
    "create_goodmem_admin_tools",
    "create_goodmem_search_tool",
]
