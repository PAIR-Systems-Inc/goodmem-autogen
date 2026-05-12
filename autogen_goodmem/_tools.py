"""Factory for the 11 GoodMem AutoGen ``FunctionTool`` instances.

Each tool is a thin wrapper around :class:`GoodMemClient`, exposed as an
AutoGen ``FunctionTool`` that an agent can call. Tool names are prefixed
with ``goodmem_`` so multiple memory backends can coexist on one agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from autogen_core.tools import FunctionTool

from ._client import GoodMemClient


TOOL_NAMES: List[str] = [
    "goodmem_list_embedders",
    "goodmem_list_spaces",
    "goodmem_get_space",
    "goodmem_create_space",
    "goodmem_update_space",
    "goodmem_delete_space",
    "goodmem_create_memory",
    "goodmem_list_memories",
    "goodmem_retrieve_memories",
    "goodmem_get_memory",
    "goodmem_delete_memory",
]


def create_goodmem_tools(client: GoodMemClient) -> List[FunctionTool]:
    """Return the 11 GoodMem ``FunctionTool`` instances bound to ``client``.

    The returned list is in the order declared in :data:`TOOL_NAMES`.
    """

    async def goodmem_list_embedders() -> List[Dict[str, Any]]:
        """List available embedder models on the GoodMem server."""
        return await client.list_embedders()

    async def goodmem_list_spaces() -> List[Dict[str, Any]]:
        """List all GoodMem spaces visible to the current API key."""
        return await client.list_spaces()

    async def goodmem_get_space(space_id: str) -> Dict[str, Any]:
        """Fetch a single GoodMem space by ID."""
        return await client.get_space(space_id)

    async def goodmem_create_space(
        name: str,
        embedder_id: str,
    ) -> Dict[str, Any]:
        """Create a GoodMem space, or reuse the existing one with the same name.

        Returns ``{spaceId, name, embedderId, reused}``.
        """
        return await client.create_space(name=name, embedder_id=embedder_id)

    async def goodmem_update_space(
        space_id: str,
        name: Optional[str] = None,
        public_read: Optional[bool] = None,
        replace_labels: Optional[Dict[str, str]] = None,
        merge_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Update a GoodMem space.

        Only ``name``, ``publicRead``, ``replaceLabels``, ``mergeLabels``, and
        ``defaultChunkingConfig`` are accepted by the server — passing a bare
        ``labels`` field returns 400, so prefer ``replace_labels`` (full
        replacement) or ``merge_labels`` (additive merge).
        """
        return await client.update_space(
            space_id,
            name=name,
            public_read=public_read,
            replace_labels=replace_labels,
            merge_labels=merge_labels,
        )

    async def goodmem_delete_space(space_id: str) -> Dict[str, Any]:
        """Permanently delete a GoodMem space and every memory inside it."""
        return await client.delete_space(space_id)

    async def goodmem_create_memory(
        space_id: str,
        content: Optional[str] = None,
        file_path: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a memory in a space from inline text or a local file.

        Exactly one of ``content`` (inline text) or ``file_path`` must be
        provided. Binary files (PDF, images) are base64-encoded; text files
        are sent inline. MIME type is auto-detected from the extension when
        ``content_type`` is omitted.
        """
        return await client.create_memory(
            space_id=space_id,
            content=content,
            file_path=file_path,
            content_type=content_type,
            metadata=metadata,
        )

    async def goodmem_list_memories(
        space_id: str,
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List memories in a space (paginated via ``nextToken``).

        Note: GoodMem has no top-level ``GET /v1/memories`` — listing is
        always scoped to a space.
        """
        return await client.list_memories(space_id, next_token=next_token)

    async def goodmem_retrieve_memories(
        message: str,
        space_ids: List[str],
        max_results: int = 5,
        fetch_memory: bool = True,
        wait_for_indexing: bool = False,
        reranker_id: Optional[str] = None,
        llm_id: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
        llm_temperature: Optional[float] = None,
        chronological_resort: bool = False,
    ) -> Dict[str, Any]:
        """Semantic retrieval against one or more GoodMem spaces.

        Optionally re-orders results with ``reranker_id`` and/or generates a
        natural-language ``abstractReply`` with ``llm_id``.
        """
        return await client.retrieve_memories(
            message=message,
            space_ids=space_ids,
            max_results=max_results,
            fetch_memory=fetch_memory,
            wait_for_indexing=wait_for_indexing,
            reranker_id=reranker_id,
            llm_id=llm_id,
            relevance_threshold=relevance_threshold,
            llm_temperature=llm_temperature,
            chronological_resort=chronological_resort,
        )

    async def goodmem_get_memory(
        memory_id: str,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """Fetch a memory's metadata, and optionally its original content."""
        return await client.get_memory(memory_id, include_content=include_content)

    async def goodmem_delete_memory(memory_id: str) -> Dict[str, Any]:
        """Permanently delete a memory and its chunks/embeddings."""
        return await client.delete_memory(memory_id)

    funcs = [
        goodmem_list_embedders,
        goodmem_list_spaces,
        goodmem_get_space,
        goodmem_create_space,
        goodmem_update_space,
        goodmem_delete_space,
        goodmem_create_memory,
        goodmem_list_memories,
        goodmem_retrieve_memories,
        goodmem_get_memory,
        goodmem_delete_memory,
    ]
    return [
        FunctionTool(
            f,
            name=f.__name__,
            description=(f.__doc__ or f.__name__).strip().splitlines()[0],
        )
        for f in funcs
    ]
