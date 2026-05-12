"""AutoGen-native memory provider backed by GoodMem.

In AutoGen, the extension point for retrieval-augmented context injection is
:class:`autogen_core.memory.Memory` — the equivalent of "BaseContextProvider"
in other frameworks. :class:`GoodMemContextProvider` plugs a single GoodMem
space into that surface so an AutoGen agent can call ``memory.add(...)`` /
``memory.query(...)`` and ``update_context`` will inject retrieved chunks as
a system message.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from autogen_core import CancellationToken, Component
from autogen_core.memory import (
    Memory,
    MemoryContent,
    MemoryMimeType,
    MemoryQueryResult,
    UpdateContextResult,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage
from typing_extensions import Self

from ._client import GoodMemClient
from ._config import GoodMemMemoryConfig

logger = logging.getLogger(__name__)


class GoodMemContextProvider(Memory, Component[GoodMemMemoryConfig]):
    """AutoGen ``Memory`` provider backed by a GoodMem space.

    Args:
        config: Connection details and space configuration.

    Example:

        .. code-block:: python

            from autogen_goodmem import GoodMemContextProvider, GoodMemMemoryConfig

            provider = GoodMemContextProvider(
                config=GoodMemMemoryConfig(
                    base_url="https://localhost:8080",
                    api_key="gm_...",
                    space_name="my-kb",
                    embedder_id="<embedder-uuid>",
                    verify_ssl=False,
                )
            )
            await provider.add(MemoryContent(content="hi", mime_type=MemoryMimeType.TEXT))
            results = await provider.query("hello")
            await provider.close()
    """

    component_config_schema = GoodMemMemoryConfig
    component_provider_override = "autogen_goodmem.GoodMemContextProvider"

    def __init__(self, config: GoodMemMemoryConfig) -> None:
        self._config = config
        self._client = GoodMemClient(
            base_url=config.base_url,
            api_key=config.api_key,
            verify_ssl=config.verify_ssl,
        )
        self._space_id: Optional[str] = None

    # ── Internals ──────────────────────────────────────────────────────

    async def _ensure_space(self) -> str:
        if self._space_id is not None:
            return self._space_id
        result = await self._client.create_space(
            self._config.space_name,
            self._config.embedder_id,
            chunking=self._config.chunking,
        )
        self._space_id = result["spaceId"]
        if result.get("reused"):
            logger.info("Reusing GoodMem space %s (%s)", self._config.space_name, self._space_id)
        else:
            logger.info("Created GoodMem space %s (%s)", self._config.space_name, self._space_id)
        return self._space_id  # type: ignore[return-value]

    # ── AutoGen Memory interface ───────────────────────────────────────

    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        last = messages[-1]
        query_text = last.content if isinstance(last.content, str) else str(last)
        query_results = await self.query(query_text)

        if query_results.results:
            lines = [f"{i}. {str(m.content)}" for i, m in enumerate(query_results.results, 1)]
            await model_context.add_message(SystemMessage(content="\nRelevant memory content:\n" + "\n".join(lines)))

        return UpdateContextResult(memories=query_results)

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """Store a text memory. Use :meth:`add_file` for binary files."""
        space_id = await self._ensure_space()
        text = content.content
        if not isinstance(text, str):
            raise ValueError(
                "GoodMemContextProvider.add() only supports text content. "
                "Use add_file() for binary files (PDF, images, etc.)."
            )

        merged: Dict[str, Any] = {}
        if self._config.metadata:
            merged.update(self._config.metadata)
        if content.metadata:
            merged.update(content.metadata)

        await self._client.create_memory(
            space_id=space_id,
            content=text,
            content_type="text/plain",
            metadata=merged or None,
        )

    async def add_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        cancellation_token: CancellationToken | None = None,
    ) -> Dict[str, Any]:
        space_id = await self._ensure_space()
        merged: Dict[str, Any] = {}
        if self._config.metadata:
            merged.update(self._config.metadata)
        if metadata:
            merged.update(metadata)
        result = await self._client.create_memory(
            space_id=space_id,
            file_path=file_path,
            metadata=merged or None,
        )
        return {
            "memoryId": result.get("memoryId"),
            "spaceId": result.get("spaceId"),
            "status": result.get("processingStatus", "PENDING"),
            "contentType": result.get("contentType"),
            "fileName": file_path.rsplit("/", 1)[-1],
        }

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        space_id = await self._ensure_space()
        space_ids: List[str] = kwargs.get("space_ids", [space_id])
        query_text = query if isinstance(query, str) else str(query.content)

        pp = self._config.post_processor
        kw: Dict[str, Any] = {}
        if pp is not None:
            kw.update(
                reranker_id=pp.reranker_id,
                llm_id=pp.llm_id,
                relevance_threshold=pp.relevance_threshold,
                llm_temperature=pp.llm_temperature,
                chronological_resort=pp.chronological_resort,
            )

        result = await self._client.retrieve_memories(
            message=query_text,
            space_ids=space_ids,
            max_results=self._config.max_results,
            fetch_memory=self._config.include_memory_definition,
            wait_for_indexing=self._config.wait_for_indexing,
            **kw,
        )

        abstract = result.get("abstractReply")
        out: List[MemoryContent] = []
        for item in result["results"]:
            meta: Dict[str, Any] = {
                "chunkId": item.get("chunkId"),
                "memoryId": item.get("memoryId"),
                "relevanceScore": item.get("relevanceScore"),
                "memoryIndex": item.get("memoryIndex"),
            }
            if abstract:
                meta["abstractReply"] = abstract
            out.append(
                MemoryContent(
                    content=item.get("chunkText", ""),
                    mime_type=MemoryMimeType.TEXT,
                    metadata=meta,
                )
            )
        return MemoryQueryResult(results=out)

    async def clear(self) -> None:
        logger.warning(
            "GoodMem does not support bulk clear; delete memories individually "
            "or recreate the space."
        )

    async def close(self) -> None:
        await self._client.close()
        self._space_id = None

    # ── Convenience pass-throughs ──────────────────────────────────────

    @property
    def client(self) -> GoodMemClient:
        """The underlying :class:`GoodMemClient` (useful for advanced ops)."""
        return self._client

    # ── Serialization ──────────────────────────────────────────────────

    def _to_config(self) -> GoodMemMemoryConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: GoodMemMemoryConfig) -> Self:
        return cls(config=config)
