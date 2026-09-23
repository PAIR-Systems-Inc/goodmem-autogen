"""AutoGen ``Memory`` implementation backed by a GoodMem space."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

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
from goodmem import AsyncGoodmem
from goodmem.errors import ConflictError
from typing_extensions import Self

from ._config import ChunkingConfig, GoodMemMemoryConfig
from ._connection import GoodMemConnection, run_cancellable
from ._results import GoodMemRetrievalError, abstract_reply, classify, hits_from_events
from ._uploads import GoodMemUploadError, resolve_upload_path
from .filters import combine


logger = logging.getLogger(__name__)

_TERMINAL = frozenset({"COMPLETED", "FAILED"})


class GoodMemIngestionError(RuntimeError):
    """A write failed. ``memory_id`` records what was accepted, if anything."""

    def __init__(self, message: str, *, memory_id: str | None = None) -> None:
        super().__init__(message)
        self.memory_id = memory_id


class GoodMemContextProvider(Memory, Component[GoodMemMemoryConfig]):
    """An AutoGen ``Memory`` backed by one GoodMem space.

    Attach it to an agent and ``update_context`` injects retrieved passages
    as a system message on each turn, which is how AutoGen's own ``ListMemory``
    and the bundled ext memories work.

    Retrieval reports what the server said: results carry ``partial`` and
    ``statuses`` in their metadata, and a search that produced nothing usable
    raises rather than returning an empty result that looks like "no matches".

    Example::

        provider = GoodMemContextProvider(
            config=GoodMemMemoryConfig(
                base_url="https://goodmem.example.com",
                api_key="gm_...",
                space_name="handbook",
                embedder_id="<embedder-uuid>",
            )
        )
        await provider.add(MemoryContent(content="...", mime_type=MemoryMimeType.TEXT))
        results = await provider.query("what is the refund policy?")
        await provider.close()
    """

    component_config_schema = GoodMemMemoryConfig
    component_provider_override = "autogen_goodmem.GoodMemContextProvider"

    def __init__(
        self,
        config: GoodMemMemoryConfig,
        *,
        client: AsyncGoodmem | None = None,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        if not config.space_id and not config.space_name:
            raise ValueError("Set either space_id or space_name on GoodMemMemoryConfig.")
        self._config = config
        self._chunking = chunking or ChunkingConfig()
        self._conn = GoodMemConnection(
            base_url=config.base_url,
            api_key=config.api_key.get_secret_value(),
            verify_ssl=config.verify_ssl,
            timeout=config.timeout,
            client=client,
        )
        self._space_id: str | None = config.space_id
        self._space_lock = asyncio.Lock()

    # ------------------------------------------------------------ internals
    async def _ensure_space(self, cancellation_token: CancellationToken | None = None) -> str:
        """Resolve the configured space, creating it by name on first use.

        A same-named space is reused only when its embedder matches the one
        configured here; a mismatch or an ambiguous name is an error rather
        than a silent attach to a space that indexes differently.
        """
        async with self._space_lock:
            if self._space_id is not None:
                return self._space_id

            name = self._config.space_name
            assert name is not None
            client = self._conn.client()

            page = await run_cancellable(client.spaces.list(name_filter=name), cancellation_token)
            # name_filter is a server-side glob, so confirm the exact name here.
            matches = [space async for space in page if space.name == name]

            if len(matches) > 1:
                raise ValueError(
                    f"{len(matches)} accessible spaces are named {name!r}. "
                    "Set space_id explicitly."
                )
            if matches:
                existing = matches[0]
                embedders = {e.embedder_id for e in (existing.space_embedders or [])}
                if self._config.embedder_id and self._config.embedder_id not in embedders:
                    raise ValueError(
                        f"Space {name!r} exists but uses a different embedder "
                        f"({', '.join(sorted(embedders)) or 'none'}). Use that embedder, "
                        "a different name, or set space_id."
                    )
                self._space_id = existing.space_id
                logger.info("Reusing GoodMem space %s (%s)", name, self._space_id)
                return self._space_id

            if not self._config.embedder_id:
                raise ValueError(
                    f"Space {name!r} does not exist and embedder_id is not set, "
                    "so it cannot be created."
                )
            try:
                created = await run_cancellable(
                    client.spaces.create(
                        name=name,
                        space_embedders=[{"embedderId": self._config.embedder_id}],
                        default_chunking_config=self._chunking.to_api(),
                    ),
                    cancellation_token,
                )
            except ConflictError:
                # Someone created it between the lookup and the create.
                async for space in await client.spaces.list(name_filter=name):
                    if space.name == name:
                        self._space_id = space.space_id
                        return self._space_id
                raise
            self._space_id = created.space_id
            logger.info("Created GoodMem space %s (%s)", name, self._space_id)
            return self._space_id

    async def _wait_for(self, memory_id: str, cancellation_token: CancellationToken | None) -> str:
        """Wait for one specific memory to finish indexing.

        Searching repeatedly cannot distinguish "not indexed yet" from
        "nothing matches", so the wait is always on a known ID.
        """
        client = self._conn.client()
        deadline = asyncio.get_running_loop().time() + self._config.indexing_timeout
        while True:
            memory = await run_cancellable(client.memories.get(id=memory_id), cancellation_token)
            status = str(memory.processing_status)
            if status in _TERMINAL:
                return status
            if asyncio.get_running_loop().time() >= deadline:
                raise GoodMemIngestionError(
                    f"Memory {memory_id} was still {status} after "
                    f"{self._config.indexing_timeout}s. It was created; check its "
                    "processing status rather than storing it again.",
                    memory_id=memory_id,
                )
            await asyncio.sleep(0.5)

    def _merged_metadata(self, extra: dict[str, Any] | None) -> dict[str, Any] | None:
        merged: dict[str, Any] = {}
        if self._config.metadata:
            merged.update(self._config.metadata)
        if extra:
            merged.update(extra)
        return merged or None

    # -------------------------------------------------- AutoGen Memory API
    async def update_context(self, model_context: ChatCompletionContext) -> UpdateContextResult:
        messages = await model_context.get_messages()
        if not messages:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        last = messages[-1]
        content = getattr(last, "content", None)
        if not isinstance(content, str) or not content.strip():
            # Only a text turn is a usable query; stringifying a structured
            # message produces a nonsense search.
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        results = await self.query(content)
        if results.results:
            # query() always yields text content; MemoryContent.content is a
            # wider union, so narrow it explicitly rather than formatting bytes.
            lines = []
            for i, memory in enumerate(results.results, 1):
                text = memory.content if isinstance(memory.content, str) else str(memory.content)
                lines.append(f"{i}. {text}")
            await model_context.add_message(
                SystemMessage(content="\nRelevant memory content:\n" + "\n".join(lines))
            )
        return UpdateContextResult(memories=results)

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """Store text. Use :meth:`add_file` for files."""
        text = content.content
        if not isinstance(text, str):
            raise TypeError(
                "GoodMemContextProvider.add() stores text. Use add_file() for files."
            )
        if not text.strip():
            raise ValueError("Cannot store empty content.")

        space_id = await self._ensure_space(cancellation_token)
        client = self._conn.client()
        mime = content.mime_type
        content_type = mime.value if isinstance(mime, MemoryMimeType) else str(mime or "text/plain")

        created = await run_cancellable(
            client.memories.create(
                space_id=space_id,
                original_content=text,
                content_type=content_type,
                metadata=self._merged_metadata(content.metadata),
            ),
            cancellation_token,
        )
        if self._config.wait_for_indexing:
            status = await self._wait_for(created.memory_id, cancellation_token)
            if status == "FAILED":
                raise GoodMemIngestionError(
                    f"Memory {created.memory_id} failed processing on the server.",
                    memory_id=created.memory_id,
                )

    async def add_file(
        self,
        file_name: str,
        metadata: dict[str, Any] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Store a file from the configured ``upload_dir``.

        ``upload_dir`` must be set. Paths resolving outside it, including via
        symlinks, are refused before the file is opened.
        """
        import base64
        import mimetypes

        path = resolve_upload_path(file_name, self._config.upload_dir)
        raw = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

        merged = dict(metadata or {})
        merged.setdefault("title", path.name)

        space_id = await self._ensure_space(cancellation_token)
        client = self._conn.client()
        kwargs: dict[str, Any] = {
            "space_id": space_id,
            "content_type": content_type,
            "metadata": self._merged_metadata(merged),
        }
        if content_type.startswith("text/"):
            kwargs["original_content"] = raw.decode("utf-8", errors="replace")
        else:
            kwargs["original_content_b64"] = base64.b64encode(raw).decode("ascii")

        created = await run_cancellable(client.memories.create(**kwargs), cancellation_token)
        status = str(created.processing_status)
        if self._config.wait_for_indexing:
            status = await self._wait_for(created.memory_id, cancellation_token)
        return {
            "memory_id": created.memory_id,
            "space_id": space_id,
            "status": status,
            "content_type": content_type,
            "file_name": path.name,
        }

    async def query(
        self,
        query: str | MemoryContent,
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        text = query if isinstance(query, str) else str(query.content)
        if not text.strip():
            raise ValueError("Query must not be empty.")

        space_id = await self._ensure_space(cancellation_token)
        space_ids: list[str] = kwargs.get("space_ids") or [space_id]
        limit = int(kwargs.get("limit") or self._config.max_results)
        expression = combine(self._config.filter, kwargs.get("filter"))

        pp = self._config.post_processor
        reranked = bool(pp and pp.reranker_id)
        if pp and pp.relevance_threshold is not None and not pp.reranker_id:
            raise ValueError(
                "relevance_threshold needs a reranker_id: a vector score is an "
                "opaque similarity, not a 0-1 relevance value."
            )

        call: dict[str, Any] = {
            "message": text,
            "requested_size": self._config.fetch_k or limit,
            "fetch_memory": True,
            "stream": False,
        }
        if expression is None:
            call["space_ids"] = space_ids
        else:
            call["space_keys"] = [{"spaceId": s, "filter": expression} for s in space_ids]
        if pp:
            if pp.reranker_id:
                call["reranker_id"] = pp.reranker_id
                call["max_results"] = limit
            if pp.llm_id:
                call["llm_id"] = pp.llm_id
                if pp.llm_temperature is not None:
                    call["llm_temp"] = pp.llm_temperature
            if pp.relevance_threshold is not None:
                call["relevance_threshold"] = pp.relevance_threshold
            if pp.chronological_resort:
                call["chronological_resort"] = True

        client = self._conn.client()
        events = list(await run_cancellable(client.memories.retrieve(**call), cancellation_token))

        statuses, degraded = classify(events)
        hits = hits_from_events(events, reranked=reranked)[:limit]

        # A failed search must not be mistaken for one that found nothing.
        if degraded and not hits:
            raise GoodMemRetrievalError(
                "; ".join(
                    f"{s.get('code', 'UNKNOWN')}: {s.get('message', '')}" for s in statuses
                )
                or "Retrieval failed",
                statuses=statuses,
            )

        reply = abstract_reply(events)
        results: list[MemoryContent] = []
        for hit in hits:
            # MemoryContent has no score field, so scores and provenance go in
            # metadata, as AutoGen's own ext memories do.
            meta: dict[str, Any] = dict(hit["metadata"])
            meta.update(
                chunk_id=hit["chunk_id"],
                memory_id=hit["memory_id"],
                space_id=hit["space_id"],
                source=hit["source"],
                score=hit["score"],
                score_kind=hit["score_kind"],
                partial=degraded,
            )
            if statuses:
                meta["statuses"] = statuses
            if reply:
                meta["abstract_reply"] = reply
            results.append(
                MemoryContent(
                    content=hit["chunk_text"],
                    mime_type=MemoryMimeType.TEXT,
                    metadata=meta,
                )
            )
        return MemoryQueryResult(results=results)

    async def clear(self) -> None:
        """Delete every memory in the configured space.

        Destructive and irreversible, so it requires ``allow_clear=True``
        rather than silently emptying a space — or silently doing nothing,
        which is worse because the caller believes it worked.
        """
        if not self._config.allow_clear:
            raise PermissionError(
                "clear() would permanently delete every memory in space "
                f"{self._space_id or self._config.space_name}. Construct the provider "
                "with allow_clear=True to permit it."
            )
        space_id = await self._ensure_space()
        client = self._conn.client()
        ids = [m.memory_id async for m in await client.memories.list(space_id=space_id)]
        for memory_id in ids:
            await client.memories.delete(id=memory_id)
        logger.info("Cleared %d memories from space %s", len(ids), space_id)

    async def close(self) -> None:
        await self._conn.close()

    # ------------------------------------------------------------ plumbing
    @property
    def space_id(self) -> str | None:
        """The resolved space ID, once :meth:`_ensure_space` has run."""
        return self._space_id

    def _to_config(self) -> GoodMemMemoryConfig:
        return self._config

    @classmethod
    def _from_config(cls, config: GoodMemMemoryConfig) -> Self:
        return cls(config=config)


__all__ = ["GoodMemContextProvider", "GoodMemIngestionError", "GoodMemUploadError"]
