"""Pure-async REST client for the GoodMem v1 API.

This client is framework-agnostic: it returns plain dictionaries (and lists of
them) and never imports anything from AutoGen. :class:`GoodMemContextProvider`
and :func:`create_goodmem_tools` are built on top of it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ._config import ChunkingConfig, PostProcessorConfig

logger = logging.getLogger(__name__)


# ── MIME type detection ─────────────────────────────────────────────────

_EXTRA_MIME: Dict[str, str] = {
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        return mime
    return _EXTRA_MIME.get(path.suffix.lower(), "application/octet-stream")


# ── Retrieve-response NDJSON/SSE parser ────────────────────────────────


def _parse_retrieve_response(
    response_text: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Any]:
    """Parse the NDJSON / SSE body returned by ``POST /v1/memories:retrieve``.

    Returns ``(results, memories, abstract_reply)`` where ``results`` is a list
    of ``{chunkId, chunkText, memoryId, relevanceScore, memoryIndex}`` dicts.
    """
    results: List[Dict[str, Any]] = []
    memories: List[Dict[str, Any]] = []
    abstract_reply: Any = None

    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("event:"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        if item.get("memoryDefinition"):
            memories.append(item["memoryDefinition"])
        elif item.get("abstractReply"):
            abstract_reply = item["abstractReply"]
        elif item.get("retrievedItem"):
            chunk_outer = item["retrievedItem"].get("chunk", {})
            chunk = chunk_outer.get("chunk", {})
            results.append(
                {
                    "chunkId": chunk.get("chunkId"),
                    "chunkText": chunk.get("chunkText"),
                    "memoryId": chunk.get("memoryId"),
                    "relevanceScore": chunk_outer.get("relevanceScore"),
                    "memoryIndex": chunk_outer.get("memoryIndex"),
                }
            )

    return results, memories, abstract_reply


# ── Client ──────────────────────────────────────────────────────────────


class GoodMemClient:
    """Async REST client for the GoodMem v1 API.

    Args:
        base_url: Base URL, e.g. ``https://localhost:8080``.
        api_key: API key sent as the ``X-API-Key`` header.
        verify_ssl: Pass ``False`` for self-signed local servers.
        timeout: Per-request timeout (seconds).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        verify_ssl: bool = True,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._client: Optional[httpx.AsyncClient] = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout, verify=self._verify_ssl)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def __aenter__(self) -> "GoodMemClient":
        self._ensure_client()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    # ── Embedders ──────────────────────────────────────────────────────

    async def list_embedders(self) -> List[Dict[str, Any]]:
        """``GET /v1/embedders`` -> list of embedder definitions."""
        client = self._ensure_client()
        resp = await client.get(f"{self._base_url}/v1/embedders", headers=self._headers)
        resp.raise_for_status()
        body = resp.json()
        return body if isinstance(body, list) else body.get("embedders", [])

    # ── Spaces ─────────────────────────────────────────────────────────

    async def list_spaces(self) -> List[Dict[str, Any]]:
        """``GET /v1/spaces`` -> list of spaces."""
        client = self._ensure_client()
        resp = await client.get(f"{self._base_url}/v1/spaces", headers=self._headers)
        resp.raise_for_status()
        body = resp.json()
        return body if isinstance(body, list) else body.get("spaces", [])

    async def get_space(self, space_id: str) -> Dict[str, Any]:
        """``GET /v1/spaces/{id}``."""
        client = self._ensure_client()
        resp = await client.get(f"{self._base_url}/v1/spaces/{space_id}", headers=self._headers)
        resp.raise_for_status()
        return resp.json()

    async def create_space(
        self,
        name: str,
        embedder_id: str,
        *,
        chunking: Optional[ChunkingConfig] = None,
    ) -> Dict[str, Any]:
        """Create a space, or reuse an existing one with the same name.

        Returns a dict that includes ``reused: True`` when a name collision was
        found. Handles a race condition where ``POST /v1/spaces`` returns 409
        between the list-check and the create call.
        """
        chunking = chunking or ChunkingConfig()
        client = self._ensure_client()

        try:
            existing = await self.list_spaces()
            for space in existing:
                if space.get("name") == name:
                    sid = space.get("spaceId") or space.get("id")
                    return {
                        "spaceId": sid,
                        "name": name,
                        "embedderId": embedder_id,
                        "reused": True,
                    }
        except Exception:
            logger.debug("list_spaces failed before create; will attempt create anyway", exc_info=True)

        body = {
            "name": name,
            "spaceEmbedders": [{"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunking.chunk_size,
                    "chunkOverlap": chunking.chunk_overlap,
                    "separators": chunking.separators,
                    "keepStrategy": chunking.keep_strategy,
                    "separatorIsRegex": chunking.separator_is_regex,
                    "lengthMeasurement": chunking.length_measurement,
                },
            },
        }
        resp = await client.post(
            f"{self._base_url}/v1/spaces",
            headers=self._headers,
            json=body,
        )
        if resp.status_code == 409:
            # Race: someone else just created it. Re-list and find by name.
            for space in await self.list_spaces():
                if space.get("name") == name:
                    sid = space.get("spaceId") or space.get("id")
                    return {
                        "spaceId": sid,
                        "name": name,
                        "embedderId": embedder_id,
                        "reused": True,
                    }
            resp.raise_for_status()

        resp.raise_for_status()
        result = resp.json()
        return {
            "spaceId": result.get("spaceId") or result.get("id"),
            "name": result.get("name", name),
            "embedderId": embedder_id,
            "chunkingConfig": body["defaultChunkingConfig"],
            "reused": False,
        }

    async def update_space(
        self,
        space_id: str,
        *,
        name: Optional[str] = None,
        public_read: Optional[bool] = None,
        replace_labels: Optional[Dict[str, str]] = None,
        merge_labels: Optional[Dict[str, str]] = None,
        chunking: Optional[ChunkingConfig] = None,
    ) -> Dict[str, Any]:
        """``PUT /v1/spaces/{id}`` with only the fields the API accepts.

        Note: the server rejects a bare ``labels`` field with 400 — use
        ``replace_labels`` (full replacement) or ``merge_labels`` (additive).
        """
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if public_read is not None:
            body["publicRead"] = public_read
        if replace_labels is not None:
            body["replaceLabels"] = replace_labels
        if merge_labels is not None:
            body["mergeLabels"] = merge_labels
        if chunking is not None:
            body["defaultChunkingConfig"] = {
                "recursive": {
                    "chunkSize": chunking.chunk_size,
                    "chunkOverlap": chunking.chunk_overlap,
                    "separators": chunking.separators,
                    "keepStrategy": chunking.keep_strategy,
                    "separatorIsRegex": chunking.separator_is_regex,
                    "lengthMeasurement": chunking.length_measurement,
                },
            }

        client = self._ensure_client()
        resp = await client.put(
            f"{self._base_url}/v1/spaces/{space_id}",
            headers=self._headers,
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    async def delete_space(self, space_id: str) -> Dict[str, Any]:
        """``DELETE /v1/spaces/{id}``."""
        client = self._ensure_client()
        resp = await client.delete(
            f"{self._base_url}/v1/spaces/{space_id}",
            headers=self._headers,
        )
        resp.raise_for_status()
        return {"spaceId": space_id, "success": True}

    # ── Memories ───────────────────────────────────────────────────────

    async def create_memory(
        self,
        *,
        space_id: str,
        content: Optional[str] = None,
        file_path: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a memory from text or a file.

        Exactly one of ``content`` (inline text) or ``file_path`` (any file —
        binary content is base64-encoded into ``originalContentB64``) must be
        supplied. When ``file_path`` is used, the MIME type is auto-detected
        from the extension unless overridden via ``content_type``.
        """
        if (content is None) == (file_path is None):
            raise ValueError("create_memory requires exactly one of content or file_path")

        body: Dict[str, Any] = {"spaceId": space_id}

        if file_path is not None:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            mime = content_type or _guess_mime(path)
            body["contentType"] = mime
            file_bytes = path.read_bytes()
            if mime.startswith("text/"):
                body["originalContent"] = file_bytes.decode("utf-8")
            else:
                body["originalContentB64"] = base64.b64encode(file_bytes).decode("ascii")
        else:
            body["contentType"] = content_type or "text/plain"
            body["originalContent"] = content  # type: ignore[assignment]

        if metadata:
            body["metadata"] = metadata

        client = self._ensure_client()
        resp = await client.post(
            f"{self._base_url}/v1/memories",
            headers=self._headers,
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    async def list_memories(
        self,
        space_id: str,
        *,
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """``GET /v1/spaces/{id}/memories`` -> ``{memories, nextToken}``.

        There is no top-level ``GET /v1/memories`` — listing is always
        scoped to a space.
        """
        params: Dict[str, Any] = {}
        if next_token:
            params["nextToken"] = next_token
        client = self._ensure_client()
        resp = await client.get(
            f"{self._base_url}/v1/spaces/{space_id}/memories",
            headers=self._headers,
            params=params,
        )
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, list):
            return {"memories": body, "nextToken": None}
        return {
            "memories": body.get("memories", []),
            "nextToken": body.get("nextToken"),
        }

    async def get_memory(
        self,
        memory_id: str,
        *,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """``GET /v1/memories/{id}`` (and optionally ``/content``).

        The ``/content`` endpoint may return plain text rather than JSON for
        text memories — both shapes are handled.
        """
        client = self._ensure_client()
        resp = await client.get(
            f"{self._base_url}/v1/memories/{memory_id}",
            headers=self._headers,
        )
        resp.raise_for_status()
        result: Dict[str, Any] = {"memory": resp.json()}

        if include_content:
            try:
                content_resp = await client.get(
                    f"{self._base_url}/v1/memories/{memory_id}/content",
                    headers=self._headers,
                )
                content_resp.raise_for_status()
                ctype = content_resp.headers.get("content-type", "")
                if "application/json" in ctype:
                    try:
                        result["content"] = content_resp.json()
                    except json.JSONDecodeError:
                        result["content"] = content_resp.text
                else:
                    result["content"] = content_resp.text
            except Exception as exc:
                result["contentError"] = f"Failed to fetch content: {exc}"

        return result

    async def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """``DELETE /v1/memories/{id}``."""
        client = self._ensure_client()
        resp = await client.delete(
            f"{self._base_url}/v1/memories/{memory_id}",
            headers=self._headers,
        )
        resp.raise_for_status()
        return {"memoryId": memory_id, "success": True}

    # ── Retrieve ────────────────────────────────────────────────────────

    async def retrieve_memories(
        self,
        *,
        message: str,
        space_ids: List[str],
        max_results: int = 5,
        fetch_memory: bool = True,
        wait_for_indexing: bool = False,
        wait_seconds: float = 60.0,
        poll_interval: float = 5.0,
        # Post-processor params
        reranker_id: Optional[str] = None,
        llm_id: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
        llm_temperature: Optional[float] = None,
        chronological_resort: bool = False,
    ) -> Dict[str, Any]:
        """``POST /v1/memories:retrieve`` with optional reranker + LLM.

        When ``wait_for_indexing=True`` and no results come back yet, polls
        every ``poll_interval`` seconds for up to ``wait_seconds`` total —
        new memories take a few seconds to embed and become searchable.

        Returns ``{"results": [...], "memories": [...], "abstractReply": ...}``.
        """
        body: Dict[str, Any] = {
            "message": message,
            "spaceKeys": [{"spaceId": sid} for sid in space_ids],
            "requestedSize": max_results,
            "fetchMemory": fetch_memory,
        }

        use_post = any(
            v is not None for v in (reranker_id, llm_id, relevance_threshold, llm_temperature)
        ) or chronological_resort
        if use_post:
            cfg: Dict[str, Any] = {"max_results": max_results}
            if reranker_id is not None:
                cfg["reranker_id"] = reranker_id
            if llm_id is not None:
                cfg["llm_id"] = llm_id
            if relevance_threshold is not None:
                cfg["relevance_threshold"] = relevance_threshold
            if llm_temperature is not None:
                cfg["llm_temp"] = llm_temperature
            if chronological_resort:
                cfg["chronological_resort"] = True
            body["postProcessor"] = {
                "name": "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
                "config": cfg,
            }

        headers = {**self._headers, "Accept": "application/x-ndjson"}
        client = self._ensure_client()

        start = asyncio.get_event_loop().time()
        while True:
            resp = await client.post(
                f"{self._base_url}/v1/memories:retrieve",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            results, memories, abstract = _parse_retrieve_response(resp.text)

            if results or not wait_for_indexing:
                return {"results": results, "memories": memories, "abstractReply": abstract}

            if asyncio.get_event_loop().time() - start >= wait_seconds:
                logger.warning("retrieve_memories: no results after %.0fs of polling", wait_seconds)
                return {"results": results, "memories": memories, "abstractReply": abstract}

            await asyncio.sleep(poll_interval)
