"""Configuration for the autogen-goodmem integration."""

from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from ._ids import UuidStr


class PostProcessorConfig(BaseModel):
    """Optional reranker / LLM post-processing applied to retrieval."""

    reranker_id: UuidStr | None = Field(default=None)
    llm_id: UuidStr | None = Field(default=None)
    llm_temperature: float | None = Field(default=None, description="0.0-2.0")
    relevance_threshold: float | None = Field(
        default=None,
        description=(
            "Minimum reranker score, applied by the server. Only meaningful "
            "with reranker_id: a raw vector score is an opaque similarity that "
            "may be negative. The scale depends on the reranker model (Voyage "
            "rerank-2.5 ~0.27..0.93, Jina jina-reranker-v3 ~-0.14..0.43 on the "
            "same documents) and is not necessarily 0-1; calibrate it for the "
            "reranker in use."
        ),
    )
    chronological_resort: bool = Field(default=False)


class GoodMemMemoryConfig(BaseModel):
    """Configuration for :class:`GoodMemContextProvider`.

    ``api_key`` is a ``SecretStr``. AutoGen serializes a memory's config
    whenever the owning agent is dumped (``AssistantAgent._to_config`` calls
    ``memory.dump_component()``), and a plain ``str`` is written out verbatim.

    ID fields must be UUIDs and are lowercased; anything else fails
    validation, because the SDK puts IDs into request paths unescaped.
    """

    base_url: str = Field(description="GoodMem API base URL, e.g. https://goodmem.example.com")
    api_key: SecretStr = Field(description="GoodMem API key (sent as X-API-Key)")
    space_id: UuidStr | None = Field(
        default=None,
        description="Space to use (its UUID). Takes precedence over space_name.",
    )
    space_name: str | None = Field(
        default=None,
        description=(
            "Space to attach to by name, created on first use if absent. "
            "Reused only when its embedder matches embedder_id; an ambiguous "
            "name is an error."
        ),
    )
    embedder_id: UuidStr | None = Field(
        default=None,
        description="Embedder for the space (its UUID). Required when creating by space_name.",
    )
    max_results: int = Field(default=5, gt=0)
    fetch_k: int | None = Field(
        default=None, gt=0, description="Candidates to fetch before reranking."
    )
    filter: str | None = Field(
        default=None,
        description="A GoodMem filter expression applied to the space on every query.",
    )
    metadata: dict[str, str] | None = Field(
        default=None, description="Metadata attached to everything this provider writes."
    )
    post_processor: PostProcessorConfig | None = Field(default=None)
    upload_dir: str | None = Field(
        default=None,
        description=(
            "Directory whose files add_file() may read. Required for file "
            "uploads; paths resolving outside it are refused."
        ),
    )
    allow_clear: bool = Field(
        default=False,
        description=(
            "Permit clear() to delete every memory in the space. Off by "
            "default so a reflexive clear() cannot empty a space."
        ),
    )
    wait_for_indexing: bool = Field(
        default=True,
        description=(
            "Wait for each written memory to finish indexing before add() "
            "returns, so a following query finds it. Searching is never used "
            "as a way to wait."
        ),
    )
    indexing_timeout: float = Field(default=120.0, gt=0)
    verify_ssl: bool = Field(default=True)
    timeout: float = Field(default=60.0, gt=0)


class ChunkingConfig(BaseModel):
    """Recursive chunking used when this provider creates a space."""

    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    keep_strategy: Literal["KEEP_END", "KEEP_START", "DISCARD"] = "KEEP_END"
    length_measurement: Literal["CHARACTER_COUNT", "TOKEN_COUNT"] = "CHARACTER_COUNT"

    def to_api(self) -> dict[str, object]:
        return {
            "recursive": {
                "chunkSize": self.chunk_size,
                "chunkOverlap": self.chunk_overlap,
                "keepStrategy": self.keep_strategy,
                "lengthMeasurement": self.length_measurement,
            }
        }
