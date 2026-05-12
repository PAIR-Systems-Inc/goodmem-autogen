"""Configuration classes for the autogen-goodmem integration."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """Recursive chunking config used when creating a space."""

    chunk_size: int = Field(default=256, description="Characters per chunk")
    chunk_overlap: int = Field(default=25, description="Overlap between consecutive chunks")
    keep_strategy: Literal["KEEP_END", "KEEP_START", "DISCARD"] = Field(default="KEEP_END")
    length_measurement: Literal["CHARACTER_COUNT", "TOKEN_COUNT"] = Field(default="CHARACTER_COUNT")
    separators: List[str] = Field(default=["\n\n", "\n", ". ", " ", ""])
    separator_is_regex: bool = Field(default=False)


class PostProcessorConfig(BaseModel):
    """Optional reranker / LLM post-processor used by retrieve_memories."""

    reranker_id: Optional[str] = Field(default=None)
    llm_id: Optional[str] = Field(default=None)
    relevance_threshold: Optional[float] = Field(default=None, description="0-1")
    llm_temperature: Optional[float] = Field(default=None, description="0-2")
    max_results: Optional[int] = Field(default=None)
    chronological_resort: bool = Field(default=False)


class GoodMemMemoryConfig(BaseModel):
    """Configuration for :class:`GoodMemContextProvider`.

    Connects an AutoGen ``Memory`` to a single GoodMem space, automatically
    creating or reusing the space by name on first use.
    """

    base_url: str = Field(description="GoodMem API base URL, e.g. https://localhost:8080")
    api_key: str = Field(description="GoodMem API key (sent as X-API-Key)")
    space_name: str = Field(description="Space to create or reuse")
    embedder_id: str = Field(description="Embedder model ID for this space")
    max_results: int = Field(default=5)
    include_memory_definition: bool = Field(default=True)
    wait_for_indexing: bool = Field(default=True, description="Poll up to ~60s for first results")
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    post_processor: Optional[PostProcessorConfig] = Field(default=None)
    metadata: Optional[Dict[str, str]] = Field(default=None)
    verify_ssl: bool = Field(default=True)
