"""autogen-goodmem: GoodMem memory and tools for the AutoGen agent framework."""

from ._config import ChunkingConfig, GoodMemMemoryConfig, PostProcessorConfig
from ._connection import GoodMemConnection
from ._context_provider import GoodMemContextProvider, GoodMemIngestionError
from ._tools import (
    ADMIN_TOOL_NAMES,
    SEARCH_TOOL_NAME,
    create_goodmem_admin_tools,
    create_goodmem_search_tool,
)
from ._uploads import GoodMemUploadError


__version__ = "0.2.0"

__all__ = [
    "ADMIN_TOOL_NAMES",
    "SEARCH_TOOL_NAME",
    "ChunkingConfig",
    "GoodMemConnection",
    "GoodMemContextProvider",
    "GoodMemIngestionError",
    "GoodMemMemoryConfig",
    "GoodMemUploadError",
    "PostProcessorConfig",
    "__version__",
    "create_goodmem_admin_tools",
    "create_goodmem_search_tool",
]
