"""autogen-goodmem: GoodMem memory + tools for the AutoGen agent framework."""

from ._client import GoodMemClient
from ._config import ChunkingConfig, GoodMemMemoryConfig, PostProcessorConfig
from ._context_provider import GoodMemContextProvider
from ._tools import TOOL_NAMES, create_goodmem_tools

# Backwards-compatible alias for code that imported GoodMemMemory from the
# autogen-ext monorepo location.
GoodMemMemory = GoodMemContextProvider

__version__ = "0.1.0"

__all__ = [
    "GoodMemClient",
    "GoodMemContextProvider",
    "GoodMemMemory",
    "GoodMemMemoryConfig",
    "ChunkingConfig",
    "PostProcessorConfig",
    "create_goodmem_tools",
    "TOOL_NAMES",
    "__version__",
]
