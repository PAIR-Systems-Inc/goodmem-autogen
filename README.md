# autogen-goodmem

[GoodMem](https://goodmem.ai) memory and tools for the [AutoGen](https://github.com/microsoft/autogen) agent framework.

`autogen-goodmem` gives an AutoGen agent two things:

1. **`GoodMemContextProvider`** — an `autogen_core.memory.Memory` implementation backed by a GoodMem space, so context retrieval is automatic on every turn.
2. **`create_goodmem_tools(client)`** — 11 `FunctionTool`s the agent can call directly to manage spaces and memories.

## Install

```bash
pip install autogen-goodmem
```

## Quickstart

### As an AutoGen `Memory`

```python
import asyncio
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_goodmem import GoodMemContextProvider, GoodMemMemoryConfig


async def main() -> None:
    provider = GoodMemContextProvider(
        config=GoodMemMemoryConfig(
            base_url="https://localhost:8080",
            api_key="gm_...",
            space_name="my-kb",
            embedder_id="<embedder-uuid>",
            verify_ssl=False,
        )
    )
    await provider.add(MemoryContent(content="The capital of France is Paris.", mime_type=MemoryMimeType.TEXT))
    results = await provider.query("What is the capital of France?")
    for r in results.results:
        print(r.content)
    await provider.close()


asyncio.run(main())
```

### As tools on an agent

```python
from autogen_goodmem import GoodMemClient, create_goodmem_tools

client = GoodMemClient(base_url="https://localhost:8080", api_key="gm_...", verify_ssl=False)
tools = create_goodmem_tools(client)  # 11 FunctionTools
# pass `tools=tools` to your AssistantAgent
```

## Tool surface

`create_goodmem_tools(client)` returns these 11 tools in order:

| Tool | Purpose |
| --- | --- |
| `goodmem_list_embedders` | List embedder models available on the server. |
| `goodmem_list_spaces` | List all spaces visible to the API key. |
| `goodmem_get_space` | Fetch one space by ID. |
| `goodmem_create_space` | Create a space (idempotent by name). |
| `goodmem_update_space` | Update name / publicRead / labels / chunking. |
| `goodmem_delete_space` | Delete a space and everything in it. |
| `goodmem_create_memory` | Add a memory from text or a local file. |
| `goodmem_list_memories` | Paginated listing of a space's memories. |
| `goodmem_retrieve_memories` | Semantic retrieval, optional reranker + LLM. |
| `goodmem_get_memory` | Fetch a memory's metadata and original content. |
| `goodmem_delete_memory` | Delete a memory and its embeddings. |

`goodmem_retrieve_memories` (and `GoodMemClient.retrieve_memories`) accept the optional post-processor params: `reranker_id`, `llm_id`, `relevance_threshold` (0-1), `llm_temperature` (0-2), `max_results`, `chronological_resort`.

## Integration tests

The tests in `tests/test_goodmem_integration.py` exercise every public method against a live GoodMem server. They are opt-in via the `integration` marker.

```bash
pip install -e ".[dev]"

export GOODMEM_API_KEY=gm_...
export GOODMEM_BASE_URL=https://localhost:8080
export GOODMEM_EMBEDDER_ID=<uuid>
export GOODMEM_RERANKER_ID=<uuid>
export GOODMEM_LLM_ID=<uuid>
export GOODMEM_PDF_PATH=/path/to/sample.pdf   # optional

python -m pytest tests/test_goodmem_integration.py -v -s -m integration
```

## License

MIT
