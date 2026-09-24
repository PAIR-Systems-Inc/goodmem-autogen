# autogen-goodmem

[GoodMem](https://goodmem.ai) memory and tools for the
[AutoGen](https://github.com/microsoft/autogen) agent framework.

Two ways in:

1. **`GoodMemContextProvider`** — an `autogen_core.memory.Memory` backed by a
   GoodMem space, so relevant passages are injected into the model context on
   every turn.
2. **`create_goodmem_search_tool` / `create_goodmem_admin_tools`** — function
   tools an agent can call directly.

Built on the official `goodmem` SDK's async client, so nothing blocks the
event loop.

## Install

```bash
pip install autogen-goodmem
```

Requires Python 3.10+, `autogen-core` 0.7.5+. Version 0.2 is a break from
0.1 — see [CHANGELOG](CHANGELOG.md) for the mapping.

## As an AutoGen `Memory`

```python
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_goodmem import GoodMemContextProvider, GoodMemMemoryConfig

provider = GoodMemContextProvider(
    config=GoodMemMemoryConfig(
        base_url="https://goodmem.example.com",
        api_key="gm_...",              # stored as SecretStr, never serialized
        space_name="handbook",         # or space_id="..." to skip the lookup
        embedder_id="<embedder-uuid>",
    )
)

await provider.add(MemoryContent(
    content="Refunds above $500 need a manager's approval.",
    mime_type=MemoryMimeType.TEXT,
    metadata={"title": "handbook", "category": "policy"},
))

results = await provider.query("who approves a large refund?")
await provider.close()
```

`add()` waits for the memory to finish indexing by default, so a query right
after it finds the result. Searching is never used as a way to wait.

Attach it to an agent and `update_context` injects retrieved passages as a
system message each turn — the same pattern AutoGen's own `ListMemory` uses.

### What a result carries

`MemoryContent` has no score field, so provenance lives in `metadata`:

```python
{
  "title": "handbook", "category": "policy",   # the memory's own metadata
  "chunk_id": "...", "memory_id": "...", "space_id": "...", "source": "...",
  "score": -0.53, "score_kind": "vector",
  "partial": False, "statuses": [],
}
```

- `partial` is `True` when part of the search did not complete — a reranker
  was unavailable, one space was unreachable. The passages are usable but may
  be incomplete, and `statuses` says why. A search that produced nothing
  usable returns an empty result and emits a warning carrying the statuses,
  so it is distinguishable from "no matches" without being raised. The
  search tool returns `partial: true` with `statuses` in its JSON for the
  same case.
- `score` is passed through exactly as GoodMem reports it. Vector scores are
  opaque similarities that may be negative; reranker scores are on a scale
  that depends on the reranker model (Voyage `rerank-2.5` ~`0.27..0.93`, Jina
  `jina-reranker-v3` ~`-0.14..0.43` on the same documents). `score_kind` says
  which you have — which is why `relevance_threshold` requires a
  `reranker_id`, and why it must be calibrated for the reranker in use rather
  than assumed to be 0–1. The threshold is applied by the server; if it
  removes every result the provider warns, since an empty result would
  otherwise read as "no matches".

## As tools

```python
from autogen_goodmem import create_goodmem_search_tool, create_goodmem_admin_tools

search = create_goodmem_search_tool(
    client, space_ids=["<space-id>"], limit=5,
    reranker_id="<reranker-uuid>",            # optional
    metadata_filter={"category": "policy"},   # optional, escaped for you
)
```

The model supplies only the query; spaces, reranking and filters are yours, so
an agent cannot redirect a search or widen it mid-run.

`create_goodmem_admin_tools(client)` adds space and memory management. These
carry the authority of the configured API key — give them only to agents that
need them. File upload is only created when you pass `upload_dir`, and paths
resolving outside that directory are refused before the file is opened.

## Cancellation

`add`, `add_file` and `query` honour an `autogen_core.CancellationToken`: an
already-cancelled token prevents the request, and cancelling mid-flight aborts
it.

## Clearing a space

`clear()` deletes every memory in the space and requires
`allow_clear=True` on the config, so a reflexive `clear()` cannot empty a
space by accident.

## Filters

A filter is a GoodMem expression applied to every configured space, e.g.
`CAST(val('$.category') AS TEXT) = 'policy'`. Pass `metadata_filter={...}` and
it is built and escaped for you. Writing one by hand: inside a quoted value
escape `'` as `\'` and `\` as `\\` — SQL-style `''` doubling is rejected by
the server.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run ruff check autogen_goodmem tests
uv run mypy autogen_goodmem
uv run pytest -m "not integration"   # offline: the real SDK over a mock transport
GOODMEM_BASE_URL=... GOODMEM_API_KEY=... GOODMEM_EMBEDDER_ID=... \
  GOODMEM_RERANKER_ID=... GOODMEM_VERIFY_SSL=false uv run pytest -m integration
```

These are the commands CI runs. `GOODMEM_RERANKER_ID` is optional — the
reranker tests skip without it; `GOODMEM_VERIFY_SSL=false` is for a local
server with a self-signed certificate.

Offline tests use event shapes captured from a live server. There is no
default API key — live tests skip unless the environment provides one.

MIT.
