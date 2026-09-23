# Changelog

## 0.2.0

0.2 is a deliberate API break. The integration uses the official `goodmem`
SDK — natively async, so nothing blocks the event loop — and narrows what an
agent is allowed to reach.

Every defect below was reproduced against the `v0.1.0` tag on a live server
(v1.0.320) or a local mock before being fixed. v0.1.0 was never published to
PyPI, so there is no released artifact to migrate from.

### Security

- **A live GoodMem API key was committed as a default in the test file** and
  had been in a public repository since the initial commit. It has been
  removed; the tests now require `GOODMEM_API_KEY` and skip when it is unset
  rather than reaching for a built-in credential. The key was rotated and the
  history rewritten separately.
- **The API key is no longer serialized.** `api_key` is a `SecretStr`.
  AutoGen calls `memory.dump_component()` for every memory attached to an
  agent (`AssistantAgent._to_config`), and a plain `str` was written out in
  clear text; it now renders as `**********`.
- **File uploads are confined to a configured directory.** `file_path` was a
  model-facing argument on the create-memory tool with no restriction, so an
  agent could name any path the process could read. Uploads now require
  `upload_dir`, refuse paths and symlinks resolving outside it, and the
  upload tool is not created at all unless a directory is configured.

### Fixed

- **A failed search no longer looks like a successful one.** Retrieval
  statuses were parsed and discarded: asking for reranking with an
  unavailable reranker returned unreranked chunks and reported nothing wrong.
  Results now carry `partial` and `statuses` in their metadata; a search
  that produced nothing usable returns empty with `partial: true` (the tool)
  or a warning carrying the statuses (the memory provider), and is never
  raised.
- **`FEATURE_DISABLED` is informational by its code alone.** The server
  defines it as "feature disabled due to missing configuration", so it never
  means a requested feature was lost; the details are not inspected.
- **Statuses from a newer server no longer break retrieval.** Codes the SDK
  does not recognize are reported as `UNKNOWN` and mark results partial; they
  never discard chunks and never raise. A truncated NDJSON line was silently
  skipped and is now reported.
- **`clear()` does what it says.** It previously logged a warning and
  returned, having deleted nothing, while satisfying an `@abstractmethod` of
  AutoGen's `Memory` ABC. It now deletes every memory in the space, and
  requires `allow_clear=True` so a reflexive `clear()` cannot empty a space
  by accident.
- **Cancellation tokens are honoured.** `add`, `add_file` and `query`
  accepted a `CancellationToken` and ignored it. Work is now wrapped with
  `CancellationToken.link_future`, so an already-cancelled token prevents the
  request and a token cancelled mid-flight aborts it.
- **Querying no longer polls.** `wait_for_indexing` defaulted to `True` on the
  provider and retried empty searches, so a query against an empty space cost
  64 seconds. `add()` waits for the memory it just wrote instead; searching is
  never used as a way to wait for indexing.
- **Space reuse is safe.** Attaching by name reused a same-named space and
  reported back the embedder that was *asked for* rather than the one the
  space uses, so a caller was told a space indexed with Voyage used Qwen3.
  Reuse now requires the embedder to match; a mismatch or an ambiguous name
  is an error, and lookup uses a server-side `name_filter` instead of
  scanning the first page.
- **`public_read` is gone.** GoodMem removed the field; sending it failed the
  whole update with HTTP 400.
- **Retrieved passages carry their source.** The memory definition was parsed
  and then thrown away, so stored `title`/`category` never reached the agent.
  Chunks are joined to their memory by UUID regardless of event order, and
  deduplicated by chunk ID so two passages from one document stay two results.
- **Get-memory reads content in one request**, decoded to text, instead of a
  second call to `/content` that reported success with the content missing
  when it failed.
- **`relevance_threshold` requires a reranker.** It was documented as a 0-1
  score; real vector scores are opaque and routinely negative (a live capture
  returned `-0.5345`). Results record `score_kind` so a reranker score is
  never compared against a vector score.
- `update_context` no longer searches for `str(message)` when the last turn
  is not plain text.

### Added

- `create_goodmem_search_tool(...)` — a read-only search tool whose only
  model-supplied argument is the query. Spaces, reranking and filters are
  developer configuration, so a model cannot redirect a search mid-run.
- Metadata filters, built by `autogen_goodmem.filters` with the escaping the
  server actually accepts (backslash; SQL-style `''` doubling is rejected),
  refusing values it cannot encode safely.
- `space_id` on the config, to attach to an existing space without a
  name lookup. Client injection, so a caller can share one `AsyncGoodmem`.

### Migration from 0.1

| 0.1 | 0.2 |
| --- | --- |
| `GoodMemClient` | The official `AsyncGoodmem` SDK client; pass one with `client=` |
| `create_goodmem_tools(client)` | `create_goodmem_search_tool(...)` and `create_goodmem_admin_tools(...)` |
| `GoodMemMemoryConfig(api_key="...")` | Same, but the field is a `SecretStr` |
| `space_name` required | `space_id` or `space_name`; `space_id` skips the lookup |
| `wait_for_indexing` on queries | Removed. `add()` waits for its own write |
| `file_path` on create-memory | `upload_dir` + `add_file()` / `goodmem_upload_file` |
| `public_read` on update-space | Removed; use labels |
| `clear()` silently did nothing | Deletes for real; needs `allow_clear=True` |
| `query()` results carried only chunk ids | Results carry the memory's metadata, `score`, `score_kind`, `partial`, `statuses` |
| `GoodMemMemory` alias | Removed; use `GoodMemContextProvider` |

## 0.1.0

Initial release: an `autogen_core.memory.Memory` provider and 11 function
tools over a hand-written REST client. Never published to PyPI.
