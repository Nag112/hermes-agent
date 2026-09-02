# Qdrant RAG Memory Provider

Replaces Hermes' file-backed `MEMORY.md` / `USER.md` injection with semantic
search against a Qdrant collection. Embeddings default to local Ollama.

Each turn injects only the top matching chunks (capped by character count).
Nothing is dumped into the system prompt.

## Requirements

- A running [Qdrant](https://qdrant.tech/) instance (REST, default `http://localhost:6333`)
- An embedding endpoint:
  - [Ollama](https://ollama.com/) with an embedding model (default `nomic-embed-text`), or
  - any OpenAI-compatible `/v1/embeddings` server

## Setup

```bash
hermes memory setup    # select "qdrant"
```

Or manually in `config.yaml`:

```yaml
memory:
  provider: qdrant
  memory_enabled: false          # stop injecting MEMORY.md
  user_profile_enabled: false    # stop injecting USER.md
  qdrant:
    qdrant_url: http://localhost:6333
    collection: hermes-memory    # or your existing RAG collection
    vector_name: ""              # set if the collection uses named vectors
    embedder: ollama             # or openai_compat
    embed_url: http://localhost:11434
    embed_model: nomic-embed-text
    top_k: 5
    score_threshold: 0.25
    max_inject_chars: 2500
    auto_ingest: false           # true = also embed each completed turn
```

If Qdrant needs a key:

```bash
echo "QDRANT_API_KEY=..." >> ~/.hermes/.env
```

Start a **new** session after changing the provider (prompt cache / toolset).

## Existing RAG collections

Point `collection` at the collection you already indexed. The provider reads
payload text from, in order: `text`, `content`, `page_content`, `document`,
`chunk`, `memory`, `body`. Override with `payload_text_keys` (list or
comma-separated string) if your schema uses another field.

The embedding model **must** match the one used to build the collection
(same dimensions). `nomic-embed-text` is 768-d.

## Tools

| Tool | Description |
|------|-------------|
| `qdrant_search` | On-demand semantic search |
| `qdrant_store` | Persist a compact fact |
| `qdrant_forget` | Delete by point id |

## Why not MCP / Mem0?

MCP search adds a tool round-trip and extra schema on every call. Mem0 OSS
runs an LLM extractor in-process. This provider embeds the user query,
searches Qdrant, and injects a few lines — no second model, no files.
