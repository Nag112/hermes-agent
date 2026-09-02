"""Qdrant RAG memory provider — payload parsing, recall caps, and tool dispatch."""

from __future__ import annotations

import json
from pathlib import Path

from plugins.memory.qdrant.client import RagClient, format_hits, payload_text
from plugins.memory.qdrant import QdrantMemoryProvider


def test_payload_text_prefers_common_rag_fields():
    assert payload_text({"page_content": "from langchain", "id": 1}) == "from langchain"
    assert payload_text({"content": "generic"}) == "generic"
    assert payload_text({"text": "canonical"}) == "canonical"
    assert payload_text({"other": "nope"}) == ""
    assert payload_text({"body": "fallback"}, extra_keys=("unused",)) == "fallback"
    assert payload_text({"note": "custom"}, extra_keys=("note",)) == "custom"


def test_format_hits_caps_injected_context():
    hits = [
        {"score": 0.91, "payload": {"text": "alpha " * 80}},
        {"score": 0.80, "payload": {"text": "beta"}},
        {"score": 0.10, "payload": {"text": "gamma"}},
    ]
    block = format_hits(hits, max_chars=120, snippet_chars=40)
    assert block.startswith("## Recalled memory")
    assert "alpha" in block
    assert len(block) <= 160


def test_format_hits_empty_when_no_text():
    assert format_hits([{"score": 1, "payload": {"id": "x"}}]) == ""


class _FakeTransport:
    def __init__(self):
        self.calls = []
        self.search_result = [
            {"id": "m1", "score": 0.88, "payload": {"text": "User prefers dark mode"}},
        ]

    def __call__(self, method, url, json=None, headers=None, timeout=8.0):
        self.calls.append((method, url, json))
        if url.endswith("/api/embeddings"):
            return {"embedding": [0.1, 0.2, 0.3]}
        if url.endswith("/v1/embeddings"):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        if url.endswith("/points/search"):
            return {"result": self.search_result}
        if "/points?wait=true" in url and method == "PUT":
            return {"result": {"status": "ok"}}
        if "/points/delete" in url:
            return {"result": {"status": "ok"}}
        if method == "GET" and "/collections/" in url:
            return {"result": {"status": "green"}}
        if method == "PUT" and url.rstrip("/").endswith("/hermes-memory"):
            return {"result": True}
        return {}


def _provider_with_fake(transport=None, **config):
    transport = transport or _FakeTransport()
    provider = QdrantMemoryProvider(config=config or {"collection": "hermes-memory"})
    provider.initialize("sess-1", platform="cli", agent_context="primary")
    provider._client = RagClient(
        collection=config.get("collection", "hermes-memory"),
        request_fn=transport,
        embed_model="nomic-embed-text",
    )
    return provider, transport


def test_prefetch_injects_top_hits_only():
    provider, _ = _provider_with_fake()
    block = provider.prefetch("what theme does the user like?")
    assert "dark mode" in block
    status = provider.recall_status()
    assert status is not None
    assert status.count == 1
    assert status.provider_label == "qdrant"


def test_prefetch_skips_trivial_prompts():
    provider, transport = _provider_with_fake()
    assert provider.prefetch("ok") == ""
    assert provider.prefetch("/help") == ""
    assert not any(url.endswith("/points/search") for _, url, _ in transport.calls)


def test_search_tool_returns_compact_json():
    provider, _ = _provider_with_fake()
    raw = provider.handle_tool_call("qdrant_search", {"query": "theme"})
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["count"] == 1
    assert payload["results"][0]["text"] == "User prefers dark mode"


def test_store_tool_upserts_with_text_payload():
    provider, transport = _provider_with_fake()
    raw = provider.handle_tool_call("qdrant_store", {"content": "Ship via Qdrant only."})
    payload = json.loads(raw)
    assert payload["success"] is True
    upserts = [c for c in transport.calls if c[0] == "PUT" and "/points" in c[1]]
    assert upserts
    point = upserts[0][2]["points"][0]
    assert point["payload"]["text"] == "Ship via Qdrant only."
    assert point["payload"]["kind"] == "fact"
    assert len(point["vector"]) == 3


def test_forget_tool_deletes_point():
    provider, transport = _provider_with_fake()
    raw = provider.handle_tool_call("qdrant_forget", {"memory_id": "m1"})
    assert json.loads(raw)["success"] is True
    deletes = [c for c in transport.calls if "/points/delete" in c[1]]
    assert deletes[0][2] == {"points": ["m1"]}


def test_auto_ingest_off_by_default():
    provider, transport = _provider_with_fake()
    provider.sync_turn("Please remember that I use neovim.", "Got it.")
    assert not any("/points" in url and method == "PUT" for method, url, _ in transport.calls)


def test_auto_ingest_stores_substantial_turns():
    provider, transport = _provider_with_fake(auto_ingest=True)
    provider.sync_turn("Please remember that I use neovim daily.", "Noted.")
    upserts = [c for c in transport.calls if c[0] == "PUT" and "/points" in c[1]]
    assert upserts
    assert "neovim" in upserts[0][2]["points"][0]["payload"]["text"].lower()


def test_cron_context_does_not_write():
    provider, transport = _provider_with_fake(auto_ingest=True)
    provider.initialize("sess-2", platform="cron", agent_context="cron")
    provider._client = RagClient(request_fn=transport)
    provider.sync_turn("Please remember that I use neovim daily.", "Noted.")
    result = json.loads(provider.handle_tool_call("qdrant_store", {"content": "should fail"}))
    assert result.get("success") is not True
    assert not any(c[0] == "PUT" and "/points" in c[1] for c in transport.calls)


def test_save_config_disables_file_memory(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model:\n  default: test\n", encoding="utf-8")

    from hermes_cli import config as config_mod

    monkeypatch.setattr(config_mod, "read_user_config_raw", lambda path=None: {
        "model": {"default": "test"},
        "memory": {"memory_enabled": True, "user_profile_enabled": True},
    })

    written = {}

    def _write(path, data, **kwargs):
        written["path"] = Path(path)
        written["data"] = data

    monkeypatch.setattr("utils.atomic_yaml_write", _write)

    provider = QdrantMemoryProvider(config={})
    provider.save_config(
        {"qdrant_url": "http://localhost:6333", "collection": "docs", "qdrant_api_key": "secret"},
        str(hermes_home),
    )

    memory = written["data"]["memory"]
    assert memory["provider"] == "qdrant"
    assert memory["memory_enabled"] is False
    assert memory["user_profile_enabled"] is False
    assert memory["qdrant"]["collection"] == "docs"
    assert "qdrant_api_key" not in memory["qdrant"]


def test_system_prompt_does_not_mention_memory_files():
    provider, _ = _provider_with_fake()
    block = provider.system_prompt_block()
    assert "Qdrant" in block
    assert "Do not write MEMORY.md" in block


def test_openai_compat_embed_path():
    transport = _FakeTransport()
    client = RagClient(embedder="openai_compat", embed_url="http://localhost:11434", request_fn=transport)
    vector = client.embed("hello")
    assert vector == [0.1, 0.2, 0.3]
    assert transport.calls[0][1].endswith("/v1/embeddings")


def test_provider_is_discovered():
    from plugins.memory import list_memory_provider_names

    assert "qdrant" in list_memory_provider_names()
