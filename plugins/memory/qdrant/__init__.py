"""Qdrant RAG memory provider — vector recall instead of MEMORY.md / USER.md.

Talks to a running Qdrant collection and an embedding endpoint (Ollama by
default). Prefetch injects only the top-k matching chunks so the system
prompt stays small. Durable writes go to Qdrant, not files.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import RecallStatus, MemoryProvider, is_trivial_prompt
from agent.secret_scope import get_secret
from tools.registry import tool_error
from utils import is_truthy_value

from .client import RagClient, QdrantClientError, format_hits, payload_text

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "qdrant_url": "http://localhost:6333",
    "collection": "hermes-memory",
    "vector_name": "",
    "embedder": "ollama",
    "embed_url": "http://localhost:11434",
    "embed_model": "nomic-embed-text",
    "top_k": 5,
    "score_threshold": 0.25,
    "max_inject_chars": 2500,
    "auto_ingest": False,
    "create_collection": True,
}

SEARCH_SCHEMA = {
    "name": "qdrant_search",
    "description": (
        "Semantic search over long-term vector memory. Use when prefetch "
        "did not cover a specific fact, preference, or prior decision."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look up."},
            "top_k": {"type": "integer", "description": "Max results (default 5, max 20)."},
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "qdrant_store",
    "description": (
        "Persist a durable fact, preference, or decision to vector memory. "
        "Keep it one idea, one sentence. Do not store secrets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to remember."},
            "kind": {
                "type": "string",
                "enum": ["fact", "preference", "decision", "context"],
                "description": "Optional category (default: fact).",
            },
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "qdrant_forget",
    "description": "Delete a stored memory by the id returned from search or store.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Qdrant point id."},
        },
        "required": ["memory_id"],
    },
}


def _load_plugin_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        memory_config = config.get("memory", {}) if isinstance(config, dict) else {}
        if not isinstance(memory_config, dict):
            return {}
        provider_config = memory_config.get("qdrant", {})
        return dict(provider_config) if isinstance(provider_config, dict) else {}
    except Exception:
        return {}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


class QdrantMemoryProvider(MemoryProvider):
    """RAG memory over Qdrant + embeddings. No file-based MEMORY.md injection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config) if config is not None else _load_plugin_config()
        self._client: Optional[RagClient] = None
        self._session_id = ""
        self._writes_enabled = True
        self._prefetch_lock = threading.Lock()
        self._prefetch_text = ""
        self._prefetch_query = ""
        self._prefetch_count = 0
        self._prefetch_thread: Optional[threading.Thread] = None
        self._last_error = ""

    @property
    def name(self) -> str:
        return "qdrant"

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        if self._last_error:
            return self._last_error
        return (
            "Qdrant RAG uses a running Qdrant server and an embedding endpoint. "
            "Set memory.qdrant.qdrant_url / embed_url in config.yaml."
        )

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "qdrant_url",
                "description": "Qdrant REST URL",
                "default": _DEFAULTS["qdrant_url"],
            },
            {
                "key": "collection",
                "description": "Qdrant collection name (existing RAG collection is fine)",
                "default": _DEFAULTS["collection"],
            },
            {
                "key": "vector_name",
                "description": "Named vector in the collection (blank = default unnamed vector)",
                "default": "",
            },
            {
                "key": "embedder",
                "description": "Embedding backend",
                "default": "ollama",
                "choices": ["ollama", "openai_compat"],
            },
            {
                "key": "embed_url",
                "description": "Embedding endpoint (Ollama or OpenAI-compatible base URL)",
                "default": _DEFAULTS["embed_url"],
            },
            {
                "key": "embed_model",
                "description": "Embedding model name",
                "default": _DEFAULTS["embed_model"],
            },
            {
                "key": "top_k",
                "description": "Memories to inject per turn",
                "default": str(_DEFAULTS["top_k"]),
            },
            {
                "key": "score_threshold",
                "description": "Drop hits below this cosine score (0-1)",
                "default": str(_DEFAULTS["score_threshold"]),
            },
            {
                "key": "max_inject_chars",
                "description": "Hard cap on recalled text injected into context",
                "default": str(_DEFAULTS["max_inject_chars"]),
            },
            {
                "key": "auto_ingest",
                "description": "Also embed completed turns into Qdrant (off = explicit store only)",
                "default": "false",
                "choices": ["true", "false"],
            },
            {
                "key": "qdrant_api_key",
                "description": "Qdrant API key (if the server requires one)",
                "secret": True,
                "env_var": "QDRANT_API_KEY",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Persist provider settings and disable file-based MEMORY.md / USER.md."""
        from hermes_cli.config import read_user_config_raw
        from utils import atomic_yaml_write

        config_path = Path(hermes_home) / "config.yaml"
        existing = read_user_config_raw(config_path)
        if not isinstance(existing, dict):
            existing = {}
        memory = existing.setdefault("memory", {})
        if not isinstance(memory, dict):
            memory = {}
            existing["memory"] = memory
        memory["provider"] = "qdrant"
        memory["memory_enabled"] = False
        memory["user_profile_enabled"] = False
        cleaned = {k: v for k, v in dict(values).items() if k != "qdrant_api_key"}
        memory["qdrant"] = cleaned
        atomic_yaml_write(config_path, existing, default_flow_style=False, sort_keys=False)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        agent_context = kwargs.get("agent_context", "")
        platform = kwargs.get("platform", "")
        self._writes_enabled = agent_context not in {"cron", "flush", "subagent"} and platform != "cron"

        cfg = self._config
        try:
            api_key = _as_str(get_secret("QDRANT_API_KEY"), "")
        except Exception:
            api_key = _as_str(os.environ.get("QDRANT_API_KEY"), "")
        try:
            embed_key = _as_str(get_secret("OPENAI_API_KEY"), "")
        except Exception:
            embed_key = _as_str(os.environ.get("OPENAI_API_KEY"), "")

        self._client = RagClient(
            qdrant_url=_as_str(cfg.get("qdrant_url"), _DEFAULTS["qdrant_url"]),
            qdrant_api_key=api_key,
            collection=_as_str(cfg.get("collection"), _DEFAULTS["collection"]),
            vector_name=_as_str(cfg.get("vector_name"), ""),
            embed_url=_as_str(cfg.get("embed_url"), _DEFAULTS["embed_url"]),
            embed_model=_as_str(cfg.get("embed_model"), _DEFAULTS["embed_model"]),
            embedder=_as_str(cfg.get("embedder"), _DEFAULTS["embedder"]),
            embed_api_key=embed_key,
            timeout_s=_as_float(cfg.get("timeout_s"), 8.0),
        )
        self._last_error = ""
        if is_truthy_value(cfg.get("create_collection"), default=True):
            try:
                info = self._client.collection_info()
                if info is None:
                    logger.info(
                        "Qdrant collection '%s' not found yet — it will be created on first store",
                        self._client.collection,
                    )
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("Qdrant collection probe failed: %s", exc)

    def system_prompt_block(self) -> str:
        collection = _as_str(self._config.get("collection"), _DEFAULTS["collection"])
        return (
            "# Vector memory (Qdrant RAG)\n"
            f"Active collection: `{collection}`. Relevant memories are injected "
            "automatically each turn — only a few top hits, not a dump.\n"
            "Use `qdrant_store` for durable facts the user would expect you to keep. "
            "Use `qdrant_search` only when you need a more specific lookup. "
            "Do not write MEMORY.md or USER.md files."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if is_trivial_prompt(query):
            with self._prefetch_lock:
                self._prefetch_text = ""
                self._prefetch_count = 0
                self._prefetch_query = query or ""
            return ""
        with self._prefetch_lock:
            cached_query = self._prefetch_query
            cached_text = self._prefetch_text
            thread = self._prefetch_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=6.0)
            with self._prefetch_lock:
                cached_query = self._prefetch_query
                cached_text = self._prefetch_text
        if cached_query == query and cached_text:
            return cached_text
        text, count = self._search_block(query)
        with self._prefetch_lock:
            self._prefetch_query = query
            self._prefetch_text = text
            self._prefetch_count = count
        return text

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if is_trivial_prompt(query) or not self._client:
            return
        with self._prefetch_lock:
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                return
            thread = threading.Thread(
                target=self._prefetch_worker,
                args=(query,),
                daemon=True,
                name="qdrant-prefetch",
            )
            self._prefetch_thread = thread
        thread.start()

    def recall_status(self) -> Optional[RecallStatus]:
        with self._prefetch_lock:
            count = self._prefetch_count
            text = self._prefetch_text
        if not text:
            return None
        return RecallStatus(provider_label="qdrant", count=count, glyph="🧭")

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not self._writes_enabled:
            return
        if not is_truthy_value(self._config.get("auto_ingest"), default=False):
            return
        user = (user_content or "").strip()
        if is_trivial_prompt(user) or len(user) < 24:
            return
        assistant = (assistant_content or "").strip()
        note = user if not assistant else f"User: {user}\nAssistant: {assistant[:400]}"
        if len(note) > 800:
            note = note[:799].rstrip() + "…"
        try:
            self._store(note, kind="context")
        except Exception as exc:
            logger.debug("Qdrant auto-ingest skipped: %s", exc)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, STORE_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "qdrant_search":
                return self._handle_search(args)
            if tool_name == "qdrant_store":
                return self._handle_store(args)
            if tool_name == "qdrant_forget":
                return self._handle_forget(args)
        except QdrantClientError as exc:
            return tool_error(str(exc))
        except Exception as exc:
            logger.exception("Qdrant tool %s failed", tool_name)
            return tool_error(str(exc))
        return tool_error(f"Unknown tool: {tool_name}")

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if action != "add" or not content or not self._writes_enabled:
            return
        kind = "preference" if target == "user" else "fact"
        try:
            self._store(content, kind=kind)
        except Exception as exc:
            logger.debug("Qdrant mirror of built-in write failed: %s", exc)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        self._session_id = new_session_id or ""
        with self._prefetch_lock:
            self._prefetch_text = ""
            self._prefetch_query = ""
            self._prefetch_count = 0

    def shutdown(self) -> None:
        with self._prefetch_lock:
            thread = self._prefetch_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._client = None

    # -- internals ----------------------------------------------------------

    def _prefetch_worker(self, query: str) -> None:
        text, count = self._search_block(query)
        with self._prefetch_lock:
            self._prefetch_query = query
            self._prefetch_text = text
            self._prefetch_count = count

    def _search_block(self, query: str) -> tuple[str, int]:
        if not self._client or not (query or "").strip():
            return "", 0
        top_k = max(1, min(_as_int(self._config.get("top_k"), _DEFAULTS["top_k"]), 20))
        threshold = _as_float(self._config.get("score_threshold"), _DEFAULTS["score_threshold"])
        max_chars = _as_int(self._config.get("max_inject_chars"), _DEFAULTS["max_inject_chars"])
        extra_keys = self._config.get("payload_text_keys") or ()
        if isinstance(extra_keys, str):
            extra_keys = [k.strip() for k in extra_keys.split(",") if k.strip()]
        try:
            hits = self._client.search(
                query,
                top_k=top_k,
                score_threshold=threshold if threshold > 0 else None,
            )
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Qdrant prefetch failed: %s", exc)
            return "", 0
        block = format_hits(
            hits,
            max_chars=max_chars,
            extra_keys=extra_keys,
        )
        return block, len(hits) if block else 0

    def _store(self, content: str, *, kind: str = "fact") -> str:
        if not self._client:
            raise QdrantClientError("Qdrant client is not initialized")
        content = (content or "").strip()
        if not content:
            raise QdrantClientError("Nothing to store")
        point_id = str(uuid.uuid4())
        payload = {
            "text": content,
            "kind": kind,
            "session_id": self._session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "hermes",
        }
        return self._client.upsert(point_id, content, payload)

    def _handle_search(self, args: Dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        top_k = _as_int(args.get("top_k"), _as_int(self._config.get("top_k"), 5))
        extra_keys = self._config.get("payload_text_keys") or ()
        if isinstance(extra_keys, str):
            extra_keys = [k.strip() for k in extra_keys.split(",") if k.strip()]
        hits = self._client.search(query, top_k=top_k) if self._client else []
        compact = [
            {
                "id": hit.get("id"),
                "score": hit.get("score"),
                "text": payload_text(hit.get("payload"), extra_keys),
            }
            for hit in hits
        ]
        return json.dumps({"success": True, "count": len(compact), "results": compact})

    def _handle_store(self, args: Dict[str, Any]) -> str:
        if not self._writes_enabled:
            return tool_error("Memory writes are disabled in this agent context")
        content = (args.get("content") or "").strip()
        kind = _as_str(args.get("kind"), "fact")
        point_id = self._store(content, kind=kind)
        return json.dumps({"success": True, "id": point_id})

    def _handle_forget(self, args: Dict[str, Any]) -> str:
        if not self._writes_enabled:
            return tool_error("Memory writes are disabled in this agent context")
        memory_id = _as_str(args.get("memory_id"), "")
        if not memory_id:
            return tool_error("memory_id is required")
        if not self._client:
            return tool_error("Qdrant client is not initialized")
        self._client.delete(memory_id)
        return json.dumps({"success": True, "id": memory_id})


def register(ctx) -> None:
    ctx.register_memory_provider(QdrantMemoryProvider())
