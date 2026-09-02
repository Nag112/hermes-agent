"""HTTP client for Qdrant REST + embedding providers (Ollama / OpenAI-compat)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

DEFAULT_PAYLOAD_TEXT_KEYS = (
    "text",
    "content",
    "page_content",
    "document",
    "chunk",
    "memory",
    "body",
)

RequestFn = Callable[..., Any]


class QdrantClientError(RuntimeError):
    """Raised when Qdrant or the embedder returns an unusable response."""


def payload_text(payload: Optional[Dict[str, Any]], extra_keys: Sequence[str] = ()) -> str:
    """Extract the human-readable chunk from a Qdrant payload."""
    if not isinstance(payload, dict):
        return ""
    keys: List[str] = []
    for key in extra_keys:
        if key and key not in keys:
            keys.append(str(key))
    for key in DEFAULT_PAYLOAD_TEXT_KEYS:
        if key not in keys:
            keys.append(key)
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def format_hits(
    hits: Sequence[Dict[str, Any]],
    *,
    max_chars: int = 2500,
    snippet_chars: int = 400,
    extra_keys: Sequence[str] = (),
) -> str:
    """Render search hits as a compact context block."""
    if not hits:
        return ""
    lines = ["## Recalled memory"]
    used = len(lines[0])
    for hit in hits:
        text = payload_text(hit.get("payload"), extra_keys)
        if not text:
            continue
        if len(text) > snippet_chars:
            text = text[: snippet_chars - 1].rstrip() + "…"
        score = hit.get("score")
        try:
            score_s = f"{float(score):.2f}"
        except (TypeError, ValueError):
            score_s = "?"
        line = f"- ({score_s}) {text}"
        extra = len(line) + 1
        if used + extra > max_chars and len(lines) > 1:
            break
        lines.append(line)
        used += extra
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


class RagClient:
    """Embed + search/upsert against a running Qdrant collection."""

    def __init__(
        self,
        *,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str = "",
        collection: str = "hermes-memory",
        vector_name: str = "",
        embed_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        embedder: str = "ollama",
        embed_api_key: str = "",
        timeout_s: float = 8.0,
        request_fn: Optional[RequestFn] = None,
    ) -> None:
        self.qdrant_url = qdrant_url.rstrip("/")
        self.qdrant_api_key = qdrant_api_key
        self.collection = collection
        self.vector_name = vector_name.strip()
        self.embed_url = embed_url.rstrip("/")
        self.embed_model = embed_model
        self.embedder = (embedder or "ollama").strip().lower()
        self.embed_api_key = embed_api_key
        self.timeout_s = timeout_s
        self._request = request_fn or _httpx_json
        self._vector_size: Optional[int] = None

    def embed(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            raise QdrantClientError("Cannot embed empty text")
        if self.embedder in {"openai", "openai_compat", "openai-compat"}:
            vector = self._embed_openai(text)
        else:
            vector = self._embed_ollama(text)
        if not vector:
            raise QdrantClientError("Embedder returned an empty vector")
        self._vector_size = len(vector)
        return vector

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        vector = self.embed(query)
        body: Dict[str, Any] = {
            "limit": max(1, min(int(top_k), 20)),
            "with_payload": True,
            "with_vector": False,
        }
        if self.vector_name:
            body["vector"] = {"name": self.vector_name, "vector": vector}
        else:
            body["vector"] = vector
        if score_threshold is not None:
            body["score_threshold"] = float(score_threshold)
        data = self._qdrant(
            "POST",
            f"/collections/{quote(self.collection, safe='')}/points/search",
            body,
        )
        result = data.get("result") if isinstance(data, dict) else None
        if not isinstance(result, list):
            return []
        hits: List[Dict[str, Any]] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            hits.append(
                {
                    "id": item.get("id"),
                    "score": item.get("score"),
                    "payload": item.get("payload") if isinstance(item.get("payload"), dict) else {},
                }
            )
        return hits

    def upsert(self, point_id: str, text: str, payload: Optional[Dict[str, Any]] = None) -> str:
        vector = self.embed(text)
        merged = dict(payload or {})
        merged.setdefault("text", text)
        point: Dict[str, Any] = {
            "id": point_id,
            "payload": merged,
        }
        if self.vector_name:
            point["vector"] = {self.vector_name: vector}
        else:
            point["vector"] = vector
        path = f"/collections/{quote(self.collection, safe='')}/points?wait=true"
        body = {"points": [point]}
        try:
            self._qdrant("PUT", path, body)
        except QdrantClientError as exc:
            if "404" not in str(exc):
                raise
            self.ensure_collection(vector_size=len(vector))
            self._qdrant("PUT", path, body)
        return point_id

    def delete(self, point_id: str) -> None:
        self._qdrant(
            "POST",
            f"/collections/{quote(self.collection, safe='')}/points/delete?wait=true",
            {"points": [point_id]},
        )

    def ensure_collection(self, vector_size: Optional[int] = None) -> None:
        """Create the collection if missing. No-op when it already exists."""
        info = self.collection_info()
        if info is not None:
            return
        size = vector_size or self._vector_size
        if not size:
            raise QdrantClientError(
                f"Collection '{self.collection}' does not exist and vector size is unknown. "
                "Store one memory first, or create the collection in Qdrant."
            )
        vectors: Dict[str, Any]
        if self.vector_name:
            vectors = {self.vector_name: {"size": int(size), "distance": "Cosine"}}
        else:
            vectors = {"size": int(size), "distance": "Cosine"}
        self._qdrant(
            "PUT",
            f"/collections/{quote(self.collection, safe='')}",
            {"vectors": vectors},
        )

    def collection_info(self) -> Optional[Dict[str, Any]]:
        try:
            data = self._qdrant(
                "GET",
                f"/collections/{quote(self.collection, safe='')}",
            )
        except QdrantClientError as exc:
            if "404" in str(exc) or "Not found" in str(exc):
                return None
            raise
        result = data.get("result") if isinstance(data, dict) else None
        return result if isinstance(result, dict) else None

    def _embed_ollama(self, text: str) -> List[float]:
        data = self._request(
            "POST",
            f"{self.embed_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
            timeout=self.timeout_s,
        )
        embedding = data.get("embedding") if isinstance(data, dict) else None
        if isinstance(embedding, list):
            return [float(x) for x in embedding]
        raise QdrantClientError("Ollama embedding response missing 'embedding'")

    def _embed_openai(self, text: str) -> List[float]:
        headers = {}
        if self.embed_api_key:
            headers["Authorization"] = f"Bearer {self.embed_api_key}"
        data = self._request(
            "POST",
            f"{self.embed_url}/v1/embeddings",
            json={"model": self.embed_model, "input": text},
            headers=headers,
            timeout=self.timeout_s,
        )
        items = data.get("data") if isinstance(data, dict) else None
        if isinstance(items, list) and items:
            embedding = items[0].get("embedding") if isinstance(items[0], dict) else None
            if isinstance(embedding, list):
                return [float(x) for x in embedding]
        raise QdrantClientError("OpenAI-compat embedding response missing data[0].embedding")

    def _qdrant(self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> Any:
        headers = {}
        if self.qdrant_api_key:
            headers["api-key"] = self.qdrant_api_key
        return self._request(
            method,
            f"{self.qdrant_url}{path}",
            json=json_body,
            headers=headers,
            timeout=self.timeout_s,
        )


def _httpx_json(
    method: str,
    url: str,
    json: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 8.0,
) -> Any:
    with httpx.Client(timeout=timeout) as client:
        response = client.request(method, url, json=json, headers=headers)
        if response.status_code >= 400:
            snippet = (response.text or "")[:300]
            raise QdrantClientError(f"{method} {url} failed ({response.status_code}): {snippet}")
        if not response.content:
            return {}
        try:
            return response.json()
        except Exception as exc:
            raise QdrantClientError(f"{method} {url} returned non-JSON") from exc
