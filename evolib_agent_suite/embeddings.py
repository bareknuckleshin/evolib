from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from evolib_agent_suite.utils import hashed_embedding


EmbeddingFunction = Callable[[str], List[float]]


class AzureOpenAIEmbeddingClient:
    """Azure OpenAI embeddings client using the api-key header."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        retries: int = 3,
        use_proxy: bool = False,
        proxy_url: Optional[str] = None,
        verify_ssl: bool = True,
    ) -> None:
        self.endpoint_url = endpoint_url or os.environ.get("AZURE_OPENAI_EMBEDDINGS_URL")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_EMBEDDINGS_API_KEY")
        self.model = model or os.environ.get("AZURE_OPENAI_EMBEDDINGS_MODEL", "text-embedding-ada-002")
        self.timeout = timeout
        self.retries = retries
        self.use_proxy = use_proxy
        self.proxy_url = proxy_url or os.environ.get("PROXY_URL")
        self.verify_ssl = verify_ssl
        if not self.endpoint_url:
            raise ValueError("Missing Azure embeddings endpoint_url.")
        if not self.api_key:
            raise ValueError("Missing Azure embeddings api_key.")

    def embed(self, text: str) -> List[float]:
        payload = {"model": self.model, "input": text}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=data,
            headers={"Content-Type": "application/json", "api-key": self.api_key},
            method="POST",
        )
        opener = self._opener()
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                with opener.open(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["data"][0]["embedding"]
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Embedding request failed after {self.retries} attempts: {last_error}")

    def _opener(self) -> urllib.request.OpenerDirector:
        handlers: List[Any] = []
        if self.use_proxy and self.proxy_url:
            handlers.append(urllib.request.ProxyHandler({"http": self.proxy_url, "https": self.proxy_url}))
        elif not self.use_proxy:
            handlers.append(urllib.request.ProxyHandler({}))
        if not self.verify_ssl:
            handlers.append(urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
        return urllib.request.build_opener(*handlers)


def build_embedding_function(config: Optional[Dict[str, Any]]) -> EmbeddingFunction:
    cfg = dict(config or {})
    provider = str(cfg.get("provider", "hashed")).lower()
    if provider in {"hashed", "hash", "local", "none"}:
        return hashed_embedding
    if provider in {"azure_openai", "azure"}:
        client = AzureOpenAIEmbeddingClient(
            endpoint_url=cfg.get("endpoint_url") or cfg.get("url"),
            api_key=cfg.get("api_key"),
            model=cfg.get("model"),
            timeout=int(cfg.get("timeout", cfg.get("timeout_seconds", 120))),
            retries=int(cfg.get("retries", 3)),
            use_proxy=_as_bool(cfg.get("use_proxy", False)),
            proxy_url=cfg.get("proxy_url"),
            verify_ssl=_as_bool(cfg.get("verify_ssl", True)),
        )
        return client.embed
    raise ValueError(f"Unknown embedding provider: {provider}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
