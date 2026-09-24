"""evren-llmapi (LLM geçidi) senkron istemcisi.

Görü modeli istemcisinden (`client.py`, `api.ssyz.org.tr`) ayrıdır: farklı
origin ve farklı anahtar türü (`evren_llm_...`) hedefler.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

import httpx

from ._version import __version__
from .llm_errors import EvrenLLMError, LLMServiceError, error_from_response, stream_error
from .llm_retry import backoff, should_retry_exc, should_retry_status
from .llm_sse import DONE, SSEBuffer, interpret
from .llm_types import ChatCompletionResult, ChatDelta, RerankResult, parse_completion, parse_rerank

DEFAULT_BASE_URL = "https://evren-llmapi.ssyz.org.tr"
ENV_API_KEY = "EVREN_LLM_API_KEY"
ENV_BASE_URL = "EVREN_LLM_BASE_URL"

Messages = list[dict[str, Any]]


class _Config:
    def __init__(self, api_key: str | None, base_url: str | None, timeout: float, max_retries: int) -> None:
        key = api_key or os.environ.get(ENV_API_KEY)
        if not key:
            raise ValueError(f"api_key verilmedi, {ENV_API_KEY} ortam değişkeni de boş")
        self.base_url = (base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
        # geçit Bearer'ı birincil kabul eder (OpenAI SDK uyumu)
        self.headers = {"Authorization": f"Bearer {key}", "User-Agent": f"evren-sdk/{__version__} (python)"}
        self.timeout = httpx.Timeout(timeout, connect=10.0)
        self.max_retries = max(0, max_retries)

    def idem_key(self, given: str | None) -> str | None:
        return given or (str(uuid.uuid4()) if self.max_retries else None)


def chat_body(model: str, messages: Messages, stream: bool, kwargs: dict[str, Any]) -> dict[str, Any]:
    body = {**kwargs, "model": model, "messages": messages, "stream": stream}
    if stream:
        body.setdefault("stream_options", {"include_usage": True})
    return body


def idem_headers(key: str | None) -> dict[str, str]:
    return {"Idempotency-Key": key} if key else {}


def transport_error(exc: Exception, *, streaming: bool = False) -> EvrenLLMError:
    if streaming:
        return stream_error({"message": f"akış bağlantısı koptu ({type(exc).__name__}); yanıt eksik"}, None)
    if isinstance(exc, httpx.TimeoutException):
        return LLMServiceError(
            f"istek zaman aşımına uğradı ({type(exc).__name__}); uzun yanıtlar için `timeout` değerini artırın",
            code="timeout", type="api_error",
        )
    return LLMServiceError(f"geçide bağlanılamadı: {exc}", code="connection_error", type="api_error")


class StreamDrain:
    """Ham baytları `ChatDelta`'ya çevirir; akışın eksik bittiğini fark eder."""

    def __init__(self, request_id: str | None) -> None:
        self._buf = SSEBuffer()
        self._rid = request_id
        self.done = False
        self._finished = False

    def feed(self, raw: bytes) -> Iterator[ChatDelta]:
        # generator: aynı pakette hata kapanışı gelse de önceki parçalar kullanıcıya ulaşır
        for payload in self._buf.feed(raw):
            item = interpret(payload, self._rid)
            if item is DONE:
                self.done = True
                return
            if isinstance(item, ChatDelta):
                self._finished = self._finished or item.finish_reason is not None
                yield item

    def close(self) -> None:
        # ne [DONE] ne finish_reason geldiyse bağlantı erken kapanmıştır
        if not self.done and not self._finished:
            raise stream_error(None, self._rid)


class EvrenLLMClient:
    """evren-llmapi senkron istemcisi.

    >>> with EvrenLLMClient() as c:          # EVREN_LLM_API_KEY ortam değişkeninden
    ...     r = c.chat("glm-5.3", [{"role": "user", "content": "Merhaba"}])
    ...     print(r.content, r.request_id)

    Args:
        api_key: `evren_llm_...` anahtarı; verilmezse `EVREN_LLM_API_KEY`.
        base_url: Varsayılan `https://evren-llmapi.ssyz.org.tr` (`EVREN_LLM_BASE_URL`).
        timeout: Okuma zaman aşımı (saniye). Düşünen modellerde uzun yanıtlar için yüksek tutulur.
        max_retries: 429/5xx ve bağlantı hatalarında tekrar sayısı. 0 kapatır.
        transport: Özel httpx transport'u (proxy, test).
    """

    def __init__(
        self, api_key: str | None = None, *, base_url: str | None = None, timeout: float = 600.0,
        max_retries: int = 2, transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._cfg = _Config(api_key, base_url, timeout, max_retries)
        self._http = httpx.Client(
            base_url=self._cfg.base_url, headers=self._cfg.headers, timeout=self._cfg.timeout, transport=transport,
        )

    def _send(self, method: str, path: str, body: dict[str, Any] | None = None, idem: str | None = None) -> httpx.Response:
        # GET doğası gereği tekrarlanabilir; POST ancak Idempotency-Key ile
        safe = method == "GET" or idem is not None
        for attempt in range(self._cfg.max_retries + 1):
            last = attempt == self._cfg.max_retries
            try:
                r = self._http.request(method, path, json=body, headers=idem_headers(idem))
            except httpx.TransportError as exc:
                if last or not should_retry_exc(exc, idempotent=safe):
                    raise transport_error(exc) from exc
                time.sleep(backoff(attempt, None))
                continue
            if r.status_code == 200:
                return r
            err = error_from_response(r)
            if last or not safe or not should_retry_status(r.status_code, stream=False):
                raise err
            time.sleep(backoff(attempt, err.retry_after))
        raise AssertionError("unreachable")

    def list_models(self) -> list[str]:
        return [m["id"] for m in self._send("GET", "/v1/models").json().get("data", [])]

    def chat(
        self, model: str, messages: Messages, *, idempotency_key: str | None = None, **kwargs: Any,
    ) -> ChatCompletionResult:
        """Akışsız sohbet tamamlaması. `tools`, `temperature`, `max_tokens` vb. OpenAI alanları `kwargs` ile geçer."""
        r = self._send("POST", "/v1/chat/completions", chat_body(model, messages, False, kwargs),
                       self._cfg.idem_key(idempotency_key))
        return parse_completion(r.json(), r.headers, model)

    def stream(self, model: str, messages: Messages, **kwargs: Any) -> Iterator[ChatDelta]:
        """Akışlı sohbet; her parça içerik, düşünme metni, araç çağrısı ve son parçada kullanım taşır.

        Raises:
            EvrenLLMError: Akış yarıda kesilirse (`code="stream_interrupted"`); gelen metin eksiktir.
        """
        body = chat_body(model, messages, True, kwargs)
        for attempt in range(self._cfg.max_retries + 1):
            last = attempt == self._cfg.max_retries
            started = False
            try:
                with self._http.stream("POST", "/v1/chat/completions", json=body) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        err = error_from_response(resp)
                        if last or not should_retry_status(resp.status_code, stream=True):
                            raise err
                        wait = backoff(attempt, err.retry_after)
                    else:
                        started = True
                        drain = StreamDrain(resp.headers.get("x-request-id"))
                        for raw in resp.iter_bytes():
                            yield from drain.feed(raw)
                            if drain.done:
                                return
                        drain.close()
                        return
            except httpx.TransportError as exc:
                if started or last or not should_retry_exc(exc, idempotent=False):
                    raise transport_error(exc, streaming=started) from exc
                wait = backoff(attempt, None)
            time.sleep(wait)

    def chat_stream(self, model: str, messages: Messages, **kwargs: Any) -> Iterator[str]:
        """Yalnızca yanıt metnini parça parça verir: `for t in c.chat_stream(...): print(t, end="")`."""
        for delta in self.stream(model, messages, **kwargs):
            if delta.content:
                yield delta.content

    def embed(self, model: str, texts: str | list[str], *, idempotency_key: str | None = None) -> list[list[float]]:
        r = self._send("POST", "/v1/embeddings", {"model": model, "input": texts}, self._cfg.idem_key(idempotency_key))
        rows = sorted(r.json().get("data", []), key=lambda row: row.get("index", 0))
        return [row["embedding"] for row in rows]

    def rerank(
        self, model: str, query: str, documents: Iterable[str], *, top_n: int | None = None,
        idempotency_key: str | None = None,
    ) -> list[RerankResult]:
        """Belgeleri sorguya göre en alakalıdan en az alakalıya sıralar."""
        body: dict[str, Any] = {"model": model, "query": query, "documents": list(documents)}
        if top_n is not None:
            body["top_n"] = top_n
        results = parse_rerank(self._send("POST", "/v1/rerank", body, self._cfg.idem_key(idempotency_key)).json())
        return results[:top_n] if top_n is not None else results

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> EvrenLLMClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
