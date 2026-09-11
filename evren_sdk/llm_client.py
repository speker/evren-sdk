"""t81 - evren-sdk'nin LLM çıkarım (evren-llmapi geçidi) desteği. `client.py`
(görü modeli, `api.ssyz.org.tr`) İLE BİRLEŞTİRİLMEDİ - AYRI bir origin/API-
key sınıfı (`evren_llm_...`) hedefler, karıştırmak kafa karıştırırdı ve
`client.py`'yi 300 satır sınırının ÇOK üstüne (zaten 835 satır) çıkarırdı."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import httpx

from .client import _auth_headers
from .exceptions import EvrenError

_LLMAPI_BASE = "https://evren-llmapi.ssyz.org.tr"


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class ChatCompletionResult:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def _raise_for_llm(resp: httpx.Response) -> None:
    try:
        body = resp.json()
        msg = body.get("error", {}).get("message", resp.text[:300])
    except Exception:  # noqa: BLE001 - hata govdesi bozuksa ham metne dus
        msg = resp.text[:300]
    raise EvrenError(msg, resp.status_code)


class _SSEBuffer:
    """MCP review'de bulundu (GERÇEK bug): ham chunk'ları TEK TEK decode
    etmek TCP parçalanmasında (bir multi-byte UTF-8 karakteri veya `\\n\\n`
    sınırını iki chunk'a bölerse) çöker/veri kaybeder - portal_envelope.py'nin
    (t75b) AYNI buffer-birleştirme deseni burada da gerekli."""

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, raw: bytes) -> Iterator[str]:
        self._buffer += raw
        while b"\n\n" in self._buffer:
            frame, self._buffer = self._buffer.split(b"\n\n", 1)
            text = frame.decode(errors="ignore").strip()
            if text.startswith("data:"):
                yield text[5:].strip()


class EvrenLLMClient:
    """evren-llmapi (public data-plane) senkron istemcisi.

    >>> with EvrenLLMClient(api_key="evren_llm_xxx") as c:
    ...     r = c.chat("glm-5.3", [{"role": "user", "content": "Merhaba"}])
    ...     print(r.content)
    """

    def __init__(self, api_key: str, *, base_url: str = _LLMAPI_BASE, timeout: float = 120.0) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=_auth_headers(api_key), timeout=timeout)

    def list_models(self) -> list[str]:
        r = self._http.get("/v1/models")
        if r.status_code != 200:
            _raise_for_llm(r)
        return [m["id"] for m in r.json().get("data", [])]

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> ChatCompletionResult:
        r = self._http.post("/v1/chat/completions", json={"model": model, "messages": messages, "stream": False, **kwargs})
        if r.status_code != 200:
            _raise_for_llm(r)
        body = r.json()
        content = body["choices"][0]["message"]["content"]
        return ChatCompletionResult(content=content, model=body.get("model", model), usage=body.get("usage", {}), raw=body)

    def chat_stream(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Iterator[str]:
        """Her çağrıldığında bir token/parça metin YIELD eder - `for chunk in
        client.chat_stream(...): print(chunk, end="")` deseni."""
        buf = _SSEBuffer()
        with self._http.stream(
            "POST", "/v1/chat/completions", json={"model": model, "messages": messages, "stream": True, **kwargs},
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                _raise_for_llm(resp)
            for raw in resp.iter_bytes():
                for line in buf.feed(raw):
                    if line == "[DONE]":
                        return
                    with suppress(ValueError, KeyError, IndexError):
                        chunk = json.loads(line)
                        delta = chunk["choices"][0].get("delta", {}).get("content")
                        if delta:
                            yield delta

    def embed(self, model: str, texts: str | list[str]) -> list[list[float]]:
        r = self._http.post("/v1/embeddings", json={"model": model, "input": texts})
        if r.status_code != 200:
            _raise_for_llm(r)
        return [row["embedding"] for row in r.json().get("data", [])]

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> EvrenLLMClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncEvrenLLMClient:
    """evren-llmapi asenkron istemcisi - `EvrenLLMClient` ile AYNI yüzey."""

    def __init__(self, api_key: str, *, base_url: str = _LLMAPI_BASE, timeout: float = 120.0) -> None:
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=_auth_headers(api_key), timeout=timeout)

    async def list_models(self) -> list[str]:
        r = await self._http.get("/v1/models")
        if r.status_code != 200:
            _raise_for_llm(r)
        return [m["id"] for m in r.json().get("data", [])]

    async def chat(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> ChatCompletionResult:
        r = await self._http.post("/v1/chat/completions", json={"model": model, "messages": messages, "stream": False, **kwargs})
        if r.status_code != 200:
            _raise_for_llm(r)
        body = r.json()
        content = body["choices"][0]["message"]["content"]
        return ChatCompletionResult(content=content, model=body.get("model", model), usage=body.get("usage", {}), raw=body)

    async def chat_stream(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        buf = _SSEBuffer()
        async with self._http.stream(
            "POST", "/v1/chat/completions", json={"model": model, "messages": messages, "stream": True, **kwargs},
        ) as resp:
            if resp.status_code != 200:
                await resp.aread()
                _raise_for_llm(resp)
            async for raw in resp.aiter_bytes():
                for line in buf.feed(raw):
                    if line == "[DONE]":
                        return
                    with suppress(ValueError, KeyError, IndexError):
                        chunk = json.loads(line)
                        delta = chunk["choices"][0].get("delta", {}).get("content")
                        if delta:
                            yield delta

    async def embed(self, model: str, texts: str | list[str]) -> list[list[float]]:
        r = await self._http.post("/v1/embeddings", json={"model": model, "input": texts})
        if r.status_code != 200:
            _raise_for_llm(r)
        return [row["embedding"] for row in r.json().get("data", [])]

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncEvrenLLMClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
