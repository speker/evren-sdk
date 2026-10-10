"""evren-llmapi asenkron istemcisi — `EvrenLLMClient` ile aynı yüzey ve aynı tekrar politikası."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx

from .llm_client import Messages, _Config, chat_body, idem_headers, transport_error
from .llm_errors import error_from_response
from .llm_media import AudioSource, ImageSource, audio_file, encode_image
from .llm_retry import backoff, should_retry_exc, should_retry_status
from .llm_sse import StreamDrain
from .llm_tasks import (
    DecisionBatchResult,
    DecisionResult,
    LLMModel,
    OcrResult,
    QuotaStatus,
    RequestStatus,
    ResponseResult,
    Transcription,
    decision_body,
    parse_decision,
    parse_decision_batch,
    parse_models,
    parse_ocr,
    parse_quota,
    parse_request_status,
    parse_response,
    parse_transcription,
    transcription_form,
)
from .llm_types import ChatCompletionResult, ChatDelta, RerankResult, parse_completion, parse_rerank


class AsyncEvrenLLMClient:
    """>>> async with AsyncEvrenLLMClient() as c:
    ...     async for t in c.chat_stream("glm-5.3", [{"role": "user", "content": "Merhaba"}]):
    ...         print(t, end="")
    """

    def __init__(
        self, api_key: str | None = None, *, base_url: str | None = None, timeout: float = 600.0,
        max_retries: int = 2, transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._cfg = _Config(api_key, base_url, timeout, max_retries)
        self._http = httpx.AsyncClient(
            base_url=self._cfg.base_url, headers=self._cfg.headers, timeout=self._cfg.timeout, transport=transport,
        )

    async def _send(
        self, method: str, path: str, body: dict[str, Any] | None = None, idem: str | None = None,
        *, form: dict[str, str] | None = None, files: dict[str, Any] | None = None,
    ) -> httpx.Response:
        safe = method == "GET" or idem is not None
        for attempt in range(self._cfg.max_retries + 1):
            last = attempt == self._cfg.max_retries
            try:
                r = await self._http.request(
                    method, path, json=body, data=form, files=files, headers=idem_headers(idem),
                )
            except httpx.TransportError as exc:
                if last or not should_retry_exc(exc, idempotent=safe):
                    raise transport_error(exc) from exc
                await asyncio.sleep(backoff(attempt, None))
                continue
            if r.status_code == 200:
                return r
            err = error_from_response(r)
            if last or not safe or not should_retry_status(r.status_code, stream=False, retry_after=err.retry_after):
                raise err
            await asyncio.sleep(backoff(attempt, err.retry_after))
        raise AssertionError("unreachable")

    async def list_models(self) -> list[str]:
        r = await self._send("GET", "/v1/models")
        return [m["id"] for m in r.json().get("data", [])]

    async def chat(
        self, model: str, messages: Messages, *, idempotency_key: str | None = None, **kwargs: Any,
    ) -> ChatCompletionResult:
        r = await self._send("POST", "/v1/chat/completions", chat_body(model, messages, False, kwargs),
                             self._cfg.idem_key(idempotency_key))
        return parse_completion(r.json(), r.headers, model)

    async def stream(self, model: str, messages: Messages, **kwargs: Any) -> AsyncIterator[ChatDelta]:
        body = chat_body(model, messages, True, kwargs)
        for attempt in range(self._cfg.max_retries + 1):
            last = attempt == self._cfg.max_retries
            started = False
            try:
                async with self._http.stream("POST", "/v1/chat/completions", json=body) as resp:
                    if resp.status_code != 200:
                        await resp.aread()
                        err = error_from_response(resp)
                        if last or not should_retry_status(resp.status_code, stream=True, retry_after=err.retry_after):
                            raise err
                        wait = backoff(attempt, err.retry_after)
                    else:
                        started = True
                        drain = StreamDrain(resp.headers.get("x-request-id"))
                        async for raw in resp.aiter_bytes():
                            for delta in drain.feed(raw):
                                yield delta
                            if drain.done:
                                return
                        drain.close()
                        return
            except httpx.TransportError as exc:
                if started or last or not should_retry_exc(exc, idempotent=False):
                    raise transport_error(exc, streaming=started) from exc
                wait = backoff(attempt, None)
            await asyncio.sleep(wait)

    async def chat_stream(self, model: str, messages: Messages, **kwargs: Any) -> AsyncIterator[str]:
        async for delta in self.stream(model, messages, **kwargs):
            if delta.content:
                yield delta.content

    async def embed(
        self, model: str, texts: str | list[str], *, idempotency_key: str | None = None,
    ) -> list[list[float]]:
        r = await self._send("POST", "/v1/embeddings", {"model": model, "input": texts},
                             self._cfg.idem_key(idempotency_key))
        rows = sorted(r.json().get("data", []), key=lambda row: row.get("index", 0))
        return [row["embedding"] for row in rows]

    async def rerank(
        self, model: str, query: str, documents: Iterable[str], *, top_n: int | None = None,
        idempotency_key: str | None = None,
    ) -> list[RerankResult]:
        body: dict[str, Any] = {"model": model, "query": query, "documents": list(documents)}
        if top_n is not None:
            body["top_n"] = top_n
        r = await self._send("POST", "/v1/rerank", body, self._cfg.idem_key(idempotency_key))
        results = parse_rerank(r.json())
        return results[:top_n] if top_n is not None else results

    async def models(self) -> list[LLMModel]:
        return parse_models((await self._send("GET", "/v1/models")).json())

    async def decide(
        self, model: str, state: Any, questions: dict[str, Any], *, images: list[ImageSource] | None = None,
        idempotency_key: str | None = None, **kwargs: Any,
    ) -> DecisionResult:
        """`EvrenLLMClient.decide` ile aynı."""
        r = await self._send("POST", "/v1/systemone", decision_body(model, state, questions, images, kwargs),
                             self._cfg.idem_key(idempotency_key))
        return parse_decision(r.json(), r.headers, model)

    async def decide_batch(
        self, model: str, states: Iterable[Any], questions: dict[str, Any], *,
        idempotency_key: str | None = None, **kwargs: Any,
    ) -> DecisionBatchResult:
        body = {**kwargs, "model": model, "states": list(states), "questions": questions}
        r = await self._send("POST", "/v1/systemone/batch", body, self._cfg.idem_key(idempotency_key))
        return parse_decision_batch(r.json(), r.headers, model)

    async def ocr(self, model: str, image: ImageSource, *, idempotency_key: str | None = None) -> OcrResult:
        r = await self._send("POST", "/v1/ocr", {"model": model, "image": encode_image(image)},
                             self._cfg.idem_key(idempotency_key))
        return parse_ocr(r.json(), r.headers, model)

    async def transcribe(
        self, model: str, audio: AudioSource, *, filename: str | None = None, language: str | None = None,
        prompt: str | None = None, response_format: str = "json", temperature: float = 0.0,
    ) -> Transcription:
        name, raw, mime = audio_file(audio, filename)
        r = await self._send("POST", "/v1/audio/transcriptions",
                             form=transcription_form(model, language, prompt, response_format, temperature),
                             files={"file": (name, raw, mime)})
        return parse_transcription(r, response_format)

    async def quota(self) -> QuotaStatus:
        return parse_quota((await self._send("GET", "/v1/quota")).json())

    async def request_status(self, request_id: str) -> RequestStatus:
        return parse_request_status((await self._send("GET", f"/v1/requests/{request_id}")).json())

    async def respond(
        self, model: str, input: Any, *, idempotency_key: str | None = None, **kwargs: Any,
    ) -> ResponseResult:
        body = {**kwargs, "model": model, "input": input, "stream": False}
        r = await self._send("POST", "/v1/responses", body, self._cfg.idem_key(idempotency_key))
        return parse_response(r.json(), r.headers, model)

    async def get_response(self, response_id: str) -> ResponseResult:
        r = await self._send("GET", f"/v1/responses/{response_id}")
        return parse_response(r.json(), r.headers, "")

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncEvrenLLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
