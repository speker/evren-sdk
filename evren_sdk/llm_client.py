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
    SpeechResult,
    Transcription,
    decision_body,
    parse_decision,
    parse_decision_batch,
    parse_models,
    parse_ocr,
    parse_quota,
    parse_request_status,
    parse_response,
    parse_speech,
    parse_transcription,
    speech_body,
    transcription_form,
)
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

    def _send(
        self, method: str, path: str, body: dict[str, Any] | None = None, idem: str | None = None,
        *, form: dict[str, str] | None = None, files: dict[str, Any] | None = None,
    ) -> httpx.Response:
        # GET doğası gereği tekrarlanabilir; POST ancak Idempotency-Key ile
        safe = method == "GET" or idem is not None
        for attempt in range(self._cfg.max_retries + 1):
            last = attempt == self._cfg.max_retries
            try:
                r = self._http.request(
                    method, path, json=body, data=form, files=files, headers=idem_headers(idem),
                )
            except httpx.TransportError as exc:
                if last or not should_retry_exc(exc, idempotent=safe):
                    raise transport_error(exc) from exc
                time.sleep(backoff(attempt, None))
                continue
            if r.status_code == 200:
                return r
            err = error_from_response(r)
            if last or not safe or not should_retry_status(r.status_code, stream=False, retry_after=err.retry_after):
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
                        if last or not should_retry_status(resp.status_code, stream=True, retry_after=err.retry_after):
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

    def models(self) -> list[LLMModel]:
        """Anahtarın erişebildiği modeller, görev (`task`) ve modaliteleriyle."""
        return parse_models(self._send("GET", "/v1/models").json())

    def decide(
        self, model: str, state: Any, questions: dict[str, Any], *, images: list[ImageSource] | None = None,
        idempotency_key: str | None = None, **kwargs: Any,
    ) -> DecisionResult:
        """Tipli karar (`laya`, `clef`): metin üretmez, soruları olasılıklarıyla yanıtlar.

        >>> r = c.decide("clef", "Faturam iki kez kesildi", {
        ...     "birim": {"type": "choice", "instructions": "Hangi birim?",
        ...               "criteria": {"fatura": "ödeme", "teknik": "hata"}},
        ...     "iptal": {"type": "noul", "instructions": "İptal tehdidi var mı?"},
        ... })
        >>> r.choice("birim"), r.noul("iptal")

        Args:
            state: Metin ya da JSON belge.
            questions: Soru kimliği -> `{type: choice|score|noul, instructions, criteria}`; en fazla 64.
                Laya her soruda `instructions` ister.
            images: Yalnız görsel kabul eden modellerde (clef); yol, bayt ya da `data:` URL, en fazla 4.
                Büyük fotoğrafları önce `encode_image(yol, max_side=1024)` ile küçültün.
            kwargs: Laya'ya özgü `lang`, `min_confidence`, `max_len` gibi alanlar.
        """
        r = self._send("POST", "/v1/systemone", decision_body(model, state, questions, images, kwargs),
                       self._cfg.idem_key(idempotency_key))
        return parse_decision(r.json(), r.headers, model)

    def decide_batch(
        self, model: str, states: Iterable[Any], questions: dict[str, Any], *,
        idempotency_key: str | None = None, **kwargs: Any,
    ) -> DecisionBatchResult:
        """Aynı soruları en fazla 64 metne tek çağrıda sorar. Yalnız `laya`; clef 400 döner."""
        body = {**kwargs, "model": model, "states": list(states), "questions": questions}
        r = self._send("POST", "/v1/systemone/batch", body, self._cfg.idem_key(idempotency_key))
        return parse_decision_batch(r.json(), r.headers, model)

    def ocr(self, model: str, image: ImageSource, *, idempotency_key: str | None = None) -> OcrResult:
        """Görseldeki metni olduğu gibi çıkarır (`dots-ocr`, `deepseek-ocr-2`)."""
        r = self._send("POST", "/v1/ocr", {"model": model, "image": encode_image(image)},
                       self._cfg.idem_key(idempotency_key))
        return parse_ocr(r.json(), r.headers, model)

    def transcribe(
        self, model: str, audio: AudioSource, *, filename: str | None = None, language: str | None = None,
        prompt: str | None = None, response_format: str = "json", temperature: float = 0.0,
    ) -> Transcription:
        """Ses kaydını metne çevirir. `response_format`: json, text, srt, verbose_json, diarized_json.

        Geçit bu uçta Idempotency-Key desteklemez; yalnız bağlantı hiç kurulamadıysa tekrar denenir.
        """
        name, raw, mime = audio_file(audio, filename)
        r = self._send("POST", "/v1/audio/transcriptions",
                       form=transcription_form(model, language, prompt, response_format, temperature),
                       files={"file": (name, raw, mime)})
        return parse_transcription(r, response_format)

    def speak(
        self, model: str, text: str, *, voice: str = "default", response_format: str = "mp3",
        speed: float = 1.0, instructions: str | None = None,
    ) -> SpeechResult:
        """Metni sese çevirir (`voxcpm2`). Biçim: mp3, wav, flac, opus, pcm. Uzun metni geçit böler.

        `voice` hazır seslerden biri: deniz, ece, emre, kaan (`default` = deniz); aynı sesle yapılan
        ayrı çağrılar aynı konuşmacıyla okunur. `instructions` verilirse ses tarife göre üretilir.
        """
        r = self._send("POST", "/v1/audio/speech", speech_body(model, text, voice, response_format, speed, instructions))
        return parse_speech(r, response_format)

    def quota(self) -> QuotaStatus:
        return parse_quota(self._send("GET", "/v1/quota").json())

    def request_status(self, request_id: str) -> RequestStatus:
        """Yanıttaki `id` ile isteğin yürütme ve ücret durumu."""
        return parse_request_status(self._send("GET", f"/v1/requests/{request_id}").json())

    def respond(
        self, model: str, input: Any, *, idempotency_key: str | None = None, **kwargs: Any,
    ) -> ResponseResult:
        """Responses API (akışsız). Zincirleme için `previous_response_id=` verin."""
        body = {**kwargs, "model": model, "input": input, "stream": False}
        r = self._send("POST", "/v1/responses", body, self._cfg.idem_key(idempotency_key))
        return parse_response(r.json(), r.headers, model)

    def get_response(self, response_id: str) -> ResponseResult:
        r = self._send("GET", f"/v1/responses/{response_id}")
        return parse_response(r.json(), r.headers, "")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> EvrenLLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
