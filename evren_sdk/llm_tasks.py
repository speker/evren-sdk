"""Sohbet dışı geçit uçlarının türleri, gövde kurucuları ve ayrıştırıcıları.

Karar (`/v1/systemone`), OCR, ses transkripsiyonu, kota, istek durumu ve
Responses API. Senkron ve asenkron istemci aynı fonksiyonları kullanır.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .llm_media import ImageSource, encode_image
from .llm_types import JSON, ResponseMeta, parse_meta


@dataclass(slots=True)
class LLMModel:
    """`/v1/models` kaydı. `task` uç seçimini belirler: decision -> `decide()`, ocr -> `ocr()` ..."""

    id: str
    task: str = "chat"
    modalities: list[str] = field(default_factory=lambda: ["text"])
    owned_by: str = "evren"
    created: int = 0
    pricing: JSON | None = None


@dataclass(slots=True)
class DecisionResult:
    """`decide()` sonucu. `answers` soru kimliğine göre ham yanıtları tutar."""

    answers: dict[str, JSON]
    model: str
    usage: JSON = field(default_factory=dict)
    id: str | None = None
    raw: JSON = field(default_factory=dict)
    meta: ResponseMeta = field(default_factory=ResponseMeta)

    @property
    def request_id(self) -> str | None:
        return self.meta.request_id

    def choice(self, question: str) -> str | None:
        value = self.answers.get(question, {}).get("choice")
        return None if value is None else str(value)

    def noul(self, question: str) -> float | None:
        """Evet olasılığı (0-1)."""
        value = self.answers.get(question, {}).get("noul")
        return None if value is None else float(value)

    def score(self, question: str) -> float | None:
        value = self.answers.get(question, {}).get("score")
        return None if value is None else float(value)

    def probabilities(self, question: str) -> dict[str, float]:
        probs = self.answers.get(question, {}).get("probabilities") or {}
        return {str(k): float(v) for k, v in probs.items()}


@dataclass(slots=True)
class DecisionBatchResult:
    results: list[DecisionResult]
    usage: JSON = field(default_factory=dict)
    id: str | None = None
    meta: ResponseMeta = field(default_factory=ResponseMeta)


@dataclass(slots=True)
class OcrResult:
    text: str
    model: str
    meta: ResponseMeta = field(default_factory=ResponseMeta)


@dataclass(slots=True)
class Transcription:
    """`text` her biçimde dolu; `srt` istendiyse altyazı metnidir. `raw` yalnız JSON biçimlerinde dolar."""

    text: str
    raw: JSON = field(default_factory=dict)
    media_id: str | None = None
    meta: ResponseMeta = field(default_factory=ResponseMeta)

    @property
    def segments(self) -> list[JSON]:
        return list(self.raw.get("segments") or [])


@dataclass(slots=True)
class QuotaStatus:
    """5 dk'lık kayan pencere + günlük tavan + kredi durumu (`/v1/quota`)."""

    level: str
    remaining_cr: float
    held_cr: float
    used_tokens: int | None = None
    cap: int | None = None
    window_minutes: int | None = None
    reset_at: str | None = None
    usage_ratio: float | None = None
    daily_used_tokens: int | None = None
    daily_limit_tokens: int | None = None
    daily_remaining_tokens: int | None = None
    daily_reset_at: str | None = None


@dataclass(slots=True)
class RequestStatus:
    """Bir isteğin yürütme ve ücret durumu; yanıt gövdesi burada dönmez."""

    request_id: str
    execution_terminal: str
    charge_terminal: bool
    is_billing_pending: bool
    state_version: int
    collected_cr: float | None = None
    outstanding_cr: float | None = None
    collection_state: str | None = None


@dataclass(slots=True)
class ResponseResult:
    """Responses API (`/v1/responses`) sonucu. `output_text` mesaj parçalarının birleşimidir."""

    id: str | None
    output_text: str
    model: str
    output: list[JSON] = field(default_factory=list)
    usage: JSON = field(default_factory=dict)
    previous_response_id: str | None = None
    raw: JSON = field(default_factory=dict)
    meta: ResponseMeta = field(default_factory=ResponseMeta)


def decision_body(
    model: str, state: Any, questions: JSON, images: Iterable[ImageSource] | None, extra: JSON,
) -> JSON:
    body: JSON = {**extra, "model": model, "state": state, "questions": questions}
    if images:
        body["images"] = [encode_image(img) for img in images]
    return body


def transcription_form(
    model: str, language: str | None, prompt: str | None, response_format: str, temperature: float,
) -> dict[str, str]:
    form = {"model": model, "response_format": response_format, "temperature": str(temperature)}
    if language:
        form["language"] = language
    if prompt:
        form["prompt"] = prompt
    return form


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def parse_models(body: JSON) -> list[LLMModel]:
    return [
        LLMModel(
            id=str(m["id"]), task=str(m.get("task") or "chat"), modalities=list(m.get("modalities") or ["text"]),
            owned_by=str(m.get("owned_by") or "evren"), created=int(m.get("created") or 0), pricing=m.get("pricing"),
        )
        for m in body.get("data") or []
    ]


def parse_decision(body: JSON, headers: httpx.Headers | None, requested_model: str) -> DecisionResult:
    return DecisionResult(
        answers=dict(body.get("answers") or {}), model=str(body.get("model") or requested_model),
        usage=dict(body.get("usage") or {}), id=body.get("id"), raw=body,
        meta=parse_meta(headers) if headers is not None else ResponseMeta(),
    )


def parse_decision_batch(body: JSON, headers: httpx.Headers, requested_model: str) -> DecisionBatchResult:
    items = [parse_decision(item, None, requested_model) for item in body.get("results") or []]
    return DecisionBatchResult(
        results=items, usage=dict(body.get("total_usage") or body.get("usage") or {}),
        id=body.get("id"), meta=parse_meta(headers),
    )


def parse_ocr(body: JSON, headers: httpx.Headers, requested_model: str) -> OcrResult:
    return OcrResult(text=str(body.get("text") or ""), model=str(body.get("model") or requested_model),
                     meta=parse_meta(headers))


def parse_transcription(resp: httpx.Response, response_format: str) -> Transcription:
    media_id = resp.headers.get("x-evren-media-id")
    if response_format in ("text", "srt"):
        return Transcription(text=resp.text, media_id=media_id, meta=parse_meta(resp.headers))
    body: JSON = resp.json()
    return Transcription(text=str(body.get("text") or ""), raw=body, media_id=media_id, meta=parse_meta(resp.headers))


def parse_quota(body: JSON) -> QuotaStatus:
    return QuotaStatus(
        level=str(body.get("level") or "unknown"),
        remaining_cr=float(body.get("remaining_cr") or 0), held_cr=float(body.get("held_cr") or 0),
        used_tokens=body.get("used_tokens"), cap=body.get("cap"), window_minutes=body.get("window_minutes"),
        reset_at=body.get("reset_at"), usage_ratio=_float(body.get("usage_ratio")),
        daily_used_tokens=body.get("daily_used_tokens"), daily_limit_tokens=body.get("daily_limit_tokens"),
        daily_remaining_tokens=body.get("daily_remaining_tokens"), daily_reset_at=body.get("daily_reset_at"),
    )


def parse_request_status(body: JSON) -> RequestStatus:
    return RequestStatus(
        request_id=str(body["request_id"]), execution_terminal=str(body.get("execution_terminal") or ""),
        charge_terminal=bool(body.get("charge_terminal")), is_billing_pending=bool(body.get("is_billing_pending")),
        state_version=int(body.get("state_version") or 0), collected_cr=_float(body.get("collected_cr")),
        outstanding_cr=_float(body.get("outstanding_cr")), collection_state=body.get("collection_state"),
    )


def _output_text(output: list[JSON]) -> str:
    parts = [
        str(part.get("text") or "")
        for item in output if item.get("type") == "message"
        for part in item.get("content") or [] if part.get("type") == "output_text"
    ]
    return "".join(parts)


def parse_response(body: JSON, headers: httpx.Headers, requested_model: str) -> ResponseResult:
    output = list(body.get("output") or [])
    return ResponseResult(
        id=body.get("id"), output_text=str(body.get("output_text") or _output_text(output)),
        model=str(body.get("model") or requested_model), output=output, usage=dict(body.get("usage") or {}),
        previous_response_id=body.get("previous_response_id"), raw=body, meta=parse_meta(headers),
    )
