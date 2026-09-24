"""LLM geçidi yanıt türleri ve gövde/başlık ayrıştırıcıları."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

JSON = dict[str, Any]

# vLLM sürümüne göre düşünme metni iki farklı alanda gelebiliyor
_REASONING_KEYS = ("reasoning", "reasoning_content")


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class ResponseMeta:
    """Geçidin yanıt başlıklarından okunan bilgiler."""

    request_id: str | None = None
    routed_model: str | None = None
    credits_remaining: float | None = None
    ratelimit_remaining_requests: int | None = None
    ratelimit_remaining_tokens: int | None = None


@dataclass(slots=True)
class ChatCompletionResult:
    """`chat()` sonucu.

    `content` hiçbir zaman None olmaz; model yalnızca araç çağırdıysa ya da
    `max_tokens` düşünme aşamasında bittiyse boş string döner.
    """

    content: str
    model: str
    usage: JSON = field(default_factory=dict)
    raw: JSON = field(default_factory=dict)
    reasoning: str | None = None
    tool_calls: list[JSON] = field(default_factory=list)
    finish_reason: str | None = None
    id: str | None = None
    meta: ResponseMeta = field(default_factory=ResponseMeta)

    @property
    def request_id(self) -> str | None:
        return self.meta.request_id


@dataclass(slots=True)
class ChatDelta:
    """Akışın tek bir parçası (`stream()`)."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[JSON] = field(default_factory=list)
    finish_reason: str | None = None
    usage: JSON | None = None


@dataclass(slots=True)
class RerankResult:
    index: int
    score: float
    document: str | None = None


def _num(raw: str | None, cast: type) -> Any:
    if raw is None:
        return None
    try:
        return cast(raw)
    except ValueError:
        return None


def parse_meta(headers: httpx.Headers) -> ResponseMeta:
    return ResponseMeta(
        request_id=headers.get("x-request-id"),
        routed_model=headers.get("x-evren-routed-model"),
        credits_remaining=_num(headers.get("x-evren-credits-remaining"), float),
        ratelimit_remaining_requests=_num(headers.get("x-ratelimit-remaining-requests"), int),
        ratelimit_remaining_tokens=_num(headers.get("x-ratelimit-remaining-tokens"), int),
    )


def _reasoning_of(obj: JSON) -> str | None:
    for key in _REASONING_KEYS:
        val = obj.get(key)
        if val:
            return str(val)
    return None


def parse_completion(body: JSON, headers: httpx.Headers, requested_model: str) -> ChatCompletionResult:
    choices = body.get("choices") or [{}]
    choice = choices[0] or {}
    msg = choice.get("message") or {}
    return ChatCompletionResult(
        content=msg.get("content") or "",
        model=body.get("model", requested_model),
        usage=body.get("usage") or {},
        raw=body,
        reasoning=_reasoning_of(msg),
        tool_calls=list(msg.get("tool_calls") or []),
        finish_reason=choice.get("finish_reason"),
        id=body.get("id"),
        meta=parse_meta(headers),
    )


def parse_delta(chunk: JSON) -> ChatDelta:
    choices = chunk.get("choices") or []
    usage = chunk.get("usage") or None
    if not choices:
        return ChatDelta(usage=usage)
    choice = choices[0] or {}
    delta = choice.get("delta") or {}
    return ChatDelta(
        content=delta.get("content") or "",
        reasoning=_reasoning_of(delta) or "",
        tool_calls=list(delta.get("tool_calls") or []),
        finish_reason=choice.get("finish_reason"),
        usage=usage,
    )


def parse_rerank(body: JSON) -> list[RerankResult]:
    out: list[RerankResult] = []
    for row in body.get("results") or []:
        doc = row.get("document")
        text = doc.get("text") if isinstance(doc, dict) else doc
        out.append(RerankResult(index=int(row["index"]), score=float(row.get("relevance_score", 0.0)), document=text))
    return sorted(out, key=lambda r: r.score, reverse=True)
