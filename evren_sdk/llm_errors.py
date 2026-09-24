"""LLM geçidi hataları.

Geçit OpenAI biçiminde `{"error": {message, type, param, code, evren}}` döner.
Sınıflar mevcut `exceptions` hiyerarşisinden de türer; `except RateLimitError`
yazan eski kod çalışmaya devam eder.
"""

from __future__ import annotations

from typing import Any

import httpx

from .exceptions import (
    AuthenticationError,
    EvrenError,
    InferenceError,
    InsufficientCreditsError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)


class EvrenLLMError(EvrenError):
    """LLM geçidinden dönen hata.

    Attributes:
        code: Makinece okunur hata kodu (`model_not_found`, `rate_limit_exceeded` ...).
        type: OpenAI hata tipi (`invalid_request_error`, `rate_limit_error` ...).
        param: Hatalı istek alanı, varsa.
        evren: EVREN'e özgü ek bilgi — `resets_at`, `suggested_models`,
            `replacement_model`, `remaining_credits_cr` gibi.
        request_id: Destek talebinde paylaşılacak istek kimliği.
        retry_after: Sunucunun önerdiği bekleme süresi (saniye), varsa.
    """

    # RateLimitError'da int, burada header'dan geldiği için None olabilir
    retry_after: Any
    required: float
    available: float

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        *,
        code: str | None = None,
        type: str | None = None,  # noqa: A002 - OpenAI alan adı
        param: str | None = None,
        evren: dict[str, Any] | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        Exception.__init__(self, message)
        self.status_code = status_code
        self.code = code
        self.type = type
        self.param = param
        self.evren = evren or {}
        self.request_id = request_id
        self.retry_after = retry_after
        # eski sınıfların beklediği alanlar
        self.required = 0.0
        self.available = float(self.evren.get("remaining_credits_cr", 0) or 0)

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (request_id={self.request_id})" if self.request_id else base


class LLMAuthenticationError(EvrenLLMError, AuthenticationError):
    pass


class LLMNotFoundError(EvrenLLMError, NotFoundError):
    pass


class LLMRateLimitError(EvrenLLMError, RateLimitError):
    pass


class LLMInsufficientCreditsError(EvrenLLMError, InsufficientCreditsError):
    pass


class LLMValidationError(EvrenLLMError, ValidationError):
    pass


class LLMServiceError(EvrenLLMError, InferenceError):
    """5xx, zaman aşımı veya akışın yarıda kesilmesi."""


class LLMConflictError(EvrenLLMError):
    """409 — aynı Idempotency-Key ile çakışan istek."""


def _error_class(status: int, code: str | None, etype: str | None) -> type[EvrenLLMError]:
    if etype == "insufficient_quota" or status == 402:
        return LLMInsufficientCreditsError
    if status in (401, 403):
        return LLMAuthenticationError
    if status == 404:
        return LLMNotFoundError
    if status == 409:
        return LLMConflictError
    if status == 429:
        return LLMRateLimitError
    if status in (400, 413, 422):
        return LLMValidationError
    if status >= 500:
        return LLMServiceError
    return EvrenLLMError


def retry_after_of(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def error_from_response(resp: httpx.Response) -> EvrenLLMError:
    err: dict[str, Any] = {}
    try:
        body = resp.json()
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            err = body["error"]
    except ValueError:
        pass
    message = str(err.get("message") or resp.text[:300] or f"HTTP {resp.status_code}")
    code, etype = err.get("code"), err.get("type")
    cls = _error_class(resp.status_code, code, etype)
    return cls(
        message, resp.status_code, code=code, type=etype, param=err.get("param"),
        evren=err.get("evren") if isinstance(err.get("evren"), dict) else None,
        request_id=resp.headers.get("x-request-id"), retry_after=retry_after_of(resp.headers),
    )


def stream_error(payload: dict[str, Any] | None, request_id: str | None) -> EvrenLLMError:
    err = payload or {}
    return LLMServiceError(
        str(err.get("message") or "akış yarıda kesildi; yanıt eksik olabilir, isteği tekrarlayın"),
        err.get("status_code") if isinstance(err.get("status_code"), int) else None,
        code=err.get("code") or "stream_interrupted", type=err.get("type") or "api_error",
        request_id=request_id,
    )
