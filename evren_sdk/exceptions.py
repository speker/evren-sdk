from __future__ import annotations


class EvrenError(Exception):

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(EvrenError):
    """Gecersiz veya suresi dolmus API anahtari (HTTP 401/403)."""


class NotFoundError(EvrenError):
    """Belirtilen model veya versiyon bulunamadi (HTTP 404)."""


class RateLimitError(EvrenError):
    """Istek limiti asildi (HTTP 429).

    ``retry_after`` saniye bekleyip tekrar deneyin.
    """

    def __init__(self, message: str, retry_after: int = 5) -> None:
        self.retry_after = retry_after
        super().__init__(message, 429)


class InferenceError(EvrenError):
    """GPU cikarim sunucusu gecici olarak kullanilamiyor (HTTP 502/503)."""


class InsufficientCreditsError(EvrenError):
    """Yetersiz kredi bakiyesi (HTTP 402).

    ``required`` ve ``available`` alanlari varsa kredi bilgisi icerir.
    """

    def __init__(self, message: str, required: float = 0, available: float = 0) -> None:
        self.required = required
        self.available = available
        super().__init__(message, 402)


class ValidationError(EvrenError):
    """Istek parametrelerinde dogrulama hatasi (HTTP 422)."""
