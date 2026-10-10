"""Yeniden deneme kararı.

Kural: istek modele ulaşmış olabilecekse aynı isteği körlemesine tekrar
göndermek çift çalıştırma (ve çift ücret) demektir. Bu yüzden:

- Akışsız isteklerde her mantıksal çağrıya bir `Idempotency-Key` verilir;
  geçit aynı anahtarla gelen tekrarı tekilleştirir (tamamlanmışsa yanıtı
  yeniden oynatır). 429/502/503/504 ve bağlantı hataları bu sayede güvenle
  tekrar denenir.
- Akışlar yeniden oynatılamaz; yalnızca geçidin isteği modele göndermeden
  reddettiği durumlarda (429, 503, bağlantı kurulamaması) tekrar denenir.
"""

from __future__ import annotations

import random

import httpx

RETRY_STATUSES = frozenset({429, 502, 503, 504})
STREAM_RETRY_STATUSES = frozenset({429, 503})
# bağlantı hiç kurulmadıysa istek sunucuya ulaşmamıştır
PRE_SEND_ERRORS: tuple[type[Exception], ...] = (httpx.ConnectError, httpx.ConnectTimeout)
# bağlantı yarıda koptu; yalnız idempotent isteklerde tekrar edilir
MID_FLIGHT_ERRORS: tuple[type[Exception], ...] = (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)

MAX_RETRY_AFTER_S = 30.0
_BASE_S = 0.5
_CAP_S = 8.0


def should_retry_status(status: int, *, stream: bool, retry_after: float | None = None) -> bool:
    # günlük kota dolunca geçit Retry-After'ı gün sonuna kadar verir; o hatayı beklemek anlamsız
    if retry_after is not None and retry_after > MAX_RETRY_AFTER_S:
        return False
    return status in (STREAM_RETRY_STATUSES if stream else RETRY_STATUSES)


def should_retry_exc(exc: Exception, *, idempotent: bool) -> bool:
    if isinstance(exc, PRE_SEND_ERRORS):
        return True
    return idempotent and isinstance(exc, MID_FLIGHT_ERRORS)


def backoff(attempt: int, retry_after: float | None) -> float:
    """`Retry-After` varsa ona uyar (üst sınırlı), yoksa jitter'lı üstel bekleme."""
    if retry_after is not None:
        return min(retry_after, MAX_RETRY_AFTER_S)
    base: float = min(_CAP_S, _BASE_S * float(2**attempt))
    return base * (0.5 + random.random() / 2)
