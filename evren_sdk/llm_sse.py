"""SSE çerçeve ayrıştırıcı + akış parçası yorumlama.

Gelen ham parçalar tek tek decode edilmez: TCP parçalanması çok baytlı bir
UTF-8 karakterini ya da çerçeve sınırını iki pakete bölebilir.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from .llm_errors import EvrenLLMError, stream_error
from .llm_types import ChatDelta, parse_delta

DONE = object()


class SSEBuffer:
    def __init__(self) -> None:
        self._buf = b""

    def feed(self, raw: bytes) -> Iterator[str]:
        """Tamamlanan her çerçevenin `data` yükünü döner (çok satırlı data birleştirilir)."""
        self._buf += raw.replace(b"\r\n", b"\n")
        while b"\n\n" in self._buf:
            frame, self._buf = self._buf.split(b"\n\n", 1)
            data = [
                line[5:].lstrip()
                for line in frame.decode("utf-8", errors="replace").split("\n")
                if line.startswith("data:")
            ]
            if data:
                yield "\n".join(data)


def interpret(payload: str, request_id: str | None) -> ChatDelta | object | None:
    """Bir `data` yükünü yorumlar: `DONE`, `ChatDelta` ya da (bozuk/boş satırda) None.

    Raises:
        EvrenLLMError: Geçit akış içinde hata bildirdiyse ya da upstream akışı
            `finish_reason="error"` ile kapattıysa. Bu durumda gelen metin eksiktir.
    """
    if payload == "[DONE]":
        return DONE
    try:
        chunk = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(chunk, dict):
        return None
    if isinstance(chunk.get("error"), dict):
        raise stream_error(chunk["error"], request_id)
    delta = parse_delta(chunk)
    if delta.finish_reason == "error":
        raise stream_error(None, request_id)
    return delta


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


__all__ = ["DONE", "EvrenLLMError", "SSEBuffer", "StreamDrain", "interpret"]
