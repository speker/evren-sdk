"""Görsel ve ses girdisi yardımcıları.

Geçit uzak URL indirmez: görseller `data:image/...;base64,...` olarak gömülür,
ses dosyaları multipart ile gönderilir.
"""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from typing import BinaryIO

ImageSource = str | Path | bytes
AudioSource = str | Path | bytes | BinaryIO

_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sniff(raw: bytes) -> str | None:
    for signature, mime in _SIGNATURES:
        if raw.startswith(signature):
            return mime
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def encode_image(source: ImageSource, *, max_side: int | None = None, mime: str | None = None) -> str:
    """Görseli geçidin kabul ettiği gömülü biçime (`data:image/...;base64,...`) çevirir.

    Karar modelleri (clef) görseli küçültmeden okur ve 32×32 piksellik her karoyu
    1 token sayar: 1024 px ≈ 1.000, 4000 px ≈ 15.600 token. Büyük fotoğraflarda
    `max_side=1024` verin; bunun için Pillow gerekir (`pip install evren-sdk[images]`).

    Args:
        source: Dosya yolu, ham bayt ya da zaten `data:` ile başlayan bir URL (olduğu gibi döner).
        max_side: Verilirse uzun kenar bu piksel değerine indirilir; daha küçük görsele dokunulmaz.
        mime: Tür kestirilemiyorsa elle verilir (`image/png` gibi).

    Raises:
        ValueError: Görsel türü anlaşılamazsa.
        ImportError: `max_side` verildi ama Pillow kurulu değilse.
    """
    if isinstance(source, str) and source.startswith("data:"):
        return source
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    if max_side is not None:
        raw, mime = _shrink(raw, max_side, mime)
    kind = mime or _sniff(raw)
    if kind is None and not isinstance(source, bytes):
        kind = mimetypes.guess_type(str(source))[0]
    if not kind or not kind.startswith("image/"):
        raise ValueError("görsel türü anlaşılamadı; PNG, JPEG veya WebP verin ya da mime belirtin")
    return f"data:{kind};base64,{base64.b64encode(raw).decode('ascii')}"


def _shrink(raw: bytes, max_side: int, mime: str | None) -> tuple[bytes, str | None]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("max_side için Pillow gerekli: pip install evren-sdk[images]") from exc
    try:
        img = Image.open(io.BytesIO(raw))
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"görsel açılamadı: {exc}") from exc
    with img:
        if max(img.size) <= max_side:
            return raw, mime
        # saydamlık JPEG'de kaybolur; varsa PNG kalır
        keep_alpha = img.mode in ("RGBA", "LA", "P")
        small = img.convert("RGBA" if keep_alpha else "RGB")
        small.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        if keep_alpha:
            small.save(buf, format="PNG")
            return buf.getvalue(), "image/png"
        small.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"


def audio_file(source: AudioSource, filename: str | None = None) -> tuple[str, bytes, str]:
    """Ses kaynağını multipart parçasına çevirir: (dosya adı, bayt, mime)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        raw, name = path.read_bytes(), filename or path.name
    elif isinstance(source, bytes):
        raw, name = source, filename or "audio.wav"
    else:
        raw = source.read()
        name = filename or Path(str(getattr(source, "name", "audio.wav"))).name
    return name, raw, mimetypes.guess_type(name)[0] or "application/octet-stream"
