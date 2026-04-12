"""EVREN WebSocket inference client — gateway'e dogrudan baglanti.

HTTP POST yerine persistent WS baglantisi uzerinden frame gonderir.
Tek connection ile tum frame'ler iletilir — connection overhead yok.

    ws = InferenceWSClient("evren_xxx", "owner/model")
    ws.connect()
    result = ws.predict_frame(jpeg_bytes)
    ws.close()
"""
from __future__ import annotations

import json
import ssl
import time
import logging
from typing import Any
from urllib.parse import urlencode

from .client import EvrenClient, _is_uuid
from .models import ClassInfo, ModelClasses, Prediction, PredictResult

log = logging.getLogger(__name__)

_GW_WS_PATH = "/inference-gw/ws/stream"

_RECONNECT_MAX = 3
_RECONNECT_DELAYS = [1.0, 2.0, 4.0]


def _build_ws_url(
    base_url: str,
    model_path: str,
    token: str,
    confidence: float,
    iou: float,
    image_size: int,
) -> str:
    host = base_url.rstrip("/")
    if host.startswith("http://"):
        ws_host = "ws://" + host[7:]
    elif host.startswith("https://"):
        ws_host = "wss://" + host[8:]
    else:
        ws_host = "wss://" + host

    # /api/v1 prefix'ini cikar — gateway /inference-gw/ altinda
    for suffix in ("/api/v1", "/api"):
        if ws_host.endswith(suffix):
            ws_host = ws_host[: -len(suffix)]
            break

    params = urlencode({
        "token": token,
        "model_path": model_path,
        "confidence": str(confidence),
        "iou": str(iou),
        "image_size": str(image_size),
    })
    return f"{ws_host}{_GW_WS_PATH}?{params}"


class InferenceWSClient:
    """Sync WebSocket client — EvrenCamera ile kullanilir."""

    __slots__ = (
        "_api_key", "_model", "_conf", "_iou", "_imgsz",
        "_base_url", "_verify", "_ws", "_connected",
        "_class_map", "_color_map", "_weights_url",
        "_reconnect_count",
    )

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        base_url: str = "https://api.ssyz.org.tr/api/v1",
        verify: bool = False,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._conf = confidence
        self._iou = iou
        self._imgsz = image_size
        self._base_url = base_url
        self._verify = verify
        self._ws = None
        self._connected = False
        self._class_map: dict[int, str] = {}
        self._color_map: dict[str, str] = {}
        self._weights_url: str | None = None
        self._reconnect_count = 0

    def connect(self) -> None:
        """Model resolve + class info fetch + WS baglantisi kur."""
        try:
            from websockets.sync.client import connect as ws_connect
        except ImportError:
            raise ImportError(
                "WebSocket modu icin websockets gerekli: "
                "pip install evren-sdk[edge]"
            )

        client = EvrenClient(
            api_key=self._api_key,
            base_url=self._base_url,
            verify=self._verify,
        )
        try:
            _vid, wurl = client.resolve_ws_params(self._model)
            if not wurl:
                raise RuntimeError(f"Model '{self._model}' icin weights_url bulunamadi")
            self._weights_url = wurl

            try:
                mc = client.model_classes(self._model)
                self._class_map = {i: c.name for i, c in enumerate(mc.classes)}
                self._color_map = {c.name: c.color for c in mc.classes}
            except Exception:
                pass
        finally:
            client.close()

        url = _build_ws_url(
            self._base_url, self._weights_url, self._api_key,
            self._conf, self._iou, self._imgsz,
        )

        ssl_ctx = None
        if url.startswith("wss://"):
            ssl_ctx = ssl.create_default_context()
            if not self._verify:
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE

        self._ws = ws_connect(
            url,
            ssl=ssl_ctx,
            open_timeout=10,
            close_timeout=5,
            max_size=50 * 1024 * 1024,
        )
        self._connected = True
        self._reconnect_count = 0
        log.debug("WS connected: %s", self._model)

    def _reconnect(self) -> bool:
        if self._reconnect_count >= _RECONNECT_MAX:
            return False
        delay = _RECONNECT_DELAYS[min(self._reconnect_count, len(_RECONNECT_DELAYS) - 1)]
        self._reconnect_count += 1
        log.debug("WS reconnect #%d (%.1fs)", self._reconnect_count, delay)
        time.sleep(delay)
        try:
            self.connect()
            return True
        except Exception:
            return False

    def predict_frame(self, jpeg_bytes: bytes) -> PredictResult:
        if not self._connected or self._ws is None:
            raise RuntimeError("WS baglantisi yok — once connect() cagirin")

        try:
            self._ws.send(jpeg_bytes)
            raw = self._ws.recv(timeout=10)
        except Exception as exc:
            self._connected = False
            if self._reconnect():
                self._ws.send(jpeg_bytes)
                raw = self._ws.recv(timeout=10)
            else:
                raise RuntimeError(f"WS baglantisi koptu: {exc}") from exc

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data: dict[str, Any] = json.loads(raw)

        if data.get("type") == "session_expired":
            self._connected = False
            if self._reconnect():
                return self.predict_frame(jpeg_bytes)
            raise RuntimeError("WS session suresi doldu")

        preds_raw = data.get("predictions", [])
        self._remap_predictions(preds_raw)

        predictions = [
            Prediction(
                class_name=p.get("class_name", ""),
                confidence=p.get("confidence", 0.0),
                bbox=p.get("bbox", []),
                color=self._color_map.get(p.get("class_name", "")) or p.get("color"),
                keypoints=p.get("keypoints"),
                mask=p.get("mask"),
                obb=p.get("obb"),
                task=p.get("task"),
            )
            for p in preds_raw
        ]

        return PredictResult(
            predictions=predictions,
            inference_ms=data.get("inference_ms", 0),
        )

    def _remap_predictions(self, preds: list[dict]) -> None:
        if not self._class_map:
            return
        db_names = set(self._class_map.values())
        for p in preds:
            cname = p.get("class_name", "")
            if cname in db_names:
                continue
            cid = p.get("class_id")
            if cid is not None:
                p["class_name"] = self._class_map.get(int(cid), cname)
            elif cname.startswith("class_") or cname.isdigit():
                try:
                    idx = int(cname.replace("class_", ""))
                    p["class_name"] = self._class_map.get(idx, cname)
                except ValueError:
                    pass

    @property
    def connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        self._connected = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def __enter__(self) -> InferenceWSClient:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        st = "connected" if self._connected else "disconnected"
        return f"<InferenceWSClient model={self._model!r} {st}>"
