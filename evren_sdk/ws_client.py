"""EVREN WebSocket inference client — `/api/v1/inference/stream`.

Backend ucu framework'ü (RF-DETR, D-FINE ...), fold'u ve sınıf adlarını HTTP
`predict()` ile aynı şekilde çözer. Gateway'e depolama yoluyla doğrudan
bağlanmak Ultralytics dışı modellerde yanlış sınıf ve kutu üretiyordu.

İki mod destekler:
  1. predict_frame() — senkron, tek frame gönder-al
  2. Pipeline — submit/next_result ile multi-frame in-flight (GPU hiç boş kalmaz)

    ws = InferenceWSClient("evren_xxx", "owner/model")
    ws.connect()
    ws.start_pipeline()
    ws.submit(jpeg1)
    ws.submit(jpeg2)       # GPU jpeg1 işlerken jpeg2 kuyrukta
    r1 = ws.next_result()  # jpeg1 sonucu
    r2 = ws.next_result()  # jpeg2 sonucu
    ws.stop_pipeline()
    ws.close()
"""
from __future__ import annotations

import contextlib
import json
import logging
import queue
import ssl
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import httpx

from .client import EvrenClient, _top1
from .exceptions import EvrenError, InferenceError
from .models import Prediction, PredictResult

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection

log = logging.getLogger(__name__)

_WS_PATH = "/inference/stream"
# sunucunun tahmin olmayan kontrol mesajları
_CONTROL_TYPES = frozenset({"session_info"})

_SKIP = object()
# sunucu bağlantı başına 1 işlenen + 2 kuyrukta kare tutar, fazlasını sessizce atar.
# Protokolde kare numarası yok: tek kayıp sonraki tüm sonuçları bir kare kaydırır.
_SERVER_MAX_INFLIGHT = 3

_RECONNECT_MAX = 3
_RECONNECT_DELAYS = [1.0, 2.0, 4.0]


def _build_ws_url(
    base_url: str,
    version_id: str,
    token: str,
    confidence: float,
    iou: float,
    image_size: int,
) -> str:
    host = base_url.rstrip("/")
    if host.startswith("http://"):
        ws_base = "ws://" + host[7:]
    elif host.startswith("https://"):
        ws_base = "wss://" + host[8:]
    else:
        ws_base = "wss://" + host
    if not ws_base.endswith("/api/v1"):
        ws_base = ws_base.removesuffix("/api") + "/api/v1"

    params = urlencode({
        "api_key": token,
        "model_version_id": version_id,
        "confidence": str(confidence),
        "iou": str(iou),
        "image_size": str(image_size),
    })
    return f"{ws_base}{_WS_PATH}?{params}"


class InferenceWSClient:

    __slots__ = (
        "_api_key", "_base_url", "_class_map", "_color_map", "_conf", "_connected",
        "_imgsz", "_inflight", "_iou", "_model", "_pipe_stop", "_pipeline_on",
        "_reconnect_count", "_recv_q", "_recv_thr", "_send_q", "_send_thr",
        "_verify", "_version_id", "_ws",
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
        verify: bool = True,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._conf = confidence
        self._iou = iou
        self._imgsz = image_size
        self._base_url = base_url
        self._verify = verify
        self._ws: ClientConnection | None = None
        self._connected = False
        self._class_map: dict[int, str] = {}
        self._color_map: dict[str, str] = {}
        self._version_id: str | None = None
        self._reconnect_count = 0
        self._send_q: queue.Queue[bytes | None] | None = None
        self._recv_q: queue.Queue[PredictResult | None] | None = None
        self._pipe_stop: threading.Event | None = None
        self._send_thr: threading.Thread | None = None
        self._recv_thr: threading.Thread | None = None
        self._pipeline_on = False
        self._inflight: threading.BoundedSemaphore | None = None

    def connect(self) -> None:
        try:
            from websockets.sync.client import connect as ws_connect
        except ImportError:
            raise ImportError(
                "WebSocket modu için websockets gerekli: "
                "pip install evren-sdk[edge]"
            )

        client = EvrenClient(
            api_key=self._api_key,
            base_url=self._base_url,
            verify=self._verify,
        )
        try:
            version_id = client._vid(self._model)
            self._version_id = version_id
            # sınıf listesi yalnız isim eşlemesi için; alınamazsa sunucunun adları kullanılır
            try:
                mc = client.model_classes(self._model)
                self._class_map = {i: c.name for i, c in enumerate(mc.classes)}
                self._color_map = {c.name: c.color for c in mc.classes}
            except (EvrenError, httpx.HTTPError) as exc:
                log.debug("model classes unavailable: %s", exc)
        finally:
            client.close()

        url = _build_ws_url(
            self._base_url, version_id, self._api_key,
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
        except Exception as exc:  # noqa: BLE001 - hangi hata olursa olsun sonuç "bağlanılamadı"
            log.debug("WS reconnect failed: %s", exc)
            return False

    # ----- parse helper -----

    def _parse_raw(self, raw: str | bytes) -> PredictResult | object | None:
        """Tahmin, `_SKIP` (kontrol mesajı) ya da None (oturum bitti) döner."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data: dict[str, Any] = json.loads(raw)

        if data.get("type") == "session_expired":
            return None
        if data.get("type") in _CONTROL_TYPES:
            return _SKIP
        if data.get("error") and not data.get("predictions"):
            raise InferenceError(f"akış çıkarımı başarısız: {data['error']}", 0)

        preds_raw = data.get("predictions", [])
        self._remap_predictions(preds_raw)

        predictions = []
        for p in preds_raw:
            name, conf = _top1(p)
            predictions.append(Prediction(
                class_name=name,
                confidence=conf,
                bbox=p.get("bbox", []),
                color=self._color_map.get(name) or p.get("color"),
                keypoints=p.get("keypoints"),
                mask=p.get("mask"),
                obb=p.get("obb"),
                task=p.get("task"),
                probs=p.get("probs"),
            ))

        return PredictResult(
            predictions=predictions,
            inference_ms=data.get("inference_ms", 0),
        )

    def _exchange(self, jpeg_bytes: bytes) -> PredictResult | None:
        ws = self._ws
        if ws is None:
            raise RuntimeError("WS bağlantısı yok — önce connect() çağırın")
        ws.send(jpeg_bytes)
        while True:
            res = self._parse_raw(ws.recv(timeout=10))
            if res is None or isinstance(res, PredictResult):
                return res

    # ----- senkron (eski uyumluluk) -----

    def predict_frame(self, jpeg_bytes: bytes) -> PredictResult:
        if not self._connected or self._ws is None:
            raise RuntimeError("WS bağlantısı yok — önce connect() çağırın")

        try:
            result = self._exchange(jpeg_bytes)
        except InferenceError:
            raise
        except Exception as exc:
            self._connected = False
            if not self._reconnect():
                raise RuntimeError(f"WS bağlantısı koptu: {exc}") from exc
            result = self._exchange(jpeg_bytes)

        if result is None:
            self._connected = False
            if self._reconnect():
                return self.predict_frame(jpeg_bytes)
            raise RuntimeError("WS session süresi doldu")

        return result

    # ----- pipeline API (multi-frame in-flight) -----

    def start_pipeline(self, max_inflight: int = 3) -> None:
        """Send/recv thread'lerini başlatır — GPU hiç boş kalmaz."""
        if self._pipeline_on:
            return
        max_inflight = max(1, min(max_inflight, _SERVER_MAX_INFLIGHT))
        self._inflight = threading.BoundedSemaphore(max_inflight)
        self._send_q = queue.Queue(maxsize=max_inflight)
        self._recv_q = queue.Queue(maxsize=max_inflight + 1)
        self._pipe_stop = threading.Event()
        self._send_thr = threading.Thread(
            target=self._pipe_send_loop, daemon=True, name="evren-ws-tx",
        )
        self._recv_thr = threading.Thread(
            target=self._pipe_recv_loop, daemon=True, name="evren-ws-rx",
        )
        self._send_thr.start()
        self._recv_thr.start()
        self._pipeline_on = True

    def _pipe_send_loop(self) -> None:
        ws, stop, send_q, recv_q = self._ws, self._pipe_stop, self._send_q, self._recv_q
        if ws is None or stop is None or send_q is None or recv_q is None:
            return
        while not stop.is_set():
            try:
                jpg = send_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if jpg is None:
                break
            try:
                ws.send(jpg)
            except Exception as exc:  # noqa: BLE001 - her gönderim hatası pipeline'ı bitirir, tüketiciye None gider
                log.debug("WS send failed: %s", exc)
                recv_q.put(None)
                break

    def _pipe_recv_loop(self) -> None:
        ws, stop, recv_q, inflight = self._ws, self._pipe_stop, self._recv_q, self._inflight
        if ws is None or stop is None or recv_q is None or inflight is None:
            return
        while not stop.is_set():
            try:
                raw = ws.recv(timeout=5)
            except Exception as exc:  # noqa: BLE001 - zaman aşımı da kopma da akışın sonu
                log.debug("WS recv ended: %s", exc)
                recv_q.put(None)
                break
            try:
                result = self._parse_raw(raw)
            except (InferenceError, ValueError) as exc:
                # kare sırası bozulmasın: hatalı kare boş sonuçla yer tutar
                log.warning("WS frame hatası: %s", exc)
                result = PredictResult(predictions=[], inference_ms=0)
            if result is _SKIP:
                continue
            if not isinstance(result, PredictResult):
                recv_q.put(None)
                break
            recv_q.put(result)
            with contextlib.suppress(ValueError):
                inflight.release()

    def submit(self, jpeg_bytes: bytes) -> None:
        """Kareyi gönderir; yolda sunucu sınırı kadar kare varsa sonuç gelene dek bekler."""
        if self._send_q is None or self._inflight is None:
            raise RuntimeError("Pipeline başlatılmadı — önce start_pipeline() çağırın")
        if not self._inflight.acquire(timeout=10):
            raise TimeoutError("WS pipeline: 10 sn içinde sonuç gelmedi")
        self._send_q.put(jpeg_bytes, timeout=10)

    def next_result(self, timeout: float = 10.0) -> PredictResult | None:
        if self._recv_q is None:
            raise RuntimeError("Pipeline başlatılmadı")
        return self._recv_q.get(timeout=timeout)

    def stop_pipeline(self) -> None:
        if not self._pipeline_on:
            return
        self._pipeline_on = False
        if self._pipe_stop:
            self._pipe_stop.set()
        if self._send_q:
            self._send_q.put(None)
        if self._send_thr:
            self._send_thr.join(timeout=3)
        if self._recv_thr:
            self._recv_thr.join(timeout=3)

    # ----- remap -----

    def _remap_predictions(self, preds: list[dict[str, Any]]) -> None:
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
        self.stop_pipeline()
        self._connected = False
        if self._ws is not None:
            # kapanışta oluşan hata kullanıcıya yansıtılmaz; bağlantı zaten bırakılıyor
            with contextlib.suppress(Exception):
                self._ws.close()
            self._ws = None

    def __enter__(self) -> InferenceWSClient:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        st = "pipeline" if self._pipeline_on else ("connected" if self._connected else "disconnected")
        return f"<InferenceWSClient model={self._model!r} {st}>"
