"""EVREN Edge — GPU'suz cihazlarda bulut tabanli gercek zamanli cikarim.

Kullanim:
    cam = EvrenCamera("evren_...", "owner/model")
    cam.run(0)   # webcam ac, ESC ile kapat

    for frame, result in cam.stream("video.mp4"):
        print(result.count, "tespit")
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from .client import EvrenClient
from .models import Prediction, PredictResult

if TYPE_CHECKING:
    import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

# BGR fallback paleti — sinif rengi yoksa dongusel kullanilir
_PAL = [
    (248, 189, 56), (94, 197, 34), (36, 191, 251),
    (94, 63, 244), (247, 85, 168), (22, 115, 249),
    (180, 230, 60), (100, 180, 220), (80, 200, 200),
    (130, 90, 240), (200, 140, 60), (60, 220, 170),
]


def _require_cv2() -> None:
    if not _CV2:
        raise ImportError(
            "Edge modulu icin opencv gerekli: pip install evren-sdk[edge]"
        )


def _hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


# -------------------------------------------------------------------
# draw
# -------------------------------------------------------------------

def draw_predictions(
    frame: np.ndarray,
    predictions: list[Prediction],
    *,
    thickness: int = 2,
    font_scale: float = 0.55,
    show_conf: bool = True,
) -> np.ndarray:
    """Tahminleri frame uzerine render eder (in-place + return)."""
    _require_cv2()
    fnt = cv2.FONT_HERSHEY_SIMPLEX

    for i, p in enumerate(predictions):
        col = _hex_bgr(p.color) if p.color else _PAL[i % len(_PAL)]

        if p.bbox and len(p.bbox) >= 4:
            x1, y1 = int(p.bbox[0]), int(p.bbox[1])
            x2, y2 = int(p.bbox[2]), int(p.bbox[3])

            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thickness)

            txt = f"{p.class_name} {p.confidence:.0%}" if show_conf else p.class_name
            (tw, th), _ = cv2.getTextSize(txt, fnt, font_scale, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), col, -1)
            cv2.putText(
                frame, txt, (x1 + 4, y1 - 5),
                fnt, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
            )

        if p.obb and all(k in p.obb for k in ("cx", "cy", "w", "h", "angle")):
            import numpy as _np
            box = cv2.boxPoints((
                (p.obb["cx"], p.obb["cy"]),
                (p.obb["w"], p.obb["h"]),
                p.obb["angle"],
            ))
            cv2.drawContours(frame, [_np.intp(box)], 0, col, thickness)

        if p.keypoints:
            for kp in p.keypoints:
                if len(kp) >= 2:
                    vis = kp[2] if len(kp) > 2 else 1.0
                    if vis > 0.3:
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, col, -1)

        if p.mask and len(p.mask) > 2:
            import numpy as _np
            pts = _np.array(p.mask, dtype=_np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], col)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            cv2.polylines(frame, [pts], True, col, 1, cv2.LINE_AA)

    return frame


def _hud(frame: np.ndarray, text: str) -> None:
    """Sol alt koseye yari-saydam HUD cizer."""
    h = frame.shape[0]
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, fnt, 0.48, 1)
    pad = 6
    y0 = h - th - pad * 3
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (tw + pad * 4, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame, text, (pad * 2, h - pad - 4),
        fnt, 0.48, (220, 220, 220), 1, cv2.LINE_AA,
    )


# -------------------------------------------------------------------
# camera
# -------------------------------------------------------------------

class EvrenCamera:
    """Herhangi bir kamera/video/RTSP kaynagini EVREN GPU'lariyla isle.

    Cikarim bulutta calisir, kullanici lokal model gibi deneyimler.

    >>> cam = EvrenCamera("evren_...", "owner/model")
    >>> cam.run(0)                          # webcam, ESC ile kapat
    >>> for f, r in cam.stream("video.mp4"):  # veya iterator
    ...     print(r.count)
    """

    __slots__ = (
        "_client", "_model", "_conf", "_iou", "_imgsz",
        "_max_fps", "_jpeg_q", "_draw",
        "_stop", "_total_frames", "_total_lat",
    )

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        max_fps: float = 15.0,
        jpeg_quality: int = 70,
        draw: bool = True,
        base_url: str = "https://api.ssyz.org.tr/api/v1",
        verify: bool = False,
    ) -> None:
        _require_cv2()
        self._client = EvrenClient(api_key=api_key, base_url=base_url, verify=verify)
        self._model = model
        self._conf = confidence
        self._iou = iou
        self._imgsz = image_size
        self._max_fps = max_fps
        self._jpeg_q = jpeg_quality
        self._draw = draw
        self._stop = threading.Event()
        self._total_frames = 0
        self._total_lat = 0.0

    # -- public api --------------------------------------------------

    def stream(
        self,
        source: int | str = 0,
    ) -> Iterator[tuple[np.ndarray, PredictResult]]:
        """Frame-by-frame cikarim. Capture ayri thread'de calisir.

        Args:
            source: 0=webcam, "video.mp4", "rtsp://..." veya gorsel yolu.

        Yields:
            (frame_bgr, PredictResult) — draw=True ise frame annotated.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Kaynak acilamadi: {source}")

        latest: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop.clear()

        def _grab():
            while not self._stop.is_set():
                ok, frm = cap.read()
                if not ok:
                    self._stop.set()
                    break
                # sadece son frame'i tut, eskiyi at
                try:
                    latest.get_nowait()
                except queue.Empty:
                    pass
                latest.put(frm)

        thr = threading.Thread(target=_grab, daemon=True, name="evren-capture")
        thr.start()
        min_dt = 1.0 / self._max_fps if self._max_fps > 0 else 0.0
        last_ts = 0.0
        fps_t = time.monotonic()
        fps_cnt = 0

        try:
            while not self._stop.is_set():
                try:
                    frame = latest.get(timeout=0.5)
                except queue.Empty:
                    continue

                now = time.monotonic()
                if now - last_ts < min_dt:
                    continue
                last_ts = now

                _, buf = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_q],
                )
                jpg_bytes = buf.tobytes()

                t0 = time.monotonic()
                try:
                    result = self._client.predict(
                        self._model, jpg_bytes,
                        confidence=self._conf,
                        iou=self._iou,
                        image_size=self._imgsz,
                    )
                except Exception as exc:
                    import sys
                    print(f"[EVREN] cikarim hatasi: {exc}", file=sys.stderr, flush=True)
                    time.sleep(0.3)
                    continue

                lat = (time.monotonic() - t0) * 1000
                self._total_frames += 1
                self._total_lat += lat

                fps_cnt += 1
                elapsed = time.monotonic() - fps_t
                fps = fps_cnt / elapsed if elapsed > 0.5 else 0.0
                if elapsed > 2.0:
                    fps_t = time.monotonic()
                    fps_cnt = 0

                if self._draw:
                    draw_predictions(frame, result.predictions)
                    _hud(frame, f"{result.count} tespit | {lat:.0f}ms | {fps:.1f} FPS")

                yield frame, result
        finally:
            self._stop.set()
            thr.join(timeout=2)
            cap.release()

    def run(
        self,
        source: int | str = 0,
        *,
        window_name: str = "EVREN Edge",
    ) -> None:
        """Pencere acar, gercek zamanli cikarim gosterir. ESC=kapat."""
        for frame, _ in self.stream(source):
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
        cv2.destroyWindow(window_name)

    def scan(
        self,
        folder: str | Path,
        *,
        pattern: str = "*.jpg",
        save_to: str | Path | None = None,
    ) -> Iterator[tuple[Path, PredictResult]]:
        """Klasordeki gorselleri toplu isle.

        Args:
            folder: gorsel dizini.
            pattern: glob deseni ("*.jpg", "*.png", "**/*.jpg").
            save_to: annotated sonuclari kaydetme dizini.

        Yields:
            (dosya_yolu, PredictResult)
        """
        src = Path(folder)
        out = Path(save_to) if save_to else None
        if out:
            out.mkdir(parents=True, exist_ok=True)

        for p in sorted(src.glob(pattern)):
            result = self._client.predict(
                self._model, str(p),
                confidence=self._conf,
                iou=self._iou,
                image_size=self._imgsz,
            )
            if out and _CV2:
                img = cv2.imread(str(p))
                if img is not None:
                    draw_predictions(img, result.predictions)
                    cv2.imwrite(str(out / p.name), img)

            yield p, result

    def record(
        self,
        source: int | str,
        output: str | Path,
        *,
        codec: str = "mp4v",
        fps: float = 0.0,
    ) -> None:
        """Annotated video kaydet.

        Args:
            source: giris kaynagi.
            output: cikis dosyasi ("sonuc.mp4").
            codec: FourCC codec ("mp4v", "XVID").
            fps: cikis fps, 0 ise kaynaktan al.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Kaynak acilamadi: {source}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out_fps = fps if fps > 0 else min(src_fps, self._max_fps)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output), fourcc, out_fps, (w, h))

        skip = max(1, int(src_fps / out_fps))
        idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % skip != 0:
                    continue

                _, buf = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_q],
                )
                try:
                    result = self._client.predict(
                        self._model, buf.tobytes(),
                        confidence=self._conf,
                        iou=self._iou,
                        image_size=self._imgsz,
                    )
                except Exception as exc:
                    import sys
                    print(f"[EVREN] cikarim hatasi: {exc}", file=sys.stderr, flush=True)
                    writer.write(frame)
                    continue

                if self._draw:
                    draw_predictions(frame, result.predictions)

                writer.write(frame)
        finally:
            writer.release()
            cap.release()

    @property
    def stats(self) -> dict[str, float]:
        """Oturum istatistikleri."""
        n = self._total_frames or 1
        return {
            "frames": self._total_frames,
            "avg_latency_ms": self._total_lat / n,
            "total_seconds": self._total_lat / 1000,
        }

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.stop()
        self._client.close()

    def __enter__(self) -> EvrenCamera:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<EvrenCamera model={self._model!r} fps={self._max_fps}>"
