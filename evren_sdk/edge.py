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

_PAL = [
    (248, 189, 56), (94, 197, 34), (36, 191, 251),
    (94, 63, 244), (247, 85, 168), (22, 115, 249),
    (180, 230, 60), (100, 180, 220), (80, 200, 200),
    (130, 90, 240), (200, 140, 60), (60, 220, 170),
]

_SENTINEL = object()
_JPARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 55] if _CV2 else []


def _require_cv2() -> None:
    if not _CV2:
        raise ImportError(
            "Edge modulu icin opencv gerekli: pip install evren-sdk[edge]"
        )


def _hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def _is_normalized(bbox: list[float], fw: int, fh: int) -> bool:
    if fw <= 1 or fh <= 1:
        return False
    return all(0.0 <= v <= 1.0 for v in bbox[:4])


def _denorm(val: float, dim: int) -> int:
    return int(val * dim) if val <= 1.0 else int(val)


def draw_predictions(
    frame: np.ndarray,
    predictions: list[Prediction],
    *,
    thickness: int = 2,
    font_scale: float = 0.55,
    show_conf: bool = True,
) -> np.ndarray:
    _require_cv2()
    fh, fw = frame.shape[:2]
    fnt = cv2.FONT_HERSHEY_SIMPLEX

    for i, p in enumerate(predictions):
        col = _hex_bgr(p.color) if p.color else _PAL[i % len(_PAL)]

        if p.bbox and len(p.bbox) >= 4:
            norm = _is_normalized(p.bbox, fw, fh)
            if norm:
                x1 = int(p.bbox[0] * fw)
                y1 = int(p.bbox[1] * fh)
                x2 = int(p.bbox[2] * fw)
                y2 = int(p.bbox[3] * fh)
            else:
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
                        kx = _denorm(kp[0], fw)
                        ky = _denorm(kp[1], fh)
                        cv2.circle(frame, (kx, ky), 4, col, -1)

        if p.mask and len(p.mask) > 2:
            import numpy as _np
            pts = []
            for pt in p.mask:
                if len(pt) >= 2:
                    px = _denorm(pt[0], fw)
                    py = _denorm(pt[1], fh)
                    pts.append([px, py])
            if pts:
                arr = _np.array(pts, dtype=_np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [arr], col)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                cv2.polylines(frame, [arr], True, col, 1, cv2.LINE_AA)

    return frame


def _hud(frame: np.ndarray, text: str) -> None:
    h = frame.shape[0]
    fnt = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, fnt, 0.48, 1)
    pad = 6
    y0 = h - th - pad * 3
    # alpha yok — 2-3ms kazanc
    cv2.rectangle(frame, (0, y0), (tw + pad * 4, h), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (pad * 2, h - pad - 4),
        fnt, 0.48, (220, 220, 220), 1, cv2.LINE_AA,
    )


def _resize_for_inference(frame: np.ndarray, target: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= target:
        return frame
    scale = target / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)


class EvrenCamera:
    """Herhangi bir kamera/video/RTSP kaynagini EVREN GPU'lariyla isle.

    >>> cam = EvrenCamera("evren_...", "owner/model")
    >>> cam.run(0)
    >>> for f, r in cam.stream("video.mp4"):
    ...     print(r.count)
    """

    __slots__ = (
        "_client", "_model", "_conf", "_iou", "_imgsz",
        "_max_fps", "_jpeg_q", "_draw", "_resize",
        "_stop", "_total_frames", "_total_lat",
        "_mode", "_api_key", "_base_url", "_verify",
        "_ws_conn",
    )

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        max_fps: float = 30.0,
        jpeg_quality: int = 55,
        draw: bool = True,
        resize: bool = True,
        mode: str = "auto",
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
        self._resize = resize
        self._mode = mode
        self._api_key = api_key
        self._base_url = base_url
        self._verify = verify
        self._ws_conn = None
        self._stop = threading.Event()
        self._total_frames = 0
        self._total_lat = 0.0

    def _is_live(self, source: int | str) -> bool:
        if isinstance(source, int):
            return True
        s = str(source).lower()
        return s.startswith("rtsp://") or s.startswith("http://") or s.startswith("https://")

    def _try_ws(self) -> bool:
        if self._mode == "http":
            return False
        try:
            from .ws_client import InferenceWSClient
            ws = InferenceWSClient(
                api_key=self._api_key,
                model=self._model,
                confidence=self._conf,
                iou=self._iou,
                image_size=self._imgsz,
                base_url=self._base_url,
                verify=self._verify,
            )
            ws.connect()
            self._ws_conn = ws
            return True
        except Exception:
            self._ws_conn = None
            if self._mode == "ws":
                raise
            return False

    def _encode(self, frame: np.ndarray) -> bytes:
        small = _resize_for_inference(frame, self._imgsz) if self._resize else frame
        _, buf = cv2.imencode(
            ".jpg", small,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_q],
        )
        return buf.tobytes()

    def _close_ws(self) -> None:
        if self._ws_conn:
            try:
                self._ws_conn.close()
            except Exception:
                pass
            self._ws_conn = None

    # -- public api --------------------------------------------------

    def stream(
        self,
        source: int | str = 0,
    ) -> Iterator[tuple[np.ndarray, PredictResult]]:
        """Frame-by-frame cikarim — her zaman pipeline modu.

        Yields:
            (frame_bgr, PredictResult) — draw=True ise frame annotated.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Kaynak acilamadi: {source}")

        ws_ok = self._try_ws()
        is_live = self._is_live(source)
        self._stop.clear()

        frame_q: queue.Queue = queue.Queue(maxsize=8)

        def _grab():
            while not self._stop.is_set():
                ok, frm = cap.read()
                if not ok:
                    break
                if is_live:
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        pass
                    frame_q.put(frm)
                else:
                    while not self._stop.is_set():
                        try:
                            frame_q.put(frm, timeout=0.5)
                            break
                        except queue.Full:
                            continue
            frame_q.put(_SENTINEL)

        cap_thr = threading.Thread(target=_grab, daemon=True, name="evren-cap")
        cap_thr.start()

        try:
            if ws_ok:
                yield from self._pipe_ws(frame_q, is_live)
            else:
                yield from self._pipe_http(frame_q, is_live)
        finally:
            self._stop.set()
            cap_thr.join(timeout=3)
            cap.release()
            self._close_ws()

    # -- WS pipeline: encode→send + recv paralel, GPU hic bos kalmaz --

    def _pipe_ws(
        self, frame_q: queue.Queue, is_live: bool,
    ) -> Iterator[tuple[np.ndarray, PredictResult]]:
        ws = self._ws_conn
        ws.start_pipeline(max_inflight=3)

        pending: queue.Queue = queue.Queue(maxsize=6)
        min_dt = 1.0 / self._max_fps if self._max_fps > 0 else 0.0

        def _encode_submit():
            last_ts = 0.0
            while not self._stop.is_set():
                try:
                    item = frame_q.get(timeout=1.0)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                if item is _SENTINEL:
                    pending.put(_SENTINEL)
                    break

                now = time.monotonic()
                if now - last_ts < min_dt:
                    if is_live:
                        continue
                jpg = self._encode(item)
                last_ts = time.monotonic()

                try:
                    ws.submit(jpg)
                    pending.put(item)
                except Exception:
                    pending.put(_SENTINEL)
                    break

        enc_thr = threading.Thread(target=_encode_submit, daemon=True, name="evren-enc")
        enc_thr.start()

        fps_t = time.monotonic()
        fps_cnt = 0

        try:
            while True:
                try:
                    item = pending.get(timeout=5.0)
                except queue.Empty:
                    break
                if item is _SENTINEL:
                    break
                frame = item

                try:
                    result = ws.next_result(timeout=10.0)
                except queue.Empty:
                    break
                if result is None:
                    break

                self._total_frames += 1
                self._total_lat += result.inference_ms
                fps_cnt += 1
                elapsed = time.monotonic() - fps_t
                fps = fps_cnt / elapsed if elapsed > 0.5 else 0.0
                if elapsed > 2.0:
                    fps_t, fps_cnt = time.monotonic(), 0

                if self._draw:
                    draw_predictions(frame, result.predictions)
                    _hud(frame, f"{result.count} tespit | {result.inference_ms:.0f}ms | {fps:.1f} FPS")

                yield frame, result
        finally:
            self._stop.set()
            ws.stop_pipeline()
            enc_thr.join(timeout=3)

    # -- HTTP pipeline: encode+predict ayri thread, main sadece draw --

    def _pipe_http(
        self, frame_q: queue.Queue, is_live: bool,
    ) -> Iterator[tuple[np.ndarray, PredictResult]]:
        out_q: queue.Queue = queue.Queue(maxsize=2)
        min_dt = 1.0 / self._max_fps if self._max_fps > 0 else 0.0

        def _worker():
            last_ts = 0.0
            while not self._stop.is_set():
                try:
                    item = frame_q.get(timeout=1.0)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                if item is _SENTINEL:
                    out_q.put(_SENTINEL)
                    break

                now = time.monotonic()
                if now - last_ts < min_dt:
                    if is_live:
                        continue
                last_ts = time.monotonic()

                jpg = self._encode(item)
                t0 = time.monotonic()
                try:
                    result = self._client.predict(
                        self._model, jpg,
                        confidence=self._conf, iou=self._iou,
                        image_size=self._imgsz,
                    )
                    lat = (time.monotonic() - t0) * 1000
                    out_q.put((item, result, lat))
                except Exception:
                    import sys
                    print("[EVREN] HTTP cikarim hatasi", file=sys.stderr, flush=True)
                    time.sleep(0.3)

        work_thr = threading.Thread(target=_worker, daemon=True, name="evren-http")
        work_thr.start()

        fps_t = time.monotonic()
        fps_cnt = 0

        try:
            while True:
                try:
                    item = out_q.get(timeout=5.0)
                except queue.Empty:
                    break
                if item is _SENTINEL:
                    break

                frame, result, lat = item
                self._total_frames += 1
                self._total_lat += lat
                fps_cnt += 1
                elapsed = time.monotonic() - fps_t
                fps = fps_cnt / elapsed if elapsed > 0.5 else 0.0
                if elapsed > 2.0:
                    fps_t, fps_cnt = time.monotonic(), 0

                if self._draw:
                    draw_predictions(frame, result.predictions)
                    _hud(frame, f"{result.count} tespit | {lat:.0f}ms | {fps:.1f} FPS")

                yield frame, result
        finally:
            self._stop.set()
            work_thr.join(timeout=3)

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
        """Annotated video kaydet — pipeline modu ile."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Kaynak acilamadi: {source}")

        ws_ok = self._try_ws()

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out_fps = fps if fps > 0 else min(src_fps, self._max_fps)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output), fourcc, out_fps, (w, h))

        skip = max(1, int(src_fps / out_fps))
        idx = 0

        if ws_ok:
            self._ws_conn.start_pipeline(max_inflight=3)

        try:
            pending_frames: list = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % skip != 0:
                    continue

                jpg = self._encode(frame)

                if ws_ok:
                    self._ws_conn.submit(jpg)
                    pending_frames.append(frame)
                    # drain sonuclari (non-blocking batch)
                    while len(pending_frames) >= 3:
                        r = self._ws_conn.next_result(timeout=10)
                        pf = pending_frames.pop(0)
                        if r and self._draw:
                            draw_predictions(pf, r.predictions)
                        writer.write(pf)
                else:
                    try:
                        result = self._client.predict(
                            self._model, jpg,
                            confidence=self._conf, iou=self._iou,
                            image_size=self._imgsz,
                        )
                        if self._draw:
                            draw_predictions(frame, result.predictions)
                    except Exception as exc:
                        import sys
                        print(f"[EVREN] cikarim hatasi: {exc}", file=sys.stderr, flush=True)
                    writer.write(frame)

            # flush kalan WS sonuclari
            if ws_ok:
                for pf in pending_frames:
                    try:
                        r = self._ws_conn.next_result(timeout=10)
                        if r and self._draw:
                            draw_predictions(pf, r.predictions)
                    except Exception:
                        pass
                    writer.write(pf)
        finally:
            writer.release()
            cap.release()
            self._close_ws()

    @property
    def stats(self) -> dict[str, float]:
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
        self._close_ws()
        self._client.close()

    def __enter__(self) -> EvrenCamera:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<EvrenCamera model={self._model!r} mode={self._mode} fps={self._max_fps}>"
