import queue

import pytest

from evren_sdk.ws_client import InferenceWSClient

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from evren_sdk.edge import EvrenCamera  # noqa: E402

VID = "019f15ec-4e75-7741-bb6c-80b332520cec"


def _camera(mode):
    return EvrenCamera("k", VID, mode=mode)


def _no_ws(monkeypatch):
    def _fail(self):
        raise OSError("ws kapalı")

    monkeypatch.setattr(InferenceWSClient, "connect", _fail)


def test_http_mode_skips_ws():
    cam = _camera("http")
    assert cam._try_ws() is None
    cam.close()


def test_auto_falls_back(monkeypatch):
    _no_ws(monkeypatch)
    cam = _camera("auto")
    assert cam._try_ws() is None and cam._ws_conn is None
    cam.close()


def test_ws_mode_raises(monkeypatch):
    _no_ws(monkeypatch)
    cam = _camera("ws")
    with pytest.raises(OSError):
        cam._try_ws()
    cam.close()


def test_record_frame_without_result():
    class _Writer:
        frames = 0

        def write(self, frame):
            self.frames += 1

    class _SlowWS:
        def next_result(self, timeout=10):
            raise queue.Empty

    cam = _camera("http")
    writer = _Writer()
    cam._write_ws_frame(writer, _SlowWS(), np.zeros((8, 8, 3), dtype=np.uint8))
    assert writer.frames == 1
    cam.close()


def test_is_live():
    cam = _camera("http")
    assert cam._is_live(0) and cam._is_live("rtsp://kamera/1") and not cam._is_live("video.mp4")
    cam.close()
