import json
import threading

import httpx
import pytest

from evren_sdk import EvrenClient, InferenceError
from evren_sdk.client import _to_prediction
from evren_sdk.ws_client import _SKIP, InferenceWSClient, _build_ws_url

VID = "019f15ec-4e75-7741-bb6c-80b332520cec"


def _client(handler):
    c = EvrenClient("evren_plt_test")
    c._http = httpx.Client(base_url="https://api.ssyz.org.tr/api/v1", transport=httpx.MockTransport(handler))
    return c


def test_download_follows_url(tmp_path):
    weights = b"\x08\x08\x12\x07" + b"x" * 5000

    def h(req):
        if req.url.path.endswith("/download"):
            return httpx.Response(200, json={
                "download_url": "https://api.ssyz.org.tr/api/v1/files/evren-models/k/best.onnx?dl=t",
                "filename": "best.onnx"})
        assert req.url.params["dl"] == "t"
        return httpx.Response(200, content=weights)

    p = _client(h).download_model(VID, tmp_path, fmt="onnx")
    assert p.name == "best.onnx"
    assert p.read_bytes() == weights


def test_download_missing_url(tmp_path):
    with pytest.raises(InferenceError):
        _client(lambda r: httpx.Response(200, json={"detail": "x"})).download_model(VID, tmp_path)


def test_classify_top1():
    p = _to_prediction({"class_name": "", "confidence": 0.0, "task": "classify", "probs": [
        {"class_name": "kedi", "confidence": 0.2}, {"class_name": "köpek", "confidence": 0.7}]})
    assert (p.class_name, p.confidence) == ("köpek", 0.7)
    assert _to_prediction({"class_name": "araba", "confidence": 0.9}).class_name == "araba"


def test_ws_url():
    u = _build_ws_url("https://api.ssyz.org.tr/api/v1", VID, "k", 0.25, 0.45, 640)
    assert u.startswith("wss://api.ssyz.org.tr/api/v1/inference/stream?")
    assert f"model_version_id={VID}" in u and "api_key=k" in u
    assert _build_ws_url("http://localhost:8000", VID, "k", 0.25, 0.45, 640).startswith(
        "ws://localhost:8000/api/v1/inference/stream?")


def test_ws_parse():
    ws = InferenceWSClient("k", VID)
    assert ws._parse_raw(json.dumps({"type": "session_info", "max_seconds": 600})) is _SKIP
    assert ws._parse_raw(json.dumps({"type": "session_expired"})) is None
    with pytest.raises(InferenceError):
        ws._parse_raw(json.dumps({"predictions": [], "inference_ms": 0, "error": "gpu"}))
    r = ws._parse_raw(json.dumps({"predictions": [{"task": "classify", "probs": [
        {"class_name": "a", "confidence": 0.9}]}], "inference_ms": 12}))
    assert r.predictions[0].class_name == "a" and r.predictions[0].probs


class _FakeWS:
    """Sunucu gibi davranır: gönderilen her kareye bir sonuç döner, yoldaki en fazla kareyi ölçer."""

    def __init__(self):
        self.pending = 0
        self.peak = 0
        self.lock = threading.Condition()

    def send(self, _):
        with self.lock:
            self.pending += 1
            self.peak = max(self.peak, self.pending)
            self.lock.notify_all()

    def recv(self, timeout=5):
        with self.lock:
            self.lock.wait_for(lambda: self.pending > 0, timeout=timeout)
            self.pending -= 1
        return json.dumps({"predictions": [], "inference_ms": 1})

    def close(self):
        pass


def test_pipeline_inflight_cap():
    ws = InferenceWSClient("k", VID)
    fake = _FakeWS()
    ws._ws, ws._connected = fake, True
    ws.start_pipeline(max_inflight=8)
    got = []
    t = threading.Thread(target=lambda: got.extend(ws.next_result(timeout=5) for _ in range(20)))
    t.start()
    for _ in range(20):
        ws.submit(b"jpg")
    t.join(timeout=10)
    ws.stop_pipeline()
    assert len(got) == 20
    assert fake.peak <= 3
