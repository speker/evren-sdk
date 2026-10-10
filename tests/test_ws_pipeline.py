import json

import pytest

import evren_sdk.ws_client as wsmod
from evren_sdk.ws_client import InferenceWSClient

VID = "019f15ec-4e75-7741-bb6c-80b332520cec"


class _ScriptedWS:
    """Sırayla verilen yanıtları döndürür; liste bitince bağlantı kopmuş gibi davranır."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def send(self, data):
        self.sent.append(data)

    def recv(self, timeout=5):
        if not self.replies:
            raise ConnectionError("kapandı")
        return self.replies.pop(0)

    def close(self):
        pass


def _ws(fake):
    ws = InferenceWSClient("k", VID)
    ws._ws, ws._connected = fake, True
    return ws


def test_predict_requires_connect():
    with pytest.raises(RuntimeError, match="connect"):
        InferenceWSClient("k", VID).predict_frame(b"jpg")


def test_pipeline_requires_start():
    ws = _ws(_ScriptedWS([]))
    with pytest.raises(RuntimeError):
        ws.submit(b"jpg")
    with pytest.raises(RuntimeError):
        ws.next_result()


def test_bad_frame_keeps_order():
    good = json.dumps({"predictions": [{"class_name": "araba", "confidence": 0.9}], "inference_ms": 3})
    ws = _ws(_ScriptedWS(["bozuk{json", good]))
    ws.start_pipeline()
    ws.submit(b"a")
    ws.submit(b"b")
    first, second = ws.next_result(timeout=5), ws.next_result(timeout=5)
    ws.stop_pipeline()
    assert first.predictions == [] and second.predictions[0].class_name == "araba"


def test_pipeline_signals_disconnect():
    ws = _ws(_ScriptedWS([]))
    ws.start_pipeline()
    assert ws.next_result(timeout=5) is None
    ws.stop_pipeline()


def test_session_expired_reconnect_fails(monkeypatch):
    monkeypatch.setattr(wsmod.time, "sleep", lambda s: None)
    ws = _ws(_ScriptedWS([json.dumps({"type": "session_info"}), json.dumps({"type": "session_expired"})]))

    def _fail(self):
        raise OSError("ağ yok")

    monkeypatch.setattr(InferenceWSClient, "connect", _fail)
    with pytest.raises(RuntimeError, match="süresi doldu"):
        ws.predict_frame(b"jpg")
    assert ws._reconnect_count == 1


def test_close_is_idempotent():
    class _Boom(_ScriptedWS):
        def close(self):
            raise OSError("zaten kapalı")

    ws = _ws(_Boom([]))
    ws.close()
    ws.close()
    assert ws._ws is None and not ws.connected
