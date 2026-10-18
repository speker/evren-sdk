import base64
import io
import json

import httpx
import pytest

import evren_sdk.llm_async_client as amod
import evren_sdk.llm_client as smod
from evren_sdk import (
    AsyncEvrenLLMClient,
    EvrenLLMClient,
    LLMRateLimitError,
    LLMValidationError,
    encode_image,
)

KEY = "evren_llm_test"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
CLEF = {
    "model": "clef", "id": "r1",
    "answers": {
        "birim": {"type": "choice", "choice": "fatura", "probabilities": {"fatura": 0.9, "teknik": 0.1}},
        "iptal": {"type": "noul", "noul": 0.97},
        "aciliyet": {"type": "score", "score": 1.4},
    },
    "usage": {"input_tokens": 389, "output_tokens": 0},
}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(smod.time, "sleep", lambda s: None)

    async def _asleep(s):
        return None

    monkeypatch.setattr(amod.asyncio, "sleep", _asleep)


def _client(handler, **kw):
    return EvrenLLMClient(KEY, transport=httpx.MockTransport(handler), **kw)


def test_decide():
    seen = {}

    def h(req):
        seen["body"] = json.loads(req.content)
        seen["idem"] = req.headers.get("idempotency-key")
        return httpx.Response(200, json=CLEF, headers={"x-request-id": "q1"})

    r = _client(h).decide("clef", "Faturam iki kez kesildi", {"birim": {"type": "choice"}}, images=[PNG], lang="tr")

    assert seen["body"]["images"][0].startswith("data:image/png;base64,")
    assert seen["body"]["lang"] == "tr" and seen["body"]["model"] == "clef"
    assert seen["idem"]
    assert r.choice("birim") == "fatura" and r.noul("iptal") == 0.97 and r.score("aciliyet") == 1.4
    assert r.probabilities("birim") == {"fatura": 0.9, "teknik": 0.1}
    assert r.choice("yok") is None and r.request_id == "q1" and r.usage["input_tokens"] == 389


def test_decide_batch():
    item = {"model": "laya", "answers": {"birim": {"type": "choice", "choice": "teknik"}}}
    body = {"id": "b1", "results": [item, item], "total_usage": {"input_tokens": 40}}

    r = _client(lambda req: httpx.Response(200, json=body)).decide_batch("laya", ["a", "b"], {"birim": {}})

    assert [x.choice("birim") for x in r.results] == ["teknik", "teknik"]
    assert r.usage == {"input_tokens": 40}


def test_daily_quota_no_retry():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(429, json={"error": {"message": "günlük kota doldu", "code": "rate_limit_exceeded"}},
                              headers={"retry-after": "36000"})

    with pytest.raises(LLMRateLimitError):
        _client(h).decide("clef", "x", {})
    assert len(calls) == 1


def test_short_429_retried():
    calls = []

    def h(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "meşgul"}}, headers={"retry-after": "2"})
        return httpx.Response(200, json=CLEF)

    assert _client(h).decide("clef", "x", {}).choice("birim") == "fatura"
    assert len(calls) == 2


def test_ocr():
    seen = {}

    def h(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"model": "dots-ocr", "text": "TOPLAM 120 TL"})

    r = _client(h).ocr("dots-ocr", PNG)
    assert seen["image"] == "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert r.text == "TOPLAM 120 TL"


def test_transcribe_srt():
    seen = {}

    def h(req):
        seen["ctype"] = req.headers["content-type"]
        seen["idem"] = req.headers.get("idempotency-key")
        seen["body"] = req.content
        return httpx.Response(200, text="1\n00:00:00,000 --> 00:00:01,000\nMerhaba\n",
                              headers={"x-evren-media-id": "m1"})

    r = _client(h).transcribe("qwen3-asr-1.7b", io.BytesIO(b"RIFFwav"), filename="ses.wav",
                              response_format="srt", language="tr")
    assert seen["ctype"].startswith("multipart/form-data") and seen["idem"] is None
    assert b'name="response_format"' in seen["body"] and b"ses.wav" in seen["body"]
    assert "Merhaba" in r.text and r.media_id == "m1" and r.raw == {}


def test_transcribe_not_retried_after_send():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "dolu"}})

    with pytest.raises(Exception):
        _client(h).transcribe("asr", b"RIFF", response_format="json")
    assert len(calls) == 1


def test_speak(tmp_path):
    seen = {}

    def h(req):
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=b"ID3mp3", headers={
            "content-type": "audio/mpeg", "x-evren-audio-chunks": "3", "x-evren-request-id": "s1",
        })

    r = _client(h).speak("voxcpm2", "Merhaba.", instructions="sakin bir ses", speed=1.25)
    assert seen["path"] == "/v1/audio/speech"
    assert seen["body"] == {
        "model": "voxcpm2", "input": "Merhaba.", "voice": "default", "response_format": "mp3",
        "speed": 1.25, "instructions": "sakin bir ses",
    }
    assert r.audio == b"ID3mp3" and r.chunks == 3 and r.request_id == "s1" and r.sample_rate == 48000
    out = tmp_path / "ses.mp3"
    r.save(str(out))
    assert out.read_bytes() == b"ID3mp3"


@pytest.mark.asyncio
async def test_async_speak():
    c = AsyncEvrenLLMClient(KEY, transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"RIFF")))
    async with c:
        r = await c.speak("voxcpm2", "Merhaba.", response_format="wav")
    assert r.audio == b"RIFF" and r.format == "wav"


def test_quota_status():
    body = {"level": "warn", "used_tokens": 400, "cap": 500, "usage_ratio": 0.8, "held_cr": "1.5",
            "remaining_cr": "98.25", "daily_limit_tokens": 15000000, "daily_remaining_tokens": 14000000}

    q = _client(lambda req: httpx.Response(200, json=body)).quota()
    assert q.level == "warn" and q.remaining_cr == 98.25 and q.held_cr == 1.5
    assert q.daily_remaining_tokens == 14000000 and q.window_minutes is None


def test_request_status():
    body = {"request_id": "019a", "execution_terminal": "COMPLETED", "charge_terminal": True,
            "is_billing_pending": False, "collected_cr": "0.0123", "outstanding_cr": None,
            "collection_state": "SETTLED", "state_version": 3}

    s = _client(lambda req: httpx.Response(200, json=body)).request_status("019a")
    assert s.execution_terminal == "COMPLETED" and s.collected_cr == 0.0123 and s.outstanding_cr is None


def test_respond():
    body = {"id": "resp_1", "model": "glm-5.3", "previous_response_id": "resp_0", "output": [
        {"type": "reasoning", "content": []},
        {"type": "message", "content": [{"type": "output_text", "text": "Anka"}, {"type": "output_text", "text": "ra"}]},
    ]}
    seen = {}

    def h(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=body)

    r = _client(h).respond("glm-5.3", "Başkent?", previous_response_id="resp_0")
    assert r.output_text == "Ankara" and r.previous_response_id == "resp_0"
    assert seen["stream"] is False and seen["previous_response_id"] == "resp_0"


def test_models():
    body = {"data": [{"id": "clef", "task": "decision", "modalities": ["text", "image"], "created": 1},
                     {"id": "glm-5.3", "created": 2}]}

    models = _client(lambda req: httpx.Response(200, json=body)).models()
    assert models[0].task == "decision" and "image" in models[0].modalities
    assert models[1].task == "chat" and models[1].modalities == ["text"]


def test_validation_error():
    resp = httpx.Response(400, json={"error": {"message": "'clef' toplu karar ucunu desteklemiyor",
                                               "type": "invalid_request_error", "code": "bad_request"}})
    with pytest.raises(LLMValidationError, match="toplu"):
        _client(lambda req: resp).decide_batch("clef", ["a"], {})


@pytest.mark.asyncio
async def test_async_decide():
    c = AsyncEvrenLLMClient(KEY, transport=httpx.MockTransport(lambda req: httpx.Response(200, json=CLEF)))
    async with c:
        r = await c.decide("clef", "x", {"birim": {"type": "choice"}})
    assert r.choice("birim") == "fatura"


def test_encode_image_formats(tmp_path):
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8 "
    assert encode_image(webp).startswith("data:image/webp;base64,")
    assert encode_image(b"\xff\xd8\xff\xe0").startswith("data:image/jpeg;base64,")
    assert encode_image("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"
    p = tmp_path / "fis.png"
    p.write_bytes(PNG)
    assert encode_image(p).startswith("data:image/png;base64,")
    with pytest.raises(ValueError):
        encode_image(b"not an image")


def test_encode_image_shrink():
    pil = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    pil.new("RGB", (3000, 2000), (200, 20, 20)).save(buf, format="PNG")

    url = encode_image(buf.getvalue(), max_side=1024)

    assert url.startswith("data:image/jpeg;base64,")
    with pil.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as img:
        assert img.size == (1024, 683)
    with pytest.raises(ValueError, match="açılamadı"):
        encode_image(b"not an image", max_side=1024)
