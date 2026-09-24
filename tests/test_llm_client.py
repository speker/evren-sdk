import json

import httpx
import pytest

import evren_sdk.llm_async_client as amod
import evren_sdk.llm_client as smod
from evren_sdk import (
    AsyncEvrenLLMClient,
    EvrenError,
    EvrenLLMClient,
    LLMRateLimitError,
    LLMServiceError,
    LLMValidationError,
    RateLimitError,
    ValidationError,
)

KEY = "evren_llm_test"
RID = {"x-request-id": "req-1"}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(smod.time, "sleep", lambda s: None)

    async def _asleep(s):
        return None

    monkeypatch.setattr(amod.asyncio, "sleep", _asleep)


def _client(handler, **kw):
    return EvrenLLMClient(KEY, transport=httpx.MockTransport(handler), **kw)


def _completion(msg, finish="stop"):
    return {"id": "c1", "model": "glm-5.3", "choices": [{"message": msg, "finish_reason": finish}],
            "usage": {"completion_tokens": 3}}


def _sse(*chunks, done=True):
    parts = [f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks]
    if done:
        parts.append("data: [DONE]\n\n")
    return "".join(parts).encode()


def test_chat_parse():
    seen = {}

    def h(req):
        seen.update(req.headers)
        return httpx.Response(200, json=_completion({
            "content": None, "reasoning_content": "düşünüyorum",
            "tool_calls": [{"id": "t1", "function": {"name": "hava", "arguments": "{}"}}],
        }, "tool_calls"), headers={**RID, "x-evren-credits-remaining": "12.5"})

    r = _client(h).chat("glm-5.3", [{"role": "user", "content": "x"}])
    assert r.content == ""
    assert r.reasoning == "düşünüyorum"
    assert r.tool_calls[0]["function"]["name"] == "hava"
    assert r.finish_reason == "tool_calls"
    assert r.request_id == "req-1" and r.meta.credits_remaining == 12.5
    assert seen["authorization"] == f"Bearer {KEY}"
    assert seen["user-agent"].startswith("evren-sdk/")
    assert seen["idempotency-key"]


def test_retry_same_idem_key():
    keys = []

    def h(req):
        keys.append(req.headers["idempotency-key"])
        if len(keys) == 1:
            return httpx.Response(503, json={"error": {"message": "meşgul"}}, headers={"retry-after": "0"})
        return httpx.Response(200, json=_completion({"content": "Ankara"}))

    assert _client(h).chat("m", []).content == "Ankara"
    assert len(keys) == 2 and keys[0] == keys[1]


def test_no_retry_on_400():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(422, json={"error": {
            "message": "geçersiz", "type": "invalid_request_error", "param": "max_tokens", "code": "validation_error",
            "evren": {"suggested_models": ["glm-5.3"]}}}, headers=RID)

    with pytest.raises(LLMValidationError) as ei:
        _client(h).chat("m", [])
    e = ei.value
    assert len(calls) == 1
    assert isinstance(e, ValidationError) and isinstance(e, EvrenError)
    assert (e.status_code, e.code, e.param, e.request_id) == (422, "validation_error", "max_tokens", "req-1")
    assert e.evren["suggested_models"] == ["glm-5.3"]


def test_rate_limit_exhausted():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(429, json={"error": {"message": "yavaş", "code": "rate_limit_exceeded"}},
                              headers={"retry-after": "7"})

    with pytest.raises(LLMRateLimitError) as ei:
        _client(h, max_retries=2).chat("m", [])
    assert len(calls) == 3
    assert isinstance(ei.value, RateLimitError) and ei.value.retry_after == 7


def test_connect_error_retried_read_timeout_not():
    n = {"c": 0}

    def flaky(req):
        n["c"] += 1
        if n["c"] == 1:
            raise httpx.ConnectError("yok", request=req)
        return httpx.Response(200, json=_completion({"content": "ok"}))

    assert _client(flaky).chat("m", []).content == "ok"

    t = {"c": 0}

    def slow(req):
        t["c"] += 1
        raise httpx.ReadTimeout("yavaş", request=req)

    with pytest.raises(LLMServiceError) as ei:
        _client(slow).chat("m", [])
    assert t["c"] == 1 and ei.value.code == "timeout"


def test_stream_split_utf8():
    body = _sse(
        {"choices": [{"delta": {"reasoning": "hmm"}}]},
        {"choices": [{"delta": {"content": "Çanakkale"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"total_tokens": 9}},
    ).replace(b"\n\n", b"\r\n\r\n")
    cut = body.index("Ç".encode()) + 1  # çok baytlı karakterin ortasından böl

    def h(req):
        assert json.loads(req.content)["stream_options"] == {"include_usage": True}
        return httpx.Response(200, stream=_chunks(body, cut), headers=RID)

    deltas = list(_client(h).stream("m", []))
    assert "".join(d.content for d in deltas) == "Çanakkale"
    assert deltas[0].reasoning == "hmm"
    assert deltas[-1].usage == {"total_tokens": 9}


class _chunks(httpx.SyncByteStream):
    def __init__(self, body, cut):
        self._parts = [body[:cut], body[cut:]]

    def __iter__(self):
        yield from self._parts


def test_stream_interrupted():
    body = _sse({"choices": [{"delta": {"content": "yarım"}}]},
                {"choices": [{"delta": {}, "finish_reason": "error"}]})
    got = []
    with pytest.raises(LLMServiceError) as ei:
        for t in _client(lambda r: httpx.Response(200, content=body, headers=RID)).chat_stream("m", []):
            got.append(t)
    assert got == ["yarım"]
    assert ei.value.code == "stream_interrupted" and ei.value.request_id == "req-1"


def test_stream_truncated():
    body = _sse({"choices": [{"delta": {"content": "yarım"}}]}, done=False)
    with pytest.raises(LLMServiceError):
        list(_client(lambda r: httpx.Response(200, content=body)).chat_stream("m", []))


def test_stream_retry_rules():
    seq = [503, 200]

    def h(req):
        s = seq.pop(0)
        if s != 200:
            return httpx.Response(s, json={"error": {"message": "dolu"}})
        return httpx.Response(200, content=_sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}))

    assert "".join(_client(h).chat_stream("m", [])) == "ok"

    calls = []

    def bad_gw(req):
        calls.append(1)
        return httpx.Response(502, text="bad gateway")

    with pytest.raises(LLMServiceError):
        list(_client(bad_gw).chat_stream("m", []))
    assert len(calls) == 1  # 502'de istek modele ulaşmış olabilir, akış tekrarlanmaz


def test_rerank_sorted():
    def h(req):
        assert json.loads(req.content)["top_n"] == 2
        return httpx.Response(200, json={"results": [
            {"index": 0, "relevance_score": 0.1}, {"index": 2, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5, "document": {"text": "b"}}]})

    res = _client(h).rerank("r", "q", ["a", "b", "c"], top_n=2)
    assert [(x.index, x.document) for x in res] == [(2, None), (1, "b")]


def test_env_key(monkeypatch):
    monkeypatch.delenv(smod.ENV_API_KEY, raising=False)
    with pytest.raises(ValueError):
        EvrenLLMClient()
    monkeypatch.setenv(smod.ENV_API_KEY, "evren_llm_env")
    c = EvrenLLMClient()
    assert c._http.headers["authorization"] == "Bearer evren_llm_env"


@pytest.mark.asyncio
async def test_async_chat_stream():
    body = _sse({"choices": [{"delta": {"content": "An"}}]},
                {"choices": [{"delta": {"content": "kara"}, "finish_reason": "stop"}]})

    def h(req):
        if json.loads(req.content)["stream"]:
            return httpx.Response(200, content=body)
        return httpx.Response(200, json=_completion({"content": "tamam"}))

    async with AsyncEvrenLLMClient(KEY, transport=httpx.MockTransport(h)) as c:
        assert (await c.chat("m", [])).content == "tamam"
        assert "".join([t async for t in c.chat_stream("m", [])]) == "Ankara"
