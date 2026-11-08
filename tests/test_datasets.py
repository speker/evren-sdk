import hashlib
import io

import httpx
import pytest

from evren_sdk import EvrenClient, EvrenError

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

DS = "01a173e7-f459-7262-9c7d-e6fc8617f100"


def _parquet(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf, row_group_size=1)
    return buf.getvalue()


FILES = {
    "train-00000.parquet": _parquet([{"item_id": 1, "text": "iyi", "labels": ["olumlu"]},
                                     {"item_id": 2, "text": "kötü", "labels": []}]),
    "test-00000.parquet": _parquet([{"item_id": 3, "text": "nötr", "labels": []}]),
}


def _api(statuses: list[str], *, bad_sha: bool = False, calls: list | None = None):
    def file(name: str) -> dict:
        body = FILES[name]
        sha = "0" * 64 if bad_sha else hashlib.sha256(body).hexdigest()
        return {"split": name.split("-")[0], "name": name, "rows": 1, "bytes": len(body), "sha256": sha,
                "url": "x", "path": f"/datasets/{DS}/records/exports/e1/files/{name}"}

    def handler(req: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append((req.method, req.url.path, req.headers.get("range")))
        path = req.url.path
        if path.endswith("/datasets/sahip/set"):
            return httpx.Response(200, json={"success": True, "data": {"id": DS}})
        if "/files/" in path:
            body = FILES[path.rsplit("/", 1)[1]]
            rng = req.headers.get("range")
            if rng:
                a, b = map(int, rng.removeprefix("bytes=").split("-"))
                return httpx.Response(206, content=body[a:b + 1])
            return httpx.Response(200, content=body)
        status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
        files = [file(n) for n in FILES] if status == "READY" else []
        return httpx.Response(200, json={"success": True, "data": {
            "id": "e1", "status": status, "format": "parquet", "rows": 3, "files": files}})
    return handler


def _client(handler) -> EvrenClient:
    c = EvrenClient("evren_plt_test")
    c._http = httpx.Client(base_url="https://api.ssyz.org.tr/api/v1", transport=httpx.MockTransport(handler))
    return c


def test_load_waits_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr("evren_sdk.datasets.time.sleep", lambda s: None)
    calls: list = []
    c = _client(_api(["PENDING", "RUNNING", "READY"], calls=calls))
    tables = c.load_dataset("sahip/set", cache_dir=tmp_path)
    assert set(tables) == {"train", "test"}
    assert tables["train"].column("labels").to_pylist() == [["olumlu"], []]
    downloads = sum(1 for m, p, _ in calls if "/files/" in p)
    assert downloads == 2
    train = c.load_dataset("sahip/set", split="train", cache_dir=tmp_path, columns=["text"])
    assert train.column_names == ["text"] and train.num_rows == 2
    # önbellek: ikinci yüklemede dosya yeniden indirilmez
    assert sum(1 for m, p, _ in calls if "/files/" in p) == downloads


def test_stream_uses_ranges():
    calls: list = []
    c = _client(_api(["READY"], calls=calls))
    rows = list(c.iter_dataset(DS, split="train", batch_size=1))
    assert [r["text"] for r in rows] == ["iyi", "kötü"]
    assert all(rng for m, p, rng in calls if "/files/" in p)


def test_bad_split_and_checksum(tmp_path):
    c = _client(_api(["READY"]))
    with pytest.raises(EvrenError, match="mevcut bölümler: test, train"):
        c.load_dataset(DS, split="dev", cache_dir=tmp_path)
    bad = _client(_api(["READY"], bad_sha=True))
    with pytest.raises(EvrenError, match="bütünlük"):
        bad.download_dataset(DS, tmp_path / "d")
    assert not list((tmp_path / "d").glob("*.parquet"))


def test_failed_export():
    c = _client(_api(["FAILED"]))
    with pytest.raises(EvrenError, match="başarısız"):
        c.export_dataset(DS)
