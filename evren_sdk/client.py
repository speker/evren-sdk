from __future__ import annotations

import asyncio
import contextlib
import io
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from .exceptions import (
    AuthenticationError,
    InferenceError,
    InsufficientCreditsError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from .models import (
    BatchResult,
    BenchmarkResult,
    ClassInfo,
    ModelClasses,
    ModelInfo,
    ModelVersion,
    Prediction,
    PredictResult,
    SybilRiskResult,
)

JSON = dict[str, Any]

_API_BASE = "https://api.ssyz.org.tr/api/v1"
_BATCH_TIMEOUT = 120.0
_DEFAULT_RETRY_AFTER = 5


def _auth_headers(key: str) -> dict[str, str]:
    if key.startswith("evren_"):
        return {"X-API-Key": key}
    return {"Authorization": f"Bearer {key}"}


def _extract_msg(resp: httpx.Response) -> str:
    # gövde JSON değilse ham metnin başı yeterli
    with contextlib.suppress(ValueError):
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err.get("detail") or resp.text[:300])
            return str(body.get("detail") or body.get("message") or resp.text[:300])
    return resp.text[:300]


def _retry_after(resp: httpx.Response) -> int:
    # Retry-After HTTP-date biçiminde de gelebilir (RFC 9110 §10.2.3); o durumda varsayılan
    try:
        return max(0, int(float(resp.headers.get("Retry-After", _DEFAULT_RETRY_AFTER))))
    except ValueError:
        return _DEFAULT_RETRY_AFTER


def _raise_for(resp: httpx.Response) -> None:
    code = resp.status_code
    msg = _extract_msg(resp)

    if code in (401, 403):
        raise AuthenticationError(msg, code)
    if code == 402:
        req = avail = 0.0
        try:
            if "Gerekli:" in msg and "Mevcut:" in msg:
                parts = msg.split(",")
                req = float(parts[0].split(":")[1].strip())
                avail = float(parts[1].split(":")[1].strip())
        except (ValueError, IndexError):
            pass
        raise InsufficientCreditsError(msg, req, avail)
    if code == 404:
        raise NotFoundError(msg, code)
    if code == 409:
        raise ValidationError(msg, code)
    if code == 422:
        raise ValidationError(msg, code)
    if code == 429:
        raise RateLimitError(msg, _retry_after(resp))
    if code >= 500:
        raise InferenceError(msg, code)
    if 400 <= code < 500:
        raise ValidationError(msg, code)
    resp.raise_for_status()


_FMT_ALIASES: dict[str, str] = {
    "pt": "pytorch", "pth": "pytorch", "torch": "pytorch",
    "trt": "tensorrt", "engine": "tensorrt",
    "lite": "tflite",
}


def _normalize_fmt(fmt: str) -> str:
    return _FMT_ALIASES.get(fmt.lower(), fmt.lower())


_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp",
    ".bmp": "image/bmp", ".tiff": "image/tiff", ".tif": "image/tiff",
}


def _read_img(source: str | Path | bytes) -> tuple[bytes, str, str]:
    if isinstance(source, bytes):
        return source, "image.jpg", "image/jpeg"
    p = Path(source)
    mime = _MIME.get(p.suffix.lower(), "image/jpeg")
    return p.read_bytes(), p.name, mime


def _is_uuid(val: str) -> bool:
    try:
        uuid.UUID(val)
        return True
    except (ValueError, AttributeError):
        return False


def _unwrap(data: Any) -> Any:
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def _build_form(version_id: str, confidence: float,
                iou: float, imgsz: int,
                classes: list[str] | None) -> dict[str, str]:
    d: dict[str, str] = {
        "model_version_id": version_id,
        "confidence_threshold": str(confidence),
        "iou_threshold": str(iou),
        "image_size": str(imgsz),
    }
    if classes:
        d["classes"] = ",".join(classes)
    return d


def _top1(raw: JSON) -> tuple[str, float]:
    # classify yanıtında class_name boş gelir, sonuç probs içindedir
    name, conf = raw.get("class_name") or "", raw.get("confidence") or 0.0
    probs = raw.get("probs") or []
    if not name and probs:
        best = max(probs, key=lambda p: p.get("confidence", 0.0))
        name, conf = best.get("class_name", ""), best.get("confidence", 0.0)
    return str(name), float(conf)


def _to_prediction(raw: JSON) -> Prediction:
    name, conf = _top1(raw)
    return Prediction(
        class_name=name,
        confidence=conf,
        bbox=raw.get("bbox", []),
        color=raw.get("color"),
        keypoints=raw.get("keypoints"),
        mask=raw.get("mask"),
        obb=raw.get("obb"),
        task=raw.get("task"),
        probs=raw.get("probs"),
    )


def _to_result(body: JSON) -> PredictResult:
    return PredictResult(
        predictions=[_to_prediction(p) for p in body.get("predictions", [])],
        inference_ms=body.get("inference_ms", 0),
        image_width=body.get("image_width"),
        image_height=body.get("image_height"),
        model_version_id=body.get("model_version_id"),
    )


def _to_sybil_result(body: JSON) -> SybilRiskResult:
    return SybilRiskResult(
        id=body["id"], series_id=body["series_id"], status=body.get("status", "PENDING"),
        years=body.get("years", [1, 2, 3, 4, 5, 6]), scores=body.get("scores"),
        attention_url=body.get("attention_url"), error_message=body.get("error_message"),
    )


def _to_model_info(m: JSON) -> ModelInfo:
    return ModelInfo(
        id=m["id"], name=m["name"], slug=m.get("slug", ""),
        architecture=m.get("architecture"),
        owner_username=m.get("owner_username"),
        owner_display_name=m.get("owner_display_name"),
        description=m.get("description"),
        modality=m.get("modality"),
        visibility=m.get("visibility"),
        stage=m.get("stage"),
        stars_count=m.get("stars_count", 0),
        forks_count=m.get("forks_count", 0),
        downloads_count=m.get("downloads_count", 0),
        versions_count=m.get("versions_count", 0),
        tags=m.get("tags") or [],
        created_at=m.get("created_at"),
    )


def _to_model_ver(v: JSON) -> ModelVersion:
    return ModelVersion(
        id=v["id"], version_tag=v.get("version_tag", ""),
        weights_url=v.get("weights_url"),
        framework=v.get("framework", "pytorch"),
        metrics=v.get("metrics", {}),
        file_size_bytes=v.get("file_size_bytes", 0),
        onnx_url=v.get("onnx_url"),
        tensorrt_url=v.get("tensorrt_url"),
        tflite_url=v.get("tflite_url"),
        job_id=v.get("job_id"),
        dataset_name=v.get("dataset_name"),
        training_architecture=v.get("training_architecture"),
        training_job_name=v.get("training_job_name"),
        gpu_count=v.get("gpu_count"),
        training_duration_hours=v.get("training_duration_hours"),
        created_at=v.get("created_at"),
    )


def _resolve_versions(body: Any) -> list[JSON]:
    data = _unwrap(body)
    rows = data if isinstance(data, list) else data.get("items", data.get("data", []))
    return [row for row in rows if isinstance(row, dict)]


def _download_target(info: JSON, output: str | Path, vid: str, fmt: str) -> tuple[str, Path]:
    """İndirme ucu dosyayı değil `{download_url, filename}` döndürür; hedef yolu hazırlar."""
    url = info.get("download_url")
    if not url:
        raise InferenceError("indirme bağlantısı alınamadı", 0)
    dest = Path(output)
    if dest.suffix == "":
        dest.mkdir(parents=True, exist_ok=True)
        dest = dest / (info.get("filename") or f"{vid}.{fmt}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
    return str(url), dest


def _same_host(url: str, base: httpx.URL) -> bool:
    u = httpx.URL(url)
    return not u.is_absolute_url or u.host == base.host


def _stream_to_file(http: httpx.Client, url: str, dest: Path) -> None:
    with http.stream("GET", url, timeout=_BATCH_TIMEOUT) as resp:
        if resp.status_code != 200:
            resp.read()
            _raise_for(resp)
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)


async def _astream_to_file(http: httpx.AsyncClient, url: str, dest: Path) -> None:
    async with http.stream("GET", url, timeout=_BATCH_TIMEOUT) as resp:
        if resp.status_code != 200:
            await resp.aread()
            _raise_for(resp)
        with dest.open("wb") as fh:
            async for chunk in resp.aiter_bytes(1 << 20):
                fh.write(chunk)


def _pick_version(versions: list[JSON], tag: str | None, slug: str) -> str:
    if not versions:
        raise NotFoundError(f"'{slug}' için versiyon yok", 404)

    if tag is None:
        return str(versions[0]["id"])

    for v in versions:
        if v.get("version_tag") == tag:
            return str(v["id"])
    raise NotFoundError(f"'{slug}' için '{tag}' versiyonu yok", 404)


# ---------------------------------------------------------------
# sync
# ---------------------------------------------------------------


class EvrenClient:
    """EVREN inference API senkron istemcisi.

    >>> with EvrenClient(api_key="evren_xxx") as c:
    ...     r = c.predict("owner/slug", "foto.jpg")
    ...     print(r.predictions)
    """

    def __init__(self, api_key: str, *,
                 base_url: str = _API_BASE,
                 timeout: float = 60.0,
                 verify: bool = True) -> None:
        self._verify = verify
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=_auth_headers(api_key),
            timeout=timeout,
            verify=verify,
        )
        self._vcache: dict[str, str] = {}

    def resolve(self, slug: str) -> str:
        """``owner/slug`` veya ``owner/slug:tag`` -> version UUID."""
        if slug in self._vcache:
            return self._vcache[slug]

        slug_part, _, raw_tag = slug.partition(":")
        tag = raw_tag or None

        r = self._http.get(f"/models/{slug_part}")
        if r.status_code != 200:
            _raise_for(r)
        mid = _unwrap(r.json())["id"]

        r2 = self._http.get(f"/models/{mid}/versions")
        if r2.status_code != 200:
            _raise_for(r2)

        vid = _pick_version(_resolve_versions(r2.json()), tag, slug)
        self._vcache[slug] = vid
        return vid

    def _vid(self, model: str) -> str:
        return model if _is_uuid(model) else self.resolve(model)

    def resolve_ws_params(self, model: str) -> tuple[str, str | None]:
        """WS bağlantısı için (version_id, weights_url) çözümler."""
        vid = self._vid(model)
        slug_part = model.partition(":")[0]
        if _is_uuid(slug_part):
            mid = slug_part
        else:
            r = self._http.get(f"/models/{slug_part}")
            if r.status_code != 200:
                _raise_for(r)
            mid = _unwrap(r.json())["id"]

        r2 = self._http.get(f"/models/{mid}/versions")
        if r2.status_code != 200:
            _raise_for(r2)
        versions = _resolve_versions(r2.json())
        for v in versions:
            if v.get("id") == vid:
                return vid, v.get("weights_url")
        if versions:
            return vid, versions[0].get("weights_url")
        return vid, None

    def predict(self, model: str, image: str | Path | bytes, *,
                confidence: float = 0.25, iou: float = 0.45,
                image_size: int = 640,
                classes: list[str] | None = None) -> PredictResult:
        raw, name, mime = _read_img(image)
        r = self._http.post(
            "/inference/predict",
            data=_build_form(self._vid(model), confidence, iou, image_size, classes),
            files={"file": (name, io.BytesIO(raw), mime)},
        )
        if r.status_code != 200:
            _raise_for(r)
        return _to_result(_unwrap(r.json()))

    def predict_batch(self, model: str, images: list[str | Path | bytes], *,
                      confidence: float = 0.25, iou: float = 0.45,
                      image_size: int = 640,
                      classes: list[str] | None = None) -> BatchResult:
        vid = self._vid(model)
        parts = []
        for i, img in enumerate(images):
            raw, fname, mime = _read_img(img)
            n = fname if fname != "image.jpg" else f"img_{i}.jpg"
            parts.append(("files", (n, io.BytesIO(raw), mime)))

        r = self._http.post(
            "/inference/predict/batch",
            data=_build_form(vid, confidence, iou, image_size, classes),
            files=parts, timeout=_BATCH_TIMEOUT,
        )
        if r.status_code != 200:
            _raise_for(r)

        body = _unwrap(r.json())
        items = [_to_result(x) for x in body.get("results", [])]
        return BatchResult(results=items,
                           total_ms=body.get("total_ms", 0),
                           count=body.get("count", len(items)))

    def predict_sybil(self, series_id: str, *, model_version_id: str | None = None,
                      poll_interval: float = 5.0, timeout: float = 900.0) -> SybilRiskResult:
        """Sybil (LDCT akciğer kanseri) çıkarımını başlatır ve tamamlanana kadar poll eder.

        Vision predict() ile KARIŞTIRILMAMALI — DICOM hacmi + asenkron Slurm işi
        gerektirir, bu yüzden tamamlanması dakikalar sürebilir.

        Args:
            series_id: EVREN medikal seri UUID'si.
            model_version_id: EVREN'de eğitilmiş model — verilmezse pretrained ensemble kullanılır.
            poll_interval: durum kontrolleri arası bekleme (saniye).
            timeout: toplam bekleme süresi (saniye) — aşılırsa InferenceError.
        """
        r = self._http.post(
            f"/medical/series/{series_id}/predict",
            json={"model_version_id": model_version_id},
        )
        if r.status_code not in (200, 202):
            _raise_for(r)
        job = _unwrap(r.json())
        result_id = job["id"]

        start = time.monotonic()
        while True:
            r2 = self._http.get(f"/medical/series/{series_id}/predict/{result_id}")
            if r2.status_code != 200:
                _raise_for(r2)
            result = _to_sybil_result(_unwrap(r2.json()))
            if result.is_done:
                return result
            if time.monotonic() - start > timeout:
                raise InferenceError(
                    f"Sybil çıkarımı {timeout:.0f}s içinde tamamlanmadı (durum: {result.status})", 0,
                )
            time.sleep(poll_interval)

    def model_classes(self, model: str) -> ModelClasses:
        vid = self._vid(model)
        r = self._http.get(f"/inference/model-classes/{vid}")
        if r.status_code != 200:
            _raise_for(r)
        b = _unwrap(r.json())
        return ModelClasses(
            model_version_id=b.get("model_version_id", vid),
            classes=[ClassInfo(name=c["name"], color=c["color"])
                     for c in b.get("classes", [])],
            model_name=b.get("model_name"),
            architecture=b.get("architecture"),
            total=b.get("total", 0),
            imgsz=b.get("imgsz", 640),
        )

    def warmup(self, models: list[str]) -> JSON:
        ids = [self._vid(m) for m in models]
        r = self._http.post("/inference/warmup", json=ids,
                            timeout=_BATCH_TIMEOUT)
        if r.status_code != 200:
            _raise_for(r)
        status: JSON = _unwrap(r.json())
        return status

    def list_models(self, limit: int = 50) -> list[ModelInfo]:
        r = self._http.get("/models",
                           params={"limit": limit, "include_public": "true"})
        if r.status_code != 200:
            _raise_for(r)
        body = _unwrap(r.json())
        rows = body if isinstance(body, list) else body.get("items", [])
        return [_to_model_info(m) for m in rows]

    def list_versions(self, model_id: str) -> list[ModelVersion]:
        r = self._http.get(f"/models/{model_id}/versions")
        if r.status_code != 200:
            _raise_for(r)
        return [_to_model_ver(v) for v in _resolve_versions(r.json())]

    def download_model(self, model: str, output: str | Path, *,
                       fmt: str = "onnx") -> Path:
        """Model agirliklarini indir.

        Args:
            model: slug veya version UUID.
            output: cikis dosya yolu.
            fmt: "pytorch", "onnx", "tensorrt", "tflite" veya kisa: "pt", "trt".
        """
        vid = self._vid(model)
        r = self._http.get(f"/models/versions/{vid}/download",
                           params={"format": _normalize_fmt(fmt)},
                           timeout=_BATCH_TIMEOUT)
        if r.status_code != 200:
            _raise_for(r)
        url, dest = _download_target(_unwrap(r.json()), output, vid, fmt)
        # anahtar yalnızca kendi API host'umuza gider; imzalı bağlantı tek başına yeterli
        if _same_host(url, self._http.base_url):
            _stream_to_file(self._http, url, dest)
        else:
            with httpx.Client(verify=self._verify) as ext:
                _stream_to_file(ext, url, dest)
        return dest

    def benchmark(self, model: str, image: str | Path | bytes, *,
                  rounds: int = 20, confidence: float = 0.25,
                  iou: float = 0.45, image_size: int = 640,
                  warmup_rounds: int = 3) -> BenchmarkResult:
        """Model performansini olc."""
        raw, name, mime = _read_img(image)
        vid = self._vid(model)
        form = _build_form(vid, confidence, iou, image_size, None)

        for _ in range(warmup_rounds):
            r = self._http.post("/inference/predict", data=form,
                                files={"file": (name, io.BytesIO(raw), mime)})
            if r.status_code != 200:
                _raise_for(r)

        latencies: list[float] = []
        total_preds = 0
        for _ in range(rounds):
            t0 = time.monotonic()
            r = self._http.post("/inference/predict", data=form,
                                files={"file": (name, io.BytesIO(raw), mime)})
            lat = (time.monotonic() - t0) * 1000
            if r.status_code != 200:
                _raise_for(r)
            latencies.append(lat)
            body = _unwrap(r.json())
            total_preds += len(body.get("predictions", []))

        latencies.sort()
        avg = sum(latencies) / len(latencies)
        p95_idx = int(len(latencies) * 0.95)
        return BenchmarkResult(
            model=model, rounds=rounds, avg_ms=avg,
            min_ms=latencies[0], max_ms=latencies[-1],
            p95_ms=latencies[min(p95_idx, len(latencies) - 1)],
            throughput_fps=1000.0 / avg if avg > 0 else 0,
            total_predictions=total_preds,
        )

    def upload_to_dataset(self, dataset_id: str,
                          image: str | Path | bytes) -> JSON:
        """Görseli veri setine yükler (presign -> upload -> confirm)."""
        import hashlib
        raw, name, mime = _read_img(image)
        sha = hashlib.sha256(raw).hexdigest()

        r1 = self._http.post(
            f"/datasets/{dataset_id}/upload/presign",
            json={"files": [{"filename": name, "sha256_hash": sha,
                              "file_size": len(raw)}]},
        )
        if r1.status_code != 200:
            _raise_for(r1)
        body = _unwrap(r1.json())
        presigned = body.get("presigned_urls", body.get("items", []))
        dupes = body.get("duplicates", [])

        if not presigned:
            return {"ok": True, "duplicate": True, "duplicates": dupes,
                    "detail": "Görsel zaten bu veri setinde mevcut"}

        item = presigned[0]
        obj_key = item.get("object_key", item.get("key", ""))
        upload_path = item.get("upload_url",
                               f"/datasets/{dataset_id}/upload/put?key={obj_key}")

        r2 = self._http.put(
            upload_path, content=raw,
            headers={"content-length": str(len(raw)),
                     "content-type": mime},
            timeout=60.0,
        )
        if r2.status_code != 200:
            _raise_for(r2)

        confirm_item: JSON = {
            "object_key": obj_key,
            "sha256_hash": sha,
            "filename": name,
            "file_size": len(raw),
        }

        r3 = self._http.post(
            f"/datasets/{dataset_id}/upload/confirm",
            json={"items": [confirm_item], "skip_dedup": False},
        )
        if r3.status_code not in (200, 201):
            _raise_for(r3)
        confirmed: JSON = _unwrap(r3.json())
        return confirmed

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> EvrenClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------
# async
# ---------------------------------------------------------------


class AsyncEvrenClient:
    """EVREN inference API asenkron istemcisi.

    >>> async with AsyncEvrenClient(api_key="evren_xxx") as c:
    ...     r = await c.predict("owner/slug", "foto.jpg")
    """

    def __init__(self, api_key: str, *,
                 base_url: str = _API_BASE,
                 timeout: float = 60.0,
                 verify: bool = True) -> None:
        self._verify = verify
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=_auth_headers(api_key),
            timeout=timeout,
            verify=verify,
        )
        self._vcache: dict[str, str] = {}

    async def resolve(self, slug: str) -> str:
        if slug in self._vcache:
            return self._vcache[slug]

        slug_part, _, raw_tag = slug.partition(":")
        tag = raw_tag or None

        r = await self._http.get(f"/models/{slug_part}")
        if r.status_code != 200:
            _raise_for(r)
        mid = _unwrap(r.json())["id"]

        r2 = await self._http.get(f"/models/{mid}/versions")
        if r2.status_code != 200:
            _raise_for(r2)

        vid = _pick_version(_resolve_versions(r2.json()), tag, slug)
        self._vcache[slug] = vid
        return vid

    async def _vid(self, model: str) -> str:
        return model if _is_uuid(model) else await self.resolve(model)

    async def resolve_ws_params(self, model: str) -> tuple[str, str | None]:
        vid = await self._vid(model)
        slug_part = model.partition(":")[0]
        if _is_uuid(slug_part):
            mid = slug_part
        else:
            r = await self._http.get(f"/models/{slug_part}")
            if r.status_code != 200:
                _raise_for(r)
            mid = _unwrap(r.json())["id"]

        r2 = await self._http.get(f"/models/{mid}/versions")
        if r2.status_code != 200:
            _raise_for(r2)
        versions = _resolve_versions(r2.json())
        for v in versions:
            if v.get("id") == vid:
                return vid, v.get("weights_url")
        if versions:
            return vid, versions[0].get("weights_url")
        return vid, None

    async def predict(self, model: str, image: str | Path | bytes, *,
                      confidence: float = 0.25, iou: float = 0.45,
                      image_size: int = 640,
                      classes: list[str] | None = None) -> PredictResult:
        raw, name, mime = _read_img(image)
        r = await self._http.post(
            "/inference/predict",
            data=_build_form(await self._vid(model), confidence, iou, image_size, classes),
            files={"file": (name, io.BytesIO(raw), mime)},
        )
        if r.status_code != 200:
            _raise_for(r)
        return _to_result(_unwrap(r.json()))

    async def predict_batch(self, model: str, images: list[str | Path | bytes], *,
                            confidence: float = 0.25, iou: float = 0.45,
                            image_size: int = 640,
                            classes: list[str] | None = None) -> BatchResult:
        vid = await self._vid(model)
        parts = []
        for i, img in enumerate(images):
            raw, fname, mime = _read_img(img)
            n = fname if fname != "image.jpg" else f"img_{i}.jpg"
            parts.append(("files", (n, io.BytesIO(raw), mime)))

        r = await self._http.post(
            "/inference/predict/batch",
            data=_build_form(vid, confidence, iou, image_size, classes),
            files=parts, timeout=_BATCH_TIMEOUT,
        )
        if r.status_code != 200:
            _raise_for(r)
        body = _unwrap(r.json())
        items = [_to_result(x) for x in body.get("results", [])]
        return BatchResult(results=items,
                           total_ms=body.get("total_ms", 0),
                           count=body.get("count", len(items)))

    async def predict_sybil(self, series_id: str, *, model_version_id: str | None = None,
                            poll_interval: float = 5.0, timeout: float = 900.0) -> SybilRiskResult:
        """Sybil (LDCT akciğer kanseri) çıkarımını başlatır ve tamamlanana kadar poll eder (async)."""
        r = await self._http.post(
            f"/medical/series/{series_id}/predict",
            json={"model_version_id": model_version_id},
        )
        if r.status_code not in (200, 202):
            _raise_for(r)
        job = _unwrap(r.json())
        result_id = job["id"]

        start = time.monotonic()
        while True:
            r2 = await self._http.get(f"/medical/series/{series_id}/predict/{result_id}")
            if r2.status_code != 200:
                _raise_for(r2)
            result = _to_sybil_result(_unwrap(r2.json()))
            if result.is_done:
                return result
            if time.monotonic() - start > timeout:
                raise InferenceError(
                    f"Sybil çıkarımı {timeout:.0f}s içinde tamamlanmadı (durum: {result.status})", 0,
                )
            await asyncio.sleep(poll_interval)

    async def model_classes(self, model: str) -> ModelClasses:
        vid = await self._vid(model)
        r = await self._http.get(f"/inference/model-classes/{vid}")
        if r.status_code != 200:
            _raise_for(r)
        b = _unwrap(r.json())
        return ModelClasses(
            model_version_id=b.get("model_version_id", vid),
            classes=[ClassInfo(name=c["name"], color=c["color"])
                     for c in b.get("classes", [])],
            model_name=b.get("model_name"),
            architecture=b.get("architecture"),
            total=b.get("total", 0),
            imgsz=b.get("imgsz", 640),
        )

    async def warmup(self, models: list[str]) -> JSON:
        ids = [await self._vid(m) for m in models]
        r = await self._http.post("/inference/warmup", json=ids,
                                  timeout=_BATCH_TIMEOUT)
        if r.status_code != 200:
            _raise_for(r)
        status: JSON = _unwrap(r.json())
        return status

    async def list_models(self, limit: int = 50) -> list[ModelInfo]:
        r = await self._http.get("/models",
                                 params={"limit": limit, "include_public": "true"})
        if r.status_code != 200:
            _raise_for(r)
        body = _unwrap(r.json())
        rows = body if isinstance(body, list) else body.get("items", [])
        return [_to_model_info(m) for m in rows]

    async def list_versions(self, model_id: str) -> list[ModelVersion]:
        r = await self._http.get(f"/models/{model_id}/versions")
        if r.status_code != 200:
            _raise_for(r)
        return [_to_model_ver(v) for v in _resolve_versions(r.json())]

    async def download_model(self, model: str, output: str | Path, *,
                             fmt: str = "onnx") -> Path:
        vid = await self._vid(model)
        r = await self._http.get(f"/models/versions/{vid}/download",
                                 params={"format": _normalize_fmt(fmt)},
                                 timeout=_BATCH_TIMEOUT)
        if r.status_code != 200:
            _raise_for(r)
        url, dest = _download_target(_unwrap(r.json()), output, vid, fmt)
        if _same_host(url, self._http.base_url):
            await _astream_to_file(self._http, url, dest)
        else:
            async with httpx.AsyncClient(verify=self._verify) as ext:
                await _astream_to_file(ext, url, dest)
        return dest

    async def benchmark(self, model: str, image: str | Path | bytes, *,
                        rounds: int = 20, confidence: float = 0.25,
                        iou: float = 0.45, image_size: int = 640,
                        warmup_rounds: int = 3) -> BenchmarkResult:
        raw, name, mime = _read_img(image)
        vid = await self._vid(model)
        form = _build_form(vid, confidence, iou, image_size, None)

        for _ in range(warmup_rounds):
            r = await self._http.post("/inference/predict", data=form,
                                      files={"file": (name, io.BytesIO(raw), mime)})
            if r.status_code != 200:
                _raise_for(r)

        latencies: list[float] = []
        total_preds = 0
        for _ in range(rounds):
            t0 = time.monotonic()
            r = await self._http.post("/inference/predict", data=form,
                                      files={"file": (name, io.BytesIO(raw), mime)})
            lat = (time.monotonic() - t0) * 1000
            if r.status_code != 200:
                _raise_for(r)
            latencies.append(lat)
            body = _unwrap(r.json())
            total_preds += len(body.get("predictions", []))

        latencies.sort()
        avg = sum(latencies) / len(latencies)
        p95_idx = int(len(latencies) * 0.95)
        return BenchmarkResult(
            model=model, rounds=rounds, avg_ms=avg,
            min_ms=latencies[0], max_ms=latencies[-1],
            p95_ms=latencies[min(p95_idx, len(latencies) - 1)],
            throughput_fps=1000.0 / avg if avg > 0 else 0,
            total_predictions=total_preds,
        )

    async def upload_to_dataset(self, dataset_id: str,
                                image: str | Path | bytes) -> JSON:
        """Görseli veri setine yükler (presign -> upload -> confirm)."""
        import hashlib
        raw, name, mime = _read_img(image)
        sha = hashlib.sha256(raw).hexdigest()

        r1 = await self._http.post(
            f"/datasets/{dataset_id}/upload/presign",
            json={"files": [{"filename": name, "sha256_hash": sha,
                              "file_size": len(raw)}]},
        )
        if r1.status_code != 200:
            _raise_for(r1)
        body = _unwrap(r1.json())
        presigned = body.get("presigned_urls", body.get("items", []))
        dupes = body.get("duplicates", [])

        if not presigned:
            return {"ok": True, "duplicate": True, "duplicates": dupes,
                    "detail": "Görsel zaten bu veri setinde mevcut"}

        item = presigned[0]
        obj_key = item.get("object_key", item.get("key", ""))
        upload_path = item.get("upload_url",
                               f"/datasets/{dataset_id}/upload/put?key={obj_key}")

        r2 = await self._http.put(
            upload_path, content=raw,
            headers={"content-length": str(len(raw)),
                     "content-type": mime},
            timeout=60.0,
        )
        if r2.status_code != 200:
            _raise_for(r2)

        confirm_item: JSON = {
            "object_key": obj_key,
            "sha256_hash": sha,
            "filename": name,
            "file_size": len(raw),
        }

        r3 = await self._http.post(
            f"/datasets/{dataset_id}/upload/confirm",
            json={"items": [confirm_item], "skip_dedup": False},
        )
        if r3.status_code not in (200, 201):
            _raise_for(r3)
        confirmed: JSON = _unwrap(r3.json())
        return confirmed

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncEvrenClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
