from __future__ import annotations

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
)

_API_BASE = "https://api.ssyz.org.tr/api/v1"
_BATCH_TIMEOUT = 120.0


def _auth_headers(key: str) -> dict[str, str]:
    if key.startswith("evren_"):
        return {"X-API-Key": key}
    return {"Authorization": f"Bearer {key}"}


def _extract_msg(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                return err.get("message") or err.get("detail") or resp.text[:300]
            return body.get("detail") or body.get("message") or resp.text[:300]
    except Exception:
        pass
    return resp.text[:300]


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
        wait = int(resp.headers.get("Retry-After", "5"))
        raise RateLimitError(msg, wait)
    if code >= 500:
        raise InferenceError(msg, code)
    if 400 <= code < 500:
        raise ValidationError(msg, code)
    resp.raise_for_status()


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


def _to_prediction(raw: dict) -> Prediction:
    return Prediction(
        class_name=raw.get("class_name", ""),
        confidence=raw.get("confidence", 0.0),
        bbox=raw.get("bbox", []),
        color=raw.get("color"),
        keypoints=raw.get("keypoints"),
        mask=raw.get("mask"),
        obb=raw.get("obb"),
        task=raw.get("task"),
        probs=raw.get("probs"),
    )


def _to_result(body: dict) -> PredictResult:
    return PredictResult(
        predictions=[_to_prediction(p) for p in body.get("predictions", [])],
        inference_ms=body.get("inference_ms", 0),
        image_width=body.get("image_width"),
        image_height=body.get("image_height"),
        model_version_id=body.get("model_version_id"),
    )


def _to_model_info(m: dict) -> ModelInfo:
    return ModelInfo(
        id=m["id"], name=m["name"], slug=m.get("slug", ""),
        architecture=m.get("architecture"),
        owner_username=m.get("owner_username"),
    )


def _to_model_ver(v: dict) -> ModelVersion:
    return ModelVersion(
        id=v["id"], version_tag=v.get("version_tag", ""),
        weights_url=v.get("weights_url"),
        framework=v.get("framework", "pytorch"),
        metrics=v.get("metrics", {}),
    )


def _resolve_versions(body: Any) -> list[dict]:
    data = _unwrap(body)
    if isinstance(data, list):
        return data
    return data.get("items", data.get("data", []))


def _pick_version(versions: list[dict], tag: str | None, slug: str) -> str:
    if not versions:
        raise NotFoundError(f"'{slug}' icin versiyon yok", 404)

    if tag is None:
        return versions[0]["id"]

    for v in versions:
        if v.get("version_tag") == tag:
            return v["id"]
    raise NotFoundError(f"'{slug}' icin '{tag}' versiyonu yok", 404)


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

        slug_part, _, tag = slug.partition(":")
        tag = tag or None

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

    def warmup(self, models: list[str]) -> dict:
        ids = [self._vid(m) for m in models]
        r = self._http.post("/inference/warmup", json=ids,
                            timeout=_BATCH_TIMEOUT)
        if r.status_code != 200:
            _raise_for(r)
        return _unwrap(r.json())

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
            fmt: "onnx", "pt", "torchscript". Varsayilan ONNX.
        """
        vid = self._vid(model)
        r = self._http.get(f"/models/versions/{vid}/download",
                           params={"format": fmt},
                           timeout=_BATCH_TIMEOUT)
        if r.status_code != 200:
            _raise_for(r)
        dest = Path(output)
        if dest.suffix == "":
            dest.mkdir(parents=True, exist_ok=True)
            cd = r.headers.get("content-disposition", "")
            fname = f"{vid}.{fmt}"
            if "filename=" in cd:
                fname = cd.split("filename=")[-1].strip('" ')
            dest = dest / fname
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
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
                          image: str | Path | bytes) -> dict:
        """Gorseli veri setine yukler (presign -> upload -> confirm)."""
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
                    "detail": "Gorsel zaten bu veri setinde mevcut"}

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

        confirm_item: dict = {
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
        return _unwrap(r3.json())

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> EvrenClient:
        return self

    def __exit__(self, *exc: Any) -> None:
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

        slug_part, _, tag = slug.partition(":")
        tag = tag or None

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

    async def warmup(self, models: list[str]) -> dict:
        ids = [await self._vid(m) for m in models]
        r = await self._http.post("/inference/warmup", json=ids,
                                  timeout=_BATCH_TIMEOUT)
        if r.status_code != 200:
            _raise_for(r)
        return _unwrap(r.json())

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
                                 params={"format": fmt},
                                 timeout=_BATCH_TIMEOUT)
        if r.status_code != 200:
            _raise_for(r)
        dest = Path(output)
        if dest.suffix == "":
            dest.mkdir(parents=True, exist_ok=True)
            cd = r.headers.get("content-disposition", "")
            fname = f"{vid}.{fmt}"
            if "filename=" in cd:
                fname = cd.split("filename=")[-1].strip('" ')
            dest = dest / fname
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
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
                                image: str | Path | bytes) -> dict:
        """Gorseli veri setine yukler (presign -> upload -> confirm)."""
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
                    "detail": "Gorsel zaten bu veri setinde mevcut"}

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

        confirm_item: dict = {
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
        return _unwrap(r3.json())

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncEvrenClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
