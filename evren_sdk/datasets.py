"""Metin ve ses kayıt setlerini etiketleriyle yükleme — `load_dataset`, akışlı okuma, indirme.

Dosyalar platformda üretilen etiketli Parquet dışa aktarımıdır (ses sütunu HF `Audio` düzeninde).
Aynı içerik bir kez hazırlanır ve yerelde önbelleklenir; ikinci çağrı dosya indirmez.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .exceptions import EvrenError

if TYPE_CHECKING:
    import pyarrow as pa

_CACHE = Path(os.environ.get("EVREN_CACHE", Path.home() / ".cache" / "evren")) / "datasets"


@dataclass
class DatasetFile:
    split: str
    name: str
    rows: int
    bytes: int
    path: str
    sha256: str | None = None


@dataclass
class DatasetExport:
    id: str
    dataset_id: str
    status: str
    format: str
    rows: int | None
    files: list[DatasetFile] = field(default_factory=list)

    @property
    def splits(self) -> list[str]:
        return sorted({f.split for f in self.files})


def _pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - ek bağımlılık
        raise EvrenError("Kayıt setleri için pyarrow gerekli: pip install 'evren-sdk[datasets]'") from exc
    return pq


def resolve_dataset(http: httpx.Client, ref: str) -> str:
    from .client import _is_uuid, _raise_for, _unwrap

    if _is_uuid(ref):
        return ref
    r = http.get(f"/datasets/{ref.strip('/')}")
    if r.status_code != 200:
        _raise_for(r)
    return str(_unwrap(r.json())["id"])


def _parse(dataset_id: str, d: dict[str, Any]) -> DatasetExport:
    files = [DatasetFile(split=f["split"], name=f["name"], rows=f["rows"], bytes=f["bytes"], path=f["path"],
                         sha256=f.get("sha256")) for f in d.get("files") or []]
    return DatasetExport(id=d["id"], dataset_id=dataset_id, status=d["status"], format=d["format"],
                         rows=d.get("rows"), files=files)


def export(http: httpx.Client, ref: str, *, format: str = "parquet", version: str | None = None,
           wait: bool = True, timeout: float = 3600.0, poll: float = 3.0) -> DatasetExport:
    """Etiketli dışa aktarımı ister; aynı içerik hazırsa ücretsiz ve anında döner."""
    from .client import _raise_for, _unwrap

    ds = resolve_dataset(http, ref)
    r = http.post(f"/datasets/{ds}/records/exports", json={"format": format, "version": version})
    if r.status_code != 200:
        _raise_for(r)
    exp = _parse(ds, _unwrap(r.json()))
    deadline = time.monotonic() + timeout
    while wait and exp.status in ("PENDING", "RUNNING"):
        if time.monotonic() > deadline:
            raise EvrenError(f"Dışa aktarım {timeout:.0f} sn içinde bitmedi: {exp.id}")
        time.sleep(poll)
        r = http.get(f"/datasets/{ds}/records/exports/{exp.id}")
        if r.status_code != 200:
            _raise_for(r)
        exp = _parse(ds, _unwrap(r.json()))
    if exp.status == "FAILED":
        raise EvrenError(f"Dışa aktarım başarısız: {exp.id}")
    return exp


def _pick(exp: DatasetExport, split: str | None) -> list[DatasetFile]:
    files = [f for f in exp.files if split is None or f.split == split]
    if split is not None and not files:
        raise EvrenError(f"Bölüm yok: {split!r}; mevcut bölümler: {', '.join(exp.splits)}")
    return files


def download(http: httpx.Client, exp: DatasetExport, dest: Path, split: str | None = None) -> list[Path]:
    out = []
    dest.mkdir(parents=True, exist_ok=True)
    for f in _pick(exp, split):
        target = dest / f.name
        if not (target.exists() and target.stat().st_size == f.bytes):
            tmp = target.with_suffix(target.suffix + ".part")
            with http.stream("GET", f.path, timeout=None) as r, open(tmp, "wb") as fh:
                if r.status_code != 200:
                    r.read()
                    from .client import _raise_for

                    _raise_for(r)
                fh.writelines(r.iter_bytes(1 << 20))
            if f.sha256 and sha256_of(tmp) != f.sha256:
                tmp.unlink(missing_ok=True)
                raise EvrenError(f"{f.name} indirildi ama bütünlük özeti tutmadı; tekrar deneyin")
            tmp.replace(target)
        out.append(target)
    return out


def load(http: httpx.Client, ref: str, *, split: str | None = None, version: str | None = None,
         columns: list[str] | None = None, cache_dir: str | Path | None = None) -> pa.Table | dict[str, pa.Table]:
    pq = _pyarrow()
    import pyarrow as pa

    exp = export(http, ref, version=version)
    root = Path(cache_dir) if cache_dir else _CACHE
    paths = download(http, exp, root / exp.dataset_id / exp.id, split)
    by_split: dict[str, list[Path]] = {}
    for f, p in zip(_pick(exp, split), paths):
        by_split.setdefault(f.split, []).append(p)
    tables = {s: pa.concat_tables([pq.read_table(p, columns=columns) for p in ps]) for s, ps in by_split.items()}
    return tables[split] if split is not None else tables


def iterate(http: httpx.Client, ref: str, *, split: str | None = None, version: str | None = None,
            columns: list[str] | None = None, batch_size: int = 1024) -> Iterator[dict[str, Any]]:
    """Akışlı okuma: dosyalar indirilmez, Parquet satır grupları HTTP aralığıyla okunur."""
    pq = _pyarrow()
    exp = export(http, ref, version=version)
    for f in _pick(exp, split):
        with io.BufferedReader(_RangeFile(http, f.path, f.bytes), buffer_size=4 << 20) as raw:
            pf = pq.ParquetFile(raw)
            for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                yield from batch.to_pylist()


class _RangeFile(io.RawIOBase):
    """Parquet okuyucusuna aranabilir dosya gibi görünen HTTP aralık okuyucusu."""

    def __init__(self, http: httpx.Client, path: str, size: int) -> None:
        self._http, self._path, self._size, self._pos = http, path, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, min(self._size, base + offset))
        return self._pos

    def readinto(self, b: Any) -> int:
        if self._pos >= self._size:
            return 0
        end = min(self._size, self._pos + len(b)) - 1
        r = self._http.get(self._path, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=None)
        if r.status_code not in (200, 206):
            from .client import _raise_for

            _raise_for(r)
        data = r.content
        n = len(data)
        b[:n] = data
        self._pos += n
        return n


def to_hf(tables: pa.Table | dict[str, pa.Table]) -> Any:
    """pyarrow tablosunu HF `datasets` nesnesine çevirir (`pip install datasets`)."""
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - ek bağımlılık
        raise EvrenError("HF biçimi için: pip install datasets") from exc
    if isinstance(tables, dict):
        return datasets.DatasetDict({k: datasets.Dataset(v) for k, v in tables.items()})
    return datasets.Dataset(tables)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
