from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Prediction:
    class_name: str = ""
    confidence: float = 0.0
    bbox: list[float] = field(default_factory=list)
    color: str | None = None
    keypoints: list[list[float]] | None = None
    mask: list[list[float]] | None = None
    obb: dict | None = None
    task: str | None = None
    probs: list[dict] | None = None

    def __repr__(self) -> str:
        return f"<Prediction {self.class_name} conf={self.confidence:.3f}>"

    def to_dict(self) -> dict:
        d: dict = {"class_name": self.class_name, "confidence": self.confidence}
        if self.bbox:
            d["bbox"] = self.bbox
        if self.obb:
            d["obb"] = self.obb
        if self.keypoints:
            d["keypoints"] = self.keypoints
        if self.mask:
            d["mask"] = self.mask
        if self.probs:
            d["probs"] = self.probs
        return d


@dataclass(slots=True)
class PredictResult:
    predictions: list[Prediction] = field(default_factory=list)
    inference_ms: float = 0.0
    image_width: int | None = None
    image_height: int | None = None
    model_version_id: str | None = None

    @property
    def count(self) -> int:
        return len(self.predictions)

    def filter(self, *,
               min_confidence: float = 0.0,
               max_confidence: float = 1.0,
               classes: list[str] | None = None) -> PredictResult:
        """Tahminleri filtrele — yeni PredictResult dondurur."""
        preds = self.predictions
        if min_confidence > 0:
            preds = [p for p in preds if p.confidence >= min_confidence]
        if max_confidence < 1.0:
            preds = [p for p in preds if p.confidence <= max_confidence]
        if classes:
            s = set(classes)
            preds = [p for p in preds if p.class_name in s]
        return PredictResult(preds, self.inference_ms, self.image_width, self.image_height, self.model_version_id)

    def to_yolo(self, class_map: dict[str, int] | list[str] | None = None) -> str:
        """YOLO annotation formatina cevir (normalized xywh).

        Args:
            class_map: sinif_adi->index dict veya sinif listesi.
                       None ise sinif adlari alfabetik siralanir.
        Returns:
            Her satir: ``class_id cx cy w h``
        """
        if class_map is None:
            names = sorted({p.class_name for p in self.predictions})
            class_map = {n: i for i, n in enumerate(names)}
        elif isinstance(class_map, list):
            class_map = {n: i for i, n in enumerate(class_map)}

        iw = self.image_width or 1
        ih = self.image_height or 1
        lines: list[str] = []

        for p in self.predictions:
            if not p.bbox or len(p.bbox) < 4:
                continue
            cid = class_map.get(p.class_name, 0)
            x1, y1, x2, y2 = p.bbox[:4]
            cx = ((x1 + x2) / 2) / iw
            cy = ((y1 + y2) / 2) / ih
            w = (x2 - x1) / iw
            h = (y2 - y1) / ih
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        return "\n".join(lines)

    def to_coco(self, image_id: int = 0, class_map: dict[str, int] | list[str] | None = None) -> list[dict]:
        """COCO annotation formatinda dict listesi dondurur."""
        if class_map is None:
            names = sorted({p.class_name for p in self.predictions})
            class_map = {n: i for i, n in enumerate(names)}
        elif isinstance(class_map, list):
            class_map = {n: i for i, n in enumerate(class_map)}

        anns = []
        for i, p in enumerate(self.predictions):
            if not p.bbox or len(p.bbox) < 4:
                continue
            x1, y1, x2, y2 = p.bbox[:4]
            anns.append({
                "id": i,
                "image_id": image_id,
                "category_id": class_map.get(p.class_name, 0),
                "category_name": p.class_name,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "score": p.confidence,
                "iscrowd": 0,
            })
        return anns

    def to_csv(self, sep: str = ",") -> str:
        """CSV formatinda dondurur."""
        header = sep.join(["class_name", "confidence", "x1", "y1", "x2", "y2"])
        rows = [header]
        for p in self.predictions:
            b = p.bbox if p.bbox and len(p.bbox) >= 4 else [0, 0, 0, 0]
            rows.append(sep.join([
                p.class_name, f"{p.confidence:.4f}",
                f"{b[0]:.1f}", f"{b[1]:.1f}", f"{b[2]:.1f}", f"{b[3]:.1f}",
            ]))
        return "\n".join(rows)

    def save(self, path: str | Path, fmt: str = "auto") -> None:
        """Sonuclari dosyaya kaydet.

        Args:
            path: cikis dosyasi.
            fmt: "yolo", "coco", "csv", "json" veya "auto" (uzantidan anla).
        """
        p = Path(path)
        if fmt == "auto":
            ext = p.suffix.lower()
            fmt = {".txt": "yolo", ".json": "json",
                   ".csv": "csv"}.get(ext, "json")

        if fmt == "yolo":
            p.write_text(self.to_yolo(), encoding="utf-8")
        elif fmt == "csv":
            p.write_text(self.to_csv(), encoding="utf-8")
        elif fmt == "coco":
            p.write_text(json.dumps(self.to_coco(), ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            data = {
                "predictions": [pr.to_dict() for pr in self.predictions],
                "inference_ms": self.inference_ms,
                "image_width": self.image_width,
                "image_height": self.image_height,
                "count": self.count,
            }
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(slots=True)
class BatchResult:
    results: list[PredictResult] = field(default_factory=list)
    total_ms: float = 0.0
    count: int = 0

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return self.count


@dataclass(slots=True, frozen=True)
class ClassInfo:
    name: str
    color: str


@dataclass(slots=True)
class ModelClasses:
    model_version_id: str
    classes: list[ClassInfo] = field(default_factory=list)
    model_name: str | None = None
    architecture: str | None = None
    total: int = 0
    imgsz: int = 640

    def __contains__(self, name: str) -> bool:
        return any(c.name == name for c in self.classes)

    def names(self) -> list[str]:
        return [c.name for c in self.classes]


@dataclass(slots=True)
class ModelInfo:
    id: str
    name: str
    slug: str
    architecture: str | None = None
    owner_username: str | None = None

    @property
    def full_slug(self) -> str:
        if self.owner_username:
            return f"{self.owner_username}/{self.slug}"
        return self.slug


@dataclass(slots=True)
class ModelVersion:
    id: str
    version_tag: str
    weights_url: str | None = None
    framework: str = "pytorch"
    metrics: dict = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkResult:
    model: str
    rounds: int
    avg_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    throughput_fps: float
    total_predictions: int

    def __repr__(self) -> str:
        return (
            f"<Benchmark {self.model} avg={self.avg_ms:.0f}ms "
            f"p95={self.p95_ms:.0f}ms {self.throughput_fps:.1f}fps>"
        )
