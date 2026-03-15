# evren-sdk

**EVREN MLOps Platform — Official Python SDK**

[![PyPI](https://img.shields.io/pypi/v/evren-sdk?color=10b981)](https://pypi.org/project/evren-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/evren-sdk)](https://pypi.org/project/evren-sdk/)
[![License](https://img.shields.io/github/license/speker/evren-sdk)](https://github.com/speker/evren-sdk/blob/main/LICENSE)

EVREN platformunda egitilmis bilgisayarli goru modellerine Python'dan cikarim yapmanizi saglayan resmi SDK.

**Nesne tespiti** · **Siniflandirma** · **Segmentasyon** · **OBB** · **Keypoint** · **Edge Inference**

## Kurulum

```bash
pip install evren-sdk            # temel SDK
pip install evren-sdk[edge]      # + OpenCV (kamera/video)
```

## Hizli Baslangic

```python
from evren_sdk import EvrenClient

client = EvrenClient(api_key="evren_xxxxx")
result = client.predict("kullanici/model-adi", "foto.jpg", confidence=0.3)

for det in result.predictions:
    print(f"{det.class_name}: {det.confidence:.0%}  bbox={det.bbox}")
```

## Ozellikler

- **Tekil & toplu cikarim** — `predict()` ve `predict_batch()` ile GPU uzerinde cikarim
- **Slug cozumleme** — `owner/model-name` veya `owner/model:v2.0` ile versiyon secimi
- **Sonuc isleme** — `filter()`, `to_yolo()`, `to_coco()`, `to_csv()`, `save()`
- **Model bilgileri** — `list_models()`, `list_versions()`, `model_classes()`
- **Warmup** — GPU'ya on-yukleme ile cold-start eliminasyonu
- **Benchmark** — Gecikme, throughput, p95 olcumu
- **Model indirme** — ONNX veya PyTorch agirliklarini indir
- **Veri seti yukleme** — SDK uzerinden veri setine gorsel yukle
- **Edge modu** — GPU'suz cihazlarda gercek zamanli cikarim (webcam, RTSP, video)
- **Async** — `AsyncEvrenClient` ile ayni API, `async/await` destegi

## Tekil Cikarim

```python
result = client.predict(
    model="kullanici/model-adi",
    image="resim.jpg",
    confidence=0.25,
    iou=0.45,
    image_size=640,
)
```

## Toplu Cikarim

```python
batch = client.predict_batch("kullanici/model", ["a.jpg", "b.jpg", "c.jpg"])
for r in batch:
    print(f"{r.count} tespit, {r.inference_ms:.0f} ms")
```

## Sonuc Export

```python
result.filter(min_confidence=0.5, classes=["araba"])
result.to_yolo()       # YOLO txt
result.to_coco()       # COCO dict
result.to_csv()        # CSV string
result.save("out.json")
```

## Edge Modu

GPU olmayan cihazlarda gercek zamanli cikarim. Cikarim bulut GPU'larinda calisir.

```bash
pip install evren-sdk[edge]
```

```python
from evren_sdk import EvrenCamera

cam = EvrenCamera("evren_...", "kullanici/model", confidence=0.3)
cam.run(0)   # webcam, ESC ile kapat
```

## Async

```python
from evren_sdk import AsyncEvrenClient

async with AsyncEvrenClient(api_key="evren_xxxxx") as client:
    result = await client.predict("kullanici/model", "foto.jpg")
```

## Hata Yonetimi

| Exception | HTTP | Aciklama |
|---|---|---|
| `AuthenticationError` | 401, 403 | Gecersiz anahtar |
| `InsufficientCreditsError` | 402 | Kredi yetersiz |
| `NotFoundError` | 404 | Model bulunamadi |
| `ValidationError` | 422 | Hatali parametre |
| `RateLimitError` | 429 | Istek limiti |
| `InferenceError` | 502, 503 | GPU sunucu hatasi |

## Dokumantasyon

Detayli dokumantasyon, diyagramlar ve calistirilabilir ornekler icin
[GitHub deposuna](https://github.com/speker/evren-sdk) bakin.

## Lisans

[Apache License 2.0](https://github.com/speker/evren-sdk/blob/main/LICENSE)
