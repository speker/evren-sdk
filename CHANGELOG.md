# Changelog

## [0.11.0] - 2026-10-18

### Added
- `speak()`: metinden sese (`voxcpm2`) — `/v1/audio/speech`. `SpeechResult.audio` istenen biçimde ikili sestir (`mp3` varsayılan, `wav`, `flac`, `opus`, `pcm`); `save(yol)` ile dosyaya yazılır. `instructions=` ile ses düz metinle tarif edilir, `speed=` 0,25-4,0. 4.096 karaktere kadar metni geçit cümle sınırlarından bölüp tek dosya döndürür
- `AsyncEvrenLLMClient.speak()`

## [0.10.0] - 2026-10-10

### Added
- `decide()` / `decide_batch()`: tipli karar modelleri (`laya`, `clef`) — `/v1/systemone`. `DecisionResult` üzerinde `choice()`, `noul()`, `score()`, `probabilities()` yardımcıları
- `decide(images=[...])`: clef için görsel girdi (en fazla 4); `encode_image(kaynak, max_side=1024)` yol, bayt ya da `data:` URL'yi gömülü biçime çevirir, istenirse küçültür (`pip install evren-sdk[images]`)
- `ocr()` (`/v1/ocr`), `transcribe()` (`/v1/audio/transcriptions`; json, text, srt, verbose_json, diarized_json)
- `respond()` / `get_response()`: Responses API (akışsız, `previous_response_id` ile zincirleme); `ResponseResult.output_text`
- `models()`: `/v1/models` kayıtlarını görev (`task`), modalite ve fiyat bilgisiyle döner (`list_models()` yalnız ad döndürmeye devam eder)
- `quota()` (5 dk pencere, günlük tavan, kredi) ve `request_status()` (yürütme/ücret durumu)
- Hepsi `AsyncEvrenLLMClient` üzerinde de mevcut

### Fixed
- Günlük kota dolduğunda geçit `Retry-After` değerini gün sonuna kadar verir; istemci bunu 30 sn ile sınırlayıp iki kez daha deniyor ve hatayı ~1 dk geç bildiriyordu. Artık 30 sn'den uzun bekleme isteyen yanıtlar tekrar denenmez
- `EvrenClient`: `Retry-After` HTTP tarihi biçiminde gelirse 429 işlenirken `ValueError` oluşuyordu
- `InferenceWSClient`: bozuk JSON karesi alma iş parçacığını sessizce durduruyor, sonraki tüm sonuçlar zaman aşımına düşüyordu; artık hatalı kare boş sonuçla yer tutar
- `InferenceWSClient`: bağlantı kurulmadan çağrılan iç yollar `NoneType` hatası veriyordu; açık `RuntimeError` döner
- `EvrenCamera.record()`: WS sonucu 10 sn içinde gelmezse kayıt `queue.Empty` ile yarıda kesiliyordu; kare kutusuz yazılıp devam edilir
- `EvrenCamera`: çıkarım hataları `print()` ile stderr'e basılıyordu; artık `logging` (`evren_sdk.edge`) kullanılır

### Changed
- PyPI bağlantıları (depo, değişiklik günlüğü, hata kayıtları, belgeler) herkese açık GitHub deposuna yönlendirildi; eski bağlantılar özel sunucuya gidiyordu
- Görü istemcisi, edge ve WS modülleri `mypy --strict` ve ruff ile temizlendi; yeni opsiyonel bağımlılık grubu `images` (Pillow)

## [0.9.2] - 2026-09-25

### Changed
- Proje tanımı "EVREN Yapay Zeka Platformu" olarak güncellendi; README girişi LLM geçidini de kapsıyor (kod değişikliği yok)

## [0.9.1] - 2026-09-25

### Fixed
- `download_model()` ağırlık dosyası yerine sunucunun döndürdüğü bağlantı JSON'unu diske yazıyordu (ör. 382 baytlık `model.onnx`); artık `download_url` izlenir, dosya parça parça diske akıtılır ve sunucudaki gerçek dosya adı kullanılır
- `InferenceWSClient` / `EvrenCamera` WS modu Ultralytics dışı modellerde (RF-DETR, D-FINE ...) yanlış sınıf ve kutu döndürüyordu; akış artık `/api/v1/inference/stream` üzerinden gider, framework, fold ve sınıf adları HTTP `predict()` ile aynı çözülür
- WS pipeline sunucunun kayıpsız taşıyabileceğinden fazla kareyi yolda tutuyordu; sunucu fazlasını attığında video akışı erken bitiyor, canlı kamerada sonuçlar yanlış karelere çiziliyordu. Yoldaki kare sayısı 3 ile sınırlandı
- Sınıflandırma modellerinde `Prediction.class_name` / `confidence` boş geliyordu; artık en olası sınıftan doldurulur (`probs` korunur, WS'de de)
- WS akış hataları sessizce boş sonuç olarak dönüyordu; `predict_frame()` artık `InferenceError` fırlatır

### Changed
- **Güvenlik:** `EvrenClient`, `AsyncEvrenClient`, `InferenceWSClient` ve `EvrenCamera` için TLS sertifika doğrulaması varsayılan olarak açık (`verify=True`). Önceki varsayılan (`False`) API anahtarını ortadaki adam saldırısına açık bırakıyordu

## [0.9.0] - 2026-09-25

0.8.0 PyPI'a yayınlanmadı; içeriği bu sürüme dahildir.

### Added
- `EvrenLLMClient` / `AsyncEvrenLLMClient`: EVREN LLM geçidi istemcisi — `chat`, `chat_stream`, `stream`, `embed`, `rerank`, `list_models`
- `stream()` ayrıntılı akış: düşünme metni (`reasoning`), araç çağrıları, `finish_reason` ve son parçada token kullanımı
- `ChatCompletionResult` alanları: `reasoning`, `tool_calls`, `finish_reason`, `id`, `meta` (`request_id`, `routed_model`, `credits_remaining`, kalan istek/token limiti)
- Otomatik tekrar: 429/502/503/504 ve bağlantı hatalarında `Retry-After` uyumlu bekleme (`max_retries`, varsayılan 2)
- Akışsız çağrılarda otomatik `Idempotency-Key`: tekrar denenen istek geçitte tekilleştirilir, çift ücretlendirilmez
- `EvrenLLMError` ve alt sınıfları (`LLMRateLimitError`, `LLMNotFoundError`, `LLMAuthenticationError`, `LLMServiceError` ...): `code`, `type`, `param`, `request_id`, `retry_after`, `evren` alanları; mevcut `RateLimitError`, `NotFoundError` vb. ile uyumlu
- `EVREN_LLM_API_KEY` / `EVREN_LLM_BASE_URL` ortam değişkenleri
- `EvrenClient.predict_sybil()` ve `SybilRiskResult`: LDCT akciğer kanseri risk çıkarımı

### Fixed
- Düşünen modellerde `content` alanı `None` dönebiliyordu; artık her zaman `str`
- Akış `finish_reason="error"` ile ya da `[DONE]` gelmeden kesildiğinde eksik yanıt tamamlanmış gibi dönüyordu; artık `LLMServiceError(code="stream_interrupted")` fırlatılır
- SSE ayrıştırıcı `\r\n` ayraçlı ve çok satırlı `data:` çerçevelerini kaçırıyordu
- LLM isteklerinde varsayılan okuma zaman aşımı 120 sn'den 600 sn'ye çıkarıldı (uzun düşünme yanıtları)

### Changed
- Sürüm tek kaynaktan okunur (`evren_sdk/_version.py`)

## [0.7.0] - 2026-04-12

### Added
- True pipeline modu: video dosyalari icin de capture/encode/predict tam paralel calisir
- WS multi-frame in-flight: send/recv ayri thread, GPU hic bos kalmaz (3 frame kuyrukta)
- `InferenceWSClient.start_pipeline()` / `stop_pipeline()` / `submit()` / `next_result()` API
- `record()` metodu da WS pipeline destekliyor

### Changed
- `_hud()` alpha blend kaldirildi — frame.copy() yerine direkt cizim (2-3ms kazanc)
- Encode islemi worker thread'e tasindi — main thread sadece draw+yield yapar
- HTTP modu da pipeline: encode+predict ayri thread'de, draw main'de

### Removed
- Eski `_stream_sequential` ve `_stream_pipeline` metodlari — tek birlesik pipeline ile degistirildi

## [0.6.1] - 2026-04-12

### Fixed
- Video stream bitişinde frame kaybı — sentinel pattern ile queue drenajı
- `_grab` thread'de `queue.Full` exception yakalanmıyor sorunu
- `_is_normalized` artık frame boyutunu da parametre olarak alır (1x1 edge case)
- `_stream_pipeline` ilk frame'de blocking wait ile sonuç garantisi
- Unused `ThreadPoolExecutor` import kaldırıldı

## [0.6.0] - 2026-04-12

### Added
- WebSocket streaming client (`InferenceWSClient`) — gateway'e persistent baglanti
- Client-side auto-resize — inference oncesi frame kucultme, payload %60 azalir
- Pipeline (double-buffer) pattern — live kaynaklarda capture/predict paralel calisir
- `EvrenCamera` `mode` parametresi: `"auto"` | `"http"` | `"ws"`
- `EvrenCamera` `resize` parametresi: client-side resize acma/kapama
- `EvrenClient.resolve_ws_params()` — model slug'dan weights_url cozumleme
- Client-side class remap (DETR modelleri icin `class_0` → gercek sinif adi)
- WS auto-reconnect (session timeout/disconnect sonrasi otomatik yeniden baglanti)

### Fixed
- `draw_predictions` normalized (0-1) koordinatlari frame boyutuna scale eder
- Mask ve keypoint koordinatlari da normalize/pixel ayrimi yapar

### Changed
- Default `jpeg_quality` 70 → 55 (inference icin yeterli, %30 daha kucuk payload)
- Default `max_fps` 15 → 30 (WS ile daha yuksek throughput mumkun)

## [0.5.3] - 2026-04-10

### Fixed
- `download_model` format alias ("pt" → "pytorch") duzeltildi
- `ModelInfo` ve `ModelVersion` dataclass'larina eksik alanlar eklendi

## [0.5.2] - 2026-04-08

### Added
- Ilk public release
- `EvrenClient` ve `AsyncEvrenClient`
- `EvrenCamera` edge modulu
- `predict`, `predict_batch`, `model_classes`, `warmup`, `benchmark`
- `download_model` (pytorch/onnx/tensorrt/tflite)
