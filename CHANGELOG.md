# Changelog

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
