# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.5.1] - 2026-03-09

### Added
- `InsufficientCreditsError` exception for HTTP 402 (insufficient credit balance).
- `e.required` and `e.available` fields on credit errors for programmatic handling.

### Fixed
- HTTP 402 responses were not caught by SDK — fell through to generic `httpx.HTTPStatusError`.

## [0.5.0] - 2026-03-09

### Added
- `PredictResult.filter()` — filter predictions by confidence range or class names.
- `PredictResult.to_yolo()` — export predictions in YOLO annotation format.
- `PredictResult.to_coco()` — export predictions in COCO annotation format.
- `PredictResult.to_csv()` — export predictions as CSV string.
- `PredictResult.save()` — save results to file (JSON, CSV, or YOLO txt).
- `Prediction.to_dict()` — dictionary serialization.
- `client.benchmark()` — measure inference latency and throughput (avg, min, max, p95, FPS).
- `client.download_model()` — download trained model weights (ONNX or PyTorch).
- `client.submit_for_review()` — send images with pre-annotations to a dataset for active learning.
- `BenchmarkResult` dataclass for performance measurement data.

## [0.4.0] - 2026-03-15

### Added
- `EvrenCamera` for real-time camera/video/RTSP inference on GPU-free edge devices.
- `EvrenCamera.stream()` — iterator yielding annotated frames with predictions.
- `EvrenCamera.run()` — one-liner OpenCV window with HUD overlay (FPS, latency, count).
- `EvrenCamera.scan()` — batch process image folders with optional annotated output.
- `EvrenCamera.record()` — process video and save annotated output file.
- `draw_predictions()` — standalone annotation renderer (bbox, OBB, keypoints, masks).
- Pipeline threading: capture thread runs parallel to inference, zero frame lag.
- Smart frame dropping: always processes the latest frame, never backlogs.
- HUD overlay: semi-transparent status bar with detection count, latency, FPS.
- `pip install evren-sdk[edge]` optional dependency group for OpenCV.

## [0.3.0] - 2026-03-14

### Added
- `predict_batch()` for GPU batch inference on multiple images.
- `warmup()` to pre-load models onto GPU and eliminate cold-start latency.
- `model_classes()` to retrieve class names, colors, and architecture info.
- `list_models()` and `list_versions()` for model discovery.
- `AsyncEvrenClient` with full async/await support via `httpx.AsyncClient`.
- `py.typed` marker for PEP 561 type-checker support.
- Custom exceptions: `AuthenticationError`, `NotFoundError`, `RateLimitError`, `ValidationError`, `InferenceError`.
- Model slug resolution with version tags (`owner/slug:v2.0`).

### Changed
- All dataclasses use `slots=True` for lower memory footprint.
- `Prediction.__repr__` shows class name and confidence.
- `BatchResult` is iterable and supports `len()`.

## [0.2.0] - 2026-03-10

### Added
- Single image prediction via `predict()`.
- API key and JWT token authentication.

## [0.1.0] - 2026-03-08

### Added
- Initial release with basic client structure.
