"""EVREN Python SDK — görüntü çıkarım (nesne tespiti) VE LLM sohbet/embedding
istemcisi. `EvrenClient`/`AsyncEvrenClient` görü modelleri (`api.ssyz.org.tr`),
`EvrenLLMClient`/`AsyncEvrenLLMClient` LLM geçidi (`evren-llmapi.ssyz.org.tr`)
için - iki AYRI API-anahtarı sınıfı, karıştırılmamalı (t81)."""

from .client import AsyncEvrenClient, EvrenClient
from .exceptions import (
    AuthenticationError,
    EvrenError,
    InferenceError,
    InsufficientCreditsError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from .llm_client import AsyncEvrenLLMClient, ChatCompletionResult, ChatMessage, EvrenLLMClient
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

__version__ = "0.8.0"

__all__ = [
    "EvrenClient",
    "AsyncEvrenClient",
    "EvrenLLMClient",
    "AsyncEvrenLLMClient",
    "ChatMessage",
    "ChatCompletionResult",
    "EvrenCamera",
    "InferenceWSClient",
    "draw_predictions",
    "EvrenError",
    "AuthenticationError",
    "InsufficientCreditsError",
    "NotFoundError",
    "RateLimitError",
    "InferenceError",
    "ValidationError",
    "Prediction",
    "PredictResult",
    "BatchResult",
    "BenchmarkResult",
    "ClassInfo",
    "ModelClasses",
    "ModelInfo",
    "ModelVersion",
]

try:
    from .edge import EvrenCamera, draw_predictions
except ImportError:
    pass

try:
    from .ws_client import InferenceWSClient
except ImportError:
    pass
