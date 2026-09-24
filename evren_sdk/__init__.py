"""EVREN Python SDK — görüntü çıkarım (nesne tespiti) VE LLM sohbet/embedding
istemcisi. `EvrenClient`/`AsyncEvrenClient` görü modelleri (`api.ssyz.org.tr`),
`EvrenLLMClient`/`AsyncEvrenLLMClient` LLM geçidi (`evren-llmapi.ssyz.org.tr`)
için - iki AYRI API-anahtarı sınıfı, karıştırılmamalı."""

from ._version import __version__
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
from .llm_async_client import AsyncEvrenLLMClient
from .llm_client import EvrenLLMClient
from .llm_errors import (
    EvrenLLMError,
    LLMAuthenticationError,
    LLMConflictError,
    LLMInsufficientCreditsError,
    LLMNotFoundError,
    LLMRateLimitError,
    LLMServiceError,
    LLMValidationError,
)
from .llm_types import ChatCompletionResult, ChatDelta, ChatMessage, RerankResult, ResponseMeta
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

__all__ = [
    "__version__",
    "EvrenClient",
    "AsyncEvrenClient",
    "EvrenLLMClient",
    "AsyncEvrenLLMClient",
    "ChatMessage",
    "ChatCompletionResult",
    "ChatDelta",
    "RerankResult",
    "ResponseMeta",
    "EvrenCamera",
    "InferenceWSClient",
    "draw_predictions",
    "EvrenError",
    "EvrenLLMError",
    "LLMAuthenticationError",
    "LLMConflictError",
    "LLMInsufficientCreditsError",
    "LLMNotFoundError",
    "LLMRateLimitError",
    "LLMServiceError",
    "LLMValidationError",
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
    "SybilRiskResult",
]

try:
    from .edge import EvrenCamera, draw_predictions
except ImportError:
    pass

try:
    from .ws_client import InferenceWSClient
except ImportError:
    pass
