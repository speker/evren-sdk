"""EVREN Python SDK — görüntü çıkarım (nesne tespiti) VE LLM geçidi istemcisi
(sohbet, embedding, rerank, tipli karar, OCR, ses transkripsiyonu). `EvrenClient`/
`AsyncEvrenClient` görü modelleri (`api.ssyz.org.tr`), `EvrenLLMClient`/
`AsyncEvrenLLMClient` LLM geçidi (`evren-llmapi.ssyz.org.tr`) için - iki AYRI
API-anahtarı sınıfı, karıştırılmamalı."""

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
from .llm_media import encode_image
from .llm_tasks import (
    DecisionBatchResult,
    DecisionResult,
    LLMModel,
    OcrResult,
    QuotaStatus,
    RequestStatus,
    ResponseResult,
    Transcription,
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
    "AsyncEvrenClient",
    "AsyncEvrenLLMClient",
    "AuthenticationError",
    "BatchResult",
    "BenchmarkResult",
    "ChatCompletionResult",
    "ChatDelta",
    "ChatMessage",
    "ClassInfo",
    "DecisionBatchResult",
    "DecisionResult",
    "EvrenCamera",
    "EvrenClient",
    "EvrenError",
    "EvrenLLMClient",
    "EvrenLLMError",
    "InferenceError",
    "InferenceWSClient",
    "InsufficientCreditsError",
    "LLMAuthenticationError",
    "LLMConflictError",
    "LLMInsufficientCreditsError",
    "LLMModel",
    "LLMNotFoundError",
    "LLMRateLimitError",
    "LLMServiceError",
    "LLMValidationError",
    "ModelClasses",
    "ModelInfo",
    "ModelVersion",
    "NotFoundError",
    "OcrResult",
    "PredictResult",
    "Prediction",
    "QuotaStatus",
    "RateLimitError",
    "RequestStatus",
    "RerankResult",
    "ResponseMeta",
    "ResponseResult",
    "SybilRiskResult",
    "Transcription",
    "ValidationError",
    "__version__",
    "draw_predictions",
    "encode_image",
]

try:
    from .edge import EvrenCamera, draw_predictions
except ImportError:
    pass

try:
    from .ws_client import InferenceWSClient
except ImportError:
    pass
