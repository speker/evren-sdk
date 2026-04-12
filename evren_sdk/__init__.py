"""EVREN Python SDK — nesne tespiti ve goruntu cikarim istemcisi."""

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

__version__ = "0.6.0"

__all__ = [
    "EvrenClient",
    "AsyncEvrenClient",
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
