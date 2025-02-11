from .client import OllamaClient
from .exceptions import OllamaError, OllamaRequestError, OllamaResponseError
from .models import (
    GenerateRequest,
    GenerateResponse,
    ChatRequest,
    ChatResponse,
    CreateModelRequest,
    ModelResponse,
    EmbeddingRequest,
    EmbeddingResponse
)

__version__ = "1.0.0"
__all__ = [
    "OllamaClient",
    "OllamaError",
    "OllamaRequestError", 
    "OllamaResponseError",
    "GenerateRequest",
    "GenerateResponse",
    "ChatRequest", 
    "ChatResponse",
    "CreateModelRequest",
    "ModelResponse",
    "EmbeddingRequest",
    "EmbeddingResponse"
]
