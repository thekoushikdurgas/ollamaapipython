from .client import OllamaClient
from .async_client import AsyncOllamaClient
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
    "AsyncOllamaClient",
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