import requests
from typing import Generator, Dict, Any, Optional, Union, List
from .config import Config
from .models import (
    GenerateRequest, GenerateResponse,
    ChatRequest, ChatResponse,
    CreateModelRequest, ModelResponse,
    EmbeddingRequest, EmbeddingResponse
)
from .exceptions import (
    OllamaRequestError, 
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaValidationError
)
from .logger import setup_logger
import json

logger = setup_logger(__name__)

class OllamaClient:
    def __init__(self, base_url: Optional[str] = None):
        """Initialize Ollama API client
        Args:
            base_url (str, optional): Base URL for Ollama API. Defaults to Config.OLLAMA_API_URL.
        """
        self.base_url = base_url or Config.OLLAMA_API_URL
        self.session = requests.Session()
        self.session.headers.update(Config.DEFAULT_HEADERS)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        timeout: int = Config.DEFAULT_TIMEOUT
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """Make HTTP request to Ollama API with proper error handling"""
        url = f"{self.base_url}{endpoint}"
        response = None

        try:
            logger.debug(f"Making {method} request to {url}")
            logger.debug(f"Request data: {data}")

            response = self.session.request(
                method=method,
                url=url,
                json=data,
                stream=stream,
                timeout=timeout
            )

            response.raise_for_status()

            if stream:
                return self._stream_response(response)

            try:
                return response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response JSON: {str(e)}")
                raise OllamaResponseError(f"Failed to parse response JSON: {str(e)}")

        except requests.Timeout:
            logger.error(f"Request timed out after {timeout} seconds")
            raise OllamaTimeoutError(f"Request timed out after {timeout} seconds")
        except requests.HTTPError as e:
            status_code = response.status_code if response else None
            logger.error(f"HTTP error occurred: {str(e)}")
            raise OllamaRequestError(f"HTTP error occurred: {str(e)}", status_code)
        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise OllamaRequestError(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise OllamaRequestError(f"Unexpected error: {str(e)}")

    def _stream_response(self, response: requests.Response) -> Generator[Dict[str, Any], None, None]:
        """Stream response from Ollama API with error handling"""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode('utf-8'))
                    yield json_response
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {str(e)}")
                    raise OllamaResponseError(f"Failed to parse JSON response: {str(e)}")
                except Exception as e:
                    logger.error(f"Error streaming response: {str(e)}")
                    raise OllamaResponseError(f"Error streaming response: {str(e)}")

    def generate(
        self, 
        request: GenerateRequest
    ) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """Generate completion using Ollama API
        Args:
            request (GenerateRequest): Request parameters for text generation
        Returns:
            Union[GenerateResponse, Generator[GenerateResponse, None, None]]: Generated response
        """
        try:
            if not request.model:
                raise OllamaValidationError("Model name is required")

            stream = request.options.stream if request.options else True
            response = self._make_request(
                "POST",
                Config.GENERATE_ENDPOINT,
                data=request.dict(exclude_none=True),
                stream=stream
            )

            if not stream:
                return GenerateResponse(**response)
            return (GenerateResponse(**chunk) for chunk in response)

        except Exception as e:
            logger.error(f"Generate request failed: {str(e)}")
            raise

    def chat(
        self, 
        request: ChatRequest
    ) -> Union[ChatResponse, Generator[ChatResponse, None, None]]:
        """Generate chat completion using Ollama API
        Args:
            request (ChatRequest): Chat request parameters
        Returns:
            Union[ChatResponse, Generator[ChatResponse, None, None]]: Chat response
        """
        try:
            if not request.model:
                raise OllamaValidationError("Model name is required")
            if not request.messages:
                raise OllamaValidationError("Messages are required")

            stream = request.stream if request.stream is not None else True
            response = self._make_request(
                "POST",
                Config.CHAT_ENDPOINT,
                data=request.dict(exclude_none=True),
                stream=stream
            )

            if not stream:
                return ChatResponse(**response)
            return (ChatResponse(**chunk) for chunk in response)

        except Exception as e:
            logger.error(f"Chat request failed: {str(e)}")
            raise

    def create_model(self, request: CreateModelRequest) -> Union[ModelResponse, Generator[ModelResponse, None, None]]:
        """Create a new model using Ollama API
        Args:
            request (CreateModelRequest): Model creation parameters
        Returns:
            Union[ModelResponse, Generator[ModelResponse, None, None]]: Creation response
        """
        try:
            if not request.model:
                raise OllamaValidationError("Model name is required")

            stream = request.stream if request.stream is not None else True
            response = self._make_request(
                "POST",
                Config.CREATE_MODEL_ENDPOINT,
                data=request.dict(exclude_none=True),
                stream=stream
            )

            if not stream:
                return ModelResponse(**response)
            return (ModelResponse(**chunk) for chunk in response)

        except Exception as e:
            logger.error(f"Create model request failed: {str(e)}")
            raise

    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            response = self._make_request("GET", Config.LIST_MODELS_ENDPOINT)
            if isinstance(response, Generator):
                return next(response)  # Get first response for non-streaming endpoint
            return response
        except Exception as e:
            logger.error(f"List models request failed: {str(e)}")
            raise

    def show_model(self, model_name: str) -> Dict[str, Any]:
        """Show model information"""
        try:
            if not model_name:
                raise OllamaValidationError("Model name is required")

            response = self._make_request(
                "POST",
                Config.SHOW_MODEL_ENDPOINT,
                data={"name": model_name}
            )
            if isinstance(response, Generator):
                return next(response)  # Get first response for non-streaming endpoint
            return response
        except Exception as e:
            logger.error(f"Show model request failed: {str(e)}")
            raise

    def copy_model(self, source: str, destination: str) -> ModelResponse:
        """Copy a model"""
        try:
            if not source or not destination:
                raise OllamaValidationError("Source and destination model names are required")

            response = self._make_request(
                "POST",
                Config.COPY_MODEL_ENDPOINT,
                data={"source": source, "destination": destination}
            )
            if isinstance(response, Generator):
                response = next(response)  # Get first response for non-streaming endpoint
            return ModelResponse(**response)
        except Exception as e:
            logger.error(f"Copy model request failed: {str(e)}")
            raise

    def delete_model(self, model_name: str) -> ModelResponse:
        """Delete a model"""
        try:
            if not model_name:
                raise OllamaValidationError("Model name is required")

            response = self._make_request(
                "DELETE",
                Config.DELETE_MODEL_ENDPOINT,
                data={"name": model_name}
            )
            if isinstance(response, Generator):
                response = next(response)  # Get first response for non-streaming endpoint
            return ModelResponse(**response)
        except Exception as e:
            logger.error(f"Delete model request failed: {str(e)}")
            raise

    def pull_model(self, model_name: str, stream: bool = True) -> Union[ModelResponse, Generator[ModelResponse, None, None]]:
        """Pull a model"""
        try:
            if not model_name:
                raise OllamaValidationError("Model name is required")

            response = self._make_request(
                "POST",
                Config.PULL_MODEL_ENDPOINT,
                data={"name": model_name},
                stream=stream
            )

            if not stream:
                return ModelResponse(**response)
            return (ModelResponse(**chunk) for chunk in response)

        except Exception as e:
            logger.error(f"Pull model request failed: {str(e)}")
            raise

    def push_model(self, model_name: str, stream: bool = True) -> Union[ModelResponse, Generator[ModelResponse, None, None]]:
        """Push a model"""
        try:
            if not model_name:
                raise OllamaValidationError("Model name is required")

            response = self._make_request(
                "POST",
                Config.PUSH_MODEL_ENDPOINT,
                data={"name": model_name},
                stream=stream
            )

            if not stream:
                return ModelResponse(**response)
            return (ModelResponse(**chunk) for chunk in response)

        except Exception as e:
            logger.error(f"Push model request failed: {str(e)}")
            raise

    def create_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using Ollama API
        Args:
            request (EmbeddingRequest): Embedding generation parameters
        Returns:
            EmbeddingResponse: Generated embeddings
        """
        try:
            if not request.model:
                raise OllamaValidationError("Model name is required")
            if not request.prompt:
                raise OllamaValidationError("Prompt is required")

            response = self._make_request(
                "POST",
                Config.EMBEDDINGS_ENDPOINT,
                data=request.dict(exclude_none=True)
            )
            if isinstance(response, Generator):
                response = next(response)  # Get first response for non-streaming endpoint
            return EmbeddingResponse(**response)
        except Exception as e:
            logger.error(f"Create embedding request failed: {str(e)}")
            raise