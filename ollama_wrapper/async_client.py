"""Async client implementation for Ollama API"""
import aiohttp
import asyncio
import os
from typing import AsyncGenerator, Dict, Any, Optional, Union
import json
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
from .utils import validate_model_name
from .mock_server import MockOllamaServer

logger = setup_logger(__name__)

class AsyncOllamaClient:
    def __init__(self, base_url: Optional[str] = None, use_mock: Optional[bool] = None):
        """Initialize Async Ollama API client
        Args:
            base_url (str, optional): Base URL for Ollama API. Defaults to Config.OLLAMA_API_URL.
            use_mock (bool, optional): Force use of mock server. Defaults to None (uses env var).
        """
        self.base_url = base_url or Config.OLLAMA_API_URL
        self.use_mock = use_mock if use_mock is not None else os.getenv('USE_MOCK_OLLAMA', '').lower() == 'true'
        self.session = None
        if self.use_mock:
            logger.info("Using mock Ollama server for development/testing")
            self.mock_server = MockOllamaServer()
        logger.info(f"Initialized Async Ollama client with base URL: {self.base_url}")

    async def _handle_mock_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Handle requests in mock mode"""
        if not data:
            data = {}

        if endpoint == Config.GENERATE_ENDPOINT:
            mock_response = self.mock_server.generate_response(
                data.get('model', ''),
                data.get('prompt', ''),
                stream=data.get('stream', True)
            )
        elif endpoint == Config.CHAT_ENDPOINT:
            mock_response = self.mock_server.chat_response(
                data.get('model', ''),
                data.get('messages', []),
                stream=data.get('stream', True)
            )
        elif endpoint == Config.CREATE_MODEL_ENDPOINT:
            mock_response = self.mock_server.create_model(data.get('model', ''), **data)
        elif endpoint == Config.LIST_MODELS_ENDPOINT:
            return self.mock_server.list_models()
        elif endpoint == Config.SHOW_MODEL_ENDPOINT:
            return self.mock_server.show_model(data.get('name', ''))
        elif endpoint == Config.EMBEDDINGS_ENDPOINT:
            return self.mock_server.create_embedding(data.get('model', ''), data.get('prompt', ''))
        elif endpoint == Config.VERSION_ENDPOINT:
            return self.mock_server.get_version()
        else:
            raise OllamaRequestError(f"Mock server does not support endpoint: {endpoint}")

        # Convert sync generator to async generator for streaming responses
        if stream:
            async def async_generator():
                for item in mock_response:
                    await asyncio.sleep(0.1)  # Simulate network delay
                    yield item
            return async_generator()

        # For non-streaming responses, get first item from generator
        return next(mock_response)

    async def __aenter__(self):
        """Create session for async context manager"""
        if not self.use_mock:
            self.session = aiohttp.ClientSession(headers=Config.DEFAULT_HEADERS)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup session for async context manager"""
        if self.session:
            await self.session.close()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        timeout: int = Config.DEFAULT_TIMEOUT
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Make async HTTP request to Ollama API with proper error handling"""
        if self.use_mock:
            return self._handle_mock_request(method, endpoint, data, stream)

        if not self.session:
            self.session = aiohttp.ClientSession(headers=Config.DEFAULT_HEADERS)

        url = f"{self.base_url}{endpoint}"
        try:
            logger.debug(f"Making async {method} request to {url}")
            if data:
                logger.debug(f"Request data: {json.dumps(data, indent=2)}")

            async with self.session.request(
                method=method,
                url=url,
                json=data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if stream:
                    return self._stream_response(response)

                if response.status >= 400:
                    error_data = await response.json()
                    error_msg = error_data.get("error", f"HTTP {response.status} error occurred")
                    raise OllamaRequestError(error_msg, status_code=response.status)

                try:
                    return await response.json()
                except aiohttp.ContentTypeError as e:
                    logger.error(f"Failed to parse response JSON: {str(e)}")
                    raise OllamaResponseError(f"Failed to parse response JSON: {str(e)}")

        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout} seconds")
            raise OllamaTimeoutError(
                f"Request to {url} timed out after {timeout} seconds. "
                "Please check if Ollama server is running and responsive."
            )
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error: {str(e)}")
            raise OllamaRequestError(
                f"Failed to connect to Ollama server at {self.base_url}. "
                "Please ensure Ollama is installed and running. "
                "Visit https://ollama.ai/download for installation instructions.",
                status_code=503
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise OllamaRequestError(f"Unexpected error: {str(e)}")

    async def _stream_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from Ollama API with error handling"""
        try:
            async for line in response.content:
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

    async def generate(
        self,
        request: GenerateRequest
    ) -> Union[GenerateResponse, AsyncGenerator[GenerateResponse, None]]:
        """Generate completion using Ollama API asynchronously
        Args:
            request (GenerateRequest): Request parameters for text generation
        Returns:
            Union[GenerateResponse, AsyncGenerator[GenerateResponse, None]]: Generated response
        """
        try:
            if not request.model:
                raise OllamaValidationError("Model name is required")

            # Validate model name format
            request.model = validate_model_name(request.model)

            stream = request.options.stream if request.options else True
            response = await self._make_request(
                "POST",
                Config.GENERATE_ENDPOINT,
                data=request.dict(exclude_none=True),
                stream=stream
            )

            if not stream:
                return GenerateResponse(**response)

            async def response_generator():
                async for chunk in response:
                    yield GenerateResponse(**chunk)
            return response_generator()

        except Exception as e:
            logger.error(f"Generate request failed: {str(e)}")
            raise

    async def chat(
        self,
        request: ChatRequest
    ) -> Union[ChatResponse, AsyncGenerator[ChatResponse, None]]:
        """Generate chat completion using Ollama API asynchronously
        Args:
            request (ChatRequest): Chat request parameters
        Returns:
            Union[ChatResponse, AsyncGenerator[ChatResponse, None]]: Chat response
        """
        try:
            if not request.model:
                raise OllamaValidationError("Model name is required")
            if not request.messages:
                raise OllamaValidationError("Messages are required")

            # Validate model name format
            request.model = validate_model_name(request.model)

            stream = request.stream if request.stream is not None else True
            response = await self._make_request(
                "POST",
                Config.CHAT_ENDPOINT,
                data=request.dict(exclude_none=True),
                stream=stream
            )

            if not stream:
                return ChatResponse(**response)

            async def response_generator():
                async for chunk in response:
                    yield ChatResponse(**chunk)
            return response_generator()

        except Exception as e:
            logger.error(f"Chat request failed: {str(e)}")
            raise

    async def create_model(
        self,
        request: CreateModelRequest
    ) -> Union[ModelResponse, AsyncGenerator[ModelResponse, None]]:
        """Create a new model using Ollama API asynchronously
        Args:
            request (CreateModelRequest): Model creation parameters
        Returns:
            Union[ModelResponse, AsyncGenerator[ModelResponse, None]]: Creation response
        """
        try:
            if not request.model:
                raise OllamaValidationError("Model name is required")

            # Validate model names format
            request.model = validate_model_name(request.model)
            if request.from_model:
                request.from_model = validate_model_name(request.from_model)

            stream = request.stream if request.stream is not None else True
            response = await self._make_request(
                "POST",
                Config.CREATE_MODEL_ENDPOINT,
                data=request.dict(exclude_none=True),
                stream=stream
            )

            if not stream:
                return ModelResponse(**response)

            async def response_generator():
                async for chunk in response:
                    yield ModelResponse(**chunk)
            return response_generator()

        except Exception as e:
            logger.error(f"Create model request failed: {str(e)}")
            raise

    async def list_models(self) -> Dict[str, Any]:
        """List available models asynchronously"""
        try:
            return await self._make_request("GET", Config.LIST_MODELS_ENDPOINT)
        except Exception as e:
            logger.error(f"List models request failed: {str(e)}")
            raise

    async def list_running_models(self) -> Dict[str, Any]:
        """List running models asynchronously"""
        try:
            return await self._make_request("GET", Config.RUNNING_MODELS_ENDPOINT)
        except Exception as e:
            logger.error(f"List running models request failed: {str(e)}")
            raise

    async def get_version(self) -> Dict[str, Any]:
        """Get Ollama version information asynchronously"""
        try:
            return await self._make_request("GET", Config.VERSION_ENDPOINT)
        except Exception as e:
            logger.error(f"Version request failed: {str(e)}")
            raise