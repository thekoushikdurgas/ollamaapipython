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
logger = setup_logger(__name__)
from .utils import validate_model_name
from .mock_server import MockOllamaServer
from .rate_limiter import RateLimiter


class AsyncOllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        use_mock: Optional[bool] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        rate_limit_requests: int = 10,
        rate_limit_capacity: int = 10,
        pool_connections: int = 100,
        pool_keepalive: int = 30,
        pool_timeout: float = 10.0
    ):
        """Initialize Async Ollama API client
        Args:
            base_url (str, optional): Base URL for Ollama API. Defaults to Config.OLLAMA_API_URL.
            use_mock (bool, optional): Force use of mock server. Defaults to None (uses env var).
            max_retries (int): Maximum number of retry attempts for failed requests
            retry_delay (float): Initial delay between retries in seconds (doubles with each retry)
            rate_limit_requests (int): Number of requests allowed per second
            rate_limit_capacity (int): Maximum burst capacity for rate limiter
            pool_connections (int): Maximum number of connections to keep in pool
            pool_keepalive (int): Keep alive timeout for pooled connections in seconds
            pool_timeout (float): Timeout for acquiring a connection from pool
        """
        self.base_url = base_url or Config.OLLAMA_API_URL
        self.use_mock = use_mock if use_mock is not None else os.getenv('USE_MOCK_OLLAMA', '').lower() == 'true'
        self.session = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limiter = RateLimiter()

        # Connection pool settings
        self.pool_connections = pool_connections
        self.pool_keepalive = pool_keepalive
        self.pool_timeout = pool_timeout

        # Configure rate limiters for different endpoints
        self._configure_rate_limiters(rate_limit_requests, rate_limit_capacity)

        if self.use_mock:
            logger.info("Using mock Ollama server for development/testing")
            self.mock_server = MockOllamaServer()
            self.mock_server.list_models()
        logger.info(f"Initialized Async Ollama client with base URL: {self.base_url}")

    def _configure_rate_limiters(self, requests_per_second: int, capacity: int):
        """Configure rate limiters for different endpoints"""
        # Configure more aggressive rate limiting for resource-intensive endpoints
        self.rate_limiter.get_bucket(Config.GENERATE_ENDPOINT, requests_per_second, capacity)
        self.rate_limiter.get_bucket(Config.CHAT_ENDPOINT, requests_per_second, capacity)
        # Less aggressive rate limiting for lightweight endpoints
        self.rate_limiter.get_bucket(Config.VERSION_ENDPOINT, requests_per_second * 2, capacity * 2)
        self.rate_limiter.get_bucket(Config.LIST_MODELS_ENDPOINT, requests_per_second * 2, capacity * 2)

    async def __aenter__(self):
        """Create session for async context manager with connection pooling"""
        if not self.use_mock:
            conn = aiohttp.TCPConnector(
                limit=self.pool_connections,
                ttl_dns_cache=300,
                keepalive_timeout=self.pool_keepalive
            )
            timeout = aiohttp.ClientTimeout(
                total=self.pool_timeout,
                connect=self.pool_timeout
            )
            self.session = aiohttp.ClientSession(
                headers=Config.DEFAULT_HEADERS,
                connector=conn,
                timeout=timeout
            )
            logger.debug(f"Created connection pool with {self.pool_connections} connections")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup session and connection pool"""
        if self.session:
            try:
                await self.session.close()
                logger.debug("Closed connection pool and cleaned up resources")
            except Exception as e:
                logger.error(f"Error closing session: {str(e)}")

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
            return self.mock_server.create_embedding(
                data.get('model', ''),
                data.get('prompt', '')
            )
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

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        timeout: int = Config.DEFAULT_TIMEOUT
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Make async HTTP request to Ollama API with proper error handling and retries"""
        if self.use_mock:
            return await self._handle_mock_request(method, endpoint, data, stream)

        # Apply rate limiting
        await self.rate_limiter.acquire(endpoint)

        retry_count = 0
        last_error = None

        while retry_count <= self.max_retries:
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession(headers=Config.DEFAULT_HEADERS)

                url = f"{self.base_url}{endpoint}"
                logger.debug(f"Making async {method} request to {url} (attempt {retry_count + 1})")

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

                    return await response.json()

            except asyncio.TimeoutError as e:
                last_error = OllamaTimeoutError(
                    f"Request to {url} timed out after {timeout} seconds. "
                    "Please check if Ollama server is running and responsive."
                )
            except aiohttp.ClientConnectorError as e:
                last_error = OllamaRequestError(
                    f"Failed to connect to Ollama server at {self.base_url}. "
                    "Please ensure Ollama is installed and running. "
                    "Visit https://ollama.ai/download for installation instructions.",
                    status_code=503
                )
            except Exception as e:
                last_error = OllamaRequestError(f"Unexpected error: {str(e)}")

            retry_count += 1
            if retry_count <= self.max_retries:
                wait_time = self.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                logger.warning(f"Request failed, retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Request failed after {self.max_retries} retries")
                raise last_error

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
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/models") as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch models: {response.status}")
                        return {"models": []}
                    
                    data = await response.json()
                    models = data.get("models", [])
                    logger.info(f"List models: {len(models)}")
                    return {"models": models}
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

    async def embeddings(
        self,
        request: EmbeddingRequest
    ) -> Union[EmbeddingResponse, AsyncGenerator[EmbeddingResponse, None]]:
        """Generate embeddings using Ollama API asynchronously
        Args:
            request (EmbeddingRequest): Request parameters for embedding generation
        Returns:
            Union[EmbeddingResponse, AsyncGenerator[EmbeddingResponse, None]]: Embedding response
        """
        try:
            if not request.model:
                raise OllamaValidationError("Model name is required")
            if not request.prompt:
                raise OllamaValidationError("Prompt is required")

            # Validate model name format
            request.model = validate_model_name(request.model)

            stream = request.stream if request.stream is not None else False
            response = await self._make_request(
                "POST",
                Config.EMBEDDINGS_ENDPOINT,
                data=request.dict(exclude_none=True),
                stream=stream
            )

            if not stream:
                return EmbeddingResponse(**response)

            async def response_generator():
                async for chunk in response:
                    yield EmbeddingResponse(**chunk)
            return response_generator()

        except Exception as e:
            logger.error(f"Embeddings request failed: {str(e)}")
            raise

    async def show_model(self, model_name: str) -> Dict[str, Any]:
        """Show details for a specific model asynchronously"""
        try:
            model_name = validate_model_name(model_name)
            return await self._make_request("GET", f"{Config.SHOW_MODEL_ENDPOINT}/{model_name}")
        except Exception as e:
            logger.error(f"Show model request failed: {str(e)}")
            raise