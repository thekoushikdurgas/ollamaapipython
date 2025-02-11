"""Mock server for Ollama API testing"""
import time
from typing import Dict, Any, Generator, Optional
# from flask import Flask, request, jsonify
# from ollama_wrapper import logger
from ollama_wrapper.config import Config
from .logger import setup_logger

logger = setup_logger(__name__)


class MockOllamaServer:
    """Mock implementation of Ollama server for testing"""

    def __init__(self):
        self.models = {}

    def generate_response(
            self,
            model: str,
            prompt: str,
            stream: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Mock generate completion response"""
        if not model or not prompt:
            raise ValueError("Model and prompt are required")

        response = {
            "model": model,
            "created_at": "2024-02-11T10:00:00Z",
            "response": f"This is a mock response for: {prompt}",
            "done": True
        }

        if stream:
            # Simulate streaming response
            words = response["response"].split()
            for word in words:
                time.sleep(0.1)  # Simulate delay
                yield {**response, "response": word + " ", "done": False}
            yield {**response, "done": True}
        else:
            yield response

    def chat_response(
            self,
            model: str,
            messages: list,
            stream: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Mock chat completion response"""
        if not model or not messages:
            raise ValueError("Model and messages are required")

        response = {
            "model": model,
            "created_at": "2024-02-11T10:00:00Z",
            "message": {
                "role": "assistant",
                "content": "This is a mock chat response"
            },
            "done": True
        }

        if stream:
            words = response["message"]["content"].split()
            for word in words:
                time.sleep(0.1)
                yield {
                    **response, "message": {
                        "role": "assistant",
                        "content": word + " "
                    },
                    "done": False
                }
            yield {**response, "done": True}
        else:
            yield response

    def create_model(self, model_name: str,
                     **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Mock model creation response"""
        if not model_name:
            raise ValueError("Model name is required")

        self.models[model_name] = {
            "name": model_name,
            "modified_at": "2024-02-11T10:00:00Z",
            "size": 4000000000,
            "digest": "sha256:mock123",
            "details": {
                "format": "gguf",
                "family": "custom",
                "parameter_size": "7B",
                "quantization_level": kwargs.get("quantize", "Q4_0")
            }
        }
        yield {"status": f"Successfully created model {model_name}"}

    def list_models(self) -> Dict[str, Any]:
        """Mock list models response"""
        try:
            response = requests.get(f"{Config.OLLAMA_API_URL}/api/models")
            if not response.ok:
                logger.error(f"Failed to fetch models: {response.status_code}")
                return {"models": []}
            
            data = response.json()
            models = data.get("models", [])
            logger.info(f"List models: {len(models)}")
            return {"models": models}
        except Exception as e:
            logger.error(f"List models request failed: {str(e)}")
            raise

    def show_model(self, model_name: str) -> Dict[str, Any]:
        """Mock show model response"""
        if not model_name:
            raise ValueError("Model name is required")
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        return self.models[model_name]

    def delete_model(self, model: str) -> Dict[str, Any]:
        """Mock delete model response"""
        if model in self.models:
            del self.models[model]
        return {"status": "success"}

    def copy_model(self, source: str, destination: str) -> Dict[str, Any]:
        """Mock copy model response"""
        if source not in self.models:
            raise ValueError(f"Source model {source} not found")
        self.models[destination] = {**self.models[source], "name": destination}
        return {"status": "success"}

    def pull_model(self, name: str, stream: bool = True) -> Generator:
        """Mock pull model response"""
        steps = ["downloading", "verifying", "extracting", "completed"]
        for step in steps:
            time.sleep(0.5)
            yield {"status": f"{step} model {name}"}

    def push_model(self, name: str, stream: bool = True) -> Generator:
        """Mock push model response"""
        steps = ["preparing", "uploading", "verifying", "completed"]
        for step in steps:
            time.sleep(0.5)
            yield {"status": f"{step} model {name}"}

    def create_embedding(self, model: str, prompt: str) -> Dict[str, Any]:
        """Mock embedding response"""
        if not model or not prompt:
            raise ValueError("Model and prompt are required")
        return {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}  # Mock 5D embedding

    def get_version(self) -> Dict[str, Any]:
        """Mock version response"""
        return {"version": "0.1.0-mock"}