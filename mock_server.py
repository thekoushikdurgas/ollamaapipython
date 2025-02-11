
"""Mock server for testing the Ollama API"""
import time
from typing import Dict, Any, Generator, Optional
from flask import Flask, request, jsonify
from ollama_wrapper.config import Config
from .logger import setup_logger

logger = setup_logger(__name__)

class MockOllamaServer:
    """Mock implementation of Ollama server for testing purposes"""

    def __init__(self):
        """Initialize mock server with empty models dictionary"""
        self.models = {}

    def _simulate_stream_delay(self, words: list, response: dict) -> Generator[Dict[str, Any], None, None]:
        """Helper method to simulate streaming response with delay"""
        for word in words:
            time.sleep(0.1)  # Simulate network delay
            yield {**response, "response": word + " ", "done": False}
        yield {**response, "done": True}

    def generate_response(self, model: str, prompt: str, stream: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Generate mock completion response"""
        if not model or not prompt:
            raise ValueError("Model and prompt are required")

        response = {
            "model": model,
            "created_at": "2024-02-11T10:00:00Z",
            "response": f"This is a mock response for: {prompt}",
            "done": True
        }

        if stream:
            yield from self._simulate_stream_delay(response["response"].split(), response)
        else:
            yield response

    def chat_response(self, model: str, messages: list, stream: bool = True) -> Generator[Dict[str, Any], None, None]:
        """Generate mock chat completion response"""
        if not model or not messages:
            raise ValueError("Model and messages are required")

        response = {
            "model": model,
            "created_at": "2024-02-11T10:00:00Z",
            "message": {"role": "assistant", "content": "This is a mock chat response"},
            "done": True
        }

        if stream:
            words = response["message"]["content"].split()
            for word in words:
                time.sleep(0.1)
                yield {**response, "message": {"role": "assistant", "content": word + " "}, "done": False}
            yield {**response, "done": True}
        else:
            yield response

    def create_model(self, model_name: str, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Create a mock model"""
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
        """List available mock models"""
        try:
            models_info = [{
                'name': model_name,
                'size': f"{(model_data.get('size', 0) / 1024 / 1024):.2f} MB",
                'format': model_data['details'].get('format'),
                'family': model_data['details'].get('family'),
                'parameter_size': model_data['details'].get('parameter_size'),
                'quantization_level': model_data['details'].get('quantization_level')
            } for model_name, model_data in self.models.items()]
            
            logger.info(f"Listed {len(models_info)} models")
            return {"models": models_info}
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            raise

    def show_model(self, model_name: str) -> Dict[str, Any]:
        """Show mock model details"""
        if not model_name:
            raise ValueError("Model name is required")
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        return self.models[model_name]

    def delete_model(self, model: str) -> Dict[str, Any]:
        """Delete a mock model"""
        if model in self.models:
            del self.models[model]
        return {"status": "success"}

    def create_embedding(self, model: str, prompt: str) -> Dict[str, Any]:
        """Create mock embedding"""
        if not model or not prompt:
            raise ValueError("Model and prompt are required")
        return {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}

    def get_version(self) -> Dict[str, Any]:
        """Get mock version info"""
        return {"version": "0.1.0-mock"}
