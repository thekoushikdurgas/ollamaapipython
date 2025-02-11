"""Mock server for Ollama API testing"""
import time
import requests
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
        return {
            "models": [
                {
                    "name": "codellama:13b",
                    "modified_at": "2024-02-11T10:00:00Z",
                    "size": 7365960935,
                    "digest": "9f438cb9cd581fc025612d27f7c1a6669ff83a8bb0ed86c94fcf4c5440555697",
                    "details": {
                        "format": "gguf",
                        "family": "llama",
                        "families": None,
                        "parameter_size": "13B",
                        "quantization_level": "Q4_0"
                    }
                },
                {
                    "name": "llama3:latest",
                    "modified_at": "2024-02-11T10:00:00Z",
                    "size": 3825819519,
                    "digest": "fe938a131f40e6f6d40083c9f0f430a515233eb2edaa6d72eb85c50d64f2300e",
                    "details": {
                        "format": "gguf",
                        "family": "llama",
                        "families": None,
                        "parameter_size": "7B",
                        "quantization_level": "Q4_0"
                    }
                }
            ]
        }

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
