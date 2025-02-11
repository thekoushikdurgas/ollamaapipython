"""Mock server for Ollama API testing"""
import time
from typing import Dict, Any, Generator

class MockOllamaServer:
    """Mock implementation of Ollama server for testing"""
    
    def __init__(self):
        self.models = {
            "llama2": {
                "name": "llama2",
                "modified_at": "2024-02-11T10:00:00Z",
                "size": 4000000000,
                "digest": "sha256:mock123",
                "details": {
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama", "llama2"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            }
        }
    
    def generate_response(self, model: str, prompt: str, stream: bool = True) -> Generator:
        """Mock generate completion response"""
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

    def chat_response(self, model: str, messages: list, stream: bool = True) -> Generator:
        """Mock chat completion response"""
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
                yield {**response, "message": {"role": "assistant", "content": word + " "}, "done": False}
            yield {**response, "done": True}
        else:
            yield response

    def create_model(self, model: str, **kwargs) -> Generator:
        """Mock model creation response"""
        self.models[model] = {
            "name": model,
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
        yield {"status": f"Successfully created model {model}"}

    def list_models(self) -> Dict[str, Any]:
        """Mock list models response"""
        return {"models": list(self.models.values())}

    def show_model(self, model: str, verbose: bool = False) -> Dict[str, Any]:
        """Mock show model response"""
        if model not in self.models:
            raise ValueError(f"Model {model} not found")
        return self.models[model]

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
        return {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}  # Mock 5D embedding

    def get_version(self) -> Dict[str, Any]:
        """Mock version response"""
        return {"version": "0.1.0-mock"}
