from typing import Dict, Any
import base64
import json

def encode_image(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def parse_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate response from Ollama API"""
    if "error" in response:
        return {"status": "error", "error": response["error"]}
    return response

def format_duration(duration_ns: int) -> float:
    """Convert duration from nanoseconds to seconds"""
    return duration_ns / 1e9 if duration_ns else 0

def validate_model_name(model_name: str) -> str:
    """Validate model name format"""
    if not model_name or ":" not in model_name:
        return f"{model_name}:latest"
    return model_name
