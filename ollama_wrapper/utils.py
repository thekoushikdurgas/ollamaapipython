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
    """Validate model name format according to Ollama conventions
    Model names follow a model:tag format where:
    - model can have an optional namespace (e.g., example/model)
    - tag is optional and defaults to 'latest'
    Examples: orca-mini:3b-q4_1, llama3:70b
    Args:
        model_name (str): The model name to validate
    Returns:
        str: Validated model name with tag (using 'latest' if not provided)
    """
    if not model_name:
        raise ValueError("Model name cannot be empty")

    # If no tag is provided, append :latest
    if ":" not in model_name:
        return f"{model_name}:latest"

    return model_name