from flask import Flask, request, jsonify
from ollama_wrapper import OllamaClient
from ollama_wrapper.models import (
    GenerateRequest, ChatRequest, CreateModelRequest,
    EmbeddingRequest, ModelOptions, Message
)
from ollama_wrapper.exceptions import (
    OllamaError, OllamaRequestError, 
    OllamaResponseError, OllamaValidationError,
    OllamaTimeoutError
)
import logging
from typing import Any, Dict, Generator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
client = OllamaClient()

def handle_streaming_response(response: Generator) -> Flask.response_class:
    """Handle streaming responses from Ollama API"""
    def generate_stream():
        try:
            for chunk in response:
                yield jsonify(chunk).get_data(as_text=True) + '\n'
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield jsonify({"error": str(e), "type": "StreamingError"}).get_data(as_text=True)
    return app.response_class(generate_stream(), mimetype='application/x-ndjson')

@app.errorhandler(OllamaError)
def handle_ollama_error(error):
    """Handle Ollama API errors"""
    if isinstance(error, OllamaRequestError):
        status_code = error.status_code or 500
        message = "Failed to connect to Ollama server. Is it running?"
    elif isinstance(error, OllamaValidationError):
        status_code = 400
        message = str(error)
    elif isinstance(error, OllamaTimeoutError):
        status_code = 504
        message = "Request to Ollama server timed out"
    else:
        status_code = 500
        message = str(error)

    error_response = {
        "error": message,
        "type": error.__class__.__name__
    }
    logger.error(f"API Error: {error_response}")
    return jsonify(error_response), status_code

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate completion endpoint"""
    try:
        data = request.get_json()
        if not data:
            raise OllamaValidationError("No JSON data provided")

        # Create generate request
        request_data = GenerateRequest(**data)
        response = client.generate(request_data)

        # Handle streaming and non-streaming responses
        if not request_data.options or request_data.options.stream:
            return handle_streaming_response(response)

        return jsonify(response.dict())

    except Exception as e:
        logger.error(f"Generate endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat completion endpoint"""
    try:
        data = request.get_json()
        if not data:
            raise OllamaValidationError("No JSON data provided")

        # Create chat request
        request_data = ChatRequest(**data)
        response = client.chat(request_data)

        # Handle streaming and non-streaming responses
        if request_data.stream:
            return handle_streaming_response(response)

        return jsonify(response.dict())

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models endpoint"""
    try:
        response = client.list_models()
        return jsonify(response)
    except Exception as e:
        logger.error(f"List models endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/show/<model_name>', methods=['GET'])
def show_model(model_name):
    """Show model information endpoint"""
    try:
        response = client.show_model(model_name)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Show model endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/models', methods=['POST'])
def create_model():
    """Create model endpoint"""
    try:
        data = request.get_json()
        if not data:
            raise OllamaValidationError("No JSON data provided")

        request_data = CreateModelRequest(**data)
        response = client.create_model(request_data)

        # Handle streaming and non-streaming responses
        if request_data.stream:
            return handle_streaming_response(response)

        return jsonify(response.dict())

    except Exception as e:
        logger.error(f"Create model endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/models/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """Delete model endpoint"""
    try:
        response = client.delete_model(model_name)
        return jsonify(response.dict())
    except Exception as e:
        logger.error(f"Delete model endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/models/copy', methods=['POST'])
def copy_model():
    """Copy model endpoint"""
    try:
        data = request.get_json()
        if not data:
            raise OllamaValidationError("No JSON data provided")

        source = data.get('source')
        destination = data.get('destination')
        if not source or not destination:
            raise OllamaValidationError("Source and destination model names are required")

        response = client.copy_model(source, destination)
        return jsonify(response.dict())
    except Exception as e:
        logger.error(f"Copy model endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/models/pull', methods=['POST'])
def pull_model():
    """Pull model endpoint"""
    try:
        data = request.get_json()
        if not data:
            raise OllamaValidationError("No JSON data provided")

        model_name = data.get('name')
        if not model_name:
            raise OllamaValidationError("Model name is required")

        stream = data.get('stream', True)
        response = client.pull_model(model_name, stream=stream)

        if stream:
            return handle_streaming_response(response)
        return jsonify(response.dict())
    except Exception as e:
        logger.error(f"Pull model endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/models/push', methods=['POST'])
def push_model():
    """Push model endpoint"""
    try:
        data = request.get_json()
        if not data:
            raise OllamaValidationError("No JSON data provided")

        model_name = data.get('name')
        if not model_name:
            raise OllamaValidationError("Model name is required")

        stream = data.get('stream', True)
        response = client.push_model(model_name, stream=stream)

        if stream:
            return handle_streaming_response(response)
        return jsonify(response.dict())
    except Exception as e:
        logger.error(f"Push model endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/embeddings', methods=['POST'])
def create_embedding():
    """Create embeddings endpoint"""
    try:
        data = request.get_json()
        if not data:
            raise OllamaValidationError("No JSON data provided")

        request_data = EmbeddingRequest(**data)
        response = client.create_embedding(request_data)
        return jsonify(response.dict())

    except Exception as e:
        logger.error(f"Create embedding endpoint error: {str(e)}")
        return handle_ollama_error(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)