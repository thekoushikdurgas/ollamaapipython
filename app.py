import os
os.environ['USE_MOCK_OLLAMA'] = 'true'  # Enable mock mode by default

from flask import Flask, request, jsonify, send_from_directory, render_template
from ollama_wrapper import OllamaClient
from ollama_wrapper.models import (
    GenerateRequest, ChatRequest, CreateModelRequest,
    EmbeddingRequest, ModelOptions, Message,
    ModelCopyRequest, ModelPullRequest, ModelPushRequest, ShowModelRequest
)
from ollama_wrapper.exceptions import (
    OllamaError, OllamaRequestError, 
    OllamaResponseError, OllamaValidationError,
    OllamaTimeoutError
)
from ollama_wrapper.utils import validate_model_name
import logging
import hashlib
from typing import Any, Dict, Generator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True # Enable template reloading
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max-limit for file uploads
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
        if "Failed to connect to Ollama server" in str(error):
            message = (
                "Failed to connect to Ollama server. "
                "Please ensure Ollama is installed and running. "
                "Visit https://ollama.ai/download for installation instructions."
            )
        else:
            message = str(error)
    elif isinstance(error, OllamaValidationError):
        status_code = 400
        message = str(error)
    elif isinstance(error, OllamaTimeoutError):
        status_code = 504
        message = (
            "Request to Ollama server timed out. "
            "Please check if the server is running and responsive."
        )
    else:
        status_code = 500
        message = str(error)

    error_response = {
        "error": message,
        "type": error.__class__.__name__,
        "status_code": status_code
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

        # Validate model name format
        if 'model' in data:
            data['model'] = validate_model_name(data['model'])

        # Handle format options
        if isinstance(data.get('format'), dict):
            # Validate JSON schema format
            if 'type' not in data['format']:
                raise OllamaValidationError("Invalid JSON schema format: missing 'type' field")
        elif data.get('format') == 'json':
            # Ensure prompt indicates JSON response is expected
            if 'prompt' in data and 'json' not in data['prompt'].lower():
                data['prompt'] += " Respond using JSON"

        # Create generate request
        request_data = GenerateRequest(**data)
        response = client.generate(request_data)

        # Handle streaming and non-streaming responses
        if request_data.stream:
            return handle_streaming_response(response)
        return jsonify(response)

    except Exception as e:
        logger.error(f"Generate endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/blobs/<digest>', methods=['POST'])
def upload_blob(digest):
    """Handle blob uploads for model files"""
    try:
        if not request.data:
            raise OllamaValidationError("No file data provided")

        # Verify the digest matches the file content
        file_hash = hashlib.sha256(request.data).hexdigest()
        expected_hash = digest.replace('sha256:', '')

        if file_hash != expected_hash:
            raise OllamaValidationError("File hash does not match provided digest")

        response = client.upload_blob(digest, request.data)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Blob upload error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/show/<model_name>', methods=['GET'])
def show_model(model_name):
    """Show model information endpoint"""
    try:
        model_name = validate_model_name(model_name)
        verbose = request.args.get('verbose', '').lower() == 'true'

        request_data = ShowModelRequest(
            model=model_name,
            verbose=verbose
        )
        response = client.show_model(request_data)
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

        # Validate model names
        if 'model' in data:
            data['model'] = validate_model_name(data['model'])
        if 'from' in data:
            data['from'] = validate_model_name(data['from'])

        # Create model request with all parameters
        request_data = CreateModelRequest(**data)
        response = client.create_model(request_data)

        # Handle streaming response
        if request_data.stream:
            return handle_streaming_response(response)
        return jsonify(response)

    except Exception as e:
        logger.error(f"Create model endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/models/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """Delete model endpoint"""
    try:
        model_name = validate_model_name(model_name)
        response = client.delete_model(model_name)
        return jsonify(response)
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

        request_data = ModelCopyRequest(**data)
        response = client.copy_model(request_data)
        return jsonify(response)
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

        request_data = ModelPullRequest(**data)
        response = client.pull_model(request_data)

        if request_data.stream:
            return handle_streaming_response(response)
        return jsonify(response)
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

        request_data = ModelPushRequest(**data)
        response = client.push_model(request_data)

        if request_data.stream:
            return handle_streaming_response(response)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Push model endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat completion endpoint"""
    try:
        data = request.get_json()
        if not data:
            raise OllamaValidationError("No JSON data provided")

        # Validate model name format
        if 'model' in data:
            data['model'] = validate_model_name(data['model'])

        # Create chat request
        request_data = ChatRequest(**data)
        response = client.chat(request_data)

        # Handle streaming and non-streaming responses
        if request_data.stream:
            return handle_streaming_response(response)

        return jsonify(response)

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

@app.route('/api/running', methods=['GET'])
def list_running_models():
    """List running models endpoint"""
    try:
        response = client.list_running_models()
        return jsonify(response)
    except Exception as e:
        logger.error(f"List running models endpoint error: {str(e)}")
        return handle_ollama_error(e)


@app.route('/api/embeddings', methods=['POST'])
def create_embedding():
    """Create embeddings endpoint"""
    try:
        data = request.get_json()
        if not data:
            raise OllamaValidationError("No JSON data provided")

        # Validate model name format
        if 'model' in data:
            data['model'] = validate_model_name(data['model'])

        request_data = EmbeddingRequest(**data)
        response = client.create_embedding(request_data)
        return jsonify(response)

    except Exception as e:
        logger.error(f"Create embedding endpoint error: {str(e)}")
        return handle_ollama_error(e)

@app.route('/api/version', methods=['GET'])
def get_version():
    """Version endpoint"""
    try:
        response = client.get_version()
        return jsonify(response)
    except Exception as e:
        logger.error(f"Version endpoint error: {str(e)}")
        return handle_ollama_error(e)

# Add static files handling
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/')
def index():
    """Render the main UI"""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)